import os
import sys
import re
import hashlib
from dataclasses import dataclass
import itertools
from typing import Iterable, Optional, Tuple, Dict, Any, List

from arango import ArangoClient
from arango.graph import Graph
from dotenv import load_dotenv

# KG generation
from kg_gen import KGGen
import json
import requests

load_dotenv()


# -----------------------------
# Environment helpers
# -----------------------------

def get_env(name: str, default: Optional[str] = None) -> str:
    val = os.getenv(name)
    return val if val is not None else (default if default is not None else "")


# -----------------------------
# Deterministic IDs
# -----------------------------

def stable_key(prefix: str, *parts: str, algo: str = "sha1") -> str:
    h = hashlib.new(algo)
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"|")
    return f"{prefix}_{h.hexdigest()}"


# -----------------------------
# Arango setup helpers
# -----------------------------

@dataclass
class ArangoConfig:
    hosts: str
    username: str
    password: str
    db_name: str


def get_arango_config() -> ArangoConfig:
    return ArangoConfig(
        hosts=get_env("ARANGO_HOSTS", "http://arangodb:8529"),
        username=get_env("ARANGO_ROOT_USER", "root"),
        password=get_env("ROOT_DB_PASSWORD", ""),
        db_name=get_env("ARANGO_DB_NAME", "winterresearch"),
    )


def get_db(cfg: ArangoConfig):
    if not cfg.password:
        print("ERROR: ROOT_DB_PASSWORD is not set.")
        sys.exit(1)

    client = ArangoClient(hosts=cfg.hosts)
    sys_db = client.db("_system", username=cfg.username, password=cfg.password)
    if not sys_db.has_database(cfg.db_name):
        sys_db.create_database(cfg.db_name)
        print(f"Created database: {cfg.db_name}")

    return client.db(cfg.db_name, username=cfg.username, password=cfg.password)


@dataclass
class GraphCollections:
    graph_name: str = "knowledge_graph"
    vertex_col: str = "kg_entities"
    edge_col: str = "kg_edges"


def ensure_graph(db, cols: GraphCollections) -> Graph:
    # Ensure collections
    if not db.has_collection(cols.vertex_col):
        db.create_collection(cols.vertex_col)
        print(f"Created collection: {cols.vertex_col}")
    if not db.has_collection(cols.edge_col):
        db.create_collection(cols.edge_col, edge=True)
        print(f"Created edge collection: {cols.edge_col}")

    # Ensure helpful indexes
    try:
        v = db.collection(cols.vertex_col)
        # Unique hash id is inherent via _key; also add non-unique index on 'key' for search/display
        try:
            # New API (avoids deprecation warning)
            v.add_index({"type": "persistent", "fields": ["key"]})
        except Exception:
            # Fallback to legacy helpers
            try:
                v.add_persistent_index(["key"])  # ignore if already exists
            except Exception:
                pass
    except Exception:
        pass

    try:
        e = db.collection(cols.edge_col)
        # Unique combination to prevent duplicate edges per doc
        try:
            e.add_index({
                "type": "persistent",
                "fields": ["_from", "_to", "relation", "doc_key"],
                "unique": True,
            })
        except Exception:
            # Compatibility fallback for older server/python-arango
            try:
                e.add_persistent_index(["_from", "_to", "relation", "doc_key"], unique=True)
            except Exception:
                try:
                    e.add_hash_index(["_from", "_to", "relation", "doc_key"], unique=True)
                except Exception:
                    pass
    except Exception:
        pass

    # Ensure graph
    if db.has_graph(cols.graph_name):
        g = db.graph(cols.graph_name)
        # Ensure edge definition exists
        if cols.edge_col not in [ed["edge_collection"] for ed in g.edge_definitions()]:
            g.create_edge_definition(
                edge_collection=cols.edge_col,
                from_vertex_collections=[cols.vertex_col],
                to_vertex_collections=[cols.vertex_col],
            )
    else:
        g = db.create_graph(cols.graph_name)
        g.create_edge_definition(
            edge_collection=cols.edge_col,
            from_vertex_collections=[cols.vertex_col],
            to_vertex_collections=[cols.vertex_col],
        )
        print(f"Created graph: {cols.graph_name}")

    return g


# -----------------------------
# KGGen setup
# -----------------------------

def configure_llm_env() -> None:
    # Configure OpenAI-compatible base URL for LiteLLM so KGGen routes via LiteLLM.
    _base_url = os.getenv("LITELLM_BASE_URL", "http://host.docker.internal:4000")
    os.environ.setdefault("OPENAI_API_BASE", _base_url)
    os.environ.setdefault("OPENAI_BASE_URL", _base_url)

    # API key for OpenAI-compatible clients (LiteLLM)
    _api_key = os.getenv("LITELLM_API_KEY") or os.getenv("LITELLM_MASTER_KEY")
    if _api_key:
        os.environ.setdefault("OPENAI_API_KEY", _api_key)

    # Allow LiteLLM to locate Ollama if needed
    _ollama_base = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    os.environ.setdefault("OLLAMA_BASE_URL", _ollama_base)


def get_available_ollama_models() -> List[str]:
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    url = f"{base}/api/tags"
    try:
        r = requests.get(url, timeout=3)
        r.raise_for_status()
        data = r.json()
        names = [m.get("name") or m.get("model") for m in data.get("models", [])]
        return [n for n in names if isinstance(n, str)]
    except Exception:
        return []


def make_kggen(model: Optional[str] = None, temperature: float = 0.0) -> KGGen:
    configure_llm_env()
    # Choose a good default model based on what Ollama has locally
    model = model or os.getenv("KGGEN_MODEL")
    if not model:
        avail = set(get_available_ollama_models())
        # Preference order (instruction-tuned first)
        candidates = [
            "ollama_chat/qwen2.5:14b-instruct",
            "ollama_chat/qwen2.5:7b-instruct",
            "ollama_chat/llama3.1:8b",
            "ollama_chat/gpt-oss:20b",
            "ollama_chat/llama3.2:3b",
        ]
        # Map to base tag names present in Ollama list (strip provider prefix)
        def is_present(c: str) -> bool:
            tag = c.split("/", 1)[1] if "/" in c else c
            return tag in avail
        chosen = next((c for c in candidates if is_present(c)), None)
        model = chosen or "ollama_chat/llama3.2:3b"
    # When using LiteLLM + Ollama, KGGen will rely on environment variables
    return KGGen(
        model=model,
        temperature=temperature,
        api_base=os.getenv("OLLAMA_BASE_URL", ""),
        api_key=os.getenv("LITELLM_MASTER_KEY") or os.getenv("LITELLM_API_KEY"),
    )


# -----------------------------
# Text chunking and retries
# -----------------------------

def chunk_text(text: str, max_chars: int) -> List[str]:
    if not text or not max_chars or len(text) <= max_chars:
        return [text]
    # Prefer to split on headings/paragraphs to keep chunks coherent
    parts: List[str] = re.split(r"\n(?=#+\s|\s*$)|\n\n+", text)
    chunks: List[str] = []
    buf: List[str] = []
    total = 0
    for p in parts:
        if not p:
            continue
        if total + len(p) + 1 > max_chars and buf:
            chunks.append("\n\n".join(buf))
            buf = [p]
            total = len(p)
        else:
            buf.append(p)
            total += len(p) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    # Guard: ensure no chunk exceeds limit; hard-split if needed
    fixed: List[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            fixed.append(c)
        else:
            for i in range(0, len(c), max_chars):
                fixed.append(c[i : i + max_chars])
    return fixed


def try_generate_kg(kg: KGGen, input_text: str, context: str, model_fallbacks: List[str]) -> Optional[Any]:
    """Try generating a KG; on failure, optionally retry with fallback models.

    Returns the kg_graph object or None if all attempts fail.
    """
    try:
        return kg.generate(input_data=input_text, context=context)
    except Exception:
        pass

    # Retry with fallback models
    for m in model_fallbacks:
        try:
            alt = make_kggen(model=m)
            return alt.generate(input_data=input_text, context=context)
        except Exception:
            continue
    return None


# -----------------------------
# Normalization helpers
# -----------------------------

def normalize_entity(e: Any) -> Optional[str]:
    # Expecting entity names as strings; if dict, try label/name
    if e is None:
        return None
    if isinstance(e, str):
        s = e.strip()
        return s or None
    if isinstance(e, dict):
        for k in ("label", "name", "text", "value"):
            v = e.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return str(e)


def normalize_relation(r: Any) -> Optional[Tuple[str, str, str]]:
    # Try common shapes: (s, p, o) tuple/list or dict with various keys
    try:
        if r is None:
            return None
        if isinstance(r, (list, tuple)):
            if len(r) >= 3:
                s, p, o = r[0], r[1], r[2]
            elif len(r) == 2:
                s, o = r
                p = "related_to"
            else:
                return None
            s = normalize_entity(s)
            p = normalize_entity(p)
            o = normalize_entity(o)
            if s and p and o:
                return s, p, o
            return None
        if isinstance(r, dict):
            keymap = {
                "s": ["subject", "source", "head", "from", "s"],
                "p": ["predicate", "relation", "type", "label", "p"],
                "o": ["object", "target", "tail", "to", "o"],
            }
            vals: Dict[str, Any] = {}
            for kcanon, cands in keymap.items():
                for c in cands:
                    if c in r:
                        vals[kcanon] = r[c]
                        break
            s = normalize_entity(vals.get("s"))
            p = normalize_entity(vals.get("p"))
            o = normalize_entity(vals.get("o"))
            if s and p and o:
                return s, p, o
            return None
        # Fallback to string pattern "A --R--> B"
        if isinstance(r, str):
            m = re.match(r"^\s*(.+?)\s*[-=]+>\s*(.+?)\s*$", r)
            if m:
                s = normalize_entity(m.group(1))
                o = normalize_entity(m.group(2))
                p = "related_to"
                if s and o:
                    return s, p, o
            return None
    except Exception:
        return None
    return None


# -----------------------------
# Persistence
# -----------------------------

def upsert_entities(db, vertex_col: str, labels: Iterable[str]) -> Dict[str, str]:
    col = db.collection(vertex_col)
    key_by_label: Dict[str, str] = {}
    for label in labels:
        if not label:
            continue
        key = stable_key("e", label.lower())
        # Store human-readable value under 'key' field to match requested schema
        doc = {"_key": key, "key": label}
        try:
            col.insert(doc, overwrite=True)  # upsert
        except Exception:
            try:
                col.update(doc)
            except Exception:
                pass
        key_by_label[label] = key
    return key_by_label


def upsert_edge(
    db,
    edge_col: str,
    vertex_col: str,
    from_key: str,
    to_key: str,
    relation: str,
    doc_key: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    ecol = db.collection(edge_col)
    _from = f"{vertex_col}/{from_key}"
    _to = f"{vertex_col}/{to_key}"
    ekey = stable_key("r", _from, relation, _to, doc_key)
    edge_doc = {
        "_key": ekey,
        "_from": _from,
        "_to": _to,
        "relation": relation,
        "doc_key": doc_key,
    }
    if extra:
        edge_doc.update(extra)
    try:
        ecol.insert(edge_doc, overwrite=True)
    except Exception:
        try:
            ecol.update(edge_doc)
        except Exception:
            pass


# -----------------------------
# Main processing
# -----------------------------

def process_documents(
    source_collection: str = "documents",
    graph_name: str = "knowledge_graph",
    vertex_col: str = "kg_entities",
    edge_col: str = "kg_edges",
    only_unprocessed: bool = False,
    limit: Optional[int] = None,
    max_chars: Optional[int] = None,
    model: Optional[str] = None,
    chunk_size: Optional[int] = 6000,
    retries: int = 1,
    retry_models: Optional[str] = None,
    fallback_cooccurrence: bool = True,
    max_cooc_pairs: int = 300,
) -> None:
    cfg = get_arango_config()
    db = get_db(cfg)
    cols = GraphCollections(graph_name=graph_name, vertex_col=vertex_col, edge_col=edge_col)
    ensure_graph(db, cols)

    if not db.has_collection(source_collection):
        print(f"ERROR: Source collection '{source_collection}' not found in DB '{cfg.db_name}'.")
        sys.exit(2)

    coll = db.collection(source_collection)

    # Build AQL query to stream documents
    filters = ["HAS(d, 'md_text') AND d.md_text != null AND LENGTH(d.md_text) > 0"]
    if only_unprocessed:
        filters.append("NOT HAS(d, 'kg_processed_ts')")
    filter_clause = " AND ".join(filters)
    aql = f"""
        FOR d IN {source_collection}
            FILTER {filter_clause}
            RETURN {{ _key: d._key, filename: d.filename, sha: d.sha256_pdf, md: d.md_text }}
    """
    if limit is not None:
        aql = aql.replace("RETURN", f"LIMIT {int(limit)} RETURN")

    cursor = db.aql.execute(aql, batch_size=50)

    kg = make_kggen(model=model)
    fallback_models: List[str] = []
    if retry_models:
        fallback_models = [m.strip() for m in retry_models.split(",") if m.strip()]
    else:
        # Sensible local defaults which tend to be JSON-friendly when prompted
        fallback_models = [
            "ollama_chat/qwen2.5:14b-instruct",
            "ollama_chat/llama3.1:8b",
            "ollama_chat/gpt-oss:20b",
            "ollama_chat/qwen2.5:7b-instruct",
        ]
    # Filter to models actually available locally
    avail = set(get_available_ollama_models())
    def present(m: str) -> bool:
        tag = m.split("/", 1)[1] if "/" in m else m
        return tag in avail
    fallback_models = [m for m in fallback_models if present(m)]

    processed = 0
    for doc in cursor:
        doc_key = doc["_key"]
        filename = doc.get("filename") or doc_key
        md_text = doc.get("md") or ""
        if max_chars and len(md_text) > max_chars:
            md_text = md_text[:max_chars]

        print(f"Processing doc key={doc_key} file='{filename}' (chars={len(md_text)}) …")
        kg_graph = None
        attempts = 0
        # Try full text first, then chunked retries
        kg_graph = try_generate_kg(kg, md_text, filename, fallback_models)
        attempts += 1
        if kg_graph is None and chunk_size and len(md_text) > chunk_size:
            print("  Falling back to chunked extraction…")
            chunks = chunk_text(md_text, chunk_size)
            merged_entities: List[Any] = []
            merged_relations: List[Any] = []
            for idx, ch in enumerate(chunks, start=1):
                if retries and attempts >= (1 + retries * len(chunks)):
                    break
                sub = try_generate_kg(kg, ch, f"{filename} (part {idx}/{len(chunks)})", fallback_models)
                attempts += 1
                if sub is None:
                    continue
                merged_entities.extend(getattr(sub, "entities", []) or [])
                merged_relations.extend(getattr(sub, "relations", []) or getattr(sub, "edges", []) or [])
            if merged_entities or merged_relations:
                class SimpleKG:
                    pass
                tmp = SimpleKG()
                tmp.entities = merged_entities
                tmp.relations = merged_relations
                tmp.edges = getattr(kg_graph, "edges", []) if kg_graph else []
                kg_graph = tmp
        if kg_graph is None:
            print(f"  KGGen failed for {doc_key}: all attempts exhausted.")
            continue

        # Collect entities
        raw_entities = getattr(kg_graph, "entities", []) or []
        entities: List[str] = []
        for e in raw_entities:
            ne = normalize_entity(e)
            if ne:
                entities.append(ne)
        # Unique-ify while preserving case (dedupe by lowercase)
        seen_lc = set()
        uniq_entities: List[str] = []
        for label in entities:
            lc = label.lower()
            if lc in seen_lc:
                continue
            seen_lc.add(lc)
            uniq_entities.append(label)

        # Persist vertices
        key_by_label = upsert_entities(db, vertex_col, uniq_entities)

        # Collect relations (triples)
        raw_relations = getattr(kg_graph, "relations", []) or []
        triples: List[Tuple[str, str, str]] = []
        for r in raw_relations:
            tr = normalize_relation(r)
            if tr:
                triples.append(tr)

        # As a fallback, derive from edges if relations empty
        if not triples:
            raw_edges = getattr(kg_graph, "edges", []) or []
            for ed in raw_edges:
                if isinstance(ed, (list, tuple)) and len(ed) >= 2:
                    s = normalize_entity(ed[0])
                    o = normalize_entity(ed[1])
                    p = normalize_entity(ed[2]) if len(ed) >= 3 else "related_to"
                    if s and p and o:
                        triples.append((s, p, o))
                elif isinstance(ed, dict):
                    tr = normalize_relation(ed)
                    if tr:
                        triples.append(tr)

        # Co-occurrence fallback if still empty
        if not triples and fallback_cooccurrence and uniq_entities:
            sent_split = re.split(r"(?<=[\.!?])\s+|\n{2,}", md_text)
            added_pairs = set()
            total_added = 0
            uniq_lc_map = {e.lower(): e for e in uniq_entities}
            uniq_lc = list(uniq_lc_map.keys())
            for sent in sent_split:
                if total_added >= max_cooc_pairs:
                    break
                present: List[str] = []
                s_lc = sent.lower()
                for lc in uniq_lc:
                    if lc in s_lc:
                        present.append(uniq_lc_map[lc])
                        if len(present) >= 12:
                            # avoid quadratic explosion per sentence
                            break
                if len(present) < 2:
                    continue
                for a, b in itertools.combinations(sorted(present, key=str.lower), 2):
                    key = (a.lower(), b.lower())
                    if key in added_pairs:
                        continue
                    triples.append((a, "co_occurs", b))
                    added_pairs.add(key)
                    total_added += 1
                    if total_added >= max_cooc_pairs:
                        break
            if total_added:
                print(f"  Co-occurrence fallback added {total_added} relation(s)")

        # Persist edges
        for (s, p, o) in triples:
            skey = key_by_label.get(s) or stable_key("e", s.lower())
            okey = key_by_label.get(o) or stable_key("e", o.lower())
            upsert_edge(
                db,
                edge_col=edge_col,
                vertex_col=vertex_col,
                from_key=skey,
                to_key=okey,
                relation=p,
                doc_key=doc_key,
                extra={"source_filename": filename},
            )

        # Mark source doc as processed if we extracted anything
        if uniq_entities or triples:
            try:
                coll.update({"_key": doc_key, "kg_processed_ts": db.time(), "kg_entities_count": len(key_by_label), "kg_relations_count": len(triples)})
            except Exception:
                pass

        processed += 1
        print(f"  Done: {filename} -> entities={len(key_by_label)} relations={len(triples)}")

    print(f"All done. Processed {processed} document(s).")


# -----------------------------
# CLI
# -----------------------------

def parse_args(argv: List[str]) -> Dict[str, Any]:
    import argparse

    p = argparse.ArgumentParser(description="Build a knowledge graph in ArangoDB from Markdown documents via KGGen.")
    p.add_argument("--source", dest="source_collection", default=get_env("ARANGO_DOCS_COLLECTION", "documents"),
                   help="ArangoDB collection holding documents with md_text (default: documents)")
    p.add_argument("--graph", dest="graph_name", default=get_env("ARANGO_GRAPH_NAME", "knowledge_graph"),
                   help="Name of the ArangoDB graph to create/use")
    p.add_argument("--vcol", dest="vertex_col", default=get_env("ARANGO_VERTEX_COLLECTION", "kg_entities"),
                   help="Vertex collection name")
    p.add_argument("--ecol", dest="edge_col", default=get_env("ARANGO_EDGE_COLLECTION", "kg_edges"),
                   help="Edge collection name")
    p.add_argument("--only-unprocessed", action="store_true", help="Only process docs missing kg_processed_ts")
    p.add_argument("--limit", type=int, default=None, help="Max number of documents to process")
    p.add_argument("--max-chars", type=int, default=None, help="Truncate md_text to this many characters")
    p.add_argument("--model", type=str, default=None, help="KGGen model override (else env KGGEN_MODEL)")
    p.add_argument("--chunk-size", type=int, default=6000, help="Split long docs into chunks of up to this many characters")
    p.add_argument("--retries", type=int, default=1, help="Number of additional retries per chunk with fallback models")
    p.add_argument(
        "--retry-models",
        type=str,
        default=os.getenv("KGGEN_RETRY_MODELS", ""),
        help="Comma-separated list of fallback models to try if the primary model fails",
    )
    p.add_argument("--no-cooc", action="store_true", help="Disable co-occurrence fallback when relations are empty")
    p.add_argument("--max-cooc-pairs", type=int, default=300, help="Cap on co-occurrence edges added per document")

    args = p.parse_args(argv)
    return vars(args)


if __name__ == "__main__":
    kwargs = parse_args(sys.argv[1:])
    if kwargs.pop("no_cooc", False):
        kwargs["fallback_cooccurrence"] = False
    if "max_cooc_pairs" in kwargs:
        # already present via parse_args
        pass
    process_documents(**kwargs)
