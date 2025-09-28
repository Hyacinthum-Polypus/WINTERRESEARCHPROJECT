import os
import sys
import re
import hashlib
from dataclasses import dataclass
from datetime import datetime
import itertools
from typing import Iterable, Optional, Tuple, Dict, Any, List, Set

from arango import ArangoClient
from arango.graph import Graph
from dotenv import load_dotenv

import json
import logging
import requests

from litellm import completion as litellm_completion

from kg_gen import KGGen
from kg_gen.models import Graph as KGGraph

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None

# Quiet DSPy JSON adapter warnings when we fall back to legacy adapters.
logging.getLogger("dspy.adapters.json_adapter").setLevel(logging.ERROR)

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


def ensure_graph(db, cols: GraphCollections, *, fresh: bool = True, debug: bool = False) -> Graph:
    if fresh and db.has_graph(cols.graph_name):
        if debug:
            print(f"Resetting existing graph '{cols.graph_name}' for a fresh run…")
        try:
            db.delete_graph(cols.graph_name, ignore_missing=True, drop_collections=True)
        except Exception as exc:
            print(f"ERROR: Unable to delete existing graph '{cols.graph_name}': {exc}")
            raise

    if fresh:
        # Ensure we don't keep stale collections when re-initializing the graph.
        for cname in (cols.vertex_col, cols.edge_col):
            if db.has_collection(cname):
                if debug:
                    print(f"Deleting stale collection '{cname}' before rebuild")
                try:
                    db.delete_collection(cname, ignore_missing=True)
                except Exception as exc:
                    print(f"ERROR: Unable to delete existing collection '{cname}': {exc}")
                    raise
    elif debug:
        print(f"Reusing existing graph '{cols.graph_name}' and related collections")

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
        # Unique hash id is inherent via _key; also add non-unique index on 'label' for search/display
        try:
            # New API (avoids deprecation warning)
            v.add_index({"type": "persistent", "fields": ["label"]})
        except Exception:
            # Fallback to legacy helpers
            try:
                v.add_persistent_index(["label"])  # ignore if already exists
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
        try:
            e.add_index({"type": "persistent", "fields": ["relation"]})
        except Exception:
            try:
                e.add_persistent_index(["relation"])
            except Exception:
                pass
    except Exception:
        pass

    # Ensure graph
    if not db.has_graph(cols.graph_name):
        g = db.create_graph(cols.graph_name)
        print(f"Created graph: {cols.graph_name}")
    else:
        g = db.graph(cols.graph_name)

    # Ensure edge definition exists (fresh runs will re-create it automatically)
    if cols.edge_col not in [ed["edge_collection"] for ed in g.edge_definitions()]:
        g.create_edge_definition(
            edge_collection=cols.edge_col,
            from_vertex_collections=[cols.vertex_col],
            to_vertex_collections=[cols.vertex_col],
        )

    return g


# -----------------------------
# LLM setup
# -----------------------------

def configure_llm_env() -> None:
    # Configure LiteLLM to route requests through OpenRouter by default.
    _base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    # Explicitly set the OpenRouter base URL so downstream helpers can find it
    os.environ.setdefault("OPENROUTER_BASE_URL", _base_url)
    os.environ.setdefault("LITELLM_BASE_URL", _base_url)
    os.environ.setdefault("OPENAI_API_BASE", _base_url)
    os.environ.setdefault("OPENAI_BASE_URL", _base_url)

    # API key priority: explicit OpenRouter key, then LiteLLM keys, then OpenAI-compatible fallback.
    _api_key = (
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("LITELLM_API_KEY")
        or os.getenv("LITELLM_MASTER_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if _api_key:
        os.environ.setdefault("OPENAI_API_KEY", _api_key)

    # Optional OpenRouter headers (helps comply with usage policy).
    site_url = os.getenv("OPENROUTER_SITE_URL")
    app_name = os.getenv("OPENROUTER_APP_NAME")
    headers = {}
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name
    if headers:
        os.environ.setdefault("LITELLM_HEADERS", json.dumps(headers))


def resolve_primary_model(model: Optional[str] = None) -> str:
    configure_llm_env()
    if model:
        return model
    env_model = os.getenv("KGGEN_MODEL")
    if env_model:
        return env_model
    # Preferred OpenRouter models (instruction-tuned for knowledge extraction).
    candidates = [
        "openrouter/x-ai/grok-4-fast:free",
        "google/gemini-2.0-flash-exp:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "openrouter/openai/gpt-oss-20b:free",
        "openrouter/qwen/qwen3-30b-a3b:free",
        "google/gemma-3-27b-it:free"
    ]
    return candidates[0]


def make_clusterer(model: str, temperature: float = 0.0) -> KGGen:
    configure_llm_env()
    cluster = KGGen(
        model=model,
        temperature=temperature,
        api_base=os.getenv("OPENROUTER_BASE_URL") or os.getenv("LITELLM_BASE_URL") or "",
        api_key=os.getenv("OPENROUTER_API_KEY")
        or os.getenv("LITELLM_MASTER_KEY")
        or os.getenv("LITELLM_API_KEY")
        or os.getenv("OPENAI_API_KEY"),
    )
    try:
        cluster.dspy.settings.configure(adapter=cluster.dspy.ChatAdapter())
    except Exception:
        pass
    cluster.config_json_adapter = True
    return cluster


def _call_openrouter(
    *,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    timeout: int,
) -> str:
    # Always prefer direct OpenRouter base if targeting an openrouter/* model
    if model.startswith("openrouter/"):
        api_base = os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    else:
        api_base = (
            os.getenv("OPENROUTER_BASE_URL")
            or os.getenv("LITELLM_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or "https://openrouter.ai/api/v1"
        )
    api_key = (
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("LITELLM_MASTER_KEY")
        or os.getenv("LITELLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    headers_env = os.getenv("LITELLM_HEADERS")
    extra_headers: Optional[Dict[str, str]] = None
    if headers_env:
        try:
            parsed = json.loads(headers_env)
            if isinstance(parsed, dict):
                extra_headers = parsed
        except Exception:
            extra_headers = None

    response = litellm_completion(
        model=model,
        messages=messages,
        timeout=timeout,
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        extra_headers=extra_headers,
    )
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("OpenRouter response missing message content")
    if not isinstance(content, str):
        raise ValueError("OpenRouter response content not a string")
    return content


def _parse_kg_json(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    for candidate in (text, repair_json(text) if repair_json else None):
        if not candidate:
            continue
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except Exception:
            continue
    # Try to locate JSON object substring
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        snippet = m.group(0)
        for candidate in (snippet, repair_json(snippet) if repair_json else None):
            if not candidate:
                continue
            try:
                result = json.loads(candidate)
                if isinstance(result, dict):
                    return result
            except Exception:
                continue
    raise ValueError("Unable to parse JSON payload from LLM response")


def extract_kg_with_model(
    *,
    model: str,
    input_text: str,
    context: str,
    temperature: float = 0.0,
    debug: bool = False,
    timeout: int = 120,
) -> Dict[str, Any]:
    system_prompt = (
        "You build concise knowledge graphs. "
        "Return only JSON with keys 'entities' (list of unique strings) and 'relations' "
        "(each object must include subject, predicate, object as strings)."
    )
    user_prompt = (
        f"Document: {context or 'unknown'}\n"
        "Extract key entities and directed relations appearing in the document. "
        "Prefer meaningful predicates in snake_case. Avoid duplicates.\n"
        "Text:\n"
        f"""{input_text}"""
    )
    content = _call_openrouter(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        timeout=timeout,
    )
    if debug:
        snippet = content[:200]
        print(f"    Raw LLM output from {model}: {snippet}{'…' if len(content) > 200 else ''}")
    data = _parse_kg_json(content)
    entities = data.get("entities") or []
    relations = data.get("relations") or []
    if not isinstance(entities, list):
        entities = []
    if not isinstance(relations, list):
        relations = []
    # Normalize relation tuples
    cleaned_relations: List[Tuple[str, str, str]] = []
    for rel in relations:
        if isinstance(rel, dict):
            subj = normalize_entity(rel.get("subject") or rel.get("source"))
            pred = normalize_entity(rel.get("predicate") or rel.get("relation"))
            obj = normalize_entity(rel.get("object") or rel.get("target"))
            if subj and pred and obj:
                cleaned_relations.append((subj, pred, obj))
        elif isinstance(rel, (list, tuple)) and len(rel) >= 3:
            subj = normalize_entity(rel[0])
            pred = normalize_entity(rel[1])
            obj = normalize_entity(rel[2])
            if subj and pred and obj:
                cleaned_relations.append((subj, pred, obj))
    cleaned_entities: List[str] = []
    for ent in entities:
        ne = normalize_entity(ent)
        if ne:
            cleaned_entities.append(ne)
    return {"entities": cleaned_entities, "relations": cleaned_relations}


def try_generate_kg(
    input_text: str,
    context: str,
    models: List[str],
    *,
    temperature: float = 0.0,
    debug: bool = False,
    timeout: int = 120,
) -> Optional[Dict[str, Any]]:
    for model in models:
        try:
            if debug:
                print(f"    Generating KG with model '{model}'")
            result = extract_kg_with_model(
                model=model,
                input_text=input_text,
                context=context,
                temperature=temperature,
                debug=debug,
                timeout=timeout,
            )
            if result.get("entities") or result.get("relations"):
                return result
        except Exception as exc:
            if debug:
                print(f"    Model '{model}' failed: {exc}")
            continue
    return None


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
# Relation fallbacks
# -----------------------------

def classify_relation_heuristic(a: str, b: str, sentence: str) -> Tuple[str, str, str]:
    """Lightweight heuristic for directional relation inference between two entities."""

    s_lc = sentence.lower()
    a_lc, b_lc = a.lower(), b.lower()
    pa = s_lc.find(a_lc)
    pb = s_lc.find(b_lc)
    if pa == -1 or pb == -1:
        return (a, "associated_with", b)

    first, second = (a, b) if pa <= pb else (b, a)
    pf, ps = (pa, pb) if pa <= pb else (pb, pa)
    seg = s_lc[pf + len(first.lower()): ps]

    forward_map = {
        "causes": "causes",
        "cause": "causes",
        "leads to": "leads_to",
        "lead to": "leads_to",
        "results in": "results_in",
        "produces": "produces",
        "increases": "increases",
        "increase": "increases",
        "raises": "increases",
        "improves": "improves",
        "boosts": "promotes",
        "promotes": "promotes",
        "drives": "drives",
        "contributes to": "contributes_to",
        "affects": "affects",
        "influences": "influences",
    }
    for phrase, pred in forward_map.items():
        if phrase in seg:
            return (first, pred, second)

    reverse_map = {
        "caused by": "causes",
        "driven by": "drives",
        "influenced by": "influences",
        "increased by": "increases",
        "reduced by": "decreases",
        "decreased by": "decreases",
        "lowered by": "decreases",
        "due to": "causes",
    }
    for phrase, pred in reverse_map.items():
        if phrase in seg:
            return (second, pred, first)

    decrease_map = {
        "reduces": "decreases",
        "reduce": "decreases",
        "decreases": "decreases",
        "decrease": "decreases",
        "lowers": "decreases",
        "mitigates": "mitigates",
        "prevents": "prevents",
    }
    for phrase, pred in decrease_map.items():
        if phrase in seg:
            return (first, pred, second)

    assoc_phrases = ["associated with", "correlated with", "linked to", "relates to"]
    if any(p in s_lc for p in assoc_phrases):
        return (first, "associated_with", second)

    if "part of" in seg or "component of" in seg:
        return (first, "part_of", second)

    return (first, "associated_with", second)


def _chunked(seq: List[Any], size: int) -> Iterable[List[Any]]:
    step = max(size, 1)
    for idx in range(0, len(seq), step):
        yield seq[idx : idx + step]


def infer_relations_with_llm(
    candidates: List[Dict[str, str]],
    *,
    doc_label: str,
    max_pairs: int,
    temperature: float = 0.0,
    debug: bool = False,
) -> List[Tuple[str, str, str]]:
    """LLM inference disabled; placeholder retains signature so callers remain intact."""

    if debug and candidates:
        print("    LLM fallback disabled; skipping")
    return []


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
        # Store human-readable value alongside Arango _key for display
        doc = {"_key": key, "label": label}
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
    reuse_graph: bool = False,
    debug: bool = False,
    only_unprocessed: bool = False,
    limit: Optional[int] = None,
    max_chars: Optional[int] = None,
    model: Optional[str] = None,
    chunk_size: Optional[int] = 6000,
    retries: int = 1,
    retry_models: Optional[str] = None,
    fallback_cooccurrence: bool = True,
    max_cooc_pairs: int = 300,
    debug_cooccurrence: bool = False,
    cluster_graph_enabled: bool = True,
    cluster_model: Optional[str] = None,
    llm_timeout: Optional[int] = None,
) -> None:
    cfg = get_arango_config()
    db = get_db(cfg)

    if llm_timeout is None:
        try:
            llm_timeout = int(os.getenv("OLLAMA_TIMEOUT", "120"))
        except ValueError:
            llm_timeout = 120

    # When not reusing, create fresh collection/graph names per run.
    if not reuse_graph:
        suffix = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base_graph, base_vertex, base_edge = graph_name, vertex_col, edge_col
        graph_name = f"{graph_name}_{suffix}"
        vertex_col = f"{vertex_col}_{suffix}"
        edge_col = f"{edge_col}_{suffix}"
        print(
            "Creating new graph resources:" 
            f" graph='{graph_name}' vertices='{vertex_col}' edges='{edge_col}'"
        )
        if debug:
            print(
                f"  Derived from base names graph='{base_graph}', vertex='{base_vertex}', edge='{base_edge}'"
            )

    cols = GraphCollections(graph_name=graph_name, vertex_col=vertex_col, edge_col=edge_col)
    ensure_graph(db, cols, fresh=not reuse_graph, debug=debug)

    if debug:
        print(
            "Starting KG build: source_collection="
            f"{source_collection}, reuse_graph={reuse_graph}, "
            f"fallback_cooccurrence={fallback_cooccurrence}, limit={limit if limit is not None else 'all'}"
        )
        print(
            f"  Active graph setup: graph='{graph_name}', vertices='{vertex_col}', edges='{edge_col}'"
        )
        print(f"  LLM timeout per request: {llm_timeout}s")

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

    primary_model = resolve_primary_model(model)
    fallback_models: List[str] = []
    if retry_models:
        fallback_models = [m.strip() for m in retry_models.split(",") if m.strip()]
    else:
        # Sensible defaults on OpenRouter (instruction-tuned)
        fallback_models = [
            "openrouter/x-ai/grok-4-fast:free",
            "google/gemini-2.0-flash-exp:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "openrouter/openai/gpt-oss-20b:free",
            "openrouter/qwen/qwen3-30b-a3b:free",
            "google/gemma-3-27b-it:free"
        ]
    fallback_models = [m for m in fallback_models if m != primary_model]
    models_to_try: List[str] = []
    for candidate in [primary_model] + fallback_models:
        if candidate and candidate not in models_to_try:
            models_to_try.append(candidate)
    if debug:
        print(f"Model order: {', '.join(models_to_try)}")

    clusterer = None
    clusterer_model = None
    if cluster_graph_enabled and models_to_try:
        clusterer_model = cluster_model or os.getenv("KG_CLUSTER_MODEL") or models_to_try[0]
        try:
            clusterer = make_clusterer(clusterer_model)
            if debug:
                print(f"  Clustering enabled via KGGen model '{clusterer_model}'")
        except Exception as exc:
            clusterer = None
            if debug:
                print(f"  Clustering disabled: unable to initialize KGGen ({exc})")

    processed = 0
    for doc in cursor:
        doc_key = doc["_key"]
        filename = doc.get("filename") or doc_key
        md_text = doc.get("md") or ""
        if max_chars and len(md_text) > max_chars:
            md_text = md_text[:max_chars]
            if debug:
                print(f"  Truncated document '{filename}' to {max_chars} characters")

        print(f"Processing doc key={doc_key} file='{filename}' (chars={len(md_text)}) …")
        if debug:
            sha = doc.get("sha", doc.get("sha256_pdf"))
            print(
                f"  Document details: sha={sha or 'n/a'}, retry_models={len(fallback_models)}"
            )
        kg_graph = None
        attempts = 0
        # Try full text first, then chunked retries
        kg_graph = try_generate_kg(
            md_text,
            filename,
            models_to_try,
            temperature=0.0,
            debug=debug,
            timeout=llm_timeout,
        )
        attempts += 1
        if kg_graph is None and chunk_size and len(md_text) > chunk_size:
            chunk_count_msg = f"  Falling back to chunked extraction…"
            if debug:
                print(f"{chunk_count_msg} (chunk_size={chunk_size})")
            else:
                print(chunk_count_msg)
            chunks = chunk_text(md_text, chunk_size)
            merged_entities: List[str] = []
            merged_relations: List[Tuple[str, str, str]] = []
            for idx, ch in enumerate(chunks, start=1):
                if debug:
                    print(
                        f"    Chunk {idx}/{len(chunks)} (chars={len(ch)})"
                    )
                if retries and attempts >= (1 + retries * len(chunks)):
                    break
                sub = try_generate_kg(
                    ch,
                    f"{filename} (part {idx}/{len(chunks)})",
                    models_to_try,
                    temperature=0.0,
                    debug=debug,
                    timeout=llm_timeout,
                )
                attempts += 1
                if sub is None:
                    if debug:
                        print(f"    Chunk {idx} returned no graph")
                    continue
                merged_entities.extend(sub.get("entities", []) or [])
                merged_relations.extend(sub.get("relations", []) or [])
            if merged_entities or merged_relations:
                kg_graph = {"entities": merged_entities, "relations": merged_relations}
                if debug:
                    print(
                        f"  Chunk merge produced entities={len(merged_entities)} relations={len(merged_relations)}"
                    )
        if kg_graph is None:
            print(f"  LLM extraction failed for {doc_key}: all attempts exhausted.")
            if debug:
                print("  Skipping document due to extraction failure")
            continue

        # Collect entities
        raw_entities = kg_graph.get("entities", []) or []
        raw_relations = kg_graph.get("relations", []) or []
        normalized_entities: List[str] = []
        for e in raw_entities:
            ne = normalize_entity(e)
            if ne:
                normalized_entities.append(ne)
        normalized_relations: List[Tuple[str, str, str]] = []
        for r in raw_relations:
            tr = normalize_relation(r)
            if tr:
                normalized_relations.append(tr)
        if debug:
            print(
                f"  LLM output summary: raw_entities={len(normalized_entities)} raw_relations={len(normalized_relations)}"
            )

        if cluster_graph_enabled and clusterer and (normalized_entities or normalized_relations):
            try:
                kg_input = KGGraph(
                    entities=set(normalized_entities),
                    edges={rel[1] for rel in normalized_relations},
                    relations=set(normalized_relations),
                )
                clustered = clusterer.cluster(kg_input, context=filename)
                if debug:
                    print(
                        "  Clustered entities {:d}→{:d}, relations {:d}→{:d}".format(
                            len(normalized_entities),
                            len(clustered.entities),
                            len(normalized_relations),
                            len(clustered.relations),
                        )
                    )
                normalized_entities = list(clustered.entities)
                normalized_relations = list(clustered.relations)
            except Exception as exc:
                if debug:
                    print(f"  Clustering failed: {exc}")
        # Unique-ify while preserving case (dedupe by lowercase)
        seen_lc = set()
        uniq_entities: List[str] = []
        for label in normalized_entities:
            lc = label.lower()
            if lc in seen_lc:
                continue
            seen_lc.add(lc)
            uniq_entities.append(label)

        # Persist vertices
        key_by_label = upsert_entities(db, vertex_col, uniq_entities)

        # Collect relations (triples)
        triples: List[Tuple[str, str, str]] = list(normalized_relations)

        # Co-occurrence fallback if still empty
        if not triples and fallback_cooccurrence and uniq_entities:
            if debug and not debug_cooccurrence:
                print("  Triggering co-occurrence fallback (enable --debug-cooc for details)")
            sent_split = re.split(r"(?<=[\.!?])\s+|\n{2,}", md_text)
            uniq_lc_map = {e.lower(): e for e in uniq_entities}
            uniq_lc = list(uniq_lc_map.keys())

            candidates: List[Dict[str, str]] = []
            heuristics_by_pair: Dict[frozenset[str], Tuple[str, str, str]] = {}
            pair_seen: Set[frozenset[str]] = set()
            sentence_considered = 0
            sentence_with_pairs = 0

            for sent in sent_split:
                if max_cooc_pairs and len(candidates) >= max_cooc_pairs:
                    break
                sentence = sent.strip()
                if not sentence:
                    continue
                s_lc = sentence.lower()
                present: List[str] = []
                for lc in uniq_lc:
                    if lc in s_lc:
                        present.append(uniq_lc_map[lc])
                        if len(present) >= 12:
                            break
                sentence_considered += 1
                if len(present) < 2:
                    continue
                sentence_with_pairs += 1
                for a, b in itertools.combinations(sorted(present, key=str.lower), 2):
                    pair_key = frozenset({a.lower(), b.lower()})
                    if pair_key in pair_seen:
                        continue
                    pair_seen.add(pair_key)
                    heuristics_by_pair[pair_key] = classify_relation_heuristic(a, b, sentence)
                    candidates.append({
                        "entity_a": a,
                        "entity_b": b,
                        "sentence": sentence,
                    })
                    if max_cooc_pairs and len(candidates) >= max_cooc_pairs:
                        break
                if max_cooc_pairs and len(candidates) >= max_cooc_pairs:
                    break

            if max_cooc_pairs and max_cooc_pairs <= len(triples):
                candidates = []

            if debug_cooccurrence:
                print(
                    "  Co-occurrence fallback: sentences considered="
                    f"{sentence_considered}, with_pairs={sentence_with_pairs}, candidates={len(candidates)}, existing_triples={len(triples)}"
                )

            added_pairs: Set[Tuple[str, str]] = set()
            llm_pairs: Set[frozenset[str]] = set()
            llm_added = 0
            heur_added = 0
            remaining_budget = max_cooc_pairs - len(triples) if max_cooc_pairs else None
            max_pairs = remaining_budget if remaining_budget is not None else len(candidates)

            if candidates and (remaining_budget is None or remaining_budget > 0):
                if debug_cooccurrence:
                    print(
                        f"  Co-occurrence fallback: invoking LLM with up to {max_pairs if max_pairs else len(candidates)} pair(s)"
                    )
                llm_relations = infer_relations_with_llm(
                    candidates,
                    doc_label=filename or doc_key,
                    max_pairs=max_pairs,
                    temperature=0.0,
                    debug=debug_cooccurrence,
                )
                for subj, pred, obj in llm_relations:
                    key = (subj.lower(), obj.lower())
                    if key in added_pairs:
                        continue
                    pair_key = frozenset({subj.lower(), obj.lower()})
                    llm_pairs.add(pair_key)
                    triples.append((subj, pred, obj))
                    added_pairs.add(key)
                    llm_added += 1
                    if remaining_budget is not None and len(triples) >= max_cooc_pairs:
                        break

            if candidates and (remaining_budget is None or len(triples) < max_cooc_pairs):
                if debug_cooccurrence and not llm_added:
                    print("  Co-occurrence fallback: falling back to heuristic relations")
                for pair_key, heur in heuristics_by_pair.items():
                    if pair_key in llm_pairs:
                        continue
                    subj, pred, obj = heur
                    key = (subj.lower(), obj.lower())
                    if key in added_pairs:
                        continue
                    triples.append((subj, pred, obj))
                    added_pairs.add(key)
                    heur_added += 1
                    if remaining_budget is not None and len(triples) >= max_cooc_pairs:
                        break

            if llm_added:
                print(f"  LLM co-occurrence fallback added {llm_added} relation(s)")
            elif heur_added:
                print(f"  Heuristic co-occurrence fallback added {heur_added} relation(s)")
            elif debug_cooccurrence and not candidates:
                print("  Co-occurrence fallback: no candidate pairs available")

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
        if debug:
            print(
                f"  Normalized entities={len(uniq_entities)} relations={len(triples)} before persistence"
            )

        if uniq_entities or triples:
            try:
                coll.update({"_key": doc_key, "kg_processed_ts": db.time(), "kg_entities_count": len(key_by_label), "kg_relations_count": len(triples)})
            except Exception:
                pass

        processed += 1
        print(f"  Done: {filename} -> entities={len(key_by_label)} relations={len(triples)}")

    print(
        f"All done. Processed {processed} document(s). "
        f"Graph '{graph_name}' with collections '{vertex_col}'/'{edge_col}' is ready."
    )


# -----------------------------
# CLI
# -----------------------------

def parse_args(argv: List[str]) -> Dict[str, Any]:
    import argparse

    p = argparse.ArgumentParser(description="Build a knowledge graph in ArangoDB from Markdown documents using local LLM extraction.")
    p.add_argument("--source", dest="source_collection", default=get_env("ARANGO_DOCS_COLLECTION", "documents"),
                   help="ArangoDB collection holding documents with md_text (default: documents)")
    p.add_argument("--graph", dest="graph_name", default=get_env("ARANGO_GRAPH_NAME", "knowledge_graph"),
                   help="Name of the ArangoDB graph to create/use")
    p.add_argument("--vcol", dest="vertex_col", default=get_env("ARANGO_VERTEX_COLLECTION", "kg_entities"),
                   help="Vertex collection name")
    p.add_argument("--ecol", dest="edge_col", default=get_env("ARANGO_EDGE_COLLECTION", "kg_edges"),
                   help="Edge collection name")
    p.add_argument("--reuse-graph", action="store_true", help="Skip resetting the graph/collections before processing")
    p.add_argument("--debug", action="store_true", help="Enable verbose debug logging for processing flow")
    p.add_argument("--only-unprocessed", action="store_true", help="Only process docs missing kg_processed_ts")
    p.add_argument("--limit", type=int, default=None, help="Max number of documents to process")
    p.add_argument("--max-chars", type=int, default=None, help="Truncate md_text to this many characters")
    p.add_argument("--model", type=str, default=None, help="Primary OpenRouter model (else KGGEN_MODEL or default openrouter/qwen2.5:14b-instruct)")
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
    p.add_argument("--debug-cooc", action="store_true", help="Enable verbose co-occurrence fallback logging")
    p.add_argument("--no-cluster", action="store_true", help="Disable KGGen-based clustering before persistence")
    p.add_argument("--cluster-model", type=str, default=os.getenv("KG_CLUSTER_MODEL", ""), help="Override model used for clustering (defaults to KG_CLUSTER_MODEL or primary model)")
    p.add_argument("--llm-timeout", type=int, default=None, help="Timeout (seconds) for each Ollama request (default 120 or OLLAMA_TIMEOUT)")

    args = p.parse_args(argv)
    return vars(args)


if __name__ == "__main__":
    kwargs = parse_args(sys.argv[1:])
    if kwargs.pop("no_cooc", False):
        kwargs["fallback_cooccurrence"] = False
    kwargs["debug"] = kwargs.pop("debug", False)
    kwargs["debug_cooccurrence"] = kwargs.pop("debug_cooc", False)
    if kwargs.pop("no_cluster", False):
        kwargs["cluster_graph_enabled"] = False
    # Normalize optional cluster model string
    cluster_model_value = kwargs.get("cluster_model")
    if isinstance(cluster_model_value, str) and not cluster_model_value.strip():
        kwargs["cluster_model"] = None
    if "max_cooc_pairs" in kwargs:
        # already present via parse_args
        pass
    process_documents(**kwargs)
