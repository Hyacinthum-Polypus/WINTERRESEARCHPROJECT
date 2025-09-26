import os
import sys
import hashlib
import mimetypes
from datetime import datetime, timezone
from pathlib import Path

from arango import ArangoClient
import pymupdf4llm

from dotenv import load_dotenv

load_dotenv()

def get_env(name: str, default: str | None = None) -> str:
    val = os.getenv(name)
    return val if val is not None else (default if default is not None else "")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_db_and_collection(client: ArangoClient, db_name: str, username: str, password: str, collection_name: str):
    sys_db = client.db("_system", username=username, password=password)
    if not sys_db.has_database(db_name):
        sys_db.create_database(db_name)
        print(f"Created database: {db_name}")

    db = client.db(db_name, username=username, password=password)
    if not db.has_collection(collection_name):
        coll = db.create_collection(collection_name)
        print(f"Created collection: {collection_name}")
        # Try to add a unique index on sha256_pdf to prevent duplicates per source file
        try:
            coll.add_persistent_index(["sha256_pdf"], unique=True)
        except Exception as e:
            # Older server/python-arango combos may use hash index helper
            try:
                coll.add_hash_index(["sha256_pdf"], unique=True)
            except Exception:
                print(f"Note: could not create unique index on sha256_pdf: {e}")
    else:
        coll = db.collection(collection_name)
    return db, coll


def upsert_document(db, collection_name: str, doc: dict) -> tuple[bool, str | None]:
    # Idempotent insert: check by sha256 first
    aql = f"""
    FOR d IN {collection_name}
        FILTER d.sha256_pdf == @sha
        RETURN d._key
    """
    cursor = db.aql.execute(aql, bind_vars={"sha": doc["sha256_pdf"]})
    existing_keys = list(cursor)

    if existing_keys:
        # Update metadata and markdown (re-extraction may improve output)
        key = existing_keys[0]
        db.collection(collection_name).update({
            "_key": key,
            "filename": doc.get("filename"),
            "size": doc.get("size"),
            "modified_ts": doc.get("modified_ts"),
            "content_type": doc.get("content_type"),
            "path_relative": doc.get("path_relative"),
            "ingested_ts": doc.get("ingested_ts"),
            "md_text": doc.get("md_text"),
            "md_chars": doc.get("md_chars"),
        })
        return False, key
    else:
        meta = db.collection(collection_name).insert(doc)
        return True, meta.get("_key") if isinstance(meta, dict) else None


def ingest_directory(
    directory: Path,
    collection_name: str = "documents_markdown",
):
    hosts = get_env("ARANGO_HOSTS", "http://arangodb:8529")
    root_user = get_env("ARANGO_ROOT_USER", "root")
    root_password = get_env("ROOT_DB_PASSWORD", "")
    db_name = get_env("ARANGO_DB_NAME", "winterresearch")

    if not root_password:
        print("ERROR: ROOT_DB_PASSWORD is not set.")
        sys.exit(1)

    client = ArangoClient(hosts=hosts)

    try:
        db, coll = ensure_db_and_collection(
            client, db_name=db_name, username=root_user, password=root_password, collection_name=collection_name
        )
    except Exception as e:
        print(f"Failed to connect/prepare ArangoDB: {e}")
        sys.exit(2)

    if not directory.exists() or not directory.is_dir():
        print(f"ERROR: Directory not found: {directory}")
        sys.exit(3)

    files = sorted(p for p in directory.iterdir() if p.is_file())
    if not files:
        print(f"No files to ingest in {directory}")
        return

    print(f"Found {len(files)} files in {directory}")
    inserted = 0
    updated = 0
    errors = 0

    for path in files:
        try:
            stat = path.stat()
            sha = sha256_file(path)
            ctype, _ = mimetypes.guess_type(path.name)
            ctype = ctype or "application/octet-stream"

            # Only process PDFs into Markdown
            if ctype != "application/pdf" and path.suffix.lower() != ".pdf":
                print(f"Skip non-PDF: {path.name} ({ctype})")
                continue

            # Extract Markdown from PDF
            md_text = pymupdf4llm.to_markdown(str(path))
            md_chars = len(md_text) if isinstance(md_text, str) else 0

            doc = {
                "filename": path.name,
                "path_relative": str(path.relative_to(directory.parent) if directory.parent in path.parents else str(path)),
                "size": stat.st_size,
                "modified_ts": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "sha256_pdf": sha,
                "content_type": "application/pdf",
                "md_text": md_text,
                "md_chars": md_chars,
                "ingested_ts": datetime.now(tz=timezone.utc).isoformat(),
            }

            is_new, key = upsert_document(db, coll.name, doc)
            if is_new:
                inserted += 1
                print(f"Inserted: {path.name} (key={key})")
            else:
                updated += 1
                print(f"Up-to-date: {path.name} (key={key})")
        except Exception as e:
            errors += 1
            print(f"ERROR ingesting {path.name}: {e}")

    print(
        f"Done. Inserted={inserted}, Updated={updated}, Errors={errors}, Total={len(files)} into collection '{coll.name}'."
    )


if __name__ == "__main__":
    # Optional args: directory [collection]
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("raw_documents")
    collection_name = sys.argv[2] if len(sys.argv) > 2 else "documents"
    ingest_directory(directory, collection_name)
