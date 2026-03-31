import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer


def _resolve_db_path(db_path: str) -> Path:
    resolved_path = Path(db_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Chroma DB path not found: {resolved_path}")

    sqlite_path = resolved_path / "chroma.sqlite3"
    if not sqlite_path.exists():
        raise FileNotFoundError(f"Chroma metadata file not found: {sqlite_path}")

    return resolved_path


def _list_collection_names(client) -> list[str]:
    try:
        return [collection.name for collection in client.list_collections()]
    except Exception:
        return []


def get_or_create_collection(db_path: str, collection_name: str):
    """
    Connects to ChromaDB and gets OR creates the collection.
    Used for Ingestion.
    """
    resolved_path = Path(db_path).expanduser().resolve()
    resolved_path.mkdir(parents=True, exist_ok=True)
    print(f"🔌 Connecting to DB (Ingest Mode): {collection_name} at {resolved_path}")
    client = chromadb.PersistentClient(path=str(resolved_path))
    return client.get_or_create_collection(collection_name)

def get_db_collection(db_path: str, collection_name: str):
    """
    Connects to an EXISTING collection.
    Used for Retrieval.
    """
    resolved_path = _resolve_db_path(db_path)
    print(f"🔌 Connecting to DB (Read Mode): {collection_name} at {resolved_path}")
    client = chromadb.PersistentClient(path=str(resolved_path))
    try:
        return client.get_collection(collection_name)
    except Exception as exc:
        available_collections = _list_collection_names(client)
        details = (
            f"Could not open collection '{collection_name}' at '{resolved_path}'. "
            f"Available collections: {available_collections or 'none'}. "
            f"Original error: {exc}"
        )
        print(f"⚠️ {details}")
        raise RuntimeError(details) from exc

def load_embedding_model(model_name: str):
    print(f"🔄 Loading Embedding Model: {model_name}")
    return SentenceTransformer(model_name)