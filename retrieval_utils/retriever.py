from ingestion_utils.load_db import load_embedding_model, get_db_collection
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# --- Internal Cache to prevent reloading model every time ---
_global_cache = {
    "encoder": None,
    "model_name": None,
    "collection": None,
    "collection_name": None
}


# --- Data Models ---

class RetrievalResult(BaseModel):
    score: float
    rank: int
    chunk_text: str
    dataset_id: str
    metadata: Dict[str, Any]


class RankedDataset(BaseModel):
    dataset_id: str
    top_score: float
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    top_chunks: List[Dict[str, Any]]


# --- Helper Function ---

def _format_query_for_model(query: str, model_name: str) -> str:
    """Add only the prefix required by the selected embedding model."""
    model_lower = model_name.lower()
    if "e5" in model_lower:
        return f"query: {query}"
    if "bge" in model_lower and "en-v1.5" in model_lower:
        return f"Represent this sentence for searching relevant passages: {query}"
    return query


def _distance_to_relevance(distance: float, distance_space: str) -> float:
    """Convert a Chroma distance to a bounded, display-safe relevance score."""
    distance = max(float(distance), 0.0)
    if distance_space == "cosine":
        return max(0.0, min(1.0, 1.0 - distance))
    # Chroma defaults to squared L2 distance. Reciprocal distance is monotonic
    # and, unlike ``1 - distance``, cannot become negative.
    return 1.0 / (1.0 + distance)


# --- Core Functions ---

def retrieve_data(
        query: str,
        db_path: str,
        collection_name: str,
        model_name: str,
        num_docs: int = 5
) -> List[RetrievalResult]:
    """Return the single best-matching chunk from each selected document."""
    if not query.strip():
        return []

    # 1. Lazy Load & Cache Resources
    global _global_cache

    # Load Model if changed or not loaded
    if _global_cache["model_name"] != model_name:
        _global_cache["encoder"] = load_embedding_model(model_name)
        _global_cache["model_name"] = model_name

    # Load Collection if changed or not loaded
    if _global_cache["collection_name"] != collection_name:
        _global_cache["collection"] = get_db_collection(db_path, collection_name)
        _global_cache["collection_name"] = collection_name

    encoder = _global_cache["encoder"]
    collection = _global_cache["collection"]

    # 2. Encode Query
    formatted_query = _format_query_for_model(query, model_name)
    query_emb = encoder.encode([formatted_query], convert_to_numpy=True)

    # Query the complete collection once, then apply stable local sorting. This
    # removes dependence on approximate-index return order and on a second set
    # of per-document searches.
    collection_size = collection.count()
    if collection_size == 0:
        return []

    raw_results = collection.query(
        query_embeddings=query_emb,
        n_results=collection_size,
        include=["documents", "metadatas", "distances"]
    )

    ids = (raw_results.get("ids") or [[]])[0]
    documents = (raw_results.get("documents") or [[]])[0]
    metadatas = (raw_results.get("metadatas") or [[]])[0]
    distances = (raw_results.get("distances") or [[]])[0]
    candidates = []
    for item_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        metadata = metadata or {}
        dataset_id = metadata.get("dataset") or metadata.get("source")
        if dataset_id and document:
            candidates.append((float(distance), str(dataset_id), str(item_id), document, metadata))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    distance_space = str((getattr(collection, "metadata", None) or {}).get("hnsw:space", "l2")).lower()
    parsed_results: List[RetrievalResult] = []
    seen_datasets = set()
    for distance, dataset_id, _, document, metadata in candidates:
        if dataset_id in seen_datasets:
            continue
        seen_datasets.add(dataset_id)
        parsed_results.append(RetrievalResult(
            score=_distance_to_relevance(distance, distance_space),
            rank=len(parsed_results) + 1,
            chunk_text=document,
            dataset_id=dataset_id,
            metadata=metadata,
        ))
        if len(parsed_results) >= num_docs:
            break

    return parsed_results


def rank_datasets(results: List[RetrievalResult]) -> List[RankedDataset]:
    """Convert one-chunk retrieval results to document-level results."""
    rankings = [
        RankedDataset(
            dataset_id=result.dataset_id,
            top_score=result.score,
            source_url=result.metadata.get("source_url"),
            source_title=result.metadata.get("source_title"),
            top_chunks=[{
                "score": result.score,
                "text": result.chunk_text,
                "source_url": result.metadata.get("source_url"),
                "source_title": result.metadata.get("source_title"),
            }],
        )
        for result in results
    ]
    rankings.sort(key=lambda x: (-x.top_score, x.dataset_id))
    return rankings
