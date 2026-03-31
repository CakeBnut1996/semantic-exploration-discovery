from io_utils.load_db import load_embedding_model, get_db_collection, get_or_create_collection
from collections import defaultdict
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
    rank: float
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

def _preprocess_query(query: str, model_name: str) -> str:
    """Handles model-specific prefixes."""
    model_lower = model_name.lower()
    if "e5" in model_lower:
        return f"query: {query}"
    if "bge" in model_lower and "en-v1.5" in model_lower:
        return f"Represent this sentence for searching relevant passages: {query}"
    return query


# --- Core Functions ---

def retrieve_data(
        query: str,
        db_path: str,
        collection_name: str,
        model_name: str,
        num_docs: int = 5,
        chunks_per_doc: int = 3
) -> List[RetrievalResult]:
    """
    Retrieves data based on string parameters.
    Automatically handles loading (and caching) the DB and Model.
    """
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
    formatted_query = _preprocess_query(query, model_name)
    query_emb = encoder.encode([formatted_query], convert_to_numpy=True)

    # 3. Broad Search (Find unique sources)
    initial_results = collection.query(
        query_embeddings=query_emb,
        n_results=num_docs * chunks_per_doc * 2,
        include=["documents", "metadatas", "embeddings", "distances"]
    )

    unique_datasets = []
    if initial_results["metadatas"] and initial_results["metadatas"][0]:
        for meta in initial_results["metadatas"][0]:
            # Support 'dataset' or 'source' key
            ds_id = meta.get("dataset") if meta else None
            if ds_id and ds_id not in unique_datasets:
                unique_datasets.append(ds_id)
            if len(unique_datasets) == num_docs:
                break

    # 4. Targeted Search (Fetch chunks per specific source)
    parsed_results = []
    global_rank = 1
    for ds_id in unique_datasets:
        doc_results = collection.query(
            query_embeddings=query_emb,
            n_results=chunks_per_doc,
            where={"dataset": ds_id},
            include=["documents", "metadatas", "distances"]
        )

        if doc_results["ids"] and doc_results["ids"][0]:
            count = len(doc_results["ids"][0])
            for i in range(count):
                dist = doc_results["distances"][0][i]
                sim_score = 1.0 - dist

                parsed_results.append(RetrievalResult(
                    score=sim_score,
                    rank=global_rank,
                    chunk_text=doc_results["documents"][0][i],
                    dataset_id=ds_id,
                    metadata=doc_results["metadatas"][0][i]
                ))
                global_rank += 1
    return parsed_results


def rank_datasets(results: List[RetrievalResult]) -> List[RankedDataset]:
    """
    Sorts the retrieved results by dataset.
    """
    if not results:
        return []

    dataset_groups = defaultdict(list)
    dataset_meta = {}
    for res in results:
        dataset_groups[res.dataset_id].append({
            "score": res.score,
            "text": res.chunk_text,
            "source_url": (res.metadata or {}).get("source_url"),
            "source_title": (res.metadata or {}).get("source_title")
        })
        if res.dataset_id not in dataset_meta:
            dataset_meta[res.dataset_id] = {
                "source_url": (res.metadata or {}).get("source_url"),
                "source_title": (res.metadata or {}).get("source_title")
            }

    rankings = []
    for ds_id, chunks in dataset_groups.items():
        chunks.sort(key=lambda x: x["score"], reverse=True)
        top_score = chunks[0]["score"]
        rankings.append(RankedDataset(
            dataset_id=ds_id,
            top_score=top_score,
            source_url=dataset_meta.get(ds_id, {}).get("source_url"),
            source_title=dataset_meta.get(ds_id, {}).get("source_title"),
            top_chunks=chunks
        ))

    rankings.sort(key=lambda x: x.top_score, reverse=True)
    return rankings