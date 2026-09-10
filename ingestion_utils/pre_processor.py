import os
from pathlib import Path
import re
from urllib.parse import urlparse
import tiktoken
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from typing import List, Any, Tuple
from ingestion_utils.load_db import load_embedding_model, get_or_create_collection


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url.strip())
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return url.strip()
    return ""


def _extract_original_url(html: str, soup: BeautifulSoup) -> str:
    # Browser-saved pages can include this marker comment.
    saved_from_match = re.search(r"saved from url=\(\d+\)(https?://[^\s\"'>]+)", html, re.IGNORECASE)
    if saved_from_match:
        candidate = _normalize_url(saved_from_match.group(1))
        if candidate:
            return candidate

    canonical = soup.find("link", rel=lambda v: v and "canonical" in " ".join(v).lower() if isinstance(v, list) else "canonical" in str(v).lower())
    if canonical and canonical.get("href"):
        candidate = _normalize_url(canonical.get("href", ""))
        if candidate:
            return candidate

    for attrs in (
        {"property": "og:url"},
        {"name": "og:url"},
        {"property": "twitter:url"},
        {"name": "twitter:url"},
    ):
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            candidate = _normalize_url(tag.get("content", ""))
            if candidate:
                return candidate

    # Local test files often have no deployable public URL. Store an empty value
    # instead of a fake/broken link; deployment can populate this metadata later.
    return ""


SKIP_LINK_TEXT = re.compile(
    r"skip\s+to\s+(?:main\s+)?content|skip\s+navigation|jump\s+to\s+content",
    re.IGNORECASE,
)


def _clean_soup(soup: BeautifulSoup) -> str:
    """Minimal cleanup for locally downloaded HTML test pages."""
    for tag in soup(["head", "script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    for control in soup.find_all(["a", "button"]):
        if SKIP_LINK_TEXT.search(control.get_text(" ", strip=True)):
            control.decompose()

    return " ".join(soup.stripped_strings)


def extract_text_and_url_from_html(path: str) -> Tuple[str, str, str]:
    if not os.path.exists(path):
        return "", "", "Untitled"

    with open(path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    original_url = _extract_original_url(html, soup)
    source_title = soup.title.get_text(strip=True) if soup.title else Path(path).stem
    return _clean_soup(soup), original_url, source_title

def extract_text_from_html(path: str) -> str:
    return extract_text_and_url_from_html(path)[0]


def chunk_text(text: str, tokenizer_name: str = "cl100k_base", max_tokens: int = 256, overlap: int = 40) -> List[str]:
    """Chunk on sentence boundaries only.

    A single sentence can exceed ``max_tokens``. In that case it is emitted as
    one oversized chunk rather than truncated, split mid-sentence, or silently
    discarded. ``overlap`` is also composed exclusively of whole sentences.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    enc = tiktoken.get_encoding(tokenizer_name)
    sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
    chunks: List[str] = []
    current: List[str] = []

    def token_count(parts: List[str]) -> int:
        return len(enc.encode(" ".join(parts)))

    def flush_current() -> None:
        nonlocal current
        if current:
            chunks.append(" ".join(current))
            current = []

    for sent in sentences:
        sent_tokens = len(enc.encode(sent))
        if sent_tokens > max_tokens:
            flush_current()
            chunks.append(sent)
            continue

        candidate_tokens = token_count([*current, sent])
        if candidate_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            overlap_sents: List[str] = []
            for s in reversed(current):
                proposed = [s, *overlap_sents]
                if token_count(proposed) <= overlap:
                    overlap_sents.insert(0, s)
                else:
                    break
            current = overlap_sents + [sent]
        else:
            current.append(sent)

    flush_current()
    return chunks


# --- Database Interaction ---

def embed_and_upsert(
    chunks: List[str],
    collection: Any,
    embedding_model: Any,
    model_name: str,
    source_filename: str,
    source_url: str,
    source_title: str
):
    if not chunks: return

    # Prefix handling for E5 models
    doc_prefix = "passage: " if "e5" in model_name.lower() else ""
    texts_to_embed = [f"{doc_prefix}{c}" for c in chunks]

    embeddings = embedding_model.encode(texts_to_embed, convert_to_numpy=True)

    ids = [f"{source_filename}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "dataset": source_filename,
            "source_url": source_url,
            "source_title": source_title
        }
        for _ in chunks
    ]

    # Replace the source as one unit so re-ingestion cannot leave stale chunks
    # behind when cleanup changes reduce the number of chunks.
    collection.delete(where={"dataset": source_filename})
    collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    print(f"   ✅ Saved {len(chunks)} chunks.")


# --- 🚀 MASTER INGESTION FUNCTION ---

def run_ingestion(
        data_dir: str,
        db_path: str,
        collection_name: str,
        embedding_model_name: str,
        tokenizer_model: str = "cl100k_base",
        chunk_size: int = 256,
        chunk_overlap: int = 40
):
    """
    Orchestrates the entire ingestion process:
    1. Initializes DB and Model.
    2. Scans directory for HTML files.
    3. Cleans, Chunks, and Embeds data.
    """

    # 1. Initialize Resources
    project_root = Path(__file__).resolve().parent.parent
    full_db_path = str((project_root / db_path).resolve()) if not os.path.isabs(db_path) else db_path
    collection = get_or_create_collection(full_db_path, collection_name)
    model = load_embedding_model(embedding_model_name)

    # 2. Find Files
    full_data_dir = str((project_root / data_dir).resolve()) if not os.path.isabs(data_dir) else data_dir

    if not os.path.exists(full_data_dir):
        print(f"❌ Error: Data directory '{full_data_dir}' not found.")
        return

    data_dir_path = Path(full_data_dir)
    files = sorted(data_dir_path.rglob("*.html"))
    print(f"\n🚀 Found {len(files)} HTML files. Starting ingestion from {full_data_dir}...\n")

    # 3. Process Loop
    for file_path in files:
        file_path = str(file_path)
        relative_path = Path(file_path).relative_to(data_dir_path)
        base_name = str(relative_path.with_suffix("")).replace("\\", "/")
        filename = relative_path.name

        print(f"📄 Processing: {filename}")

        # Local HTML is minimally cleaned during extraction, then chunked once.
        raw_text, original_url, source_title = extract_text_and_url_from_html(file_path)
        chunks = chunk_text(
            raw_text,
            tokenizer_name=tokenizer_model,
            max_tokens=chunk_size,
            overlap=chunk_overlap
        )

        # Database Upsert
        embed_and_upsert(
            chunks=chunks,
            collection=collection,
            embedding_model=model,
            model_name=embedding_model_name,
            source_filename=base_name,
            source_url=original_url,
            source_title=source_title
        )

    count = collection.count()
    print(f"\n✅ Ingestion Complete! Collection '{collection_name}' now contains {count} chunks.")


def main():
    import yaml

    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config.yaml"

    if not config_path.exists():
        print(f"❌ Config file not found at {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    active_emb_key = cfg["retrieval"]["active_embedding"]
    active_emb_model = cfg["embeddings"][active_emb_key]["model"]
    active_db_key = cfg["retrieval"]["active_db"]
    collection_name = cfg["db"][active_db_key]["collection"]

    run_ingestion(
        data_dir=cfg["data"]["data_to_db"],
        db_path=cfg["data"]["db_path"],
        collection_name=collection_name,
        embedding_model_name=active_emb_model,
        chunk_size=cfg["data"]["chunk_size"],
        chunk_overlap=cfg["data"]["chunk_overlap"]
    )


if __name__ == "__main__":
    main()
