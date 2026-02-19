import os
import re
import hashlib
import tiktoken
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from typing import List, Any
from io_utils.load_db import load_embedding_model, get_db_collection, get_or_create_collection


def extract_text_and_url_from_html(path: str):
    if not os.path.exists(path):
        return "", ""

    with open(path, "r", encoding="utf-8") as f:
        html = f.read()

    # 1. Extract Original URL from the "saved from" comment
    url_match = re.search(r'', html)
    original_url = url_match.group(1) if url_match else "Unknown Source"

    # 2. Extract Text
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.extract()

    text = soup.get_text(separator="\n", strip=True)
    return text, original_url

# --- Text Processing Functions (Same as before) ---

def extract_text_from_html(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.extract()
    return soup.get_text(separator="\n", strip=True)


def clean_text(text: str) -> str:
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\f', '', text)
    return text.strip()


def filter_noise(text: str) -> str:
    lines = text.split("\n")
    clean_lines = []
    for ln in lines:
        s = ln.strip()
        if not s: continue
        if re.match(r'^\d+[\.\)]', s): continue
        if len(s) < 30 and s.isupper(): continue
        if "REFERENCES" in s.upper() or "TABLE" in s.upper(): continue
        clean_lines.append(ln)
    return "\n".join(clean_lines)


def _deduplicate_chunks(chunks: List[str]) -> List[str]:
    seen = set()
    unique = []
    for c in chunks:
        h = hashlib.md5(c.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(c)
    return unique


def chunk_text(text: str, tokenizer_name: str = "cl100k_base", max_tokens: int = 256, overlap: int = 40) -> List[str]:
    enc = tiktoken.get_encoding(tokenizer_name)
    sentences = sent_tokenize(text)
    chunks = []
    current = []
    current_tokens = 0

    for sent in sentences:
        sent_len = len(enc.encode(sent))
        if sent_len > max_tokens: continue
        if current_tokens + sent_len > max_tokens:
            full_chunk = " ".join(current)
            chunks.append(full_chunk)
            overlap_txt = full_chunk[-overlap:] if len(full_chunk) > overlap else full_chunk
            current = [overlap_txt]
            current_tokens = len(enc.encode(overlap_txt))
        current.append(sent)
        current_tokens += sent_len

    if current: chunks.append(" ".join(current))
    return _deduplicate_chunks(chunks)


# --- Database Interaction ---

def embed_and_upsert(chunks: List[str], collection: Any, embedding_model: Any, model_name: str, source_filename: str):
    if not chunks: return

    # Prefix handling for E5 models
    doc_prefix = "passage: " if "e5" in model_name.lower() else ""
    texts_to_embed = [f"{doc_prefix}{c}" for c in chunks]

    embeddings = embedding_model.encode(texts_to_embed, convert_to_numpy=True)

    ids = [f"{source_filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"dataset": source_filename}] * len(chunks)

    collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
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
    collection = get_or_create_collection(db_path, collection_name)
    model = load_embedding_model(embedding_model_name)

    # 2. Find Files
    if not os.path.exists(data_dir):
        print(f"❌ Error: Data directory '{data_dir}' not found.")
        return

    files = [f for f in os.listdir(data_dir) if f.endswith(".html")]
    print(f"\n🚀 Found {len(files)} HTML files. Starting ingestion...\n")

    # 3. Process Loop
    for filename in files:
        file_path = os.path.join(data_dir, filename)
        base_name = os.path.splitext(filename)[0]  # e.g. "report_q3"

        print(f"📄 Processing: {filename}")

        # Pipeline: Extract -> Clean -> Filter -> Chunk
        raw_text = extract_text_from_html(file_path)
        clean_txt = clean_text(raw_text)
        filtered_txt = filter_noise(clean_txt)

        chunks = chunk_text(
            filtered_txt,
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
            source_filename=base_name
        )

    count = collection.count()
    print(f"\n✅ Ingestion Complete! Collection '{collection_name}' now contains {count} chunks.")