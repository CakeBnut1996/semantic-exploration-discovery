import sys
import re
import yaml
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

from retrieval_utils.retriever import retrieve_data, rank_datasets
from generation_utils.generator import (
    StudentGenerator,
    ground_response,
    select_ranked_context,
    serialize_ranked_context,
)
from generation_utils.schema import Response

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"


def load_system():
    """Loads config, resolves active models/DBs, and initializes the Student generator."""
    config_path = BASE_DIR / "config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    active_emb = cfg["embeddings"][cfg["retrieval"]["active_embedding"]]
    active_stu = cfg["llm"][cfg["generation"]["active_student"]]
    active_db = cfg["db"][cfg["retrieval"]["active_db"]]

    system_config = {
        "DB_PATH": str((BASE_DIR / cfg["data"]["db_path"]).resolve()),
        "COLLECTION_NAME": active_db["collection"],
        "EMBEDDING_MODEL": active_emb["model"],
        "NUM_DOCS": cfg["retrieval"]["num_docs"],
    }

    student_agent = StudentGenerator(
        provider=active_stu["provider"],
        model_name=active_stu["model"],
    )

    return system_config, student_agent


def sanitize_filename(query: str, max_length: int = 40) -> str:
    """Sanitizes the query string to create a valid filename slug."""
    slug = re.sub(r"[^\w\s-]", "", query.strip().lower())
    slug = re.sub(r"[-\s]+", "_", slug).strip("_")
    if not slug:
        slug = "search_result"
    return slug[:max_length]


def _is_public_http_url(value: str | None) -> bool:
    if not value:
        return False
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def format_markdown_response(query: str, answer_obj: Response, dataset_meta_map: dict) -> str:
    """Formats the answer and supporting evidence into a Markdown string."""
    lines = [
        f"# Search Result: {query}\n",
        "## 🧠 Answer Summary\n",
        f"{answer_obj.answer or 'No answer generated.'}\n",
    ]

    if answer_obj.evidence_status == "insufficient":
        lines.append("## Supporting Evidence\n")
        lines.append("No retrieved source directly supports the requested fact or time period.\n")
        return "\n".join(lines)

    name_top = getattr(answer_obj, "name_top", "Unnamed Dataset")
    top_meta = dataset_meta_map.get(name_top, {})
    top_title = top_meta.get("source_title") or name_top
    top_url = top_meta.get("source_url")

    lines.append("### Top Source")
    if _is_public_http_url(top_url):
        lines.append(f"- **Title:** [{top_title}]({top_url})")
        lines.append(f"- **Source URL:** {top_url}")
    else:
        lines.append(f"- **Title:** {top_title}")
    lines.append(f"- **Dataset ID:** `{name_top}`\n")
    top_score = top_meta.get("relevance_score")
    if top_score is not None:
        lines.append(f"- **Retrieval relevance:** {top_score:.0%} ({top_score:.2f})\n")

    lines.append("## 📚 Supporting Evidence & Results\n")
    datasets = getattr(answer_obj, "supporting_datasets", [])
    if not datasets:
        lines.append("No supporting evidence returned.\n")
    else:
        for idx, ds in enumerate(datasets, 1):
            ds_meta = dataset_meta_map.get(ds.name, {})
            title = ds_meta.get("source_title") or ds.name
            url = ds_meta.get("source_url")

            lines.append(f"### {idx}. {title}")
            lines.append(f"- **Dataset ID:** `{ds.name}`")
            if _is_public_http_url(url):
                lines.append(f"- **Source URL:** {url}")
            score = ds_meta.get("relevance_score")
            if score is not None:
                lines.append(f"- **Retrieval relevance:** {score:.0%} ({score:.2f})")
            if ds.summary:
                lines.append(f"- **Summary:** {ds.summary}")
            if ds.quote:
                lines.append(f"- **Quote:** > {ds.quote}")
            lines.append("")

    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("==========================================")
    print(" 🔍 SEED Semantic Search (CLI Interface)")
    print("==========================================\n")

    if len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:]).strip()
    else:
        query_text = input("Enter your search question: ").strip()

    if not query_text:
        print("❌ Error: Query cannot be empty.")
        return

    print(f"\n🔍 Searching for: '{query_text}'...")
    print("Loading system configuration and models...")
    sys_cfg, student = load_system()

    try:
        # A. Retrieval
        retrieved_data = retrieve_data(
            query=query_text,
            db_path=sys_cfg["DB_PATH"],
            collection_name=sys_cfg["COLLECTION_NAME"],
            model_name=sys_cfg["EMBEDDING_MODEL"],
            num_docs=sys_cfg["NUM_DOCS"],
        )

        dataset_meta_map = {}
        for item in retrieved_data:
            if item.dataset_id not in dataset_meta_map:
                dataset_meta_map[item.dataset_id] = {
                    "source_url": (item.metadata or {}).get("source_url"),
                    "source_title": (item.metadata or {}).get("source_title"),
                    "relevance_score": item.score,
                }

        title_to_dataset_id = {
            meta["source_title"]: dataset_id
            for dataset_id, meta in dataset_meta_map.items()
            if meta.get("source_title")
        }

        # B. Ranking
        ranked_data = select_ranked_context(query_text, rank_datasets(retrieved_data))
        for ranked_dataset in ranked_data:
            dataset_meta_map[ranked_dataset.dataset_id]["relevance_score"] = ranked_dataset.top_score

        # C. Generation
        context_str = serialize_ranked_context(ranked_data)
        answer_object = student.generate(
            query=query_text,
            context=context_str,
            schema=Response,
        )

        if isinstance(answer_object, str):
            print(f"❌ Error from model: {answer_object}")
            return

        answer_object = cast(Response, answer_object)
        answer_object = ground_response(answer_object, ranked_data)

        # Map dataset names back if needed
        if (
            answer_object.evidence_status == "supported"
            and getattr(answer_object, "name_top", None) not in dataset_meta_map
        ):
            mapped_top = title_to_dataset_id.get(getattr(answer_object, "name_top", ""))
            if mapped_top:
                answer_object.name_top = mapped_top
            elif ranked_data:
                answer_object.name_top = ranked_data[0].dataset_id

        for ds in getattr(answer_object, "supporting_datasets", []):
            if ds.name not in dataset_meta_map:
                mapped_name = title_to_dataset_id.get(ds.name)
                if mapped_name:
                    ds.name = mapped_name

        # Sort supporting datasets strictly in rank order determined by vector retrieval
        rank_order = {rds.dataset_id: i for i, rds in enumerate(ranked_data)}
        answer_object.supporting_datasets.sort(key=lambda ds: rank_order.get(ds.name, 999))

        # D. Save to Markdown
        file_slug = sanitize_filename(query_text)
        output_path = OUTPUT_DIR / f"{file_slug}.md"

        markdown_content = format_markdown_response(query_text, answer_object, dataset_meta_map)
        output_path.write_text(markdown_content, encoding="utf-8")

        print("\n✅ Search complete!")
        print(f"📄 Result saved to: {output_path}")
        print("\n--- Summary ---")
        print(answer_object.answer or "No relevant information found in the retrieved documents.")
        print("----------------")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")


if __name__ == "__main__":
    main()
