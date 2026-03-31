import streamlit as st
import yaml
from typing import cast

# --- Standard Imports (No Reloading) ---
from retrieval_utils.retriever import retrieve_data, rank_datasets
from generation_utils.generator import StudentGenerator
from generation_utils.schema import Response
from display_utils.ui_components import (
    apply_custom_css,
    render_header,
    render_search_bar,
    render_answer_section,
    render_supporting_evidence
)

# --- Page Configuration ---
st.set_page_config(page_title="Semantic Search Demo", layout="wide")
apply_custom_css()
render_header()


# --- Load Logic & Config ---
@st.cache_resource
def load_system():
    """
    Loads config, resolves active models/DBs, and initializes the Student.
    Cached so it doesn't re-run on every UI interaction.
    """
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Resolve Active Settings based on your new structure
    active_emb = cfg['embeddings'][cfg["retrieval"]['active_embedding']]
    active_stu = cfg['llm'][cfg["generation"]['active_student']]
    active_db = cfg['db'][cfg["retrieval"]['active_db']]

    # 2. Set Operational Variables
    system_config = {
        "DB_PATH": cfg['data']['db_path'],
        "COLLECTION_NAME": active_db['collection'],
        "EMBEDDING_MODEL": active_emb['model'],
        "NUM_DOCS": cfg['retrieval']['num_docs'],
        "CHUNKS_PER_DOC": cfg['retrieval']['chunks_per_doc']
    }

    # 3. Initialize Student Generator
    student_agent = StudentGenerator(
        provider=active_stu['provider'],
        model_name=active_stu['model']
    )

    return system_config, student_agent


# Load everything once
sys_cfg, student = load_system()

# --- Main UI Loop ---
query_text, search_btn = render_search_bar()

if search_btn and query_text:
    with st.spinner("🔍 Searching & Generating Answer..."):
        try:
            # A. Retrieval
            retrieved_data = retrieve_data(
                query=query_text,
                db_path=sys_cfg["DB_PATH"],
                collection_name=sys_cfg["COLLECTION_NAME"],
                model_name=sys_cfg["EMBEDDING_MODEL"],
                num_docs=sys_cfg["NUM_DOCS"],
                chunks_per_doc=sys_cfg["CHUNKS_PER_DOC"]
            )

            dataset_meta_map = {}
            for item in retrieved_data:
                if item.dataset_id not in dataset_meta_map:
                    dataset_meta_map[item.dataset_id] = {
                        "source_url": (item.metadata or {}).get("source_url"),
                        "source_title": (item.metadata or {}).get("source_title")
                    }
            title_to_dataset_id = {
                meta["source_title"]: dataset_id
                for dataset_id, meta in dataset_meta_map.items()
                if meta.get("source_title")
            }

            # B. Ranking
            ranked_data = rank_datasets(retrieved_data)

            # C. Generation (Structured)
            # Convert ranked objects to string context for the LLM
            context_str = str(ranked_data)

            answer_object = student.generate(
                query=query_text,
                context=context_str,
                schema=Response
            )

            if isinstance(answer_object, str):
                st.error(answer_object)
                st.stop()
            answer_object = cast(Response, answer_object)

            # Map model-generated names back to dataset IDs when possible.
            if getattr(answer_object, "name_top", None) not in dataset_meta_map:
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

            # --- Display Results ---
            col_left, col_right = st.columns([2, 2])

            with col_left:
                render_answer_section(answer_object, dataset_meta_map)

            with col_right:
                render_supporting_evidence(answer_object, dataset_meta_map)

        except Exception as e:
            st.error(f"An error occurred: {e}")