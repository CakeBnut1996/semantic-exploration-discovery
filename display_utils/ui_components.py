# display/ui_components.py
import html
import streamlit as st


def apply_custom_css():
    """Injects custom CSS to match the 'big input' and 'green theme' design."""
    st.markdown("""
    <style>
    /* Bigger font for the text input */
    div[data-baseweb="input"] input {
        font-size: 1.3rem !important;
        font-family: 'Source Sans Pro', sans-serif !important;
        padding: 0.5rem 1rem !important;
    }

    /* Search button styling */
    div.stButton > button {
        font-size: 1.3rem !important;
        font-family: 'Source Sans Pro', sans-serif !important;
        border-radius: 8px !important;
        height: 3rem; /* Attempt to match input height */
    }

    /* Green highlight tags */
    .kdf-tag {
        background-color: #E8F8EE;
        padding: 6px 10px;
        border-radius: 6px;
        display: inline-block;
        font-size: 1rem;
        color: #0A8A34;
        text-decoration: underline;
        margin-bottom: 5px;
    }

    /* Top result tag (slightly different style) */
    .kdf-tag-top {
        padding: 6px 10px;
        border-radius: 6px;
        display: inline-block;
        font-size: 1.1rem;
        color: #0A8A34;
        font-weight: bold;
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    st.title("🔍 Semantic Search Demo")


def _render_source_tag(label_class: str, dataset_name: str, source_url: str | None = None):
    safe_name = html.escape(dataset_name)
    safe_url = html.escape(source_url) if source_url else ""
    if source_url and source_url != "Unknown Source":
        st.markdown(
            f'<div class="{label_class}">📂 <a href="{safe_url}" target="_blank">{safe_name}</a></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="{label_class}">📂 {safe_name}</div>',
            unsafe_allow_html=True
        )


def render_search_bar():
    """Renders the search input and button side-by-side."""
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            label="Search Query",
            label_visibility="collapsed",
            placeholder="Enter your semantic search query",
        )
    with col2:
        # use_container_width makes the button fill the column width
        clicked = st.button("Search", use_container_width=True)

    return query, clicked


def render_answer_section(answer_obj, dataset_meta_map=None):
    """Renders the left column: Main Answer + Top Source."""
    st.subheader("🧠 Answer Summary")

    if not answer_obj:
        st.info("No answer generated.")
        return

    st.info(answer_obj.answer)

    # Render the top source tag
    name_top = getattr(answer_obj, 'name_top', "Unnamed Dataset")
    meta_map = dataset_meta_map or {}
    top_meta = meta_map.get(name_top, {})

    source_title = top_meta.get("source_title") or name_top
    source_url = top_meta.get("source_url")
    _render_source_tag("kdf-tag-top", f"Top Source: {source_title}", source_url)


def render_supporting_evidence(answer_obj, dataset_meta_map=None):
    """Renders the right column: List of supporting datasets."""
    st.subheader("Results")

    datasets = getattr(answer_obj, 'supporting_datasets', [])
    meta_map = dataset_meta_map or {}

    if not datasets:
        st.warning("No supporting evidence found.")
        return

    for ds in datasets:
        ds_meta = meta_map.get(ds.name, {})
        source_title = ds_meta.get("source_title") or ds.name
        source_url = ds_meta.get("source_url")

        # Use HTML for the green tag style.
        _render_source_tag("kdf-tag", source_title, source_url)
        st.markdown(f"**Summary:** {ds.summary}")
        st.markdown(f"**Source text excerpt:** \n> *{ds.quote}*")
        st.markdown("---")