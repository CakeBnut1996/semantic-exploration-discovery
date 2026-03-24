# Local Semantic Search and Summarization

I develoepd a local semantic search function for a website using a Retrieval-Augmented Generation (RAG) system. It provides end-to-end data ingestion pipelines, embedding generation, vector database management, and LLM-powered retrieval. 
It allows users to:
- Embed text datasets into a vector database (ChromaDB).
- Query semantically, i.e., ask natural-language questions instead of using keywords.
- Retrieve and summarize the most relevant dataset using LLM API.
- While the example data are crawled HTML pages, the approach works with any text corpus (research papers, reports, knowledge bases, etc.).

## Repository structure
```
semantic-search-engine/
├── chroma_db/ # embedded data
├── retrieval_utils/ # retrieval functions
├── generation_utils/ # generation functions
├── display_utils/ # streamlit display functions
├── io_utils/ # read and save files
├── app.py # visualization
```

## Usage
1. Clone the repository

2. Install dependencies
```
pip install uv
uv sync
```

3. Configure your environment

Add your API keys (e.g., Gemini, OpenAI) into the config file (e.g., config.yaml). Ensure your data or vector database directories exist (e.g., data/, chroma_db/).

4. Run the application
```
streamlit run app.py
```
## Output description
1. AI Summary: The system generates a concise LLM-powered summary of the extracted content.
2. Reference: The engine identifies the exact source dataset or document on the website that matches the user’s query.
3. Direct Quote: Once a relevant reference is found, the system extracts the exact most relevant sentence or paragraph from the original document.
   
## Illustration
![Description of Image](./images/illustration.png)
