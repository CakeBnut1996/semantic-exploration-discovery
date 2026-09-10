# SEED Deployment Handoff

## Purpose

This document explains what the current repository does and what the deployment team will need to adapt when connecting it to the production database.

The current implementation is a local proof of concept. It reads downloaded HTML pages, creates embeddings in a local Chroma vector store, retrieves relevant documents, and uses an LLM to generate a cited response.

## Workflow

```text
                    INGESTION / SYNCHRONIZATION

 Production source database
 (datasets and metadata)
             │
             │  New, changed, or deleted records
             ▼
 Source database adapter              Deployment team adds this
             │
             │  Clean content + selected metadata
             ▼
 Sentence chunking → Embedding model → Production vector store
                                           │
                                           │  Embeddings, source IDs,
                                           │  metadata, and permissions
                                           ▼

                         SEARCH / GENERATION

 User question → Query embedding → Vector search
                                      │
                                      │  Best document matches
                                      │  One excerpt per document
                                      ▼
                              Grounded LLM response
                                      │
                                      ▼
                         Answer + sources + relevance scores
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                 Markdown output             Separate UI/API
```

The ingestion flow and search flow should be deployed separately. Ingestion keeps the vector store synchronized. Search reads from that vector store and calls the LLM.

## Summary of Current Setups and Expected Changes for Production

| Area | Current repository | Deployment responsibility |
|---|---|---|
| Source data | Downloaded HTML files in `data_raw/` | Connect to the production source database. |
| Content | Retrieved everything from each HTML page in `data_raw/`. | Suggest read the database's clean content fields directly. |
| Embeddings | Stored in local Chroma database under folder `chroma_db/` | Select and configure the production vector database. This may be part of the source database or a separate vector database. |
| Synchronization | Manual ingestion command by running `ingestion_utils/pre_-_processor.py` | Suggest run ingestion to vector database on a schedule or from raw database change events. Handle additions, updates, and deletions. |
| LLM access | Local environment variable | Supply the approved production LLM token. |
| Output | Run `main.py` and results are in `output/` | Results need to be displayed through a separately designed UI. |

## Details
### 1. Production Ingestion and Synchronization

The current ingestion entry point is:

```bash
uv run python -m ingestion_utils.pre_processor
```

What it does now is:

```text
Read source record → map content and metadata → chunk → embed → upsert vectors
```

It currently scans the configured HTML directory. The deployment team will need to replace the local folder (`data_raw`) with the actual production database.

Ingestion mgiht need to run whenever a dataset is added, changed, or removed. 

Each record needs a stable unique source ID. That ID should also identify its vectors so updates and deletions can be applied safely.

Embeddings should be saved in a separate table.

### 2. Production Content and Metadata

Local testing uses nearly all visible text from each downloaded HTML page. This is why webpage-only text such as navigation labels can appear in test data. A small local cleanup currently removes common navigation elements.

The deployment needs to connect to the actual production database, confirm the production schema, and finalize the field mapping.

The current vector metadata contains:

- Dataset/source identifier
- Page title
- Source URL

The production database may have additional fields, such as:

- Record or dataset ID
- Dataset name and description
- Source organization
- Public URL
- Category or topic
- Geographic coverage
- Time coverage
- Created and updated timestamps
- Version or content hash
- Access classification or permission groups


### 3. Embedding Configuration

The default embedding model is selected in `config.yaml`:

```yaml
retrieval:
  active_embedding: "mini_lm"
```

The same embedding model and preprocessing rules must be used for both ingestion and user queries. Changing the embedding model requires rebuilding all stored embeddings.

The default chunk size and sentence overlap are also configured in `config.yaml`. Changing either value requires re-ingestion because the stored chunks will change.

The default number of documents returned is 5:

```yaml
retrieval:
  num_docs: 5
```

### 4. LLM Access Token and Model Selection

The deployment team should store the LLM token in its secret manager (here I used .env) and inject it as an environment variable. 

Recognized token names are:

| Provider | Environment variable |
|---|---|
| Groq | `GROQ_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Gemini | `GEMINI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |

Example for a temporary local shell only:

```bash
export GROQ_API_KEY="<token-from-secret-manager>"
```

I currently used the free tier LLM API, openai/gpt-oss-120b, as shown on https://console.groq.com/docs/rate-limits:

```config.yaml
generation:
  active_student: "groq_gpt_oss"
```

The selected name must match an entry under `llm`, for example:

```yaml
llm:
  groq_gpt_oss:
    provider: "groq"
    model: "openai/gpt-oss-120b"
```

Before deployment, confirm that the chosen model supports the structured response format used by this application and that its use is approved for the source data's classification.


### 5. Running a Query

Run the command-line entry point with a question:

```bash
uv run python main.py "What information is available about biomass resources?"
```

The response is written as a Markdown file in:

```text
output/
```