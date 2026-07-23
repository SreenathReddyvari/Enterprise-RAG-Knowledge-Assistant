# Enterprise RAG Knowledge Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers questions from enterprise
documents (HR policies, IT security, travel policy, benefits guides, etc.) with grounded,
source-cited answers instead of hallucinated responses.

## Architecture

```
Document (PDF/DOCX/TXT/CSV) -> clean -> chunk (LangChain splitter) -> embed (Sentence-Transformers)
   -> vector store (FAISS or ChromaDB)

Question -> embed -> similarity search -> metadata filter -> context compression
   -> cross-encoder rerank -> LLM (Ollama or OpenAI) -> answer + sources
```

Metadata (document registry, chat history) is stored in SQLite by default and PostgreSQL in
Docker Compose / production, via `DATABASE_URL`.

## Project structure

```
src/
  data/        loader.py, preprocess.py        - document ingestion & cleaning
  rag/         chunking, embeddings, vector_store, retriever, reranker, llm_service, rag_pipeline
  api/         app.py, schemas.py              - FastAPI service
  dashboard/   streamlit_app.py                - Streamlit UI
  database/    db.py                           - SQLAlchemy models
  utils/       config.py, logger.py, helper.py
  monitoring/  health.py
input/         sample knowledge_base.csv, document_metadata.csv, sample_questions.csv
tests/         pytest suite (api, retriever, rag pipeline, embeddings)
docker/        Dockerfile, docker-compose.yml (API + Streamlit + Postgres)
```

## Scope notes

- **Reranking**: cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) with an automatic
  fallback to lexical overlap scoring if the model can't be downloaded (offline/restricted
  network) — see `USE_RERANKER` in `.env`.
- **LlamaIndex** was deliberately left out of this build to keep the dependency surface
  minimal; chunking uses LangChain's `RecursiveCharacterTextSplitter` and everything else
  (retrieval, reranking, orchestration) is plain Python, per the project's "keep it minimal"
  guidance. LangChain integration is present; LlamaIndex can be added as an alternate
  chunking/indexing backend later without changing the API contract.
- **Azure AI Search** (optional enterprise vector store) was not implemented — FAISS and
  ChromaDB satisfy the core deliverable.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and adjust if needed (defaults work out of the box with a
   local Ollama model called `llama3.2-local`).
3. Make sure Ollama is running locally (or set `LLM_PROVIDER=openai` and `OPENAI_API_KEY`).

## Run it

**Terminal 1 — API:**
```
uvicorn src.api.app:app --reload --port 8020
```

**Terminal 2 — Dashboard:**
```
streamlit run src/dashboard/streamlit_app.py --server.port 8520
```

Open http://localhost:8520, go to the **Upload Documents** tab and click
**"Load sample knowledge base (input/\*.csv)"** to index the four sample HR/IT/Finance
documents, then ask questions in the **Ask** tab (e.g. *"What is the annual leave policy?"*).

API docs: http://localhost:8020/docs

## Run with Docker

```
docker compose -f docker/docker-compose.yml up --build
```

API on http://localhost:8020, dashboard on http://localhost:8520, Postgres on port 5433.

## Tests

```
pytest -v
```

## Sample API call

```
curl -X POST http://localhost:8020/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the annual leave policy?"}'
```

```json
{
  "question": "What is the annual leave policy?",
  "answer": "Employees are entitled to 20 annual leave days...",
  "sources": [{"document": "HR Policy", "document_id": "DOC001", "score": 0.87}],
  "retrieved_chunks": [...]
}
```
