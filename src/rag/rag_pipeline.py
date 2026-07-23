"""End-to-end RAG orchestration: ingestion and question answering."""
from pathlib import Path

from src.data.loader import load_document, load_knowledge_base_csv, load_metadata_csv
from src.data.preprocess import preprocess
from src.rag.chunking import chunk_text
from src.rag.llm_service import generate_answer
from src.rag.reranker import rerank
from src.rag.retriever import retrieve
from src.rag.vector_store import Chunk, get_vector_store
from src.utils.helper import new_id


def ingest_text(
    document_id: str,
    title: str,
    text: str,
    category: str = "",
    department: str = "",
    doc_type: str = "",
    access: str = "",
) -> int:
    """Clean, chunk, embed and store one document. Returns number of chunks added."""
    cleaned = preprocess(text)
    pieces = chunk_text(cleaned)
    chunks = [
        Chunk(
            id=new_id("chunk"),
            text=piece,
            document_id=document_id,
            title=title,
            category=category,
            department=department,
            doc_type=doc_type,
            access=access,
        )
        for piece in pieces
    ]
    get_vector_store().add(chunks)
    return len(chunks)


def ingest_file(path: Path, document_id: str, title: str, **metadata) -> int:
    text = load_document(path)
    return ingest_text(document_id, title, text, **metadata)


def ingest_knowledge_base_csv(kb_path: Path, metadata_path: Path | None = None) -> dict[str, int]:
    """Bulk-load the sample knowledge_base.csv (+ optional document_metadata.csv).

    Returns a map of document_id -> chunks added, so callers can persist per-document counts.
    """
    rows = load_knowledge_base_csv(kb_path)
    metadata = load_metadata_csv(metadata_path) if metadata_path and metadata_path.exists() else {}
    chunks_per_doc: dict[str, int] = {}
    for row in rows:
        doc_id = row["Document_ID"]
        meta = metadata.get(doc_id, {})
        chunks_per_doc[doc_id] = ingest_text(
            document_id=doc_id,
            title=row["Title"],
            text=row["Content"],
            category=row.get("Category", ""),
            department=meta.get("Department", ""),
            doc_type=meta.get("Type", ""),
            access=meta.get("Access", ""),
        )
    return chunks_per_doc


def answer_question(
    question: str,
    department: str | None = None,
    doc_type: str | None = None,
    category: str | None = None,
    top_k: int = 5,
) -> dict:
    retrieved = retrieve(question, top_k=top_k, department=department, doc_type=doc_type, category=category)
    ranked = rerank(question, retrieved)

    if not ranked:
        return {
            "question": question,
            "answer": "I don't have enough information in the knowledge base to answer that.",
            "sources": [],
            "retrieved_chunks": [],
        }

    context = "\n\n".join(f"[{c.title}] {c.text}" for c in ranked)
    answer = generate_answer(question, context)

    sources = [
        {"document": c.title, "document_id": c.document_id, "score": round(c.score, 4)} for c in ranked
    ]
    retrieved_chunks = [
        {"document": c.title, "text": c.text, "score": round(c.score, 4)} for c in ranked
    ]
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
    }
