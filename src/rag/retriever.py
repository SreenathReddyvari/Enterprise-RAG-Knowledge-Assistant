"""Retrieval: similarity search + metadata filtering + context compression."""
from src.rag.vector_store import Chunk, get_vector_store
from src.utils.config import settings


def _compress_context(chunks: list[Chunk], min_score: float = 0.15) -> list[Chunk]:
    """Drop near-duplicate and low-relevance chunks before reranking/LLM."""
    seen_text: set[str] = set()
    compressed = []
    for chunk in chunks:
        signature = chunk.text[:80].strip().lower()
        if signature in seen_text:
            continue
        if chunk.score < min_score:
            continue
        seen_text.add(signature)
        compressed.append(chunk)
    return compressed


def retrieve(
    query: str,
    top_k: int | None = None,
    department: str | None = None,
    doc_type: str | None = None,
    category: str | None = None,
) -> list[Chunk]:
    store = get_vector_store()
    filters = {"department": department, "doc_type": doc_type, "category": category}
    chunks = store.search(query, top_k=top_k or settings.top_k, filters=filters)
    return _compress_context(chunks)
