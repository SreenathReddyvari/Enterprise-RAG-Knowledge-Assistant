"""Cross-encoder reranking of retrieved chunks, with a lexical-overlap fallback."""
from functools import lru_cache

from src.rag.vector_store import Chunk
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _get_cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model)


def _lexical_overlap_score(query: str, text: str) -> float:
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    if not query_words:
        return 0.0
    return len(query_words & text_words) / len(query_words)


def rerank(query: str, chunks: list[Chunk], top_n: int | None = None) -> list[Chunk]:
    if not chunks:
        return []
    top_n = top_n or settings.rerank_top_n
    used_cross_encoder = False
    if settings.use_reranker:
        try:
            model = _get_cross_encoder()
            pairs = [(query, c.text) for c in chunks]
            scores = model.predict(pairs)
            for chunk, score in zip(chunks, scores):
                chunk.score = float(score)
            used_cross_encoder = True
        except Exception as exc:  # model download/network issues
            logger.warning("Cross-encoder reranker unavailable (%s); using lexical overlap fallback.", exc)
            for chunk in chunks:
                chunk.score = _lexical_overlap_score(query, chunk.text)
    ranked = sorted(chunks, key=lambda c: c.score, reverse=True)

    if used_cross_encoder:
        # Cross-encoder logits: positive means relevant. Drop clearly irrelevant
        # chunks, but always keep the single best match so an answer can still be formed.
        relevant = [c for c in ranked if c.score > 0]
        ranked = relevant or ranked[:1]

    return ranked[:top_n]
