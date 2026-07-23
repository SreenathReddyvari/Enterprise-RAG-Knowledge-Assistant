"""Sentence-Transformers embedding wrapper (with lazy singleton loading)."""
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.config import settings


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_embedder()
    return np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False))


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
