import numpy as np

from src.rag.embeddings import embed_query, embed_texts


def test_embed_texts_shape():
    vectors = embed_texts(["hello world", "another sentence"])
    assert vectors.shape[0] == 2
    assert vectors.shape[1] > 0


def test_embed_texts_normalized():
    vectors = embed_texts(["a short sentence"])
    norm = np.linalg.norm(vectors[0])
    assert abs(norm - 1.0) < 1e-3


def test_embed_query_similarity():
    v1 = embed_query("annual leave policy")
    v2 = embed_query("annual leave policy")
    v3 = embed_query("completely unrelated database migration")
    same_sim = float(np.dot(v1, v2))
    diff_sim = float(np.dot(v1, v3))
    assert same_sim > diff_sim
