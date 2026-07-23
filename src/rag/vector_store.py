"""Vector store abstraction with two interchangeable backends: FAISS and ChromaDB."""
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

from src.rag.embeddings import embed_texts
from src.utils.config import settings


@dataclass
class Chunk:
    id: str
    text: str
    document_id: str
    title: str
    category: str = ""
    department: str = ""
    doc_type: str = ""
    access: str = ""
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


class BaseVectorStore:
    def add(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    def search(self, query: str, top_k: int, filters: dict | None = None) -> list[Chunk]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


def _matches_filters(chunk: Chunk, filters: dict | None) -> bool:
    if not filters:
        return True
    for key, value in filters.items():
        if not value:
            continue
        if str(getattr(chunk, key, "")).lower() != str(value).lower():
            return False
    return True


class FaissVectorStore(BaseVectorStore):
    """Flat inner-product FAISS index with a parallel chunk metadata store."""

    def __init__(self, persist_dir: Path | None = None):
        self.persist_dir = persist_dir or settings.vector_db_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.persist_dir / "faiss.index"
        self.meta_path = self.persist_dir / "faiss_meta.pkl"
        self.dim = 384  # all-MiniLM-L6-v2 output size
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
        self.chunks: dict[int, Chunk] = {}
        self._next_id = 0
        self._load()

    def _load(self) -> None:
        if self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.meta_path, "rb") as f:
                data = pickle.load(f)
                self.chunks = data["chunks"]
                self._next_id = data["next_id"]

    def _save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "next_id": self._next_id}, f)

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = embed_texts([c.text for c in chunks]).astype("float32")
        ids = np.arange(self._next_id, self._next_id + len(chunks))
        self.index.add_with_ids(vectors, ids)
        for idx, chunk in zip(ids, chunks):
            self.chunks[int(idx)] = chunk
        self._next_id += len(chunks)
        self._save()

    def search(self, query: str, top_k: int, filters: dict | None = None) -> list[Chunk]:
        if self.index.ntotal == 0:
            return []
        query_vec = embed_texts([query]).astype("float32")
        # Over-fetch to allow post-filtering by metadata.
        fetch_k = min(self.index.ntotal, max(top_k * 5, top_k))
        scores, ids = self.index.search(query_vec, fetch_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            chunk = self.chunks.get(int(idx))
            if chunk is None or not _matches_filters(chunk, filters):
                continue
            chunk.score = float(score)
            results.append(chunk)
            if len(results) >= top_k:
                break
        return results

    def count(self) -> int:
        return self.index.ntotal

    def clear(self) -> None:
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
        self.chunks = {}
        self._next_id = 0
        self._save()


class ChromaVectorStore(BaseVectorStore):
    """Thin wrapper around a persistent ChromaDB collection."""

    def __init__(self, persist_dir: Path | None = None):
        import chromadb

        self.persist_dir = persist_dir or (settings.vector_db_dir / "chroma")
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection("rag_chunks")

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = embed_texts([c.text for c in chunks]).tolist()
        self.collection.add(
            ids=[c.id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "document_id": c.document_id,
                    "title": c.title,
                    "category": c.category,
                    "department": c.department,
                    "doc_type": c.doc_type,
                    "access": c.access,
                }
                for c in chunks
            ],
        )

    def search(self, query: str, top_k: int, filters: dict | None = None) -> list[Chunk]:
        if self.collection.count() == 0:
            return []
        query_vec = embed_texts([query]).tolist()
        where = {k: v for k, v in (filters or {}).items() if v} or None
        result = self.collection.query(
            query_embeddings=query_vec, n_results=min(top_k, self.collection.count()), where=where
        )
        chunks = []
        for i, doc_id in enumerate(result["ids"][0]):
            meta = result["metadatas"][0][i]
            distance = result["distances"][0][i]
            chunks.append(
                Chunk(
                    id=doc_id,
                    text=result["documents"][0][i],
                    document_id=meta.get("document_id", ""),
                    title=meta.get("title", ""),
                    category=meta.get("category", ""),
                    department=meta.get("department", ""),
                    doc_type=meta.get("doc_type", ""),
                    access=meta.get("access", ""),
                    score=1.0 - distance,
                )
            )
        return chunks

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        self.client.delete_collection("rag_chunks")
        self.collection = self.client.get_or_create_collection("rag_chunks")


_store_instance: BaseVectorStore | None = None


def get_vector_store() -> BaseVectorStore:
    global _store_instance
    if _store_instance is None:
        if settings.vector_store_backend == "chroma":
            _store_instance = ChromaVectorStore()
        else:
            _store_instance = FaissVectorStore()
    return _store_instance
