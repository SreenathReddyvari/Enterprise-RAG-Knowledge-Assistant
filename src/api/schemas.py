"""Pydantic request/response models for the FastAPI service."""
from pydantic import BaseModel, ConfigDict


class AskRequest(BaseModel):
    question: str
    department: str | None = None
    doc_type: str | None = None
    category: str | None = None
    top_k: int = 5


class Source(BaseModel):
    document: str
    document_id: str
    score: float


class RetrievedChunk(BaseModel):
    document: str
    text: str
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    retrieved_chunks: list[RetrievedChunk]


class DocumentOut(BaseModel):
    id: str
    title: str
    category: str
    department: str
    doc_type: str
    access: str
    chunk_count: int

    model_config = ConfigDict(from_attributes=True)


class UploadResponse(BaseModel):
    document_id: str
    title: str
    chunks_added: int
