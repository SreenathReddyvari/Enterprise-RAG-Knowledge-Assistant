"""Split cleaned document text into overlapping chunks using LangChain's splitter."""
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config import settings


def chunk_text(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c for c in splitter.split_text(text) if c.strip()]
