"""Load raw documents (PDF, DOCX, TXT, CSV knowledge base) into plain text."""
import csv
from pathlib import Path

import docx
from pypdf import PdfReader


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_docx(path: Path) -> str:
    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".docx":
        return load_docx(path)
    if suffix in (".txt", ".md"):
        return load_txt(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def load_knowledge_base_csv(path: Path) -> list[dict]:
    """Load the sample knowledge_base.csv into one 'document' per row."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_metadata_csv(path: Path) -> dict:
    """Load document_metadata.csv keyed by Document_ID."""
    with open(path, newline="", encoding="utf-8") as f:
        return {row["Document_ID"]: row for row in csv.DictReader(f)}
