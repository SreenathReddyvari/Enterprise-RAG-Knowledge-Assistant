"""Small shared helpers."""
import re
import uuid


def new_id(prefix: str = "doc") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
