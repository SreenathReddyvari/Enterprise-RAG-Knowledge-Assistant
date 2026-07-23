"""Text preprocessing before chunking."""
from src.utils.helper import clean_text


def preprocess(text: str) -> str:
    return clean_text(text)
