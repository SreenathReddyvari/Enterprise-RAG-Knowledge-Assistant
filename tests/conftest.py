"""Shared pytest fixtures: isolate vector store, database and reranker for fast, deterministic tests."""
import os
from pathlib import Path

import pytest

_TEST_DB_PATH = Path(__file__).parent / "test_rag_metadata.db"
_TEST_DB_PATH.unlink(missing_ok=True)

os.environ["USE_RERANKER"] = "false"
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB_PATH}"

from src.database.db import ChatHistory, Document, get_session, init_db  # noqa: E402
from src.rag import vector_store as vs_module  # noqa: E402
from src.utils.config import settings  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_vector_store(tmp_path, monkeypatch):
    """Point the vector store at a throwaway directory and reset the singleton per test."""
    monkeypatch.setattr(settings, "vector_db_dir", tmp_path / "vector_db")
    vs_module._store_instance = None
    yield
    vs_module._store_instance = None


@pytest.fixture(autouse=True)
def isolated_database():
    init_db()
    with get_session() as session:
        session.query(ChatHistory).delete()
        session.query(Document).delete()
        session.commit()
    yield
