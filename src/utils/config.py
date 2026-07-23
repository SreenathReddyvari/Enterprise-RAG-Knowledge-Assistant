"""Central configuration loaded from environment variables (.env)."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    # Paths
    base_dir: Path = BASE_DIR
    input_dir: Path = BASE_DIR / "input"
    upload_dir: Path = BASE_DIR / "uploads"
    vector_db_dir: Path = BASE_DIR / "vector_db"

    # Vector store
    vector_store_backend: str = os.getenv("VECTOR_STORE_BACKEND", "faiss")  # faiss | chroma
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "5"))
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "3"))
    use_reranker: bool = os.getenv("USE_RERANKER", "true").lower() == "true"
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # LLM
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")  # ollama | openai
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2-local")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Database
    database_url: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'rag_metadata.db'}")

    # API / Dashboard
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8020"))
    streamlit_port: int = int(os.getenv("STREAMLIT_PORT", "8520"))
    api_base_url: str = os.getenv("API_BASE_URL", f"http://localhost:{api_port}")

    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
