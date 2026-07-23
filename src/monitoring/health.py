"""Application health checks."""
import httpx

from src.rag.vector_store import get_vector_store
from src.utils.config import settings


def check_vector_store() -> dict:
    try:
        store = get_vector_store()
        return {"status": "ok", "backend": settings.vector_store_backend, "chunks": store.count()}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


def check_llm_provider() -> dict:
    if settings.llm_provider == "openai":
        return {"status": "ok" if settings.openai_api_key else "not_configured", "provider": "openai"}
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        response.raise_for_status()
        return {"status": "ok", "provider": "ollama", "model": settings.ollama_model}
    except Exception as exc:
        return {"status": "error", "provider": "ollama", "detail": str(exc)}


def get_health_report() -> dict:
    vector = check_vector_store()
    llm = check_llm_provider()
    overall = "ok" if vector["status"] == "ok" and llm["status"] in ("ok", "not_configured") else "degraded"
    return {"status": overall, "vector_store": vector, "llm": llm}
