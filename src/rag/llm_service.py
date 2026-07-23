"""LLM call to generate a grounded answer from retrieved context (Ollama or OpenAI)."""
import httpx

from src.utils.config import settings

SYSTEM_PROMPT = (
    "You are an enterprise knowledge assistant. Answer the question using ONLY the "
    "provided context from company documents. If the context does not contain the "
    "answer, say you don't have enough information. Be concise and factual."
)


class ProviderNotConfiguredError(Exception):
    pass


def _build_prompt(question: str, context: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def _call_ollama(question: str, context: str) -> str:
    prompt = _build_prompt(question, context)
    try:
        response = httpx.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": settings.ollama_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
    except httpx.ConnectError as exc:
        raise ProviderNotConfiguredError(f"Ollama not reachable at {settings.ollama_base_url}") from exc


def _call_openai(question: str, context: str) -> str:
    if not settings.openai_api_key:
        raise ProviderNotConfiguredError("OPENAI_API_KEY is not set")
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    completion = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(question, context)},
        ],
    )
    return completion.choices[0].message.content.strip()


def generate_answer(question: str, context: str, provider: str | None = None) -> str:
    provider = provider or settings.llm_provider
    if provider == "openai":
        return _call_openai(question, context)
    return _call_ollama(question, context)
