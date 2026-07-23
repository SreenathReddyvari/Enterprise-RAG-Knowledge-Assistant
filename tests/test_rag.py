from src.rag.chunking import chunk_text
from src.rag.rag_pipeline import answer_question, ingest_text


def test_chunk_text_splits_long_document():
    text = "sentence. " * 200
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 150 for c in chunks)  # allow overlap slack


def test_answer_question_uses_retrieved_context(monkeypatch):
    ingest_text(
        "DOC001",
        "HR Policy",
        "Employees are entitled to 20 annual leave days per year.",
        department="HR",
    )

    captured = {}

    def fake_generate_answer(question, context, provider=None):
        captured["context"] = context
        return "Employees get 20 annual leave days."

    monkeypatch.setattr("src.rag.rag_pipeline.generate_answer", fake_generate_answer)

    result = answer_question("What is the annual leave policy?")

    assert "20 annual leave days" in result["answer"]
    assert result["sources"][0]["document"] == "HR Policy"
    assert "annual leave" in captured["context"].lower()


def test_answer_question_no_matches_returns_fallback():
    result = answer_question("What is the meaning of life?")
    assert result["sources"] == []
    assert "don't have enough information" in result["answer"]
