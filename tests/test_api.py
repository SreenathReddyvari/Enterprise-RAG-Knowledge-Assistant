from fastapi.testclient import TestClient

from src.api import app as app_module
from src.api.app import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_load_sample_and_ask(monkeypatch):
    monkeypatch.setattr(app_module, "answer_question", lambda **kwargs: {
        "question": kwargs["question"],
        "answer": "Employees get 20 annual leave days.",
        "sources": [{"document": "HR Policy", "document_id": "DOC001", "score": 0.9}],
        "retrieved_chunks": [],
    })

    response = client.post("/documents/load-sample")
    assert response.status_code == 200
    assert response.json()["documents_loaded"] > 0

    docs_response = client.get("/documents")
    assert docs_response.status_code == 200
    assert len(docs_response.json()) > 0

    ask_response = client.post("/ask", json={"question": "What is the annual leave policy?"})
    assert ask_response.status_code == 200
    body = ask_response.json()
    assert "20 annual leave days" in body["answer"]


def test_upload_rejects_unsupported_file_type():
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.xyz", b"some content")},
        data={"title": "Bad file"},
    )
    assert response.status_code == 400
