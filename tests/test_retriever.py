from src.rag.rag_pipeline import ingest_text
from src.rag.retriever import retrieve


def test_retrieve_returns_relevant_chunk():
    ingest_text(
        document_id="DOC001",
        title="HR Policy",
        text="Employees are entitled to 20 annual leave days per year. Leave beyond five "
        "consecutive days requires manager approval.",
        department="HR",
        doc_type="Policy",
    )
    ingest_text(
        document_id="DOC002",
        title="IT Security",
        text="Passwords expire every 90 days and multi-factor authentication is mandatory "
        "for all internal systems.",
        department="IT",
        doc_type="Security",
    )

    results = retrieve("What is the annual leave policy?", top_k=2)

    assert results
    assert results[0].document_id == "DOC001"


def test_retrieve_applies_department_filter():
    ingest_text("DOC001", "HR Policy", "Annual leave is 20 days per year.", department="HR")
    ingest_text("DOC002", "IT Security", "Passwords expire every 90 days.", department="IT")

    results = retrieve("policy", top_k=5, department="IT")

    assert all(r.department == "IT" for r in results)
