"""FastAPI service for the Enterprise RAG Knowledge Assistant."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.api.schemas import AskRequest, AskResponse, DocumentOut, UploadResponse
from src.database.db import ChatHistory, Document, get_session, init_db
from src.monitoring.health import get_health_report
from src.rag.rag_pipeline import answer_question, ingest_file, ingest_knowledge_base_csv
from src.utils.config import settings
from src.utils.helper import new_id
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Enterprise RAG Knowledge Assistant", version="1.0.0", lifespan=lifespan)


@app.get("/")
def root():
    return {"service": "Enterprise RAG Knowledge Assistant", "status": "running"}


@app.get("/health")
def health():
    return get_health_report()


@app.post("/documents/upload", response_model=UploadResponse)
def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    category: str = Form(""),
    department: str = Form(""),
    doc_type: str = Form(""),
    access: str = Form(""),
):
    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in (".pdf", ".docx", ".txt", ".md"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    dest = settings.upload_dir / file.filename
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(file.file.read())

    document_id = new_id("doc")
    doc_title = title or file.filename
    chunks_added = ingest_file(
        dest, document_id, doc_title, category=category, department=department, doc_type=doc_type, access=access
    )

    with get_session() as session:
        session.add(
            Document(
                id=document_id,
                title=doc_title,
                category=category,
                department=department,
                doc_type=doc_type,
                access=access,
                chunk_count=chunks_added,
            )
        )
        session.commit()

    return UploadResponse(document_id=document_id, title=doc_title, chunks_added=chunks_added)


@app.post("/documents/load-sample")
def load_sample_knowledge_base():
    """Convenience endpoint: bulk-ingest the sample CSVs shipped in input/."""
    kb_path = settings.input_dir / "knowledge_base.csv"
    metadata_path = settings.input_dir / "document_metadata.csv"
    if not kb_path.exists():
        raise HTTPException(status_code=404, detail=f"{kb_path} not found")

    chunks_per_doc = ingest_knowledge_base_csv(kb_path, metadata_path)

    from src.data.loader import load_knowledge_base_csv, load_metadata_csv

    rows = load_knowledge_base_csv(kb_path)
    metadata = load_metadata_csv(metadata_path) if metadata_path.exists() else {}
    with get_session() as session:
        for row in rows:
            doc_id = row["Document_ID"]
            if session.get(Document, doc_id):
                continue
            meta = metadata.get(doc_id, {})
            session.add(
                Document(
                    id=doc_id,
                    title=row["Title"],
                    category=row.get("Category", ""),
                    department=meta.get("Department", ""),
                    doc_type=meta.get("Type", ""),
                    access=meta.get("Access", ""),
                    chunk_count=chunks_per_doc.get(doc_id, 0),
                )
            )
        session.commit()

    return {"documents_loaded": len(rows), "chunks_added": sum(chunks_per_doc.values())}


@app.get("/documents", response_model=list[DocumentOut])
def list_documents():
    with get_session() as session:
        return session.query(Document).all()


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = answer_question(
        question=request.question,
        department=request.department,
        doc_type=request.doc_type,
        category=request.category,
        top_k=request.top_k,
    )
    with get_session() as session:
        session.add(ChatHistory(question=result["question"], answer=result["answer"]))
        session.commit()
    return result
