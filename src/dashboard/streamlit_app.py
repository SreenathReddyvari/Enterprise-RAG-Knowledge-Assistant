"""Streamlit dashboard for the Enterprise RAG Knowledge Assistant."""
import sys
from pathlib import Path

# `streamlit run` puts this file's own directory on sys.path, not the project
# root, so the `src` package isn't importable unless we add it ourselves.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx
import streamlit as st

from src.utils.config import settings

API_URL = settings.api_base_url

st.set_page_config(page_title="Enterprise RAG Knowledge Assistant", layout="wide")
st.title("Enterprise RAG Knowledge Assistant")

tab_ask, tab_upload, tab_docs, tab_health = st.tabs(["Ask", "Upload Documents", "Documents", "Health"])

with tab_ask:
    st.subheader("Ask a question")
    question = st.text_input("Your question", placeholder="What is the annual leave policy?")

    col1, col2, col3 = st.columns(3)
    department = col1.text_input("Filter by department (optional)")
    doc_type = col2.text_input("Filter by document type (optional)")
    category = col3.text_input("Filter by category (optional)")

    if st.button("Ask", type="primary") and question:
        with st.spinner("Retrieving context and generating answer..."):
            try:
                response = httpx.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question,
                        "department": department or None,
                        "doc_type": doc_type or None,
                        "category": category or None,
                    },
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPError as exc:
                st.error(f"Request failed: {exc}")
                data = None

        if data:
            st.markdown("### Answer")
            st.write(data["answer"])

            st.markdown("### Sources")
            if data["sources"]:
                st.table(data["sources"])
            else:
                st.info("No sources retrieved.")

            with st.expander("Retrieved chunks"):
                for chunk in data["retrieved_chunks"]:
                    st.markdown(f"**{chunk['document']}** (score: {chunk['score']})")
                    st.caption(chunk["text"])

with tab_upload:
    st.subheader("Upload a document")
    uploaded_file = st.file_uploader("Choose a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt", "md"])
    title = st.text_input("Title (optional)")
    dep = st.text_input("Department")
    dtype = st.text_input("Document type")
    access = st.selectbox("Access", ["Public", "Internal", "Restricted"])
    category = st.text_input("Category")

    if st.button("Upload & Ingest") and uploaded_file:
        with st.spinner("Uploading and indexing..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            data = {
                "title": title,
                "category": category,
                "department": dep,
                "doc_type": dtype,
                "access": access,
            }
            try:
                response = httpx.post(f"{API_URL}/documents/upload", files=files, data=data, timeout=120)
                response.raise_for_status()
                st.success(f"Ingested: {response.json()}")
            except httpx.HTTPError as exc:
                st.error(f"Upload failed: {exc}")

    st.divider()
    if st.button("Load sample knowledge base (input/*.csv)"):
        with st.spinner("Loading sample data..."):
            try:
                response = httpx.post(f"{API_URL}/documents/load-sample", timeout=120)
                response.raise_for_status()
                st.success(response.json())
            except httpx.HTTPError as exc:
                st.error(f"Load failed: {exc}")

with tab_docs:
    st.subheader("Indexed documents")
    try:
        response = httpx.get(f"{API_URL}/documents", timeout=30)
        response.raise_for_status()
        docs = response.json()
        if docs:
            st.dataframe(docs, use_container_width=True)
        else:
            st.info("No documents indexed yet.")
    except httpx.HTTPError as exc:
        st.error(f"Could not load documents: {exc}")

with tab_health:
    st.subheader("System health")
    try:
        response = httpx.get(f"{API_URL}/health", timeout=10)
        response.raise_for_status()
        st.json(response.json())
    except httpx.HTTPError as exc:
        st.error(f"Health check failed: {exc}")
