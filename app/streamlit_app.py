"""Streamlit front-end for the AI PDF Assistant.

Talks to the FastAPI backend over HTTP so the UI and inference can scale
independently. Set BACKEND_URL to point at your API (default: localhost:8000).
"""

from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="AI PDF Assistant", page_icon="📄", layout="wide")
st.title("📄 AI PDF Assistant")
st.caption("Upload a PDF, then ask questions, summarize, extract tables/dates, or generate a report.")

if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "filename" not in st.session_state:
    st.session_state.filename = None


def _post(path: str, **kwargs):
    resp = requests.post(f"{BACKEND_URL}{path}", timeout=120, **kwargs)
    resp.raise_for_status()
    return resp.json()


with st.sidebar:
    st.header("1 · Upload")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded and st.button("Ingest PDF", use_container_width=True):
        with st.spinner("Extracting, chunking and embedding…"):
            try:
                data = _post("/upload", files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")})
                st.session_state.doc_id = data["doc_id"]
                st.session_state.filename = data["filename"]
                st.success(f"Ingested: {data['filename']}")
            except Exception as exc:
                st.error(f"Upload failed: {exc}")

    if st.session_state.doc_id:
        st.info(f"Active document: **{st.session_state.filename}**\n\n`{st.session_state.doc_id}`")


if not st.session_state.doc_id:
    st.warning("Upload and ingest a PDF from the sidebar to begin.")
    st.stop()

tab_ask, tab_tools = st.tabs(["💬 Ask", "🛠️ Tools"])

with tab_ask:
    question = st.text_input("Ask a question about the document")
    if st.button("Ask", type="primary") and question:
        with st.spinner("Thinking…"):
            try:
                st.markdown(_post("/ask", json={"question": question, "doc_id": st.session_state.doc_id})["answer"])
            except Exception as exc:
                st.error(f"Query failed: {exc}")

with tab_tools:
    cols = st.columns(4)
    tools = [
        ("Summarize", "/summarize"),
        ("Extract Tables", "/tables"),
        ("Find Dates", "/dates"),
        ("Generate Report", "/report"),
    ]
    for col, (label, path) in zip(cols, tools):
        if col.button(label, use_container_width=True):
            with st.spinner(f"{label}…"):
                try:
                    st.markdown(_post(path, json={"doc_id": st.session_state.doc_id})["result"])
                except Exception as exc:
                    st.error(f"{label} failed: {exc}")
