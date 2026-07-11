"""FastAPI backend exposing the RAG pipeline over HTTP."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from .errors import PDFAssistantError
from .factory import build_pipeline
from .logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def create_app():
    """Application factory. Imported lazily so tests without FastAPI still run."""
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from pydantic import BaseModel

    configure_logging()
    app = FastAPI(title="AI PDF Assistant", version="1.0.0")
    pipeline = build_pipeline()

    class QueryRequest(BaseModel):
        question: str
        doc_id: Optional[str] = None

    class TaskRequest(BaseModel):
        doc_id: Optional[str] = None

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/upload")
    async def upload(file: UploadFile = File(...)) -> dict:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(400, "Only PDF files are supported.")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(await file.read())
                tmp_path = Path(tmp.name)
            doc_id = pipeline.ingest(tmp_path)
            tmp_path.unlink(missing_ok=True)
            return {"doc_id": doc_id, "filename": file.filename}
        except PDFAssistantError as exc:
            raise HTTPException(422, str(exc))

    @app.post("/ask")
    def ask(req: QueryRequest) -> dict:
        try:
            return {"answer": pipeline.ask(req.question, req.doc_id)}
        except PDFAssistantError as exc:
            raise HTTPException(422, str(exc))

    def _task_route(fn):
        def handler(req: TaskRequest) -> dict:
            try:
                return {"result": fn(req.doc_id)}
            except PDFAssistantError as exc:
                raise HTTPException(422, str(exc))
        return handler

    app.post("/summarize")(_task_route(pipeline.summarize))
    app.post("/tables")(_task_route(pipeline.extract_tables))
    app.post("/dates")(_task_route(pipeline.find_dates))
    app.post("/report")(_task_route(pipeline.generate_report))

    return app


# Uvicorn entry point: `uvicorn pdf_assistant.api:app`
# Guarded so importing this module never forces FastAPI/heavy deps to load.
try:  # pragma: no cover
    app = create_app()
except Exception:  # pragma: no cover
    app = None
