"""End-to-end pipeline tests using in-memory fakes (no heavy deps/network)."""

from __future__ import annotations

import pytest

from pdf_assistant import prompts
from pdf_assistant.errors import DocumentNotFoundError
from pdf_assistant.vector_store import Retrieved


def _seed(store, embedder, doc_id="doc1", n=3):
    texts = [f"Contract clause number {i} dated 2026-0{i}-01." for i in range(1, n + 1)]
    embs = embedder.embed(texts)
    store.add(
        ids=[f"{doc_id}:{i}" for i in range(n)],
        embeddings=embs,
        documents=texts,
        metadatas=[{"doc_id": doc_id, "page": i + 1, "source": "c.pdf"} for i in range(n)],
    )


def test_retrieve_returns_chunks(pipeline):
    pipe, embedder, store, _ = pipeline
    _seed(store, embedder)
    results = pipe.retrieve("clause")
    assert results and all(isinstance(r, Retrieved) for r in results)
    assert len(results) <= pipe._settings.top_k


def test_retrieve_empty_raises(pipeline):
    pipe, *_ = pipeline
    with pytest.raises(DocumentNotFoundError):
        pipe.retrieve("anything")


def test_ask_wires_qa_prompt(pipeline):
    pipe, embedder, store, llm = pipeline
    _seed(store, embedder)
    answer = pipe.ask("What is clause 1?")
    assert answer.startswith("ANSWER::")
    assert llm.last_system == prompts.SYSTEM_QA
    assert "Question: What is clause 1?" in llm.last_user


def test_summarize_uses_summary_system(pipeline):
    pipe, embedder, store, llm = pipeline
    _seed(store, embedder)
    pipe.summarize()
    assert llm.last_system == prompts.SYSTEM_SUMMARY


def test_all_tasks_run(pipeline):
    pipe, embedder, store, llm = pipeline
    _seed(store, embedder)
    for fn, sys_prompt in [
        (pipe.extract_tables, prompts.SYSTEM_TABLES),
        (pipe.find_dates, prompts.SYSTEM_DATES),
        (pipe.generate_report, prompts.SYSTEM_REPORT),
    ]:
        fn()
        assert llm.last_system == sys_prompt


def test_doc_filter(pipeline):
    pipe, embedder, store, _ = pipeline
    _seed(store, embedder, doc_id="A", n=2)
    _seed(store, embedder, doc_id="B", n=2)
    results = pipe.retrieve("clause", doc_id="B")
    assert all(r.metadata["doc_id"] == "B" for r in results)


def test_build_context_tags_pages():
    chunks = [Retrieved(text="hello", metadata={"page": 5}, distance=0.1)]
    ctx = prompts.build_context(chunks)
    assert "p.5" in ctx and "hello" in ctx
