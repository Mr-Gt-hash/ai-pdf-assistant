"""Prompt templates for the assistant's tasks.

Kept separate so prompt wording can be tuned without touching pipeline logic.
"""

from __future__ import annotations

from typing import List

from .vector_store import Retrieved

SYSTEM_QA = (
    "You are a precise assistant that answers questions using ONLY the provided "
    "context from a PDF document. If the answer is not in the context, say you "
    "could not find it. Cite page numbers in square brackets like [p.3]."
)

SYSTEM_SUMMARY = (
    "You are a summarization assistant. Produce a clear, structured summary of "
    "the provided document context using short bullet points grouped by theme."
)

SYSTEM_TABLES = (
    "You extract tabular data. From the provided context, reconstruct any tables "
    "you find as GitHub-flavored Markdown tables. If none exist, say so."
)

SYSTEM_DATES = (
    "You extract dates and their significance. List every date found in the "
    "context with a short note on what it refers to, as a Markdown list."
)

SYSTEM_REPORT = (
    "You are a report writer. Using the provided context, produce a concise "
    "structured report with sections: Overview, Key Points, Risks, and Actions."
)


def build_context(chunks: List[Retrieved]) -> str:
    """Render retrieved chunks into a numbered, page-tagged context block."""
    lines: List[str] = []
    for i, ch in enumerate(chunks, start=1):
        page = ch.metadata.get("page", "?")
        lines.append(f"[Chunk {i} | p.{page}]\n{ch.text}")
    return "\n\n".join(lines)


def qa_user_prompt(question: str, context: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def task_user_prompt(context: str) -> str:
    return f"Context:\n{context}\n\nProduce the requested output:"
