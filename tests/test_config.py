"""Tests for settings loading."""

from __future__ import annotations

from pdf_assistant.config import Settings


def test_defaults() -> None:
    s = Settings()
    assert s.llm_model == "claude-opus-4-8"
    assert s.chunk_size > s.chunk_overlap
    assert s.top_k >= 1


def test_from_env(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "abc123")
    monkeypatch.setenv("TOP_K", "7")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    s = Settings.from_env()
    assert s.anthropic_api_key == "abc123"
    assert s.top_k == 7
    assert s.chunk_size == 500


def test_from_env_ignores_bad_int(monkeypatch) -> None:
    monkeypatch.setenv("TOP_K", "notanumber")
    assert Settings.from_env().top_k == Settings.top_k
