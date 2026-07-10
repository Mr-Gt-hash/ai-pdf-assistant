"""Centralized logging setup. Only the app entry points configure handlers."""

from __future__ import annotations

import logging
import sys

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger once (idempotent)."""
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
