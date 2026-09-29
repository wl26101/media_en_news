"""LangSmith tracing helpers.

Tracing is opt-in through the standard LangSmith environment variables:

    export LANGCHAIN_TRACING_V2=true
    export LANGCHAIN_API_KEY=lsv2_...
    export LANGCHAIN_PROJECT=media_en_news

Without them the pipeline runs untraced and `traced` degrades to the plain
function, so nothing here is required to use the pipeline.
"""

from __future__ import annotations

import logging
import os
from typing import Callable

logger = logging.getLogger(__name__)


def _api_key() -> str | None:
    return os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")


def tracing_enabled() -> bool:
    flag = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or ""
    return flag.strip().lower() in ("1", "true", "yes", "on") and bool(_api_key())


def setup_tracing() -> bool:
    """Report whether LangSmith tracing is on; returns the flag."""
    if tracing_enabled():
        logger.info("LangSmith tracing enabled (project=%s)", os.getenv("LANGCHAIN_PROJECT", "default"))
        return True
    logger.info("LangSmith tracing disabled (set LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY to enable)")
    return False


def traced(fn: Callable) -> Callable:
    """Wrap a tool with `langsmith.traceable` when tracing is enabled."""
    if not tracing_enabled():
        return fn
    try:
        from langsmith import traceable

        return traceable(fn)
    except Exception as exc:  # never let tracing break the pipeline
        logger.warning("Could not enable tracing for %s: %s", getattr(fn, "__name__", fn), exc)
        return fn
