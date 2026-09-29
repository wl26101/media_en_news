"""TTS tool: synthesize one sentence into a WAV file."""

from __future__ import annotations

import logging
from pathlib import Path

import requests
from ai.settings import settings
import ai.manage_model as manage_model

from .. import config
from ..tracing import traced

logger = logging.getLogger(__name__)


@traced
def synthesize(
    text: str,
    out_path: str | Path,
    language: str = config.TTS_LANGUAGE,
    tone: str | None = config.TTS_TONE,
    timeout: int = config.TTS_TIMEOUT,
    model_manage: manage_model.Manage_Model | None = None,
) -> str:
    """Render `text` to `out_path` and return the path. Raises on failure."""
    if model_manage is not None:
        model_manage.launch_model("tts")

    payload: dict = {"text": text, "language": language}
    if tone:
        payload["tone"] = tone

    resp = requests.post(f"http://127.0.0.1:{settings.tts_port}/v1/tts", json=payload, timeout=timeout)
    resp.raise_for_status()
    if not resp.content.startswith(b"RIFF"):
        raise RuntimeError(f"TTS did not return WAV audio (content-type: {resp.headers.get('content-type', '?')})")

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    return str(path)
