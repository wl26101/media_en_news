"""Agentic English news-video pipeline (langgraph + langsmith).

Puts the project root on sys.path so the package can reuse the shared
`ai.manage_model` servers and the ffmpeg helpers in `step5.py` no matter how
it is launched.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
