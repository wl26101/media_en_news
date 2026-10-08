"""Text helpers: sentence splitting and the resource file layout.

The splitter mirrors step2.py/step3.py so the scripts agree on what a
sentence is.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .. import config
from ..state import Sentence

logger = logging.getLogger(__name__)


def split_sentences(text: str) -> list[str]:
    """Split into complete sentences on . ! ? or line breaks, preserving the
    sentence-final punctuation. A trailing fragment without punctuation is
    merged into the previous sentence."""
    parts = [p.strip() for p in re.split(r'(?<=[.!?]")\s+|(?<=[.!?])\s+|\n+', text) if p.strip()]
    if not parts:
        return []
    if parts[-1][-1] not in ".!?":
        tail = parts.pop()
        if parts:
            parts[-1] = f"{parts[-1]} {tail}"
        else:
            parts.append(tail + ".")
    return parts


def cut_at_sentence_boundary(text: str) -> str:
    """Drop the final fragment when the model was cut off mid-sentence.

    A reply that holds no sentence boundary at all is returned unchanged: one
    long unterminated run is still the caller's to judge, and dropping it would
    turn a truncation into an empty script.
    """
    stripped = text.strip()
    sentences = [p for p in re.split(r"(?<=[.!?])\s+|\n+", stripped) if p.strip()]
    if len(sentences) > 1 and not sentences[-1].rstrip().endswith((".", "!", "?")):
        sentences = sentences[:-1]
    return " ".join(sentences).strip() or stripped

def count_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9'’-]+", text))


def read_source(path: Path | str = config.SOURCE_FILE) -> str:
    """Read the input news text."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"source file is empty: {path}")
    return text


def write_text(path: Path | str, text: str) -> str:
    """Write text plus a trailing newline; returns the path as a string."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return str(path)


def sentence_name(index: int, count: int, suffix: str) -> str:
    """`news_007.txt` — zero padded so files sort in script order."""
    pad = max(3, len(str(count)))
    return f"{config.STEM}_{index:0{pad}d}{suffix}"


def sibling(text_file: str, directory: Path, suffix: str) -> Path:
    """`resource/image_prompt/news_001.txt` for `resource/text/news_001.txt`."""
    return Path(directory) / f"{Path(text_file).stem}{suffix}"


def write_sentence_files(sentences: list[str]) -> list[Sentence]:
    """Write one file per sentence into resource/text/, replacing any earlier
    attempt (a rewrite may produce fewer sentences)."""
    for stale in config.TEXT_DIR.glob(f"{config.STEM}_*.txt"):
        stale.unlink()
    count = len(sentences)
    result: list[Sentence] = []
    for i, sentence in enumerate(sentences, start=1):
        path = config.TEXT_DIR / sentence_name(i, count, ".txt")
        write_text(path, sentence)
        result.append({"index": i, "text": sentence, "text_file": str(path)})
    return result


def clean_outputs() -> list[str]:
    """Delete the artifacts of a previous run so stale files cannot be paired
    with fresh ones. source.txt is never touched."""
    removed: list[str] = []
    for directory, pattern in (
        (config.TEXT_DIR, f"{config.STEM}_*.txt"),
        (config.PROMPT_DIR, f"{config.STEM}_*.txt"),
        (config.AUDIO_DIR, f"{config.STEM}_*.wav"),
        (config.IMAGE_DIR, f"{config.STEM}_*.png"),
    ):
        for path in sorted(directory.glob(pattern)):
            path.unlink()
            removed.append(str(path))
    return removed
