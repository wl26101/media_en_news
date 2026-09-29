#!/usr/bin/env python3
"""Step 2 — Summary text → per-sentence audio files.

Reads every summary text file under a directory, splits each into sentences,
and synthesizes one WAV file per sentence via the local TTS server. A JSON
manifest (<stem>_sentences.json) records each sentence and its audio file so
a later step can concatenate them in order.

Note: the LLM curl shown under step2 in CLAUDE.md is the same boilerplate
example as step1; step2's actual work is sentence-splitting + TTS.

Usage:
    python step2.py [SUMMARY_DIR] [AUDIO_DIR]

Defaults:
    SUMMARY_DIR = outputs/
    AUDIO_DIR   = outputs/

Example (day-specific folders):
    python step2.py 260828_news/outputs 260828_news/outputs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TTS_URL = "http://localhost:8000/v1/tts"
TTS_LANGUAGE = "English"
TTS_TONE = None  # JSON null → default voice
TTS_TIMEOUT = 120  # seconds; TTS of one sentence can take a while


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """Split into complete sentences on . ! ? (or line breaks), preserving
    sentence-final punctuation. A trailing fragment that ends without
    punctuation is merged into the previous sentence."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p.strip()]
    if not parts:
        return []
    if parts[-1][-1] not in ".!?":
        tail = parts.pop()
        if parts:
            parts[-1] = f"{parts[-1]} {tail}"
        else:
            parts.append(tail + ".")
    return parts


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------


def call_tts(text: str, out_path: Path, language: str = TTS_LANGUAGE, tone: str | None = TTS_TONE) -> None:
    """Synthesize one sentence and write the WAV to out_path."""
    payload = {"text": text, "language": language, "tone": tone}
    request = urllib.request.Request(
        TTS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=TTS_TIMEOUT) as resp:
        if resp.status != 200:
            raise RuntimeError(f"TTS returned HTTP {resp.status}")
        data = resp.read()
        if not data.startswith(b"RIFF"):
            content_type = resp.headers.get("content-type", "?")
            raise RuntimeError(f"TTS did not return WAV audio (content-type: {content_type})")
    out_path.write_bytes(data)


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------


def read_summary_files(summary_dir: Path) -> list[Path]:
    files = sorted(p for p in summary_dir.glob("*.txt") if not p.name.startswith("."))
    if not files:
        raise FileNotFoundError(f"No .txt summary files found in {summary_dir}")
    return files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize per-sentence audio (step 2).")
    parser.add_argument("summary_dir", nargs="?", default="outputs", help="directory with summary text files")
    parser.add_argument("audio_dir", nargs="?", default="outputs", help="where to write WAV files")
    parser.add_argument("--language", default=TTS_LANGUAGE, help="TTS language (default: English)")
    parser.add_argument("--tone", default=TTS_TONE, help="optional TTS tone string (default: none)")
    args = parser.parse_args()

    summary_dir = Path(args.summary_dir)
    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    for path in read_summary_files(summary_dir):
        print(f"[*] Processing {path}")
        text = path.read_text(encoding="utf-8")
        sentences = split_sentences(text)
        if not sentences:
            print("    ! no sentences found, skipping", file=sys.stderr)
            continue

        stem = path.stem
        pad = len(str(len(sentences)))
        manifest = {"source": path.name, "sentences": []}
        failures = 0
        for i, sentence in enumerate(sentences, start=1):
            audio_name = f"{stem}_{i:0{pad}d}.wav"
            audio_path = audio_dir / audio_name
            try:
                call_tts(sentence, audio_path, language=args.language, tone=args.tone)
                manifest["sentences"].append({"index": i, "text": sentence, "audio": audio_name, "status": "ok"})
                print(f"    [{i}/{len(sentences)}] {audio_name}: {sentence[:60]}{'...' if len(sentence) > 60 else ''}")
            except Exception as exc:  # log and continue; batch job
                failures += 1
                manifest["sentences"].append({"index": i, "text": sentence, "audio": audio_name, "status": f"error: {exc}"})
                print(f"    [{i}/{len(sentences)}] FAILED {audio_name}: {exc}", file=sys.stderr)

        manifest_path = audio_dir / f"{stem}_sentences.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"    -> {manifest_path} ({len(sentences)} sentences, {failures} failed)")

    print("Done.")


if __name__ == "__main__":
    main()
