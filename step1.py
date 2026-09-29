#!/usr/bin/env python3
"""Step 1 — News text → 1-minute spoken English summary.

Reads every file under a source directory, sends each to the local
OpenAI-compatible LLM, and writes a de-duplicated summary to the output
directory.

Usage:
    python step1.py [SOURCE_DIR] [OUTPUT_DIR]

Defaults:
    SOURCE_DIR = source/
    OUTPUT_DIR = outputs/

Example (day-specific folders):
    python step1.py 260828_news/source 260828_news/outputs

Note: functions are plain and side-effect free so they can later be wrapped
as langgraph nodes (see CLAUDE.md tech stack).
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

LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
TEMPERATURE = 0.2
# The local model rambles and rarely emits a clean "stop" on its own, so we
# keep a fixed budget and cut at the last complete sentence if it truncates.
# ~500 tokens ≈ a comfortable 1-minute spoken summary.
MAX_TOKENS = 500

# ‑ is the U+2011 non-breaking hyphen used in the CLAUDE.md prompt.
PROMPT = (
    "Generate a 1‑minute spoken English summary of the source text. "
    "Respond in English only. "
    "Only output bare objective facts: time, location, relevant persons, and events. "
    "No rhetoric, comments, analysis or promotional language. "
    "State each fact exactly once; never repeat the same sentence, fact, or phrase."
)

# Optional second pass: ask the LLM to drop sentences that restate an earlier
# fact. Lexical de-dup alone misses rephrased repeats (e.g. "3 dead" vs
# "3 deaths"), and the local model understands the text better than token
# overlap ever will. Disable with --no-refine.
REFINE_PROMPT = (
    "The text below is an English news summary. Rewrite it so that every fact "
    "is stated exactly once: drop any sentence that restates a fact already "
    "given, and keep only bare objective facts (time, location, persons, "
    "events). Add nothing new. Return the revised summary only."
)
REFINE_MAX_TOKENS = 400

# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------


def build_payload(source_text: str, max_tokens: int) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are a concise news summarizer. Always reply in English."},
            {"role": "user", "content": PROMPT + "\n\nSource text:\n" + source_text},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
    }


def call_llm(payload: dict, timeout: int = 120) -> dict:
    """POST to the local LLM; returns the parsed JSON response."""
    request = urllib.request.Request(
        LLM_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"LLM returned HTTP {resp.status}")
        return json.loads(resp.read().decode("utf-8"))


def cut_at_sentence_boundary(text: str) -> str:
    """Drop the final fragment if the model was cut off mid-sentence."""
    sentences = re.split(r"(?<=\.)\s+", text.strip())
    if sentences and not sentences[-1].rstrip().endswith("."):
        sentences = sentences[:-1]
    return " ".join(sentences).strip()


def summarize(text: str) -> str:
    """Call the LLM once. If the model hits the token cap mid-sentence, cut
    the output at the last complete sentence instead of retrying."""
    data = call_llm(build_payload(text, MAX_TOKENS))
    choice = data["choices"][0]
    content = choice["message"]["content"].strip()
    if choice.get("finish_reason") == "length":
        print(f"    ... hit {MAX_TOKENS}-token cap, cutting at last sentence", file=sys.stderr)
        return cut_at_sentence_boundary(content)
    return content


def refine_summary(summary: str) -> str:
    """Second LLM pass: drop sentences that restate an earlier fact."""
    payload = {
        "messages": [
            {"role": "system", "content": "You are a concise news editor. Always reply in English."},
            {"role": "user", "content": REFINE_PROMPT + "\n\nSummary:\n" + summary},
        ],
        "temperature": 0.2,
        "max_tokens": REFINE_MAX_TOKENS,
    }
    data = call_llm(payload)
    choice = data["choices"][0]
    content = choice["message"]["content"].strip()
    if choice.get("finish_reason") == "length":
        print(f"    ... refine hit {REFINE_MAX_TOKENS}-token cap, cutting at last sentence", file=sys.stderr)
        return cut_at_sentence_boundary(content)
    return content


# ---------------------------------------------------------------------------
# De-duplication
# ---------------------------------------------------------------------------


def normalize_sentence(sentence: str) -> str:
    """Lowercase, strip punctuation and whitespace; letters and digits only."""
    lowered = sentence.lower()
    stripped = re.sub(r"[^a-z0-9 ]", " ", lowered)
    return " ".join(stripped.split())


def extract_numbers(sentence: str) -> set[str]:
    """Significant numbers in a sentence. Clock times (8:00, 10:30 AM) are
    skipped so timestamps don't over-trigger the de-dup rule; ordinals like
    '27th' fold to '27'."""
    without_times = re.sub(r"\b\d{1,2}:\d{2}(?:\s*[ap]m)?\b", " ", sentence, flags=re.IGNORECASE)
    return set(re.findall(r"\d+", without_times.replace(",", "")))


def is_duplicate(
    candidate_tokens: set[str],
    candidate_numbers: set[str],
    seen_entries: list[tuple[set[str], set[str]]],
    threshold: float = 0.6,
) -> bool:
    """True if the candidate sentence repeats one already kept, using:
      - Jaccard overlap of word tokens (verbatim / near-verbatim repeats), or
      - high coverage (candidate is basically a subset of a kept sentence), or
      - shared significant numbers + moderate coverage (rephrased fact repeats
        like "3 dead" vs "3 deaths")."""
    if not candidate_tokens:
        return True  # empty / punctuation-only fragments are dropped
    for other_tokens, other_numbers in seen_entries:
        overlap = len(candidate_tokens & other_tokens)
        union = len(candidate_tokens | other_tokens)
        jaccard = overlap / union if union else 0.0
        coverage = overlap / len(candidate_tokens)
        shared_numbers = len(candidate_numbers & other_numbers)
        if jaccard >= threshold:
            return True
        if coverage >= 0.75:
            return True
        if shared_numbers >= 2 and coverage >= 0.4:
            return True
    return False


def dedupe_sentences(text: str) -> str:
    """Remove duplicate / near-duplicate sentences, preserving order and
    paragraph breaks."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    seen: list[tuple[set[str], set[str]]] = []
    kept_paragraphs: list[str] = []
    for paragraph in paragraphs:
        sentences = re.split(r"(?<=\.)\s+", paragraph.strip())
        kept: list[str] = []
        for sentence in sentences:
            norm = normalize_sentence(sentence)
            tokens = set(norm.split())
            if not tokens or is_duplicate(tokens, extract_numbers(sentence), seen):
                continue
            seen.append((tokens, extract_numbers(sentence)))
            kept.append(sentence.strip())
        if kept:
            kept_paragraphs.append(" ".join(kept))
    return "\n\n".join(kept_paragraphs).strip()


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------


def read_source_files(source_dir: Path) -> list[Path]:
    files = sorted(p for p in source_dir.iterdir() if p.is_file() and not p.name.startswith("."))
    if not files:
        raise FileNotFoundError(f"No files found in {source_dir}")
    return files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate English news summaries (step 1).")
    parser.add_argument("source_dir", nargs="?", default="source", help="directory with source text files")
    parser.add_argument("output_dir", nargs="?", default="outputs", help="where to write summaries")
    parser.add_argument("--no-refine", action="store_true",
                        help="skip the second LLM pass that removes restated facts")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in read_source_files(source_dir):
        print(f"[*] Summarizing {path}")
        text = path.read_text(encoding="utf-8")
        summary = dedupe_sentences(summarize(text))
        if not args.no_refine:
            summary = dedupe_sentences(refine_summary(summary))
        out_path = output_dir / f"{path.stem}_summary.txt"
        out_path.write_text(summary + "\n", encoding="utf-8")
        print(f"    -> {out_path} ({len(summary)} chars)")

    print("Done.")


if __name__ == "__main__":
    main()
