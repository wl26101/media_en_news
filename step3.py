#!/usr/bin/env python3
"""Step 3 — Sentence → image prompt (American comic book style).

For every sentence produced by step2 (read from the *_sentences.json
manifests in the summary dir; falling back to splitting the summary .txt
files), calls the local LLM to write a detailed image-generation prompt, and
saves one prompt file per sentence plus a *_prompts.json manifest into the
image-gen directory.

Usage:
    python step3.py [SUMMARY_DIR] [IMAGE_DIR]

Defaults:
    SUMMARY_DIR = outputs/
    IMAGE_DIR   = image_gen/

Example (day-specific folders):
    python step3.py 260828_news/outputs 260828_news/image_gen
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
# The model writes long prose prompts and never emits a clean stop, so we use
# a fixed budget and cut at the last complete sentence if it truncates.
MAX_TOKENS = 250
STYLE = "American comic book"


def cut_at_sentence_boundary(text: str) -> str:
    """Cut a truncated prompt at the last complete sentence, keeping trailing
    closing punctuation. Returns the original text if there is no clean
    sentence boundary."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) <= 1:
        return text  # keyword-style prompt, no sentence structure; keep it all
    last = sentences[-1].rstrip()
    core = last.rstrip('"\')\]}»”’')
    if not core.endswith((".", "!", "?")):
        sentences = sentences[:-1]
    return " ".join(sentences).strip()


def build_image_prompt(sentence: str) -> str:
    return (
        "Write one detailed English image-generation prompt for the following "
        "news sentence. Render it in American comic book style: bold ink "
        "outlines, dramatic shading and hatching, halftone dot textures, "
        "saturated colors, and dynamic, expressive compositions. Describe the "
        "scene, setting, characters, weather, and any objects or text visible "
        "in the frame. Output only the prompt text itself, nothing else.\n\n"
        f"Sentence: {sentence}"
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


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


def generate_image_prompt(sentence: str) -> str:
    """Ask the LLM for one image prompt for a sentence."""
    payload = {
        "messages": [
            {"role": "system", "content": "You write image-generation prompts. Always reply in English."},
            {"role": "user", "content": build_image_prompt(sentence)},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    data = call_llm(payload)
    choice = data["choices"][0]
    content = choice["message"]["content"].strip()
    if choice.get("finish_reason") == "length":
        print(f"    ... prompt truncated at {MAX_TOKENS} tokens, cutting at last sentence", file=sys.stderr)
        return cut_at_sentence_boundary(content)
    return content


# ---------------------------------------------------------------------------
# Sentence gathering
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """Fallback splitter (mirrors step2): complete sentences on . ! ? or line
    breaks; a trailing fragment without punctuation is merged into the
    previous sentence."""
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


def gather_sentences(summary_dir: Path) -> list[tuple[str, list[str]]]:
    """Return [(stem, [sentence, ...]), ...] for each source.

    Prefers the step2 manifests (*_sentences.json) so the image prompts line
    up 1:1 with the synthesized audio; falls back to splitting the summary
    .txt files when no manifests exist yet."""
    manifests = sorted(summary_dir.glob("*_sentences.json"))
    if manifests:
        result: list[tuple[str, list[str]]] = []
        for manifest_path in manifests:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            sentences = [s["text"] for s in data.get("sentences", []) if s.get("status") == "ok"]
            stem = manifest_path.stem[: -len("_sentences")]
            result.append((stem, sentences))
        return result

    files = sorted(summary_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No *_sentences.json or *.txt files found in {summary_dir}")
    return [(p.stem, split_sentences(p.read_text(encoding="utf-8"))) for p in files]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate image prompts per sentence (step 3).")
    parser.add_argument("summary_dir", nargs="?", default="outputs", help="directory with step2 manifests / summary text files")
    parser.add_argument("image_dir", nargs="?", default="image_gen", help="where to write image prompts")
    parser.add_argument("--style", default=STYLE, help="image style to bake into each prompt")
    args = parser.parse_args()

    summary_dir = Path(args.summary_dir)
    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    for stem, sentences in gather_sentences(summary_dir):
        if not sentences:
            print(f"    ! {stem}: no sentences, skipping", file=sys.stderr)
            continue
        print(f"[*] Processing {stem} ({len(sentences)} sentences)")
        pad = len(str(len(sentences)))
        manifest = {"source": f"{stem}.txt", "style": args.style, "prompts": []}
        failures = 0
        for i, sentence in enumerate(sentences, start=1):
            prompt_name = f"{stem}_{i:0{pad}d}_prompt.txt"
            prompt_path = image_dir / prompt_name
            try:
                prompt = generate_image_prompt(sentence)
                prompt_path.write_text(prompt + "\n", encoding="utf-8")
                manifest["prompts"].append({
                    "index": i, "sentence": sentence, "prompt": prompt,
                    "prompt_file": prompt_name, "status": "ok",
                })
                print(f"    [{i}/{len(sentences)}] {prompt_name}: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
            except Exception as exc:  # log and continue; batch job
                failures += 1
                manifest["prompts"].append({
                    "index": i, "sentence": sentence, "prompt_file": prompt_name, "status": f"error: {exc}",
                })
                print(f"    [{i}/{len(sentences)}] FAILED {prompt_name}: {exc}", file=sys.stderr)

        manifest_path = image_dir / f"{stem}_prompts.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"    -> {manifest_path} ({len(sentences)} prompts, {failures} failed)")

    print("Done.")


if __name__ == "__main__":
    main()
