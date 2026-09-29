"""The langgraph workflow: news text in, narrated video out.

    START ─▶ summarize ─┬─(no usable draft)─▶ summarize
                        └─▶ review ─┬─(approved)─▶ image_prompts ─▶ tts ─▶ images ─▶ video ─▶ report ─▶ END
                                    └─(revise)───▶ summarize

`summarize ⇄ review` is the agentic part: an editor LLM approves the English
script or hands back notes, and the writer rewrites it — at most
config.MAX_SUMMARY_ATTEMPTS times — before the expensive media steps run. A
draft the writer cannot produce at all is retried with a corrective note, and a
failed rewrite keeps the last usable draft rather than ending the run.
After that the graph is deliberately deterministic: one image prompt, one WAV
and one PNG per sentence, then ffmpeg pairs each image with its audio.

Nodes never raise: a failing sentence is recorded in the state and skipped, so
one bad TTS or image call cannot throw away the rest of the run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from ai import manage_model

from . import config
from .state import NewsState, Sentence
from .tools import image, llm, text as text_tools, tts, video

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Script: write, critique, rewrite
# ---------------------------------------------------------------------------
model_serve = manage_model.Manage_Model()

def summarize_node(state: NewsState) -> dict:
    """Write the one-minute English script and split it into sentence files.

    A failed attempt never throws the run away: the last draft that reached
    resource/text/ stays in the state, and when there is no previous draft the
    node hands back a corrective note so the next attempt can retry.
    """
    attempts = state.get("attempts", 0) + 1
    previous = state.get("summary", "")
    try:
        source_text = text_tools.read_source(state.get("source_file") or config.SOURCE_FILE)
        script = llm.write_script(source_text, feedback=state.get("feedback", ""), model_manage=model_serve)
        sentences = text_tools.split_sentences(script)
        if not sentences:
            raise RuntimeError("the model returned no usable sentence")
        records = text_tools.write_sentence_files(sentences)
    except Exception as exc:
        logger.exception("summarize failed")
        error = f"summarize attempt {attempts}: {exc}"
        if previous:
            logger.warning("keeping the previous draft (%d words)", text_tools.count_words(previous))
            return {"attempts": attempts, "feedback": "", "errors": [error],
                    "log": [f"attempt {attempts} failed ({exc}); keeping the previous draft"]}
        return {"attempts": attempts, "summary": "", "sentences": [], "approved": False,
                "feedback": config.RETRY_FEEDBACK, "errors": [error],
                "log": [f"attempt {attempts} failed ({exc})"]}

    words = text_tools.count_words(script)
    logger.info("script attempt %d: %d words, %d sentences", attempts, words, len(sentences))
    logger.debug("script attempt %d content: %s", attempts, script)
    return {"summary": script, "sentences": records, "attempts": attempts, "feedback": "",
            "log": [f"attempt {attempts}: {words} words, {len(sentences)} sentences"]}


def review_node(state: NewsState) -> dict:
    """Check the script's length, then have the editor LLM approve it."""
    script = state.get("summary", "")
    words = text_tools.count_words(script)
    if not config.MIN_WORDS <= words <= config.MAX_WORDS:
        feedback = config.LENGTH_FEEDBACK.format(
            words=words, target=config.TARGET_WORDS, low=config.MIN_WORDS, high=config.MAX_WORDS)
        return {"approved": False, "feedback": feedback, "log": [f"review: rejected — {feedback}"]}

    try:
        source_text = text_tools.read_source(state.get("source_file") or config.SOURCE_FILE)
        approved, feedback = llm.review_script(script, source_text, model_manage=model_serve)
    except Exception as exc:
        logger.error("editor LLM unavailable (%s); accepting the script", exc)
        return {"approved": True, "feedback": "", "errors": [f"review: {exc}"],
                "log": ["review: editor unavailable, accepting the script"]}

    note = "approved by the editor" if approved else f"rewrite requested — {feedback}"
    logger.info("review: %s", note)
    logger.debug("review result: approved=%s, feedback=%s", approved, feedback)
    return {"approved": approved, "feedback": feedback, "log": [f"review: {note}"]}


def route_after_summarize(state: NewsState) -> str:
    if state.get("summary"):
        return "review"
    if state.get("attempts", 0) >= config.MAX_SUMMARY_ATTEMPTS:
        return "report"
    return "retry"


def route_after_review(state: NewsState) -> str:
    if state.get("approved"):
        return "media"
    if state.get("attempts", 0) >= config.MAX_SUMMARY_ATTEMPTS:
        logger.warning("script not approved after %d attempts; using the last draft",
                       state.get("attempts", 0))
        return "media"
    return "revise"


# ---------------------------------------------------------------------------
# Media: one prompt, one WAV and one PNG per sentence
# ---------------------------------------------------------------------------


def _map_sentences(sentences: list[Sentence], label: str, action) -> tuple[list[Sentence], list[str]]:
    """Run `action(sentence)` for every sentence and merge its updates.

    A failing sentence keeps the error and the loop continues.
    """
    updated: list[Sentence] = []
    errors: list[str] = []
    for sentence in sentences:
        try:
            updated.append({**sentence, **action(sentence)})
        except Exception as exc:
            logger.error("%s failed for sentence %d: %s", label, sentence["index"], exc)
            updated.append({**sentence, "error": f"{label}: {exc}"})
            errors.append(f"{label} sentence {sentence['index']}: {exc}")
    return updated, errors


def image_prompts_node(state: NewsState) -> dict:
    """Turn every sentence into a comic-book-style image prompt."""
    def action(sentence: Sentence) -> dict:
        prompt = llm.write_image_prompt(sentence["text"], model_manage=model_serve)
        path = text_tools.write_text(text_tools.sibling(sentence["text_file"], config.PROMPT_DIR, ".txt"), prompt)
        return {"prompt": prompt, "prompt_file": path}

    sentences = state.get("sentences", [])
    updated, errors = _map_sentences(sentences, "image prompt", action)
    return {"sentences": updated, "errors": errors,
            "log": [f"image prompts: {len(sentences) - len(errors)}/{len(sentences)}"]}


def tts_node(state: NewsState) -> dict:
    """Synthesize one WAV per sentence."""
    def action(sentence: Sentence) -> dict:
        path = text_tools.sibling(sentence["text_file"], config.AUDIO_DIR, ".wav")
        return {"audio_file": tts.synthesize(sentence["text"], path, model_manage=model_serve)}

    sentences = state.get("sentences", [])
    updated, errors = _map_sentences(sentences, "tts", action)
    return {"sentences": updated, "errors": errors,
            "log": [f"audio: {len(sentences) - len(errors)}/{len(sentences)}"]}


def images_node(state: NewsState) -> dict:
    """Generate one z-Image-Turbo image per sentence."""
    def action(sentence: Sentence) -> dict:
        # a sentence whose prompt failed falls back to the sentence itself
        prompt = sentence.get("prompt") or sentence["text"]
        path = text_tools.sibling(sentence["text_file"], config.IMAGE_DIR, ".png")
        return {"image_file": image.generate_image(prompt=prompt, out_path=path, model_manage=model_serve)}

    sentences = state.get("sentences", [])
    updated, errors = _map_sentences(sentences, "image", action)
    return {"sentences": updated, "errors": errors,
            "log": [f"images: {len(sentences) - len(errors)}/{len(sentences)}"]}


def video_node(state: NewsState) -> dict:
    """Pair each image with its audio and render the final mp4."""
    sentences = state.get("sentences", [])
    try:
        path = video.build_video(expected_segments=len(sentences))
    except Exception as exc:
        logger.exception("video rendering failed")
        return {"video_file": "", "errors": [f"video: {exc}"]}
    return {"video_file": path, "log": [f"video: {path}"]}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report_node(state: NewsState) -> dict:
    """Write the run manifest and log what came out of the pipeline."""
    rows = []
    for sentence in state.get("sentences", []):
        complete = bool(sentence.get("audio_file") and sentence.get("image_file"))
        rows.append({
            "index": sentence["index"],
            "text": sentence["text"],
            "text_file": sentence.get("text_file"),
            "prompt_file": sentence.get("prompt_file"),
            "audio_file": sentence.get("audio_file"),
            "image_file": sentence.get("image_file"),
            "status": "ok" if complete else sentence.get("error", "incomplete"),
        })

    video_file = state.get("video_file", "")
    duration = video.probe_duration(Path(video_file)) if video_file else None
    manifest = {
        "source": state.get("source_file") or str(config.SOURCE_FILE),
        "video": video_file,
        "duration_s": duration,
        "words": text_tools.count_words(state.get("summary", "")),
        "attempts": state.get("attempts", 0),
        "sentences": rows,
        "errors": state.get("errors", []),
    }

    manifest_path = config.VIDEO_DIR / f"{config.STEM}_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = sum(1 for row in rows if row["status"] == "ok")
    summary = (f"{ok}/{len(rows)} sentences complete, "
               f"video={'yes' if video_file else 'no'}"
               + (f" ({duration:.1f}s)" if duration else ""))
    logger.info("done: %s", summary)
    return {"log": [f"manifest: {manifest_path}", f"done: {summary}"]}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_graph():
    """Compile the news-video workflow."""
    graph = StateGraph(NewsState)
    graph.add_node("summarize", summarize_node)
    graph.add_node("review", review_node)
    graph.add_node("image_prompts", image_prompts_node)
    graph.add_node("tts", tts_node)
    graph.add_node("images", images_node)
    graph.add_node("video", video_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "summarize")
    graph.add_conditional_edges("summarize", route_after_summarize,
                                {"review": "review", "retry": "summarize", "report": "report"})
    graph.add_conditional_edges("review", route_after_review, {"media": "image_prompts", "revise": "summarize"})
    graph.add_edge("image_prompts", "tts")
    graph.add_edge("tts", "images")
    graph.add_edge("images", "video")
    graph.add_edge("video", "report")
    graph.add_edge("report", END)
    return graph.compile()
