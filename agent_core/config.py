"""Configuration for the agent_core news-video pipeline.

Layout (all under <project>/resource/):

    text/source.txt      input news text (Chinese)
    text/news_001.txt    step 1 — one file per sentence of the English script
    image_prompt/news_001.txt   step 2 — image prompt per sentence
    audio/news_001.wav          step 3 — TTS per sentence
    images/news_001.png         step 4 — z-Image-Turbo image per sentence
    video/news.mp4              step 5 — one image + one audio per segment

Everything shares the STEM + index naming (news_001, news_002, ...) so the
video step can pair images with audio by file stem alone.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RESOURCE_DIR = ROOT / "resource"

TEXT_DIR = RESOURCE_DIR / "text"
SOURCE_FILE = TEXT_DIR / "source.txt"
PROMPT_DIR = RESOURCE_DIR / "image_prompt"
AUDIO_DIR = RESOURCE_DIR / "audio"
IMAGE_DIR = RESOURCE_DIR / "images"
VIDEO_DIR = RESOURCE_DIR / "video"

# Shared file stem for every generated artifact of one run.
STEM = "news"

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# ~1 minute of spoken English is roughly 150 words. 2 min is roughly 300 words.
TARGET_WORDS = 300
MIN_WORDS = 220
MAX_WORDS = 350

# Reflection loop: the script is critiqued and rewritten at most this often.
MAX_SUMMARY_ATTEMPTS = 3

# A 300-word script is ~400 tokens and the model overshoots it, so a 500-token
# budget cut the reply off mid-script and the draft was judged on a truncation.
LLM_MAX_TOKENS = 800
LLM_TEMPERATURE = 0.2
LLM_TIMEOUT = 180

# Below this share of Latin letters the reply is not English at all.
MIN_ENGLISH_RATIO = 0.8

IMAGE_PROMPT_MAX_TOKENS = 250

TTS_LANGUAGE = "English"
TTS_TONE = None
TTS_TIMEOUT = 120

IMAGE_TIMEOUT = 600  # seconds to wait for one ComfyUI generation
VIDEO_FPS = 25

# Model server startup budgets (seconds). ComfyUI needs the longest.
READY_TIMEOUTS = {"llm": 180, "tts": 180, "image": 300}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SUMMARY_PROMPT = f"""\
You will receive multiple separate news articles. DO NOT simply summarize each article one by one and concatenate them in input order.
Integrate all key information across all provided news texts, merge overlapping topics, remove duplicate facts, then rewrite into a cohesive 2-minute spoken English news broadcast script.

Requirements:
- Output only English. No Chinese characters.
- Target word count: {TARGET_WORDS}, suitable for natural reading aloud in around 2 minutes.
- Strictly objective factual reporting only: preserve key time, location, subjects and events.
- No commentary, analysis, speculation, subjective opinions, metaphors or persuasive language.
- Each unique fact appears exactly once; avoid repeated sentences, facts or redundant phrases.
- Deliver as one continuous plain prose paragraph. No headings, bullet points, direct quotes, markdown formatting.
- Output nothing except the final news script.
"""

REVIEW_PROMPT = f"""\
You are a news editor. Check the English news script below against the source text.

Reply with exactly one word first:
- APPROVE if the script is English, factual, about {TARGET_WORDS} words \
long, and repeats no fact.
- REVISE if anything is wrong.

If you reply REVISE, add one short sentence saying what to fix. Nothing else.
"""

IMAGE_PROMPT = (
    "Write one detailed English image-generation prompt for the following news "
    "sentence. Render it in American comic book style: bold ink outlines, "
    "dramatic shading and hatching, halftone dot textures, saturated colors, "
    "and dynamic, expressive compositions. Describe the scene, setting, "
    "characters, weather, and any objects or text visible in the frame. "
    "Output only the prompt text itself, nothing else.\n\nSentence: "
)

SYSTEM_PROMPT = "You are a concise English news writer. Always reply in English."

# Sent back to the writer when its reply could not be turned into a script at
# all — usually because it answered in the source language.
RETRY_FEEDBACK = """\
Your previous reply could not be used: it was not English prose. Reply with the \
English news script only, in plain English sentences. No Chinese characters, no \
headings, no commentary."""

# Editor note when the script is too short or too long for the target length.
LENGTH_FEEDBACK = "the script is {words} words; rewrite it to about {target} words ({low}-{high})"
