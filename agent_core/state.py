"""Shared state for the news-video graph."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class Sentence(TypedDict, total=False):
    """One sentence of the script and the artifacts generated from it."""

    index: int          # 1-based position in the script
    text: str           # the sentence itself
    text_file: str      # resource/text/news_001.txt
    prompt: str         # image prompt written by the LLM
    prompt_file: str    # resource/image_prompt/news_001.txt
    audio_file: str     # resource/audio/news_001.wav
    image_file: str     # resource/images/news_001.png
    error: str          # last failure for this sentence, if any


class NewsState(TypedDict, total=False):
    """State passed between graph nodes."""

    source_file: str
    summary: str        # current English script
    feedback: str       # critic feedback fed into the next rewrite
    attempts: int       # how many times the script was written
    approved: bool      # critic verdict on the current script
    sentences: list[Sentence]
    video_file: str     # final mp4, empty when rendering failed
    # Annotated reducers: nodes append only their own new entries.
    errors: Annotated[list[str], operator.add]
    log: Annotated[list[str], operator.add]
