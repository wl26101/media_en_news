"""LLM tool: talk to the local llama.cpp server (OpenAI-compatible API)."""

from __future__ import annotations

import logging
import time

import requests

from ai.settings import settings
import ai.manage_model as manage_model

from .. import config
from ..tracing import traced
from .text import cut_at_sentence_boundary

logger = logging.getLogger(__name__)


def _endpoint() -> str:
    return f"http://127.0.0.1:{settings.llm_port}/v1/chat/completions"


@traced
def chat(
    prompt: str,
    system: str = config.SYSTEM_PROMPT,
    temperature: float = config.LLM_TEMPERATURE,
    max_tokens: int = config.LLM_MAX_TOKENS,
    timeout: int = config.LLM_TIMEOUT,
    retries: int = 2,
    model_manage: manage_model.Manage_Model | None = None,
) -> str:
    """Send one prompt to the local LLM and return the reply text.

    Raises requests.HTTPError / requests.RequestException when every attempt
    fails, so callers can record the error and carry on.
    """
    if model_manage is not None:
            model_manage.launch_model("llm")
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                _endpoint(),
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            choice = resp.json()["choices"][0]
            content = choice["message"]["content"].strip()
            if choice.get("finish_reason") == "length":
                logger.warning("LLM hit the %d-token cap; truncating at the last sentence", max_tokens)
                content = cut_at_sentence_boundary(content)
            return content
        except Exception as exc:
            last_error = exc
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(2)
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_error}")


@traced
def write_script(source_text: str, feedback: str = "", model_manage: manage_model.Manage_Model | None = None) -> str:
    """Write the one-minute English news script for the source text.

    `feedback` carries the editor's notes from a previous attempt.
    """
    prompt = config.SUMMARY_PROMPT + "\nSource text:\n" + source_text
    if feedback:
        prompt += (
            "\n\nYour previous draft was rejected by the editor.\n"
            f"Editor's note: {feedback}\n"
            "Rewrite the script in English and fix exactly that problem."
        )
        logger.warning("Previous draft was rejected by the editor: %s", feedback)
    return chat(prompt=prompt, model_manage=model_manage)


@traced
def review_script(script: str, source_text: str, model_manage: manage_model.Manage_Model | None = None) -> tuple[bool, str]:
    """Ask the editor LLM to approve or reject the script.

    Returns (approved, feedback). A reply that does not start with APPROVE or
    REVISE is treated as approval so a chatty model cannot stall the loop.
    """
    prompt = f"{config.REVIEW_PROMPT}\nSource text:\n{source_text}\n\nNews script:\n{script}"
    reply = chat(prompt, temperature=0.0, max_tokens=120, model_manage=model_manage).strip()
    verdict = reply.split(maxsplit=1)[0].strip(".:*#").upper() if reply else ""
    if verdict == "REVISE":
        feedback = reply[len(reply.split(maxsplit=1)[0]):].strip(" .:-\n") or "Improve the script."
        return False, feedback
    if verdict != "APPROVE":
        logger.warning("Editor reply was neither APPROVE nor REVISE (%r); accepting the script", reply[:80])
    return True, ""


@traced
def write_image_prompt(sentence: str, model_manage: manage_model.Manage_Model | None = None) -> str:
    """Ask the LLM for one comic-book-style image prompt for a sentence."""
    return chat(
        config.IMAGE_PROMPT + sentence,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.IMAGE_PROMPT_MAX_TOKENS,
        model_manage=model_manage,
    )
