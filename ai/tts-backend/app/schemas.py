"""Request / response models for the TTS HTTP API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    """Body of POST /v1/tts.

    Fields map to the arguments of the underlying qwen_tts generation API.
    Which fields matter depends on the loaded checkpoint type (see /health):

    * ``base``        -> voice cloning: needs ``ref_audio`` (defaults to the
                         built-in reference voice) + ``ref_text``. ``tone`` is
                         NOT supported.
    * ``voice_design``-> needs ``tone`` (natural-language voice description).
    * ``custom_voice``-> needs ``speaker``; ``tone`` is optional.
    """

    text: str = Field(
        ...,
        min_length=1,
        description="The sentence to synthesize.",
        examples=["I am solving the equation: x = [-b ± sqrt(b²-4ac)] / 2a."],
    )
    language: str = Field(
        "Auto",
        description=(
            "Language of the target text. One of the model's supported "
            "languages, or 'Auto' to let the model decide. "
            "e.g. Chinese / English / Japanese / Korean / German / French / "
            "Russian / Portuguese / Spanish / Italian."
        ),
    )
    tone: Optional[str] = Field(
        None,
        max_length=500,
        description=(
            "Tone / style description in natural language, e.g. "
            "'speak in a very sad tone, slowly'. Only honored by the "
            "VoiceDesign and CustomVoice checkpoints; the Base checkpoint "
            "raises 422 if this is set."
        ),
    )

    # --- voice-clone inputs (Base model) ------------------------------------
    ref_audio: Optional[str] = Field(
        None,
        description=(
            "Reference audio for voice cloning: a local wav path, an http(s) "
            "URL, or a base64 data URI. Defaults to the built-in voice."
        ),
    )
    ref_text: Optional[str] = Field(
        None,
        description="Transcript of ref_audio (required for ICL mode on Base).",
    )
    x_vector_only: bool = Field(
        False,
        description="Base model: use only the speaker embedding, ignore ref_text.",
    )

    # --- custom-voice inputs (CustomVoice model) -----------------------------
    speaker: Optional[str] = Field(
        None,
        description="Predefined speaker name (CustomVoice checkpoint only).",
    )

    # --- generation parameters (all optional) --------------------------------
    response_format: Literal["wav", "json"] = Field(
        "wav",
        description="'wav' streams audio; 'json' returns a base64 payload.",
    )
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(None, ge=1)
    top_p: Optional[float] = Field(None, gt=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(None, ge=0.0)
    max_new_tokens: Optional[int] = Field(None, ge=1)

    model_config = {"extra": "forbid"}


class HealthResponse(BaseModel):
    """Body of GET /health."""

    status: Literal["ok"] = "ok"
    model_path: str
    model_type: Literal["base", "voice_design", "custom_voice"]
    tone_supported: bool
    supported_languages: list[str]
    supported_speakers: list[str]
    sample_rate: Optional[int] = None
    device: str


class TTSJsonResponse(BaseModel):
    """Body when ``response_format=json``."""

    base64_wav: str
    sample_rate: int
    duration_s: float
    model_type: str
