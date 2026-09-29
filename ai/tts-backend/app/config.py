"""Configuration for the Qwen3-TTS backend.

Every value can be overridden through environment variables so the same
codebase can point at different Qwen3-TTS checkpoints without editing files.
"""
from __future__ import annotations

import os
from pathlib import Path

# tts-backend/
BASE_DIR = Path(__file__).resolve().parent.parent

# Parent dir contains the downloaded checkpoints (../models/Qwen3-TTS-12Hz-0.6B-Base).
MODELS_DIR = BASE_DIR.parent.parent / "models"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


class Settings:
    """Runtime settings, all overridable via environment variables."""

    def __init__(self) -> None:
        # --- Model -----------------------------------------------------------
        # Path to a Qwen3-TTS checkpoint (Base / VoiceDesign / CustomVoice).
        # e.g. MODEL_PATH=../Qwen3-TTS-12Hz-1.7B-VoiceDesign
        self.model_path: str = _env(
            "MODEL_PATH",
            str(MODELS_DIR / "Qwen3-TTS-12Hz-0.6B-Base"),
        )
        # cuda:0 / cpu
        self.device: str = _env("DEVICE", "cuda:0")
        # bfloat16 / float16 / float32
        self.dtype: str = _env("DTYPE", "bfloat16")
        # "flash_attention_2" if flash-attn is installed, otherwise empty (sdpa)
        self.attn_implementation: str = _env("ATTN_IMPL", "").strip()

        # --- Voice-clone defaults (same reference as test.py) ----------------
        # Used by the Base model when the request does not supply ref_audio/ref_text.
        self.default_ref_audio: str = _env(
            "DEFAULT_REF_AUDIO",
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav",
        )
        self.default_ref_text: str = _env(
            "DEFAULT_REF_TEXT",
            "Okay. Yeah. I resent you. I love you. I respect you. "
            "But you know what? You blew it! And thanks to you.",
        )

        # --- Server ----------------------------------------------------------
        self.host: str = _env("HOST", "0.0.0.0")
        self.port: int = _env_int("PORT", 8000)
        self.output_dir: Path = Path(_env("OUTPUT_DIR", str(BASE_DIR / "outputs")))
        self.frontend_file: Path = BASE_DIR / "frontend" / "index.html"

        # --- Generation defaults (fall through to model's generation_config) --
        self.temperature: float | None = _env_float("TEMPERATURE")
        self.top_p: float | None = _env_float("TOP_P")
        self.top_k: int | None = _env_int_opt("TOP_K")
        self.repetition_penalty: float | None = _env_float("REPETITION_PENALTY")
        self.max_new_tokens: int | None = _env_int_opt("MAX_NEW_TOKENS")

        # Serialize concurrent generations; TTS is GPU-bound so queueing is safer
        # than risking OOM from parallel decode.
        self.serialize_requests: bool = _env_bool("SERIALIZE_REQUESTS", True)

    @property
    def dtype_torch(self):
        import torch

        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.dtype]


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    return float(raw) if raw else None


def _env_int_opt(name: str) -> int | None:
    raw = os.environ.get(name)
    return int(raw) if raw else None


settings = Settings()
