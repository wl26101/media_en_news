"""TTS engine: loads a Qwen3-TTS checkpoint once and dispatches to the right
generation API based on the checkpoint type.

Qwen3-TTS ships several 0.6B/1.7B checkpoints, each specialized differently:

* ``base``         -> voice cloning  -> ``generate_voice_clone(text, language,
                      ref_audio, ref_text, x_vector_only_mode)``
* ``voice_design`` -> description    -> ``generate_voice_design(text, language,
                      instruct=tone)``
* ``custom_voice`` -> preset speaker -> ``generate_custom_voice(text, speaker,
                      language, instruct=tone)``

The checkpoint type is read from its ``config.json`` (``tts_model_type``), so
swapping ``MODEL_PATH`` to a different checkpoint changes behavior with no code
changes. The Base checkpoint (the one used by test.py) does NOT support
free-text tone control -- its tone follows the style of the reference audio.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from qwen_tts import Qwen3TTSModel

from .config import Settings

SUPPORTED_TYPES = ("base", "voice_design", "custom_voice")

# Message shown when `tone` is requested from a model that cannot do it.
TONE_UNSUPPORTED_MSG = (
    "The loaded Qwen3-TTS '{model_type}' checkpoint does not support free-text "
    "tone control. Tone/description control requires a VoiceDesign checkpoint "
    "(e.g. Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) -- point MODEL_PATH at it and "
    "restart. With the Base checkpoint, the tone follows the reference audio's "
    "style; upload a ref_audio that already sounds the way you want."
)


@dataclass
class GenerationResult:
    wavs: list
    sample_rate: int
    model_type: str
    duration_s: float


class TTSService:
    """Holds the loaded model and exposes one ``generate()`` entry point."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: Optional[Qwen3TTSModel] = None
        self.model_type: Optional[str] = None
        self.sr: Optional[int] = None
        self._lock = None
        self.load()

    # ------------------------------------------------------------------ setup
    def load(self) -> None:
        cfg = self.settings
        kwargs: dict[str, Any] = dict(device_map=cfg.device, dtype=cfg.dtype_torch)
        if cfg.attn_implementation:
            kwargs["attn_implementation"] = cfg.attn_implementation

        self.model = Qwen3TTSModel.from_pretrained(cfg.model_path, **kwargs)

        self.model_type = getattr(self.model.model, "tts_model_type", None)
        if self.model_type not in SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported Qwen3-TTS model type: {self.model_type!r}. "
                f"Expected one of {SUPPORTED_TYPES}."
            )

        if cfg.serialize_requests:
            import threading

            self._lock = threading.Lock()

    def unload(self) -> None:
        if self.model is not None:
            self.model = None
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- queries
    def supported_languages(self) -> list[str]:
        fn = getattr(self.model.model, "get_supported_languages", None)
        return list(fn()) if callable(fn) and fn() else []

    def supported_speakers(self) -> list[str]:
        fn = getattr(self.model.model, "get_supported_speakers", None)
        return list(fn()) if callable(fn) and fn() else []

    def get_generation_status(self) -> str:
        if self.model is None:
            return "not_ready"
        return "completed"

    @property
    def tone_supported(self) -> bool:
        return self.model_type in ("voice_design", "custom_voice")

    @property
    def sample_rate(self) -> Optional[int]:
        return self.sr

    # ------------------------------------------------------------- generation
    def generate(
        self,
        text: str,
        language: str = "Auto",
        tone: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        x_vector_only: bool = False,
        speaker: Optional[str] = None,
        **gen_kwargs: Any,
    ) -> GenerationResult:
        """Synthesize audio, dispatching on the loaded checkpoint type."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        tone = (tone or "").strip() or None
        text = text.strip()
        if not text:
            raise ValueError("'text' must not be empty.")

        # Keep only non-None kwargs; the wrapper merges in its defaults.
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        if self.model_type == "base":
            wavs, sr = self._generate_voice_clone(
                text, language, tone, ref_audio, ref_text, x_vector_only, gen_kwargs
            )
        elif self.model_type == "voice_design":
            wavs, sr = self._generate_voice_design(text, language, tone, gen_kwargs)
        else:  # custom_voice
            wavs, sr = self._generate_custom_voice(
                text, language, speaker, tone, gen_kwargs
            )

        duration_s = float(len(wavs[0]) / sr) if len(wavs[0]) else 0.0
        self.sr = int(sr)
        return GenerationResult(
            wavs=wavs,
            sample_rate=int(sr),
            model_type=self.model_type,
            duration_s=duration_s,
        )

    # ------------------------------------------------------- base / voice clone
    def _generate_voice_clone(
        self,
        text: str,
        language: str,
        tone: Optional[str],
        ref_audio: Optional[str],
        ref_text: Optional[str],
        x_vector_only: bool,
        gen_kwargs: dict,
    ):
        if tone is not None:
            raise ValueError(TONE_UNSUPPORTED_MSG.format(model_type="base"))

        if ref_audio is None:
            ref_audio = self.settings.default_ref_audio
        if ref_text is None and not x_vector_only:
            ref_text = self.settings.default_ref_text

        with self._guard():
            return self.model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=bool(x_vector_only),
                **gen_kwargs,
            )

    # ------------------------------------------------------ voice_design / tone
    def _generate_voice_design(
        self, text: str, language: str, tone: Optional[str], gen_kwargs: dict
    ):
        if not tone:
            raise ValueError(
                "The VoiceDesign checkpoint requires a 'tone' description "
                "(e.g. 'speak in a sad tone, slowly')."
            )
        with self._guard():
            return self.model.generate_voice_design(
                text=text,
                language=language,
                instruct=tone,
                **gen_kwargs,
            )

    # ------------------------------------------------------- custom_voice / tone
    def _generate_custom_voice(
        self,
        text: str,
        language: str,
        speaker: Optional[str],
        tone: Optional[str],
        gen_kwargs: dict,
    ):
        if not speaker:
            raise ValueError(
                "The CustomVoice checkpoint requires a 'speaker' name. "
                f"Supported: {self.supported_speakers()}"
            )
        with self._guard():
            return self.model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=tone,
                **gen_kwargs,
            )

    # ------------------------------------------------------------------ guard
    def _guard(self):
        """Serialize generation if configured (default on)."""
        if self._lock is not None:
            return self._lock
        return _nullcontext()


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
