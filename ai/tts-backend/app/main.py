"""Qwen3-TTS HTTP backend.

Run (use the tts conda env's python, not the system python):

    cd tts-backend
    /home/wl26/miniconda3/envs/tts/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /            demo frontend page
    GET  /health      model type, supported languages, tone support
    POST /v1/tts      generate speech from {text, language, tone, ...}
"""
from __future__ import annotations

import base64
import io
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from .config import Settings
from .schemas import HealthResponse, TTSJsonResponse, TTSRequest
from .tts_service import TTSService

settings = Settings()
service: TTSService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = TTSService(settings)
    app.state.service = service
    print(
        f"[tts] loaded {service.model_type} checkpoint: {settings.model_path}",
        flush=True,
    )
    try:
        yield
    finally:
        service.unload()
        print("[tts] model unloaded", flush=True)


app = FastAPI(
    title="Qwen3-TTS Backend",
    description="Text-to-speech via Qwen3-TTS. Supports voice cloning (Base), "
    "tone/description control (VoiceDesign) and preset speakers (CustomVoice).",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the demo page and any local frontend to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def index():
    if settings.frontend_file.exists():
        return FileResponse(settings.frontend_file, media_type="text/html")
    return Response(
        "Frontend not found. See tts-backend/frontend/index.html",
        media_type="text/plain",
    )


@app.get("/health", response_model=HealthResponse)
def health():
    """Model info so clients can adapt (e.g. hide tone input if unsupported)."""
    return HealthResponse(
        status="ok",
        model_path=settings.model_path,
        model_type=service.model_type,
        tone_supported=service.tone_supported,
        supported_languages=service.supported_languages(),
        supported_speakers=service.supported_speakers(),
        sample_rate=service.sample_rate,
        device=settings.device,
    )


@app.post("/v1/tts")
def tts(req: TTSRequest):
    """Synthesize speech. Returns WAV audio (default) or a base64 JSON body."""
    if service is None:
        raise HTTPException(status_code=503, detail="Model is not ready yet.")

    gen_kwargs = dict(
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        max_new_tokens=req.max_new_tokens,
    )

    try:
        result = service.generate(
            text=req.text,
            language=req.language,
            tone=req.tone,
            ref_audio=req.ref_audio,
            ref_text=req.ref_text,
            x_vector_only=req.x_vector_only,
            speaker=req.speaker,
            **gen_kwargs,
        )
    except ValueError as e:
        # e.g. unsupported language, tone requested on the Base checkpoint
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:  # noqa: BLE001 - report whatever the model threw
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    if req.response_format == "json":
        buf = _wav_bytes(result.wavs[0], result.sample_rate)
        return TTSJsonResponse(
            base64_wav=base64.b64encode(buf).decode("ascii"),
            sample_rate=result.sample_rate,
            duration_s=round(result.duration_s, 3),
            model_type=result.model_type,
        )

    buf = _wav_bytes(result.wavs[0], result.sample_rate)
    return Response(
        content=buf,
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'inline; filename="qwen3-tts.wav"',
            "X-Sample-Rate": str(result.sample_rate),
            "X-Duration-S": f"{result.duration_s:.3f}",
        },
    )

@app.get("/v1/tts/status")
def tts_status():
    """Check if the TTS generation is completed."""
    if service is None:
        raise HTTPException(status_code=503, detail="Model is not ready yet.")

    status = service.get_generation_status()
    return {"status": status}

def _wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    return buf.getvalue()
