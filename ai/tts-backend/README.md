# Qwen3-TTS Backend

FastAPI backend that turns **sentence + language + tone** into speech using
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), built around the same
`generate_voice_clone` API as `../test.py`.

It ships with the **Base** checkpoint (`Qwen3-TTS-12Hz-0.6B-Base`) for 3-second
voice cloning, and automatically adapts to the other checkpoints in the family:

| Checkpoint (`MODEL_PATH`)        | `model_type`    | `tone` (description) | `ref_audio` voice cloning |
|----------------------------------|-----------------|----------------------|---------------------------|
| `Qwen3-TTS-12Hz-0.6B-Base`       | `base`          | ❌ (422 with guidance)| ✅ (default built-in voice)|
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `voice_design` | ✅ (required)   | ❌                         |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | `custom_voice` | ✅ (optional)   | ❌ (preset speakers)       |

The type is read from each checkpoint's `config.json` — no code changes needed.

## Run

Use the `tts` conda env's Python (the system `python3` is a different 3.14 env
without `qwen_tts`):

```bash
cd /home/wl26/project/models/tts-backend

# Optional: configure via env
cp .env.example .env   # edit MODEL_PATH / DEVICE / etc.

/home/wl26/miniconda3/envs/tts/bin/python -m uvicorn app.main:app \
    --host 0.0.0.0 --port 8000 --env-file .env
```

Then open <http://localhost:8000> for the demo page, or hit the API directly:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world, this is a test.",
    "language": "English",
    "tone": null
  }' \
  -o out.wav
```

> The first request downloads/loads the built-in reference voice and warms the
> GPU — allow a few seconds.

## API

### `GET /health`
Model info so clients can adapt their UI:

```json
{
  "status": "ok",
  "model_path": ".../Qwen3-TTS-12Hz-0.6B-Base",
  "model_type": "base",
  "tone_supported": false,
  "supported_languages": ["chinese", "english", "french", "german", "italian", "japanese", "korean", "portuguese", "russian", "spanish"],
  "supported_speakers": [],
  "sample_rate": 24000,
  "device": "cuda:0"
}
```

### `POST /v1/tts`
Request fields:

| Field                | Type     | Default   | Notes |
|----------------------|----------|-----------|-------|
| `text`               | string   | required  | Sentence to synthesize |
| `language`           | string   | `Auto`    | Chinese/English/Japanese/Korean/German/French/Russian/Portuguese/Spanish/Italian |
| `tone`               | string?  | `null`    | Tone/style description. **Base model → 422**; needs VoiceDesign/CustomVoice |
| `ref_audio`          | string?  | built-in  | Voice to clone: wav path / http(s) URL / base64 `data:audio` URI (Base only) |
| `ref_text`           | string?  | built-in  | Transcript of `ref_audio` (Base, ICL mode) |
| `x_vector_only`      | bool     | `false`   | Base: speaker vector only, ignores `ref_text` |
| `speaker`            | string?  | `null`    | Preset speaker (CustomVoice only) |
| `response_format`    | `wav`\|`json` | `wav` | `json` returns base64 WAV |
| `temperature`, `top_k`, `top_p`, `repetition_penalty`, `max_new_tokens` | number? | model defaults | Sampling knobs |

**Responses**
- `200 audio/wav` — WAV bytes (`Content-Disposition: inline; filename="qwen3-tts.wav"`, plus `X-Sample-Rate` / `X-Duration-S` headers).
- `200 application/json` — with `response_format=json`:
  ```json
  { "base64_wav": "...", "sample_rate": 24000, "duration_s": 3.2, "model_type": "base" }
  ```
- `422` — validation / unsupported `tone` on Base / unsupported language.
- `500` — model error.

## Adding tone control (VoiceDesign)

The Base checkpoint cannot follow free-text tone instructions — tone follows the
reference audio's style. For real description-driven control, download the
VoiceDesign checkpoint and restart:

```bash
cd /home/wl26/project/models
/home/wl26/miniconda3/envs/tts/bin/huggingface-cli download \
    Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --local-dir Qwen3-TTS-12Hz-1.7B-VoiceDesign

# then point the backend at it
MODEL_PATH=/home/wl26/project/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  /home/wl26/miniconda3/envs/tts/bin/python -m uvicorn app.main:app --port 8000
```

Now `tone` works: `{"text": "...", "language": "English", "tone": "speak in a very sad tone, slowly"}`.

## Layout

```
app/
  config.py        env-driven settings (model path, device, defaults)
  schemas.py       Pydantic request/response models
  tts_service.py   model loading + per-type dispatch (base / voice_design / custom_voice)
  main.py          FastAPI routes, WAV encoding
frontend/
  index.html       demo page (served at /)
```
