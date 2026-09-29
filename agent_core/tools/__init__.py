"""Tools used by the news-video graph nodes.

Each module wraps one external capability and raises on failure, so nodes can
record a per-sentence error and keep going:

    text.py   sentence splitting + the resource file layout
    llm.py    local llama.cpp server (script, review, image prompts)
    tts.py    local TTS server (one WAV per sentence)
    image.py  z-Image-Turbo on ComfyUI (one PNG per sentence)
    video.py  ffmpeg pairing of image + audio into the final mp4
    models.py start/stop of the three local model servers
"""
