"""Video tool: render one mp4 from the paired images and audio files.

Each image is shown for the length of its audio (one image pairs one audio),
then the segments are concatenated. The ffmpeg helpers are the ones from
step5.py — imported rather than copied so both entry points stay in sync.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from .video_util import concat_segments, discover_pairs, make_segment, probe_duration

import ai.manage_model as manage_model
from .. import config
from ..tracing import traced

logger = logging.getLogger(__name__)


@traced
def build_video(
    source: str = config.STEM,
    fps: int = config.VIDEO_FPS,
    expected_segments: int | None = None,
    model_manage: manage_model.Manage_Model | None = None,
) -> str:
    """Render resource/video/<source>.mp4 from resource/images + resource/audio.

    Returns the video path; raises when no pair is found or ffmpeg fails.
    """
    if model_manage is not None:
            model_manage.launch_model("video")
    pairs = discover_pairs(config.IMAGE_DIR, config.AUDIO_DIR).get(source, [])
    if not pairs:
        raise RuntimeError(f"no image/audio pairs found for {source!r} in {config.IMAGE_DIR} + {config.AUDIO_DIR}")
    if expected_segments is not None and len(pairs) < expected_segments:
        logger.warning("rendering %d of %d sentences (failed steps have no image/audio pair)",
                       len(pairs), expected_segments)

    out_path = config.VIDEO_DIR / f"{source}.mp4"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"agent_core_{source}_") as tmp:
        tmp_dir = Path(tmp)
        segments: list[Path] = []
        for index, image, audio in pairs:
            segment = tmp_dir / f"{source}_{index:03d}.mp4"
            logger.info("segment %d/%d: %s + %s", index, len(pairs), image.name, audio.name)
            # cap the segment at the narration so no silence trails the audio
            make_segment(image, audio, segment, fps=fps, duration=probe_duration(audio))
            segments.append(segment)

        list_file = tmp_dir / "concat.txt"
        list_file.write_text("".join(f"file '{p.resolve()}'\n" for p in segments), encoding="utf-8")
        concat_segments(list_file, out_path)

    duration = probe_duration(out_path)
    logger.info("video %s (%.1fs, %d segments)", out_path, duration or 0.0, len(pairs))
    return str(out_path)
