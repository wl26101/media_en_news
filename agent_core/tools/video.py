"""Video tool: render one mp4 from the paired images and audio files.

Each image is shown for the length of its audio (one image pairs one audio),
then the segments are concatenated. The ffmpeg helpers are the ones from
step5.py — imported rather than copied so both entry points stay in sync.

The sentence of each image (resource/text/<stem>.txt) is burned onto a
temporary copy of the image first, so the video shows subtitles. The clean
PNG in resource/images/ is never modified.
"""

from __future__ import annotations

import importlib.util
import logging
import tempfile
from pathlib import Path

from .video_util import concat_segments, discover_pairs, make_segment, probe_duration

import ai.manage_model as manage_model
from .. import config
from ..tracing import traced

logger = logging.getLogger(__name__)


def _load_add_caption():
    """Import tools/add_caption.py by path.

    Both the project's root tools/ and this package are named `tools`. When
    agent_core/ is on sys.path — `python main.py` run from inside it — a plain
    `import tools.add_caption` resolves to agent_core.tools instead and fails,
    so load the file directly.
    """
    path = config.ROOT / "tools" / "add_caption.py"
    spec = importlib.util.spec_from_file_location("add_caption", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the caption tool from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.add_caption


add_caption = _load_add_caption()


def caption_image(image: Path, work_dir: Path) -> Path:
    """Draw the sentence that belongs to `image` onto a temporary copy.

    The caption is the sibling of the image in resource/text/ (news_001.png ↔
    news_001.txt). Returns the original image when it has no sentence file or
    captioning fails — a missing subtitle must not cost the run its video.
    """
    text_file = config.TEXT_DIR / f"{image.stem}.txt"
    if not text_file.exists():
        logger.warning("no caption for %s (%s missing); using the plain image",
                       image.name, text_file.name)
        return image
    try:
        return Path(add_caption(
            image,
            work_dir / f"{image.stem}_captioned.png",
            text_file.read_text(encoding="utf-8"),
            box=True,  # a translucent box keeps the text readable over the art
        ))
    except Exception as exc:
        logger.error("captions failed for %s: %s; using the plain image", image.name, exc)
        return image


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
            frame = caption_image(image, tmp_dir) if config.CAPTION_IMAGES else image
            # cap the segment at the narration so no silence trails the audio
            make_segment(frame, audio, segment, fps=fps, duration=probe_duration(audio))
            segments.append(segment)

        list_file = tmp_dir / "concat.txt"
        list_file.write_text("".join(f"file '{p.resolve()}'\n" for p in segments), encoding="utf-8")
        concat_segments(list_file, out_path)

    duration = probe_duration(out_path)
    logger.info("video %s (%.1fs, %d segments)", out_path, duration or 0.0, len(pairs))
    return str(out_path)
