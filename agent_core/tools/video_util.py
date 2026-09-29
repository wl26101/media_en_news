#!/usr/bin/env python3
"""Step 5 — Images + paired audio → video (ffmpeg).

For every source in the imgs dir, pairs each image (*.png) with the matching
audio file (*.wav of the same stem, e.g. s1_summary_1.png ↔
s1_summary_1.wav), renders one video segment per pair (the image displayed
for the duration of its audio), and concatenates the segments into a single
<source>.mp4.

Usage:
    python step5.py [IMGS_DIR] [OUTPUTS_DIR] [VIDEOS_DIR]

Defaults:
    IMGS_DIR    = imgs/
    OUTPUTS_DIR = outputs/
    VIDEOS_DIR  = videos/

Example (day-specific folders):
    python step5.py 260828_news/imgs 260828_news/outputs 260828_news/videos
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FPS = 25
AUDIO_BITRATE = "192k"
FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------


def run(cmd: list[str]) -> int:
    """Run a command, echoing stderr on failure."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print("    ffmpeg stderr:", proc.stderr.decode(errors="replace"), file=sys.stderr)
    return proc.returncode


def make_segment(image: Path, audio: Path, seg_path: Path, fps: int = FPS,
                 duration: float | None = None) -> None:
    """Render one still-image + audio pair into a short mp4 segment.

    Pass `duration` (the audio's length in seconds) to cap the segment: with a
    looped still image, -shortest alone still overshoots the audio by ~2s and
    leaves that much silence at the end of the segment."""
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-loop", "1", "-framerate", str(fps), "-i", str(image),
        "-i", str(audio),
        "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", AUDIO_BITRATE,
        "-shortest",
    ]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    if run(cmd + [str(seg_path)]) != 0:
        raise RuntimeError(f"failed to render segment {seg_path.name}")


def concat_segments(list_file: Path, out_path: Path) -> None:
    """Concatenate segments via the concat demuxer. Tries stream copy first,
    falls back to re-encoding if the streams aren't concatenatable."""
    base = [FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(list_file)]
    if run(base + ["-c", "copy", str(out_path)]) == 0:
        return
    print("    ... stream copy failed, re-encoding", file=sys.stderr)
    out_path.unlink(missing_ok=True)
    if run(base + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out_path)]) != 0:
        raise RuntimeError(f"failed to concatenate {out_path.name}")


def probe_duration(video: Path) -> float | None:
    """Return the video duration in seconds via ffprobe, or None on failure."""
    cmd = [FFPROBE, "-v", "error", "-show_entries", "format=duration",
           "-of", "json", str(video)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        return None
    try:
        return float(json.loads(proc.stdout.decode("utf-8"))["format"]["duration"])
    except (KeyError, ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Pair discovery
# ---------------------------------------------------------------------------


def discover_pairs(imgs_dir: Path, outputs_dir: Path) -> dict[str, list[tuple[int, Path, Path]]]:
    """Group image/audio files into {source: [(index, image, audio), ...]}.

    Files are paired by matching stem; the trailing '_<number>' is the index
    (so s1_summary_1.png pairs with s1_summary_1.wav, source = s1_summary)."""
    images = {p.stem: p for p in imgs_dir.glob("*.png")}
    audios = {p.stem: p for p in outputs_dir.glob("*.wav")}
    groups: dict[str, list[tuple[int, Path, Path]]] = {}
    for stem in sorted(images.keys() & audios.keys()):
        match = re.match(r"^(.*)_(\d+)$", stem)
        if not match:
            print(f"  ! skipping unindexed pair {stem}", file=sys.stderr)
            continue
        prefix, index = match.group(1), int(match.group(2))
        groups.setdefault(prefix, []).append((index, images[stem], audios[stem]))
    for files in groups.values():
        files.sort(key=lambda item: item[0])
    return groups


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine paired images and audio into videos (step 5).")
    parser.add_argument("imgs_dir", nargs="?", default="imgs", help="directory with *.png images")
    parser.add_argument("outputs_dir", nargs="?", default="outputs", help="directory with *.wav audio")
    parser.add_argument("videos_dir", nargs="?", default="videos", help="where to write the mp4 videos")
    parser.add_argument("--fps", type=int, default=FPS, help="frames per second for the still images")
    args = parser.parse_args()

    imgs_dir = Path(args.imgs_dir)
    outputs_dir = Path(args.outputs_dir)
    videos_dir = Path(args.videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)

    groups = discover_pairs(imgs_dir, outputs_dir)
    if not groups:
        raise SystemExit(f"No matching image/audio pairs found between {imgs_dir} and {outputs_dir}")

    manifest = []
    for source, pairs in sorted(groups.items()):
        print(f"[*] {source}: {len(pairs)} segment(s)")
        out_path = videos_dir / f"{source}.mp4"
        with tempfile.TemporaryDirectory(prefix=f"step5_{source}_") as tmp:
            tmp = Path(tmp)
            segments: list[Path] = []
            for index, image, audio in pairs:
                segment = tmp / f"{source}_{index:03d}.mp4"
                print(f"    [{index}/{len(pairs)}] {image.name} + {audio.name}")
                make_segment(image, audio, segment, fps=args.fps)
                segments.append(segment)
            list_file = tmp / "concat.txt"
            list_file.write_text("".join(f"file '{p.resolve()}'\n" for p in segments), encoding="utf-8")
            concat_segments(list_file, out_path)
        duration = probe_duration(out_path)
        print(f"    -> {out_path} ({duration:.1f}s)" if duration else f"    -> {out_path}")
        manifest.append({
            "source": source,
            "video": out_path.name,
            "duration_s": duration,
            "segments": [{"index": idx, "image": img.name, "audio": aud.name} for idx, img, aud in pairs],
        })

    manifest_path = videos_dir / "videos_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {manifest_path} ({len(manifest)} video(s))")
    print("Done.")


if __name__ == "__main__":
    main()
