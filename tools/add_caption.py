#!/usr/bin/env python3
"""Burn a caption (subtitle) onto an image with ffmpeg.

The workflow needs each sentence drawn on its image so the rendered video
shows subtitles. A pure black band as tall as the image is appended below it
and the caption is drawn in that band, near its top, so the artwork stays
untouched and the text sits on plain black. The raw command

    ffmpeg -i input.jpg -vf "pad=iw:ih*2:0:0:color=black,drawtext=fontfile=...\
           :text='...':fontcolor=white:..." -vframes 1 output.jpg

is painful to call from a workflow: the caption must be escaped for the
filtergraph (quotes, colons, commas, `%`), the font path differs per machine,
and drawtext does not wrap a long sentence.

Usage:
    python tools/add_caption.py INPUT OUTPUT --text "The bank raised rates."
    python tools/add_caption.py INPUT OUTPUT --text-file resource/text/news_001.txt
    echo "The bank raised rates." | python tools/add_caption.py INPUT OUTPUT --text -
    python tools/add_caption.py --batch resource/images --captions-dir resource/text \
        --out-dir resource/captioned

The same options are available as add_caption() for in-process callers.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# Preferred fonts, in order. Bold faces first: a caption sits on top of an
# arbitrary photo and needs the weight to stay readable.
LATIN_FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
)
CJK_FONTS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)

# Font size is a fraction of the image height so one setting fits any
# resolution (0.055 ≈ 40px on a 720p frame).
FONT_SIZE_RATIO = 0.055
MIN_FONT_SIZE, MAX_FONT_SIZE = 16, 120


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return 0x2E80 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF or 0xFF00 <= code <= 0xFFEF


def _has_cjk(text: str) -> bool:
    return any(_is_cjk(c) for c in text)


def resolve_font(text: str, font: str | None = None) -> str:
    """Pick a font file that can render `text`; ask fontconfig as a last resort."""
    if font:
        if Path(font).exists():
            return font
        raise FileNotFoundError(f"font not found: {font}")

    candidates = CJK_FONTS + LATIN_FONTS if _has_cjk(text) else LATIN_FONTS + CJK_FONTS
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate

    family = "Noto Sans CJK SC" if _has_cjk(text) else "DejaVu Sans"
    match = shutil.which("fc-match")
    if match:
        proc = subprocess.run([match, "-f", "%{file}", family],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        path = proc.stdout.decode(errors="replace").strip()
        if proc.returncode == 0 and path and Path(path).exists():
            return path
    raise FileNotFoundError("no usable font found; pass one with --font")


def probe_size(path: Path) -> tuple[int, int]:
    """Return (width, height) of the first video/image stream, 1280x720 if unknown."""
    cmd = [FFPROBE, "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height", "-of", "json", str(path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stream = json.loads(proc.stdout.decode("utf-8"))["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
        if width > 0 and height > 0:
            return width, height
    except (KeyError, IndexError, ValueError, TypeError, json.JSONDecodeError):
        pass
    print(f"  ! could not probe {path.name}; assuming 1280x720", file=sys.stderr)
    return 1280, 720


# Approximate glyph widths in em for the DejaVu/Noto faces above. Measuring
# with PIL is not an option (it is not a dependency), and a flat "average
# character width" overflowed the frame on capital-heavy news sentences.
_NARROW = set("iljI.,:;'|!()[]· ")
_WIDE = set("mwMW@%")
BOLD_FACTOR = 1.10  # the preferred faces are bold; bold is wider than the table


def _char_width(char: str) -> float:
    """Rough advance width of `char` in em."""
    if _is_cjk(char):
        return 1.0
    if char in _NARROW:
        return 0.30
    if char in _WIDE:
        return 0.92
    if char.isupper():
        return 0.72
    if char.isdigit():
        return 0.60
    return 0.58


def _text_width(text: str) -> float:
    return sum(_char_width(c) for c in text) * BOLD_FACTOR


def wrap_text(text: str, max_em: float) -> str:
    """Word wrap so no line is wider than `max_em` em, then hard-wrap any token
    that alone exceeds it (a URL, a run of CJK)."""
    if max_em <= 0:
        return text
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            if _text_width(current) + _char_width(" ") + _text_width(word) <= max_em:
                current += " " + word
            else:
                lines.append(current)
                current = word
        lines.append(current)

    wrapped: list[str] = []
    for line in lines:
        while _text_width(line) > max_em and len(line) > 1:
            cut = len(line)
            while cut > 1 and _text_width(line[:cut]) > max_em:
                cut -= 1
            wrapped.append(line[:cut])
            line = line[cut:]
        wrapped.append(line)
    return "\n".join(wrapped)


def _escape(value: str) -> str:
    """Escape a value for the filtergraph (the text itself goes in a file)."""
    return "".join("\\" + c if c in "\\':,;[]" else c for c in value)


def build_filter(
    *,
    font: str,
    textfile: Path,
    fontsize: int,
    margin: int,
    position: str,
    color: str,
    border_color: str,
    border_width: int,
    line_spacing: int,
    box: bool,
    box_color: str,
) -> str:
    """Assemble the pad + drawtext chain: a pure black band of the image's own
    height is appended below the image and the caption is drawn inside it.

    The caption is read from `textfile`, so no escaping of the caption is
    needed and `expansion=none` keeps a literal `%`."""
    options = [
        f"fontfile={_escape(font)}",
        f"textfile={_escape(str(textfile))}",
        "expansion=none",
        f"fontcolor={color}",
        f"fontsize={fontsize}",
        f"borderw={border_width}",
        f"bordercolor={border_color}",
        f"line_spacing={line_spacing}",
        "x=(w-text_w)/2",
    ]
    if box:
        options += ["box=1", f"boxcolor={box_color}", f"boxborderw={max(4, margin // 2)}"]
    # drawtext sees the padded frame, so h/2 is the top edge of the black band
    # whatever the input resolution; positions are relative to that band.
    y = {
        "top": f"h/2+{margin}",
        "center": "h/2+(h/2-text_h)/2",
        "bottom": f"h-text_h-{margin}",
    }[position]
    options.append(f"y={y}")
    return ("pad=iw:ih*2:0:0:color=black,drawtext=" + ":".join(options))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_caption(
    image: str | Path,
    output: str | Path,
    text: str,
    *,
    font: str | None = None,
    fontsize: int | None = None,
    position: str = "top",
    margin: int | None = None,
    color: str = "white",
    border_color: str = "black",
    border_width: int = 3,
    line_spacing: int | None = None,
    wrap_chars: int | None = None,
    box: bool = False,
    box_color: str = "black@0.5",
    ffmpeg: str = FFMPEG,
) -> str:
    """Draw `text` under `image` and write `output`; returns the output path.

    A pure black band as tall as the image is appended below it and the caption
    is drawn inside that band, so the output is twice the input's height.
    `position` places the caption within the band.

    Raises RuntimeError when ffmpeg is missing or fails.
    """
    image, output = Path(image), Path(output)
    if not image.exists():
        raise FileNotFoundError(f"input image not found: {image}")
    text = text.strip()
    if not text:
        raise ValueError("caption text is empty")

    width, height = probe_size(image)
    fontsize = fontsize or max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, round(height * FONT_SIZE_RATIO)))
    margin = margin if margin is not None else max(12, round(fontsize * 0.6))
    line_spacing = line_spacing if line_spacing is not None else max(4, fontsize // 5)
    # Usable width: the margins, the outline (borderw grows the glyph box on
    # both sides) and, with --box, the box padding.
    usable_px = width - 2 * margin - 2 * border_width
    if box:
        usable_px -= 2 * max(4, margin // 2)
    max_em = wrap_chars * 0.58 if wrap_chars is not None else usable_px / fontsize

    font_path = resolve_font(text, font)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="add_caption_") as tmp:
        textfile = Path(tmp) / "caption.txt"
        textfile.write_text(wrap_text(text, max_em), encoding="utf-8")
        vf = build_filter(
            font=font_path, textfile=textfile, fontsize=fontsize, margin=margin,
            position=position, color=color, border_color=border_color,
            border_width=border_width, line_spacing=line_spacing,
            box=box, box_color=box_color,
        )
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(image),
               "-vf", vf, "-frames:v", "1"]
        cmd += ["-q:v", "2"] if output.suffix.lower() in (".jpg", ".jpeg") else []
        cmd.append(str(output))
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed for {image.name}: "
                f"{proc.stderr.decode(errors='replace').strip()}"
            )

    return str(output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Burn a caption onto an image (ffmpeg drawtext).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:", 1)[1],
    )
    parser.add_argument("image", nargs="?", help="input image")
    parser.add_argument("output", nargs="?", help="output image")
    parser.add_argument("-t", "--text", help="caption text ('-' reads stdin)")
    parser.add_argument("--text-file", help="read the caption from this file")

    parser.add_argument("--batch", metavar="IMAGES_DIR",
                        help="caption every image in a directory")
    parser.add_argument("--captions-dir", metavar="TEXT_DIR",
                        help="batch: per-image caption from <stem>.txt in this directory")
    parser.add_argument("--out-dir", metavar="DIR", help="batch: where captioned images go")

    parser.add_argument("--font", help="font file (default: auto-detected)")
    parser.add_argument("--fontsize", type=int, help="pixels (default: ~5.5%% of image height)")
    parser.add_argument("--position", choices=("top", "center", "bottom"), default="top",
                        help="caption placement within the black band (default: top)")
    parser.add_argument("--margin", type=int, help="pixels between caption and the band's edge")
    parser.add_argument("--color", default="white", help="text color, e.g. white or #ffcc00")
    parser.add_argument("--border-color", default="black")
    parser.add_argument("--border-width", type=int, default=3)
    parser.add_argument("--line-spacing", type=int)
    parser.add_argument("--wrap", type=int, dest="wrap_chars",
                        help="approximate max characters per line (0 disables wrapping)")
    parser.add_argument("--box", action="store_true", help="draw a background box")
    parser.add_argument("--box-color", default="black@0.5")
    return parser


def _options(args) -> dict:
    """The add_caption() keyword arguments shared by single and batch mode."""
    return dict(
        font=args.font, fontsize=args.fontsize, position=args.position,
        margin=args.margin, color=args.color, border_color=args.border_color,
        border_width=args.border_width, line_spacing=args.line_spacing,
        wrap_chars=args.wrap_chars, box=args.box, box_color=args.box_color,
    )


def _read_caption(args) -> str:
    if args.text == "-":
        return sys.stdin.read().strip()
    if args.text:
        return args.text
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8").strip()
    raise SystemExit("no caption given: use --text, --text-file or --text -")


def _run_batch(args) -> int:
    images_dir, out_dir = Path(args.batch), Path(args.out_dir or "")
    if not args.out_dir:
        raise SystemExit("--batch needs --out-dir")
    images = sorted(p for p in images_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"no images found in {images_dir}")

    captions_dir = Path(args.captions_dir) if args.captions_dir else None
    # one caption for every image; read once so stdin is not drained by the
    # first iteration
    shared = None if captions_dir else _read_caption(args)
    failed = 0
    for image in images:
        if captions_dir is not None:
            caption_file = captions_dir / f"{image.stem}.txt"
            if not caption_file.exists():
                print(f"  ! no caption for {image.name}; skipping", file=sys.stderr)
                failed += 1
                continue
            text = caption_file.read_text(encoding="utf-8").strip()
        else:
            text = shared
        try:
            out = add_caption(image, out_dir / image.name, text, **_options(args))
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"  ! {image.name}: {exc}", file=sys.stderr)
            failed += 1
            continue
        print(f"  {image.name} -> {out}")
    print(f"captioned {len(images) - failed}/{len(images)} image(s)")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.batch:
        return _run_batch(args)

    if not args.image or not args.output:
        raise SystemExit("usage: add_caption.py INPUT OUTPUT (--text ...)")
    try:
        out = add_caption(args.image, args.output, _read_caption(args), **_options(args))
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
