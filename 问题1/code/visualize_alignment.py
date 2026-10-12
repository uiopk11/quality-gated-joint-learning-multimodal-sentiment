from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

from q1lib import load_config

# Prefer classical serif faces for paper figures; fall back gracefully on Windows/Linux.
_SERIF_CANDIDATES = (
    "times.ttf",
    "Times New Roman.ttf",
    "TimesNewRoman.ttf",
    "georgia.ttf",
    "Georgia.ttf",
    "cambria.ttc",
    "Cambria.ttf",
    "DejaVuSerif.ttf",
    "LiberationSerif-Regular.ttf",
)
_SERIF_BOLD_CANDIDATES = (
    "timesbd.ttf",
    "Times New Roman Bold.ttf",
    "TimesNewRomanBold.ttf",
    "georgiab.ttf",
    "Georgia Bold.ttf",
    "cambriab.ttf",
    "DejaVuSerif-Bold.ttf",
    "LiberationSerif-Bold.ttf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a verifiable alignment figure without Matplotlib")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--max-frames", type=int, default=6)
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.28,
        help="Scale all figure fonts; 1.28 balances serif paper typography with page-width reduction.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


@lru_cache(maxsize=8)
def _resolve_font_path(bold: bool = False) -> Path | None:
    names = _SERIF_BOLD_CANDIDATES if bold else _SERIF_CANDIDATES
    search_dirs = (
        Path(r"C:\Windows\Fonts"),
        Path("/usr/share/fonts/truetype/liberation"),
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts"),
    )
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def load_academic_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    size = max(1, int(round(size)))
    path = _resolve_font_path(bold=bold)
    if path is None and bold:
        path = _resolve_font_path(bold=False)
    if path is not None:
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def safe_row_norm(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1)
    maximum = float(norms.max(initial=0.0))
    return norms / maximum if maximum > 0 else norms


def word_text_features(data: np.lib.npyio.NpzFile) -> np.ndarray:
    token_features = data["text_full"]
    word_ids = data["word_ids_full"]
    word_count = len(data["words"])
    result = np.zeros((word_count, token_features.shape[1]), dtype=np.float32)
    for word_index in range(word_count):
        selected = token_features[word_ids == word_index]
        if len(selected):
            result[word_index] = selected.mean(axis=0)
    return result


def read_frame(video_path: Path, timestamp: float) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(video_path))
    try:
        # Seeking to the exact container end can fail even when the final frame exists.
        # Retry slightly earlier; this changes only the verification figure, not features.
        for fallback_seconds in (0.0, 0.05, 0.15, 0.30, 0.50):
            candidate = max(0.0, float(timestamp) - fallback_seconds)
            capture.set(cv2.CAP_PROP_POS_MSEC, candidate * 1000.0)
            ok, frame = capture.read()
            if ok:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    finally:
        capture.release()


def color_scale(value: float) -> tuple[int, int, int]:
    value = float(np.clip(value, 0.0, 1.0))
    if value < 0.5:
        alpha = value * 2.0
        return (int(32 + 40 * alpha), int(70 + 130 * alpha), int(120 + 20 * alpha))
    alpha = (value - 0.5) * 2.0
    return (int(72 + 183 * alpha), int(200 + 35 * alpha), int(140 - 90 * alpha))


def fit_image(
    array: np.ndarray,
    width: int,
    height: int,
    *,
    valign: str = "center",
    pad: bool = True,
) -> Image.Image:
    image = Image.fromarray(array).convert("RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    if not pad:
        return image
    canvas = Image.new("RGB", (width, height), "white")
    x = (width - image.width) // 2
    if valign == "top":
        y = 0
    elif valign == "bottom":
        y = height - image.height
    else:
        y = (height - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def main() -> None:
    args = parse_args()
    if args.font_scale <= 0:
        raise ValueError("--font-scale must be positive")

    def scaled_font(base_size: int, *, bold: bool = False) -> ImageFont.ImageFont:
        return load_academic_font(base_size * args.font_scale, bold=bold)

    config = load_config(args.config)
    sample_dir = config.output_root / "samples" / args.sample_id
    metadata_path = sample_dir / "metadata.json"
    feature_path = sample_dir / "features.npz"
    alignment_path = sample_dir / "word_alignment.csv"
    if not metadata_path.is_file() or not feature_path.is_file() or not alignment_path.is_file():
        raise FileNotFoundError(f"sample output is incomplete: {sample_dir}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    alignment = pd.read_csv(alignment_path)
    data = np.load(feature_path, allow_pickle=False)
    words = data["words"].astype(str)
    starts = data["word_starts"].astype(float)
    ends = data["word_ends"].astype(float)
    valid_intervals = (starts >= 0) & (ends > starts)
    centers = np.where(valid_intervals, (starts + ends) / 2.0, np.nan)

    text_norm = safe_row_norm(word_text_features(data))
    audio_norm = safe_row_norm(data["audio_word_features"])
    vision_norm = safe_row_norm(data["vision_word_features"])
    heatmap = np.vstack([text_norm, audio_norm, vision_norm])
    masks = np.vstack([
        np.ones(len(words), dtype=np.uint8), data["official_word_match_mask"],
        data["audio_word_valid_mask"], data["vision_word_valid_mask"],
    ])

    waveform, sample_rate = sf.read(metadata["files"]["wav"], always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    duration = float(metadata["media"]["duration_seconds"] or len(waveform) / sample_rate)

    width = max(1800, min(3200, 320 + len(words) * 95))
    canvas_probe = Image.new("RGB", (8, 8), "white")
    probe = ImageDraw.Draw(canvas_probe)
    body_font = scaled_font(17)
    title_font = scaled_font(25, bold=True)
    section_font = scaled_font(19, bold=True)
    word_font = scaled_font(13)
    caption_font = scaled_font(13)
    label_font = scaled_font(15)
    left, right = 150, width - 50
    plot_width = right - left

    def text_height(font: ImageFont.ImageFont, sample: str = "Ag") -> int:
        box = probe.textbbox((0, 0), sample, font=font)
        return max(1, box[3] - box[1])

    section_h = text_height(section_font, "Validity masks (white = valid)")
    title_gap = 24  # pixels between section-title bottom and panel top border

    wave_top, wave_bottom = 118, 390
    heat_title_y = wave_bottom + 20
    heat_top = heat_title_y + section_h + title_gap
    heat_rows, row_height = 3, 68
    heat_bottom = heat_top + heat_rows * row_height
    mask_title_y = heat_bottom + 16
    mask_top = mask_title_y + section_h + title_gap
    mask_row_h = 40
    mask_bottom = mask_top + 4 * mask_row_h
    frame_top = mask_bottom + 8
    frame_height_cap = 250
    caption_block = text_height(caption_font) * 2 + 12

    frame_count = min(max(1, args.max_frames), len(words))
    valid_word_indices = np.flatnonzero(valid_intervals)
    if len(valid_word_indices):
        selected = valid_word_indices[
            np.linspace(0, len(valid_word_indices) - 1, frame_count, dtype=int)
        ]
        frame_times = [float(centers[index]) for index in selected]
    else:
        selected = np.linspace(0, len(words) - 1, frame_count, dtype=int)
        frame_times = np.linspace(0.0, max(0.0, duration - 1e-3), frame_count).tolist()

    gap = 16
    frame_width = int((plot_width - gap * (frame_count - 1)) / frame_count)
    # Fit frames without bottom letterbox so captions can sit tightly under content.
    fitted_frames: list[Image.Image | None] = []
    content_heights: list[int] = []
    for frame_time in frame_times:
        frame = read_frame(Path(metadata["source"]["video_path"]), frame_time)
        if frame is None:
            fitted_frames.append(None)
            content_heights.append(frame_height_cap)
        else:
            fitted = fit_image(frame, frame_width, frame_height_cap, pad=False)
            fitted_frames.append(fitted)
            content_heights.append(fitted.height)
    frame_content_h = max(content_heights) if content_heights else frame_height_cap
    height = frame_top + frame_content_h + caption_block + 10

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((left, 12), f"{args.sample_id} | {metadata['quality']['actual_alignment_type']}",
              fill="#1a1a1a", font=title_font)
    draw.text((left, 50), f"Official text: {metadata['official_text']}",
              fill="#1a1a1a", font=body_font)
    draw.text((left, 78), f"ASR text: {metadata.get('asr_text', '')}",
              fill="#4a4a4a", font=body_font)

    draw.rectangle((left, wave_top, right, wave_bottom), outline="#777777", width=1)
    sample_positions = np.linspace(0, max(0, len(waveform) - 1), plot_width).astype(int)
    sampled = waveform[sample_positions] if len(waveform) else np.zeros(plot_width)
    scale = float(np.max(np.abs(sampled), initial=1e-6))
    mid = (wave_top + wave_bottom) // 2
    points = [(left + x, int(mid - sampled[x] / scale * (wave_bottom - wave_top) * 0.42))
              for x in range(plot_width)]
    if len(points) > 1:
        draw.line(points, fill="#244a73", width=1)
    for word_index, (word, start, end) in enumerate(zip(words, starts, ends)):
        if start < 0 or end <= start:
            continue
        x0 = left + int(start / duration * plot_width)
        x1 = left + int(end / duration * plot_width)
        draw.line((x0, wave_top, x0, wave_bottom), fill="#e9a15c", width=1)
        draw.line((x1, wave_top, x1, wave_bottom), fill="#f3c497", width=1)
        label_y = wave_top + 4 + (word_index % 2) * 22
        draw.text((x0 + 2, label_y), word[:9], fill="#6b3f19", font=word_font)
    draw.text((20, wave_top + 118), "audio", fill="#1a1a1a", font=label_font)

    cell_width = plot_width / max(1, len(words))
    draw.text(
        (left, heat_title_y),
        "Normalized feature-vector norms by official word",
        fill="#1a1a1a",
        font=section_font,
    )
    for row, label in enumerate(("BERT", "OpenSMILE", "OpenFace")):
        y0 = heat_top + row * row_height
        draw.text((20, y0 + 20), label, fill="#1a1a1a", font=label_font)
        for column, value in enumerate(heatmap[row]):
            x0 = left + int(column * cell_width)
            x1 = left + int((column + 1) * cell_width)
            draw.rectangle((x0, y0, x1, y0 + row_height - 2), fill=color_scale(float(value)))

    draw.text(
        (left, mask_title_y),
        "Validity masks (white = valid)",
        fill="#1a1a1a",
        font=section_font,
    )
    for row, label in enumerate(("text", "official-ASR", "audio valid", "vision valid")):
        y0 = mask_top + row * mask_row_h
        draw.text((18, y0 + 8), label, fill="#1a1a1a", font=word_font)
        for column, value in enumerate(masks[row]):
            x0 = left + int(column * cell_width)
            x1 = left + int((column + 1) * cell_width)
            color = "white" if int(value) else "black"
            draw.rectangle((x0, y0, x1, y0 + mask_row_h - 2), fill=color, outline="#999999")

    for column, (word_index, frame_time, fitted) in enumerate(
        zip(selected, frame_times, fitted_frames)
    ):
        x0 = left + column * (frame_width + gap)
        content_h = content_heights[column]
        if fitted is None:
            draw.rectangle(
                (x0, frame_top, x0 + frame_width, frame_top + content_h),
                fill="#eeeeee",
                outline="#777777",
            )
            draw.text(
                (x0 + 10, frame_top + content_h // 2),
                "frame unavailable",
                fill="#1a1a1a",
                font=caption_font,
            )
        else:
            paste_x = x0 + (frame_width - fitted.width) // 2
            canvas.paste(fitted, (paste_x, frame_top))
        interval_text = (
            f"{starts[word_index]:.2f}-{ends[word_index]:.2f}s"
            if valid_intervals[word_index]
            else f"unaligned {frame_time:.2f}s"
        )
        audio_count = int(alignment.iloc[word_index]["audio_frame_count"])
        vision_count = int(alignment.iloc[word_index]["vision_frame_count"])
        caption_y = frame_top + content_h + 2
        draw.text(
            (x0, caption_y),
            f"{words[word_index]} | {interval_text}",
            fill="#1a1a1a",
            font=caption_font,
        )
        draw.text(
            (x0, caption_y + text_height(caption_font) + 2),
            f"A:{audio_count}  V:{vision_count}",
            fill="#333333",
            font=caption_font,
        )

    output = args.output or (config.output_root / "figures" / f"{args.sample_id}_alignment.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    print(output.resolve())


if __name__ == "__main__":
    main()
