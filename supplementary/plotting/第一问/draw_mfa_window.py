"""Draw Fig. q1_mfa_window: windowed MFA time-coordinate transform.

Uses real sample -3g5yACwYnA__13 (read-only outside this folder):
  WAV, asr_words.csv, word_alignment.csv, mfa_alignment_windows.csv, TextGrid.

Upper lane: full-timeline waveform + ASR word spans (prefix extras highlighted).
Lower lane: candidate window, local MFA intervals, and MFA intervals shifted
back by window start ell_i onto the presented video timeline.

Writes PNG only under cursor_draw/figures/.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
Q1_ROOT = HERE.parent
SAMPLE_ID = "-3g5yACwYnA__13"
SAMPLE_DIR = Q1_ROOT / "输出" / "samples" / SAMPLE_ID
WINDOWS_CSV = Q1_ROOT / "输出" / "summary" / "mfa_alignment_windows.csv"
OUT_DIR = HERE / "figures"
OUT_PATH = OUT_DIR / "q1_mfa_window.png"

DPI = 2
W = 1680 * DPI
MARGIN_L = 148 * DPI
MARGIN_R = 42 * DPI
MARGIN_T = 48 * DPI
MARGIN_B = 32 * DPI

COLOR_WAVE = (36, 74, 120)
COLOR_ASR_PREFIX = (176, 112, 56)
COLOR_ASR_MATCH = (90, 120, 150)
COLOR_ASR_FILL_PREFIX = (255, 236, 212)
COLOR_ASR_FILL_MATCH = (220, 232, 242)
COLOR_WINDOW = (63, 124, 168)
COLOR_WINDOW_FILL = (210, 228, 242, 110)
COLOR_LOCAL = (120, 90, 160)
COLOR_LOCAL_FILL = (232, 222, 245)
COLOR_SHIFT = (46, 120, 90)
COLOR_SHIFT_FILL = (210, 236, 220)
COLOR_AXIS = (70, 76, 84)
COLOR_GRID = (220, 224, 230)
COLOR_PANEL = (248, 249, 251)
COLOR_MUTE = (150, 154, 160)


def _load_cn(size: int, bold: bool = False) -> ImageFont.ImageFont:
    windir = Path(r"C:\Windows\Fonts")
    names = (
        ["msyhbd.ttc", "simhei.ttf", "msyh.ttc"]
        if bold
        else ["msyh.ttc", "simhei.ttf", "simsun.ttc"]
    )
    for name in names:
        path = windir / name
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default(size=size)


def _load_times(size: int, bold: bool = False) -> ImageFont.ImageFont:
    windir = Path(r"C:\Windows\Fonts")
    names = ["timesbd.ttf", "times.ttf"] if bold else ["times.ttf", "timesbd.ttf"]
    for name in names:
        path = windir / name
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default(size=size)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_textgrid_words(path: Path) -> list[tuple[float, float, str]]:
    text = path.read_text(encoding="utf-8")
    words_tier = re.search(
        r'name\s*=\s*"words"\s*(.*?)(?:item\s*\[\s*\d+\s*\]:|\Z)',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    block = words_tier.group(1) if words_tier else text
    pattern = re.compile(
        r"xmin\s*=\s*([0-9.]+)\s*"
        r"xmax\s*=\s*([0-9.]+)\s*"
        r'text\s*=\s*"([^"]*)"',
        flags=re.IGNORECASE,
    )
    words: list[tuple[float, float, str]] = []
    for xmin_s, xmax_s, token in pattern.findall(block):
        token = token.strip()
        if not token:
            continue
        words.append((float(xmin_s), float(xmax_s), token))
    return words


def _x_map(t: float, x0: int, x1: int, t0: float, t1: float) -> int:
    if t1 <= t0:
        return x0
    return int(round(x0 + (t - t0) / (t1 - t0) * (x1 - x0)))


def _draw_rounded(draw: ImageDraw.ImageDraw, box, fill, outline, width: int = 2, radius: int = 10):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def load_sample() -> dict:
    meta = json.loads((SAMPLE_DIR / "metadata.json").read_text(encoding="utf-8"))
    asr = _read_csv(SAMPLE_DIR / "asr_words.csv")
    alignment = _read_csv(SAMPLE_DIR / "word_alignment.csv")
    windows = _read_csv(WINDOWS_CSV)
    window_rows = [row for row in windows if row["sample_id"] == SAMPLE_ID]
    if not window_rows:
        raise FileNotFoundError(f"{SAMPLE_ID} missing in {WINDOWS_CSV}")
    window = window_rows[0]

    wav_path = Path(meta["files"]["wav"])
    if not wav_path.is_file():
        raise FileNotFoundError(wav_path)
    signal, sample_rate = sf.read(str(wav_path), always_2d=False)
    if getattr(signal, "ndim", 1) > 1:
        signal = signal.mean(axis=1)
    signal = np.asarray(signal, dtype=np.float32)

    tg_path = Path(meta["files"]["textgrid"])
    local_words = _parse_textgrid_words(tg_path) if tg_path.is_file() else []

    duration = float(meta["media"]["duration_seconds"])
    window_start = float(window["window_start"])
    window_end = float(window["window_end"])
    anchor_start = float(window["official_anchor_start"])
    anchor_end = float(window["official_anchor_end"])

    first = alignment[0]
    if abs(float(first["start"]) - 1.45) > 1e-6 or abs(float(first["end"]) - 1.58) > 1e-6:
        raise ValueError(
            f"first official MFA interval unexpected: {first['start']}-{first['end']}"
        )
    if abs(window_start - 1.41) > 1e-6 or abs(window_end - 5.399125) > 1e-6:
        raise ValueError(f"window unexpected: {window_start}-{window_end}")
    if abs(duration - 5.513997) > 1e-3:
        raise ValueError(f"duration unexpected: {duration}")
    if local_words:
        shifted0 = local_words[0][0] + window_start
        shifted1 = local_words[0][1] + window_start
        if abs(shifted0 - float(first["start"])) > 0.02 or abs(shifted1 - float(first["end"])) > 0.02:
            raise ValueError(
                f"TextGrid+ell_i mismatch: {shifted0:.3f}-{shifted1:.3f} vs "
                f"{first['start']}-{first['end']}"
            )

    return {
        "meta": meta,
        "asr": asr,
        "alignment": alignment,
        "window": window,
        "signal": signal,
        "sample_rate": int(sample_rate),
        "duration": duration,
        "window_start": window_start,
        "window_end": window_end,
        "anchor_start": anchor_start,
        "anchor_end": anchor_end,
        "local_words": local_words,
    }


def draw_figure(data: dict) -> Path:
    fonts = {
        "title": _load_cn(44, bold=True),
        "section": _load_cn(36, bold=True),
        "lane": _load_cn(38, bold=True),
        "small": _load_cn(28),
        "box_cn": _load_cn(34, bold=True),
        "box_cn_sm": _load_cn(28, bold=True),
        "num": _load_times(32),
        "num_sm": _load_times(28),
        "box_num": _load_times(32, bold=True),
        "formula": _load_times(34, bold=True),
        "word": _load_times(34, bold=True),
        "word_sm": _load_times(26, bold=True),
    }

    duration = data["duration"]
    window_start = data["window_start"]
    window_end = data["window_end"]
    t0, t1 = 0.0, duration

    title_h = 48 * DPI
    upper_h = 250 * DPI
    lower_h = 430 * DPI
    legend_h = 52 * DPI
    gap = 24 * DPI
    height = MARGIN_T + title_h + upper_h + gap + lower_h + 12 * DPI + legend_h + MARGIN_B

    canvas = Image.new("RGB", (W, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")
    x0, x1 = MARGIN_L, W - MARGIN_R
    plot_w = x1 - x0
    label_x = x0 - 6 * DPI

    def side_label(text: str, y_mid: int) -> None:
        draw.text((label_x, y_mid), text, font=fonts["lane"], fill=COLOR_AXIS, anchor="rm")

    def fit_word(
        label: str,
        box_w: int,
        prefer: ImageFont.ImageFont | None = None,
        fallback: ImageFont.ImageFont | None = None,
    ) -> tuple[str, ImageFont.ImageFont]:
        prefer = prefer or fonts["word"]
        fallback = fallback or fonts["word_sm"]
        if int(prefer.getlength(label)) <= max(4, box_w - 4):
            return label, prefer
        if int(fallback.getlength(label)) <= max(4, box_w - 4):
            return label, fallback
        clipped = label
        while len(clipped) > 1 and int(fallback.getlength(clipped)) > max(4, box_w - 4):
            clipped = clipped[:-1]
        return clipped, fallback

    def labeled_box(
        left: int,
        right: int,
        top: int,
        bot: int,
        label: str,
        *,
        fill,
        outline,
        x_max: int,
        font: ImageFont.ImageFont | None = None,
        mode: str = "widen_right",
    ) -> tuple[int, int]:
        """Keep the true start (left); widen only rightward; left-align text inside."""
        font = font or fonts["word"]
        pad = 14 * DPI
        tb = draw.textbbox((0, 0), label, font=font)
        need = int((tb[2] - tb[0]) * 1.08) + 2 * pad
        if mode == "widen_right":
            right = max(right, min(x_max, left + need))
        draw.rectangle([left, top, right, bot], fill=fill, outline=outline, width=3)
        # Left-align so bold glyphs never spill past the true time start.
        draw.text((left + pad, (top + bot) // 2), label, font=font, fill=outline, anchor="lm")
        return left, right

    y = MARGIN_T
    align_type = data["meta"]["quality"]["actual_alignment_type"]
    draw.text(
        (x0, y),
        f"{SAMPLE_ID}  |  {align_type}",
        font=fonts["title"],
        fill=(30, 34, 40),
    )
    y += title_h

    upper_top = y
    upper_bot = y + upper_h
    _draw_rounded(
        draw,
        [x0 - 8, upper_top - 6, x1 + 8, upper_bot + 6],
        fill=COLOR_PANEL,
        outline=(210, 214, 220),
        width=2,
        radius=12,
    )
    draw.text(
        (x0, upper_top + 6),
        "正式播放时间轴 · 真实音轨与 ASR 词时间",
        font=fonts["section"],
        fill=(40, 46, 54),
    )

    wave_top = upper_top + 50 * DPI
    wave_bot = wave_top + 118 * DPI
    asr_top = wave_bot + 18 * DPI
    asr_bot = upper_bot - 10 * DPI

    signal = data["signal"]
    n_plot = max(2, plot_w)
    idx = np.linspace(0, max(0, len(signal) - 1), n_plot).astype(int)
    sampled = signal[idx] if len(signal) else np.zeros(n_plot)
    peak = float(np.max(np.abs(sampled), initial=1e-6))
    mid = (wave_top + wave_bot) // 2
    amp = 0.42 * (wave_bot - wave_top)
    points = [
        (x0 + i, int(mid - float(sampled[i]) / peak * amp))
        for i in range(n_plot)
    ]
    draw.rectangle([x0, wave_top, x1, wave_bot], outline=(180, 186, 194), width=1)
    if len(points) > 1:
        draw.line(points, fill=COLOR_WAVE, width=2)

    wx0 = _x_map(window_start, x0, x1, t0, t1)
    wx1 = _x_map(window_end, x0, x1, t0, t1)
    overlay = Image.new("RGBA", (W, height), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rectangle([wx0, wave_top, wx1, wave_bot], fill=COLOR_WINDOW_FILL)
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.line([(wx0, wave_top), (wx0, wave_bot)], fill=COLOR_WINDOW, width=3)
    draw.line([(wx1, wave_top), (wx1, wave_bot)], fill=COLOR_WINDOW, width=3)

    asr_rows = data["asr"]
    anchor_start = data["anchor_start"]
    asr_label_items: list[tuple[int, int, str, tuple[int, int, int], tuple[int, int, int]]] = []
    for row in asr_rows:
        s, e = float(row["start"]), float(row["end"])
        word = row["word"]
        ax0 = _x_map(s, x0, x1, t0, t1)
        ax1 = max(ax0 + 2, _x_map(e, x0, x1, t0, t1))
        is_prefix = e <= anchor_start + 1e-6
        fill = COLOR_ASR_FILL_PREFIX if is_prefix else COLOR_ASR_FILL_MATCH
        outline = COLOR_ASR_PREFIX if is_prefix else COLOR_ASR_MATCH
        draw.rectangle([ax0, asr_top, ax1, asr_bot], fill=fill, outline=outline, width=1)
        if word in {"that", "we", "do", "They've", "table"} or is_prefix:
            asr_label_items.append((ax0, ax1, word, fill, outline))
    # Redraw labeled ASR boxes: left edge fixed, widen right only.
    for ax0, ax1, word, fill, outline in asr_label_items:
        labeled_box(ax0, ax1, asr_top, asr_bot, word, fill=fill, outline=outline, x_max=x1, mode="widen_right")

    side_label("音轨", (wave_top + wave_bot) // 2)
    side_label("ASR", (asr_top + asr_bot) // 2)

    y = upper_bot + gap
    lower_top = y
    lower_bot = y + lower_h
    _draw_rounded(
        draw,
        [x0 - 8, lower_top - 6, x1 + 8, lower_bot + 6],
        fill=COLOR_PANEL,
        outline=(210, 214, 220),
        width=2,
        radius=12,
    )
    draw.text(
        (x0, lower_top + 6),
        "候选窗口 · 局部 MFA · 回移至正式播放时间",
        font=fonts["section"],
        fill=(40, 46, 54),
    )

    lane_h = 72 * DPI
    lane_gap = 52 * DPI
    lane1_top = lower_top + 58 * DPI
    lane1_bot = lane1_top + lane_h
    lane2_top = lane1_bot + lane_gap
    lane2_bot = lane2_top + lane_h
    lane3_top = lane2_bot + lane_gap
    lane3_bot = lane3_top + lane_h

    for lane_top, lane_bot in ((lane1_top, lane1_bot), (lane2_top, lane2_bot), (lane3_top, lane3_bot)):
        draw.line(
            [(x0, (lane_top + lane_bot) // 2), (x1, (lane_top + lane_bot) // 2)],
            fill=COLOR_GRID,
            width=2,
        )

    side_label("窗口", (lane1_top + lane1_bot) // 2)
    draw.rectangle(
        [wx0, lane1_top, wx1, lane1_bot],
        fill=(200, 222, 238),
        outline=COLOR_WINDOW,
        width=3,
    )
    win_cn = "MFA 候选窗口"
    win_num = f"[{window_start:.3f}, {window_end:.3f}] s"
    cn_font = fonts["box_cn"]
    num_font = fonts["box_num"]
    gap_px = 12 * DPI
    total_w = int(cn_font.getlength(win_cn)) + gap_px + int(num_font.getlength(win_num))
    if total_w > (wx1 - wx0 - 16):
        win_cn = "候选窗口"
        cn_font = fonts["box_cn_sm"]
        total_w = int(cn_font.getlength(win_cn)) + gap_px + int(num_font.getlength(win_num))
    mid_y = (lane1_top + lane1_bot) // 2
    text_left = (wx0 + wx1 - total_w) // 2
    draw.text((text_left, mid_y), win_cn, font=cn_font, fill=COLOR_WINDOW, anchor="lm")
    draw.text(
        (text_left + int(cn_font.getlength(win_cn)) + gap_px, mid_y),
        win_num,
        font=num_font,
        fill=COLOR_WINDOW,
        anchor="lm",
    )
    if wx0 > x0:
        draw.rectangle(
            [x0, lane1_top + 10, wx0, lane1_bot - 10],
            fill=(232, 234, 238),
            outline=COLOR_MUTE,
            width=1,
        )
        cut_label, cut_font = fit_word(
            "截去", wx0 - x0, prefer=fonts["box_cn"], fallback=fonts["box_cn_sm"]
        )
        draw.text(
            ((x0 + wx0) // 2, (lane1_top + lane1_bot) // 2),
            cut_label,
            font=cut_font,
            fill=COLOR_MUTE,
            anchor="mm",
        )
    if wx1 < x1:
        draw.rectangle(
            [wx1, lane1_top + 10, x1, lane1_bot - 10],
            fill=(232, 234, 238),
            outline=COLOR_MUTE,
            width=1,
        )

    side_label("局部", (lane2_top + lane2_bot) // 2)
    alignment = data["alignment"]
    highlight_local = {0, len(alignment) // 2, len(alignment) - 1} if alignment else set()
    local_boxes: list[tuple[int, int, int]] = []
    local_meta: list[tuple[int, int, int, str]] = []
    # Use the same global times as word_alignment (t' + l_i) so local/回移 share left edges.
    for i, row in enumerate(alignment):
        token = str(row["word"])
        gx0 = _x_map(float(row["start"]), x0, x1, t0, t1)
        gx1 = max(gx0 + 2, _x_map(float(row["end"]), x0, x1, t0, t1))
        local_meta.append((i, gx0, gx1, token))
        local_boxes.append((gx0, gx1, i))
        if i not in highlight_local:
            draw.rectangle(
                [gx0, lane2_top, gx1, lane2_bot],
                fill=COLOR_LOCAL_FILL,
                outline=COLOR_LOCAL,
                width=1,
            )
    for i, gx0, gx1, token in local_meta:
        # Index 0 is redrawn later with the shift lane as an identical pair.
        if i not in highlight_local or i == 0:
            continue
        gx0, gx1 = labeled_box(
            gx0,
            gx1,
            lane2_top,
            lane2_bot,
            token,
            fill=COLOR_LOCAL_FILL,
            outline=COLOR_LOCAL,
            x_max=x1,
            mode="widen_right",
        )
        local_boxes[i] = (gx0, gx1, i)
    draw.text(
        (x0 + 4, lane2_top - 22 * DPI),
        "局部 MFA（t'+l_i）",
        font=fonts["small"],
        fill=COLOR_LOCAL,
    )

    side_label("回移", (lane3_top + lane3_bot) // 2)
    highlight_shift = {0, len(alignment) // 2, len(alignment) - 1}
    shift_boxes: list[tuple[int, int, int]] = []
    shift_meta: list[tuple[int, int, int, str]] = []
    for i, row in enumerate(alignment):
        s, e = float(row["start"]), float(row["end"])
        word = str(row["word"])
        sx0 = _x_map(s, x0, x1, t0, t1)
        sx1 = max(sx0 + 2, _x_map(e, x0, x1, t0, t1))
        shift_meta.append((i, sx0, sx1, word))
        shift_boxes.append((sx0, sx1, i))
        if i not in highlight_shift:
            draw.rectangle(
                [sx0, lane3_top, sx1, lane3_bot],
                fill=COLOR_SHIFT_FILL,
                outline=COLOR_SHIFT,
                width=1,
            )
    for i, sx0, sx1, word in shift_meta:
        if i not in highlight_shift or i == 0:
            continue
        sx0, sx1 = labeled_box(
            sx0,
            sx1,
            lane3_top,
            lane3_bot,
            word,
            fill=COLOR_SHIFT_FILL,
            outline=COLOR_SHIFT,
            x_max=x1,
            mode="widen_right",
        )
        shift_boxes[i] = (sx0, sx1, i)

    # Shared geometry for the first official word on both local and shift lanes.
    first = alignment[0]
    fs, fe = float(first["start"]), float(first["end"])
    first_left = _x_map(fs, x0, x1, t0, t1)
    first_right = max(first_left + 2, _x_map(fe, x0, x1, t0, t1))
    first_word = str(first["word"])

    # Redraw first-word boxes on both lanes with identical left/right after widen.
    first_left, first_right_l = labeled_box(
        first_left,
        first_right,
        lane2_top,
        lane2_bot,
        first_word,
        fill=COLOR_LOCAL_FILL,
        outline=COLOR_LOCAL,
        x_max=x1,
        mode="widen_right",
    )
    _, first_right_s = labeled_box(
        first_left,
        first_right,
        lane3_top,
        lane3_bot,
        first_word,
        fill=COLOR_SHIFT_FILL,
        outline=COLOR_SHIFT,
        x_max=x1,
        mode="widen_right",
    )
    # Keep the two first boxes the same width (use the wider of the two).
    common_right = max(first_right_l, first_right_s)
    draw.rectangle(
        [first_left, lane2_top, common_right, lane2_bot],
        fill=COLOR_LOCAL_FILL,
        outline=COLOR_LOCAL,
        width=3,
    )
    draw.text(
        (first_left + 14 * DPI, (lane2_top + lane2_bot) // 2),
        first_word,
        font=fonts["word"],
        fill=COLOR_LOCAL,
        anchor="lm",
    )
    draw.rectangle(
        [first_left, lane3_top, common_right, lane3_bot],
        fill=COLOR_SHIFT_FILL,
        outline=COLOR_SHIFT,
        width=3,
    )
    draw.text(
        (first_left + 14 * DPI, (lane3_top + lane3_bot) // 2),
        first_word,
        font=fonts["word"],
        fill=COLOR_SHIFT,
        anchor="lm",
    )

    # Straight connector on the shared true start of the first official word.
    draw.line(
        [(first_left, lane2_bot + 2), (first_left, lane3_top - 2)],
        fill=COLOR_SHIFT,
        width=3,
    )

    note = f'"{first_word}"  {fs:.2f}-{fe:.2f} s  =  t\' + l_i'
    note_x = min(max(first_left + 10 * DPI, wx0 + 40 * DPI), x1 - 520 * DPI)
    note_y = lane3_bot + 10 * DPI
    draw.text((note_x, note_y), note, font=fonts["formula"], fill=COLOR_SHIFT, anchor="lt")

    # Draw ell_i last so it stays visible; it must stay left of the first word.
    draw.line([(wx0, lane1_top - 4), (wx0, lane3_bot + 2)], fill=COLOR_WINDOW, width=3)
    draw.text(
        (wx0 + 10 * DPI, lane1_top - 12),
        f"l_i = {window_start:.2f} s",
        font=fonts["formula"],
        fill=COLOR_WINDOW,
        anchor="lb",
    )

    axis_y = note_y + 40 * DPI
    draw.line([(x0, axis_y), (x1, axis_y)], fill=COLOR_AXIS, width=2)
    for tick in np.linspace(0.0, duration, 6):
        tx = _x_map(float(tick), x0, x1, t0, t1)
        draw.line([(tx, axis_y - 6), (tx, axis_y + 6)], fill=COLOR_AXIS, width=2)
        draw.text((tx, axis_y + 8), f"{tick:.1f}", font=fonts["num"], fill=COLOR_AXIS, anchor="mt")
    draw.text((x1, axis_y + 34), "t / s  (正式播放)", font=fonts["small"], fill=COLOR_AXIS, anchor="rt")

    ly = axis_y + 54 * DPI
    items = [
        (COLOR_ASR_FILL_PREFIX, COLOR_ASR_PREFIX, "ASR 额外前缀"),
        (COLOR_ASR_FILL_MATCH, COLOR_ASR_MATCH, "ASR 对齐段"),
        ((200, 222, 238), COLOR_WINDOW, "MFA 候选窗口"),
        (COLOR_LOCAL_FILL, COLOR_LOCAL, "局部 MFA (t')"),
        (COLOR_SHIFT_FILL, COLOR_SHIFT, "回移后官方词 (t'+l_i)"),
    ]
    lx = x0
    for fill, outline, label in items:
        draw.rectangle([lx, ly, lx + 22 * DPI, ly + 16 * DPI], fill=fill, outline=outline, width=2)
        draw.text(
            (lx + 28 * DPI, ly + 8 * DPI),
            label,
            font=fonts["small"],
            fill=(50, 54, 60),
            anchor="lm",
        )
        lx += 28 * DPI + int(fonts["small"].getlength(label)) + 30 * DPI

    content_bottom = ly + 24 * DPI + MARGIN_B
    if content_bottom < height:
        canvas = canvas.crop((0, 0, W, content_bottom))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(OUT_PATH, format="PNG", optimize=True)
    return OUT_PATH


def main() -> None:
    data = load_sample()
    path = draw_figure(data)
    print(
        f"[ok] window=[{data['window_start']:.6f}, {data['window_end']:.6f}] "
        f"duration={data['duration']:.6f} first={data['alignment'][0]['start']}-{data['alignment'][0]['end']}"
    )
    print(f"[ok] local_mfa_words={len(data['local_words'])} asr_words={len(data['asr'])}")
    print(f"[saved] {path.resolve()} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
