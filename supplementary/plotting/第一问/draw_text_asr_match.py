"""Draw Fig. q1_text_asr_match: official vs ASR sequential matching.

Two real samples (read-only quality.json / asr_words.csv / word_alignment.csv):
  (a) -3g5yACwYnA__13  — ASR has extra preamble before the official text
  (b) -9y-fZ3swSY__0   — only local contiguous word spans match

Writes PNG only under cursor_draw/figures/.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
Q1_ROOT = HERE.parent
SAMPLE_ROOT = Q1_ROOT / "输出" / "samples"
OUT_DIR = HERE / "figures"
OUT_PATH = OUT_DIR / "q1_text_asr_match.png"

SAMPLE_A = "-3g5yACwYnA__13"
SAMPLE_B = "-9y-fZ3swSY__0"

DPI_SCALE = 2
W = 1680 * DPI_SCALE
MARGIN_X = 48 * DPI_SCALE
MARGIN_TOP = 18 * DPI_SCALE
MARGIN_BOTTOM = 28 * DPI_SCALE
ROW_GAP = 24 * DPI_SCALE
BOX_H = 70 * DPI_SCALE
BOX_PAD_X = 18 * DPI_SCALE
BOX_GAP = 16 * DPI_SCALE
LINE_GAP = 22 * DPI_SCALE
LANE_GAP = 82 * DPI_SCALE
ANCHOR_HEADROOM = 44 * DPI_SCALE
LEGEND_BAND = 50 * DPI_SCALE

# Layout font sizes the box; render font is larger Times inside the same box.
WORD_LAYOUT_SIZE = 36
WORD_RENDER_SIZE = 54

COLOR_MATCHED = (46, 110, 158)
COLOR_MATCHED_FILL = (214, 234, 244)
COLOR_UNMATCHED = (140, 140, 140)
COLOR_UNMATCHED_FILL = (238, 238, 238)
COLOR_EXTRA = (176, 112, 56)
COLOR_EXTRA_FILL = (255, 236, 212)
COLOR_ANCHOR = (168, 72, 24)
COLOR_LINK = (96, 132, 168)
COLOR_BG = (255, 255, 255)
COLOR_PANEL = (250, 251, 253)
COLOR_SHADOW = (210, 214, 220)


def _load_cn_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
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
    return ImageFont.load_default()


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
    return _load_cn_font(size, bold=bold)


def _draw_text(draw, xy, text, font, fill=(20, 20, 20), anchor="lt") -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def _text_width(font: ImageFont.ImageFont, text: str) -> int:
    bbox = font.getbbox(text)
    return int(bbox[2] - bbox[0])


def load_sample(sample_id: str) -> dict:
    sample_dir = SAMPLE_ROOT / sample_id
    quality = json.loads((sample_dir / "quality.json").read_text(encoding="utf-8"))
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))

    official_words: list[str] = []
    with (sample_dir / "word_alignment.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            official_words.append(row["word"])

    asr_words: list[dict] = []
    asr_csv = sample_dir / "asr_words.csv"
    if asr_csv.is_file():
        with asr_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                asr_words.append(
                    {
                        "word": row["word"],
                        "start": float(row["start"]),
                        "end": float(row["end"]),
                    }
                )
    else:
        for item in quality.get("asr_words", []):
            asr_words.append(
                {
                    "word": item["word"],
                    "start": float(item["start"]),
                    "end": float(item["end"]),
                }
            )

    mapping = [int(v) for v in quality["official_to_asr"]]
    match_mask = [int(v) for v in quality["official_match_mask"]]
    if len(mapping) != len(official_words) or len(match_mask) != len(official_words):
        raise ValueError(f"{sample_id}: mapping/mask length mismatch")

    mapped_asr = {j for j in mapping if j >= 0}
    return {
        "sample_id": sample_id,
        "official_words": official_words,
        "asr_words": asr_words,
        "mapping": mapping,
        "match_mask": match_mask,
        "mapped_asr": mapped_asr,
        "consistency_score": float(quality.get("consistency_score", 0.0)),
        "alignment_type": metadata["quality"].get("actual_alignment_type")
        or quality.get("alignment_type"),
        "first_anchor": bool(match_mask[0]) if match_mask else False,
        "last_anchor": bool(match_mask[-1]) if match_mask else False,
    }


def layout_words(
    words: list[str],
    *,
    roles: list[str],
    x0: int,
    y0: int,
    max_x: int,
    font: ImageFont.ImageFont,
) -> tuple[list[dict], int]:
    """Single-line layout: grow gaps to fill width, or shrink widths only if necessary."""
    n = len(words)
    if n == 0:
        return [], y0

    natural_w = []
    for word in words:
        tw = _text_width(font, word)
        # Extra room for long words / apostrophes so text is not cramped
        bonus = int(6 * DPI_SCALE)
        if "'" in word or "’" in word:
            bonus += int(14 * DPI_SCALE)
        if len(word) >= 7:
            bonus += int(12 * DPI_SCALE)
        if len(word) >= 9:
            bonus += int(12 * DPI_SCALE)
        natural_w.append(max(tw + 2 * BOX_PAD_X + bonus, int(0.85 * BOX_H)))

    gap0 = BOX_GAP
    width_sum = sum(natural_w)
    avail = max(1, max_x - x0)
    gaps_n = max(n - 1, 0)

    if gaps_n == 0:
        bw = min(natural_w[0], avail)
        return (
            [
                {
                    "index": 0,
                    "word": words[0],
                    "role": roles[0],
                    "x": x0,
                    "y": y0,
                    "w": bw,
                    "h": BOX_H,
                    "cx": x0 + bw // 2,
                    "cy": y0 + BOX_H // 2,
                }
            ],
            y0 + BOX_H,
        )

    # Prefer keeping natural box widths; spend leftover on gaps (fill right blank).
    # Only shrink boxes if even minimum gaps cannot fit.
    min_gap = max(6, int(0.45 * BOX_GAP))
    if width_sum + min_gap * gaps_n > avail:
        scale = (avail - min_gap * gaps_n) / max(width_sum, 1)
        widths = [max(20, int(round(w * scale))) for w in natural_w]
        gap = min_gap
        # Fix rounding drift on the last box
        used = sum(widths) + gap * gaps_n
        widths[-1] = max(20, widths[-1] + (avail - used))
    else:
        widths = list(natural_w)
        leftover = avail - width_sum
        gap = leftover // gaps_n
        rem = leftover - gap * gaps_n
        # Distribute remainder pixels into the first few gaps
        gap_list = [gap + (1 if i < rem else 0) for i in range(gaps_n)]
        boxes: list[dict] = []
        x = x0
        for index, (word, role, bw) in enumerate(zip(words, roles, widths)):
            boxes.append(
                {
                    "index": index,
                    "word": word,
                    "role": role,
                    "x": x,
                    "y": y0,
                    "w": bw,
                    "h": BOX_H,
                    "cx": x + bw // 2,
                    "cy": y0 + BOX_H // 2,
                }
            )
            if index < gaps_n:
                x += bw + gap_list[index]
        return boxes, y0 + BOX_H

    boxes = []
    x = x0
    for index, (word, role, bw) in enumerate(zip(words, roles, widths)):
        boxes.append(
            {
                "index": index,
                "word": word,
                "role": role,
                "x": x,
                "y": y0,
                "w": bw,
                "h": BOX_H,
                "cx": x + bw // 2,
                "cy": y0 + BOX_H // 2,
            }
        )
        x += bw + gap
    return boxes, y0 + BOX_H


def _style_for(role: str) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
    if role == "matched":
        return COLOR_MATCHED_FILL, COLOR_MATCHED, 2
    if role in {"anchor_first", "anchor_last"}:
        return COLOR_MATCHED_FILL, COLOR_ANCHOR, 3
    if role == "extra":
        return COLOR_EXTRA_FILL, COLOR_EXTRA, 2
    return COLOR_UNMATCHED_FILL, COLOR_UNMATCHED, 2


def draw_boxes(
    draw: ImageDraw.ImageDraw,
    boxes: list[dict],
    word_font: ImageFont.ImageFont,
    label_font: ImageFont.ImageFont,
) -> None:
    for box in boxes:
        fill, outline, width = _style_for(box["role"])
        x0, y0 = box["x"], box["y"]
        x1, y1 = x0 + box["w"], y0 + box["h"]

        # Soft shadow
        draw.rounded_rectangle(
            [x0 + 2, y0 + 3, x1 + 2, y1 + 3],
            radius=11,
            fill=COLOR_SHADOW,
        )
        # Main pill
        draw.rounded_rectangle(
            [x0, y0, x1, y1],
            radius=11,
            fill=fill,
            outline=outline,
            width=width,
        )
        # Thin inner rim for a cleaner edge
        inset = 3
        if x1 - x0 > 2 * inset + 8 and y1 - y0 > 2 * inset + 8:
            draw.rounded_rectangle(
                [x0 + inset, y0 + inset, x1 - inset, y1 - inset],
                radius=8,
                outline=(255, 255, 255),
                width=1,
            )

        _draw_text(
            draw,
            (box["cx"], box["cy"]),
            box["word"],
            word_font,
            fill=(24, 28, 32),
            anchor="mm",
        )

        if box["role"] == "anchor_first":
            _draw_text(draw, (box["cx"], y0 - 6), "首锚点", label_font, fill=COLOR_ANCHOR, anchor="mb")
        elif box["role"] == "anchor_last":
            _draw_text(draw, (box["cx"], y0 - 6), "末锚点", label_font, fill=COLOR_ANCHOR, anchor="mb")


def draw_links(
    draw: ImageDraw.ImageDraw,
    official_boxes: list[dict],
    asr_boxes: list[dict],
    mapping: list[int],
) -> None:
    asr_by_index = {b["index"]: b for b in asr_boxes}
    for ob in official_boxes:
        j = mapping[ob["index"]]
        if j < 0 or j not in asr_by_index:
            continue
        ab = asr_by_index[j]
        draw.line(
            [(ob["cx"], ob["y"] + ob["h"]), (ab["cx"], ab["y"])],
            fill=COLOR_LINK,
            width=2,
        )


def draw_legend(draw, right: int, y: int, fonts: dict) -> int:
    """Draw enlarged legend right-aligned outside the a/b panels. Returns bottom y."""
    items = [
        (COLOR_MATCHED_FILL, COLOR_MATCHED, "匹配"),
        (COLOR_UNMATCHED_FILL, COLOR_UNMATCHED, "未匹配官方词"),
        (COLOR_EXTRA_FILL, COLOR_EXTRA, "额外ASR词"),
        (COLOR_MATCHED_FILL, COLOR_ANCHOR, "首/末锚点"),
    ]
    swatch = 48
    swatch_h = 40
    gap_label = 14
    gap_item = 36
    widths = []
    total = 0
    for _, _, label in items:
        tw = _text_width(fonts["legend"], label)
        item_w = swatch + gap_label + tw
        widths.append(item_w)
        total += item_w
    total += gap_item * (len(items) - 1)

    x_cursor = right - total
    for (fill, outline, label), item_w in zip(items, widths):
        draw.rounded_rectangle(
            [x_cursor, y, x_cursor + swatch, y + swatch_h],
            radius=10,
            fill=fill,
            outline=outline,
            width=4 if outline == COLOR_ANCHOR else 3,
        )
        _draw_text(
            draw,
            (x_cursor + swatch + gap_label, y + swatch_h // 2),
            label,
            fonts["legend"],
            fill=(40, 40, 40),
            anchor="lm",
        )
        x_cursor += item_w + gap_item
    return y + swatch_h


def official_roles(sample: dict) -> list[str]:
    roles: list[str] = []
    n = len(sample["official_words"])
    for i, matched in enumerate(sample["match_mask"]):
        if matched:
            if i == 0:
                roles.append("anchor_first")
            elif i == n - 1:
                roles.append("anchor_last")
            else:
                roles.append("matched")
        else:
            roles.append("unmatched")
    return roles


def asr_roles(sample: dict) -> list[str]:
    return ["matched" if i in sample["mapped_asr"] else "extra" for i in range(len(sample["asr_words"]))]


def _panel_header_height() -> int:
    return int(36 * DPI_SCALE)


def draw_sample_panel(
    draw: ImageDraw.ImageDraw,
    sample: dict,
    *,
    top: int,
    left: int,
    right: int,
    tag: str,
    fonts: dict,
) -> int:
    y = top
    title = f"{tag}  {sample['sample_id']}"
    _draw_text(draw, (left, y), title, fonts["title"], fill=(15, 15, 15))
    y += int(36 * DPI_SCALE)

    _draw_text(draw, (left, y), "官方词", fonts["lane"], fill=(30, 30, 30))
    y += int(30 * DPI_SCALE) + ANCHOR_HEADROOM

    o_boxes, o_bottom = layout_words(
        sample["official_words"],
        roles=official_roles(sample),
        x0=left,
        y0=y,
        max_x=right,
        font=fonts["word_layout"],
    )
    if o_boxes and not sample["first_anchor"]:
        b = o_boxes[0]
        _draw_text(draw, (b["cx"], b["y"] - 6), "无首锚点", fonts["anchor"], fill=(150, 60, 60), anchor="mb")
    if o_boxes and not sample["last_anchor"]:
        b = o_boxes[-1]
        _draw_text(draw, (b["cx"], b["y"] - 6), "无末锚点", fonts["anchor"], fill=(150, 60, 60), anchor="mb")

    asr_top = o_bottom + LANE_GAP
    _draw_text(draw, (left, asr_top - int(30 * DPI_SCALE)), "ASR词", fonts["lane"], fill=(30, 30, 30))
    a_boxes, a_bottom = layout_words(
        [w["word"] for w in sample["asr_words"]],
        roles=asr_roles(sample),
        x0=left,
        y0=asr_top,
        max_x=right,
        font=fonts["word_layout"],
    )

    draw_links(draw, o_boxes, a_boxes, sample["mapping"])
    draw_boxes(draw, o_boxes, fonts["word"], fonts["anchor"])
    draw_boxes(draw, a_boxes, fonts["word"], fonts["anchor"])
    return a_bottom + int(18 * DPI_SCALE)


def _estimate_height(sample: dict, fonts: dict, left: int, right: int) -> int:
    y = _panel_header_height()
    y += int(30 * DPI_SCALE) + ANCHOR_HEADROOM
    _, o_bottom = layout_words(
        sample["official_words"],
        roles=official_roles(sample),
        x0=left,
        y0=y,
        max_x=right,
        font=fonts["word_layout"],
    )
    asr_top = o_bottom + LANE_GAP
    _, a_bottom = layout_words(
        [w["word"] for w in sample["asr_words"]],
        roles=asr_roles(sample),
        x0=left,
        y0=asr_top,
        max_x=right,
        font=fonts["word_layout"],
    )
    return a_bottom + int(18 * DPI_SCALE)


def draw_figure(samples: list[dict], out_path: Path) -> Path:
    fonts = {
        "title": _load_cn_font(40, bold=True),
        "body": _load_cn_font(32),
        "lane": _load_cn_font(34, bold=True),
        "legend": _load_cn_font(42),
        "anchor": _load_cn_font(36),
        "word_layout": _load_times(WORD_LAYOUT_SIZE),
        "word": _load_times(WORD_RENDER_SIZE),
    }

    left = MARGIN_X
    right = W - MARGIN_X
    heights = [_estimate_height(sample, fonts, left, right) for sample in samples]
    height = (
        MARGIN_TOP
        + LEGEND_BAND
        + sum(heights)
        + ROW_GAP * (len(samples) - 1)
        + MARGIN_BOTTOM
    )
    img = Image.new("RGB", (W, height), COLOR_BG)
    draw = ImageDraw.Draw(img)

    # Legend outside panels a/b, close above panel (a), right-aligned.
    legend_y = MARGIN_TOP
    draw_legend(draw, right, legend_y, fonts)

    y = MARGIN_TOP + LEGEND_BAND
    tags = ["(a)", "(b)"]
    for i, sample in enumerate(samples):
        panel_top = y - int(10 * DPI_SCALE)
        est_bottom = y + heights[i]
        draw.rounded_rectangle(
            [left - 18, panel_top, right + 18, est_bottom + 10],
            radius=14,
            fill=COLOR_PANEL,
            outline=(215, 220, 226),
            width=2,
        )
        y = draw_sample_panel(
            draw,
            sample,
            top=y,
            left=left,
            right=right,
            tag=tags[i],
            fonts=fonts,
        )
        if i < len(samples) - 1:
            y += ROW_GAP

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG", optimize=True)
    return out_path


def main() -> None:
    sample_a = load_sample(SAMPLE_A)
    sample_b = load_sample(SAMPLE_B)
    for s in (sample_a, sample_b):
        print(
            f"[ok] {s['sample_id']}: {s['alignment_type']} S={s['consistency_score']:.4f} "
            f"match={sum(s['match_mask'])}/{len(s['official_words'])} "
            f"anchors=({s['first_anchor']},{s['last_anchor']})"
        )
    path = draw_figure([sample_a, sample_b], OUT_PATH)
    print(f"[saved] {path}")
    print(f"[size]  {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
