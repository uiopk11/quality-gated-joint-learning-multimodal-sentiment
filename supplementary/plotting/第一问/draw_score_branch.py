"""绘制 100 条样本的“一致性分数—最终分支”结果图。

数据来源：
1. ``quality_audit.csv``：``sample_id``、``consistency_score``、``speech_present``；
2. ``all_samples.csv``：``sample_id``、最终 ``alignment_type``。

脚本只按 ``sample_id`` 一对一关联真实结果，不从分支计数反推或构造分数。
使用 Pillow 绘制立体感散点（本机 Matplotlib Agg 不稳定）。
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont


HERE = Path(__file__).resolve().parent
Q1_ROOT = HERE.parent
SUMMARY_DIR = Q1_ROOT / "输出" / "summary"
DEFAULT_OUTPUT = HERE / "figures" / "q1_score_branch"

TAU_LOW = 0.35
TAU_HIGH = 0.75

BRANCH_ORDER = [
    "full_mfa",
    "full_mfa_windowed",
    "partial_asr_timestamp_fallback",
    "partial_asr_timestamp",
    "no_word_alignment",
]
BRANCH_LABELS = {
    "full_mfa": "完整 MFA",
    "full_mfa_windowed": "窗口化 MFA",
    "partial_asr_timestamp_fallback": "高分降级局部 ASR",
    "partial_asr_timestamp": "直接局部 ASR",
    "no_word_alignment": "无词级对齐",
}

STYLES = {
    "full_mfa": {"color": (47, 111, 94), "light": (160, 220, 195), "shape": "sphere"},
    "full_mfa_windowed": {"color": (79, 143, 106), "light": (175, 225, 185), "shape": "cone"},
    "partial_asr_timestamp_fallback": {"color": (184, 74, 74), "light": (240, 180, 180), "shape": "diamond"},
    "partial_asr_timestamp": {"color": (196, 132, 42), "light": (245, 210, 150), "shape": "cube"},
    "no_word_alignment": {"color": (94, 101, 112), "light": (190, 196, 205), "shape": "sphere"},
}

DPI = 2
W = 1460 * DPI
H = 960 * DPI
MARGIN_L = 175 * DPI
MARGIN_R = 360 * DPI  # 缩短横轴
MARGIN_T = 72 * DPI
MARGIN_B = 185 * DPI
MARKER_SCALE = 16
LEGEND_MARKER_SCALE = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-audit", type=Path, default=SUMMARY_DIR / "quality_audit.csv")
    parser.add_argument("--all-samples", type=Path, default=SUMMARY_DIR / "all_samples.csv")
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _as_bool(series: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "1": True, "false": False, "0": False}
    invalid = sorted(set(normalized) - set(mapping))
    if invalid:
        raise ValueError(f"{column} 含无法解析的布尔值: {invalid}")
    return normalized.map(mapping).astype(bool)


def load_and_validate(quality_path: Path, samples_path: Path) -> pd.DataFrame:
    quality = pd.read_csv(quality_path)
    samples = pd.read_csv(samples_path)

    quality_required = {"sample_id", "consistency_score", "speech_present"}
    samples_required = {
        "sample_id",
        "text_audio_consistency_score",
        "speech_present",
        "alignment_type",
    }
    if missing := quality_required - set(quality.columns):
        raise ValueError(f"quality_audit.csv 缺少字段: {sorted(missing)}")
    if missing := samples_required - set(samples.columns):
        raise ValueError(f"all_samples.csv 缺少字段: {sorted(missing)}")
    if len(quality) != 100 or len(samples) != 100:
        raise ValueError(
            f"正式结果应各含 100 行，实际 quality={len(quality)}, all_samples={len(samples)}"
        )
    if quality["sample_id"].duplicated().any() or samples["sample_id"].duplicated().any():
        raise ValueError("sample_id 必须在两张表中分别唯一")

    left = quality[["sample_id", "consistency_score", "speech_present"]].rename(
        columns={"speech_present": "speech_present_audit"}
    )
    right = samples[
        ["sample_id", "text_audio_consistency_score", "speech_present", "alignment_type"]
    ].rename(
        columns={
            "speech_present": "speech_present_summary",
            "alignment_type": "final_branch",
        }
    )
    merged = left.merge(right, on="sample_id", how="inner", validate="one_to_one")
    if len(merged) != 100:
        raise ValueError("sample_id 未完整对应")
    if merged.isna().any().any():
        raise ValueError("关联结果存在缺失值")

    merged["consistency_score"] = pd.to_numeric(merged["consistency_score"], errors="raise")
    merged["text_audio_consistency_score"] = pd.to_numeric(
        merged["text_audio_consistency_score"], errors="raise"
    )
    if not merged["consistency_score"].between(0.0, 1.0).all():
        raise ValueError("consistency_score 必须全部位于 [0, 1]")
    if not np.allclose(
        merged["consistency_score"],
        merged["text_audio_consistency_score"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("两张表中的一致性分数不一致")

    merged["speech_present_audit"] = _as_bool(
        merged["speech_present_audit"], "quality_audit.speech_present"
    )
    merged["speech_present_summary"] = _as_bool(
        merged["speech_present_summary"], "all_samples.speech_present"
    )
    if not merged["speech_present_audit"].equals(merged["speech_present_summary"]):
        raise ValueError("两张表中的 speech_present 状态不一致")
    merged["speech_present"] = merged["speech_present_audit"]

    unknown = sorted(set(merged["final_branch"]) - set(BRANCH_ORDER))
    if unknown:
        raise ValueError(f"发现未知最终分支: {unknown}")
    return merged.sort_values(["final_branch", "consistency_score", "sample_id"]).reset_index(
        drop=True
    )


def _stable_unit_interval(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big")
    return integer / float(2**64 - 1)


def add_vertical_jitter(data: pd.DataFrame, amplitude: float = 0.13) -> pd.DataFrame:
    plotted = data.copy()
    y_base = {branch: float(len(BRANCH_ORDER) - 1 - i) for i, branch in enumerate(BRANCH_ORDER)}
    plotted["y_base"] = plotted["final_branch"].map(y_base)
    plotted["y"] = plotted["y_base"]

    for branch, group in plotted.groupby("final_branch", sort=False):
        rounded = group["consistency_score"].round(6)
        for _, same_score in group.groupby(rounded, sort=False):
            ordered = same_score.sort_values("sample_id")
            if len(ordered) > 1:
                offsets = np.linspace(-amplitude, amplitude, len(ordered))
            else:
                sid = str(ordered.iloc[0]["sample_id"])
                offsets = np.array([(_stable_unit_interval(f"{branch}|{sid}") - 0.5) * amplitude])
            plotted.loc[ordered.index, "y"] = plotted.loc[ordered.index, "y_base"] + offsets
    return plotted


def _load_font(size: int, *, bold: bool = False, serif: bool = False) -> ImageFont.ImageFont:
    windir = Path(r"C:\Windows\Fonts")
    if serif:
        names = ["georgia.ttf", "cambria.ttc", "times.ttf"]
    elif bold:
        names = ["msyhbd.ttc", "seguisb.ttf", "calibrib.ttf", "msyh.ttc"]
    else:
        names = ["msyh.ttc", "segoeui.ttf", "calibri.ttf", "simhei.ttf"]
    for name in names:
        path = windir / name
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _blend(c: tuple[int, int, int], t: float, toward=(255, 255, 255)) -> tuple[int, int, int]:
    return tuple(int(round(np.clip(a + (b - a) * t, 0, 255))) for a, b in zip(c, toward))


def _dark(c: tuple[int, int, int], amount: float = 0.35) -> tuple[int, int, int]:
    return _blend(c, amount, (15, 18, 22))


def _x_map(s: float, x0: int, x1: int) -> int:
    return int(round(x0 + float(np.clip(s, 0.0, 1.0)) * (x1 - x0)))


def _y_map(v: float, y0: int, y1: int, vmin: float, vmax: float) -> int:
    ratio = (v - vmin) / (vmax - vmin)
    return int(round(y1 - ratio * (y1 - y0)))


def _draw_sphere(draw: ImageDraw.ImageDraw, x: int, y: int, r: int, color, light) -> None:
    # shadow
    draw.ellipse([x - r + 3, y - r + 5, x + r + 3, y + r + 5], fill=(170, 176, 184, 90))
    # body
    draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=_dark(color))
    # highlight
    hr = max(2, r // 3)
    draw.ellipse(
        [x - r + 3, y - r + 2, x - r + 3 + 2 * hr, y - r + 2 + 2 * hr],
        fill=light,
    )


def _draw_cone(draw: ImageDraw.ImageDraw, x: int, y: int, r: int, color, light) -> None:
    draw.polygon(
        [(x + 3, y + r + 3), (x - r + 3, y + r + 3), (x + 3, y - r + 1)],
        fill=(170, 176, 184),
    )
    pts = [(x, y - r - 1), (x + r + 1, y + r), (x - r - 1, y + r)]
    draw.polygon(pts, fill=color, outline=_dark(color))
    # soft top highlight tip
    draw.ellipse([x - 3, y - r, x + 3, y - r + 6], fill=light)


def _draw_diamond(draw: ImageDraw.ImageDraw, x: int, y: int, r: int, color, light) -> None:
    draw.polygon(
        [(x + 3, y - r + 2), (x + r + 3, y + 3), (x + 3, y + r + 4), (x - r + 3, y + 3)],
        fill=(170, 176, 184),
    )
    pts = [(x, y - r - 1), (x + r + 1, y), (x, y + r + 1), (x - r - 1, y)]
    draw.polygon(pts, fill=color, outline=_dark(color))
    # upper facet highlight
    draw.polygon([(x, y - r + 2), (x + r // 2, y - 1), (x, y + 2), (x - r // 2, y - 1)], fill=light)


def _draw_cube(draw: ImageDraw.ImageDraw, x: int, y: int, r: int, color, light) -> None:
    # isometric-ish cube suggestion
    side = _blend(color, 0.18)
    top = light
    front = color
    ox, oy = 4, 3
    # shadow
    draw.rectangle([x - r + 4, y - r + 6, x + r + 4, y + r + 6], fill=(170, 176, 184))
    # front face
    draw.rectangle([x - r, y - r + 2, x + r - 2, y + r], fill=front, outline=_dark(color))
    # top face
    draw.polygon(
        [
            (x - r, y - r + 2),
            (x - r + ox, y - r - oy),
            (x + r - 2 + ox, y - r - oy),
            (x + r - 2, y - r + 2),
        ],
        fill=top,
        outline=_dark(color, 0.2),
    )
    # side face
    draw.polygon(
        [
            (x + r - 2, y - r + 2),
            (x + r - 2 + ox, y - r - oy),
            (x + r - 2 + ox, y + r - oy),
            (x + r - 2, y + r),
        ],
        fill=side,
        outline=_dark(color, 0.25),
    )


def _draw_hollow_x(draw: ImageDraw.ImageDraw, x: int, y: int, r: int, color) -> None:
    w = max(3, r // 4)
    inset = max(3, r // 3)
    draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=w)
    draw.line([(x - r + inset, y - r + inset), (x + r - inset, y + r - inset)], fill=color, width=w)
    draw.line([(x - r + inset, y + r - inset), (x + r - inset, y - r + inset)], fill=color, width=w)


def draw_marker(draw, x, y, *, shape: str, color, light, hollow: bool = False, scale: int = 11) -> None:
    r = scale
    if hollow:
        _draw_hollow_x(draw, x, y, r + 1, color)
        return
    if shape == "sphere":
        _draw_sphere(draw, x, y, r, color, light)
    elif shape == "cone":
        _draw_cone(draw, x, y, r, color, light)
    elif shape == "diamond":
        _draw_diamond(draw, x, y, r, color, light)
    elif shape == "cube":
        _draw_cube(draw, x, y, r - 1, color, light)
    else:
        _draw_sphere(draw, x, y, r, color, light)


def _paste_text(
    base: Image.Image,
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    *,
    xy: tuple[int, int],
    angle: float = 0.0,
    anchor: str = "mm",
) -> None:
    """Draw text onto RGBA canvas; supports rotation (degrees, counter-clockwise)."""
    bbox = font.getbbox(text)
    pad = 4
    tw, th = bbox[2] - bbox[0] + 2 * pad, bbox[3] - bbox[1] + 2 * pad
    canvas = Image.new("RGBA", (max(tw, 2), max(th, 2)), (0, 0, 0, 0))
    ImageDraw.Draw(canvas).text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=(*fill, 255))
    if abs(angle) > 1e-6:
        canvas = canvas.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    cw, ch = canvas.size
    ax, ay = xy
    # Interpret anchor relative to the (possibly rotated) bitmap.
    if "l" in anchor:
        px = ax
    elif "r" in anchor:
        px = ax - cw
    else:
        px = ax - cw // 2
    if "t" in anchor:
        py = ay
    elif "b" in anchor:
        py = ay - ch
    else:
        py = ay - ch // 2
    base.alpha_composite(canvas, (int(px), int(py)))


def draw_figure(data: pd.DataFrame, output_stem: Path) -> list[Path]:
    plotted = add_vertical_jitter(data)
    counts = plotted["final_branch"].value_counts()

    fonts = {
        "axis": _load_font(40, bold=True),
        "tick": _load_font(36),
        "num": _load_font(34, serif=True),
        "legend": _load_font(36),
        "tau": _load_font(40, serif=True),
    }

    base = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    draw = ImageDraw.Draw(base)

    plot_x0, plot_x1 = MARGIN_L, W - MARGIN_R
    plot_y0, plot_y1 = MARGIN_T, H - MARGIN_B
    vmin, vmax = -0.55, len(BRANCH_ORDER) - 0.45

    for i, branch in enumerate(reversed(BRANCH_ORDER)):
        y_lo = _y_map(i - 0.48, plot_y0, plot_y1, vmin, vmax)
        y_hi = _y_map(i + 0.48, plot_y0, plot_y1, vmin, vmax)
        top, bot = min(y_lo, y_hi), max(y_lo, y_hi)
        fill = (238, 241, 245, 255) if i % 2 == 0 else (247, 248, 250, 255)
        draw.rectangle([plot_x0, top, plot_x1, bot], fill=fill)

    for i in range(len(BRANCH_ORDER)):
        yy = _y_map(float(i), plot_y0, plot_y1, vmin, vmax)
        draw.line([(plot_x0, yy), (plot_x1, yy)], fill=(213, 218, 224), width=2)

    draw.rounded_rectangle(
        [plot_x0, plot_y0, plot_x1, plot_y1],
        radius=10,
        outline=(138, 145, 154),
        width=3,
    )

    # Threshold dashed lines (clear dash pattern)
    for thr, color in [(TAU_LOW, (196, 122, 58)), (TAU_HIGH, (63, 124, 168))]:
        xx = _x_map(thr, plot_x0, plot_x1)
        y = plot_y0
        dash, gap = 16, 12
        while y < plot_y1:
            y2 = min(y + dash, plot_y1)
            draw.line([(xx, y), (xx, y2)], fill=color, width=4)
            y = y2 + gap
        label = f"τ_L={thr:.2f}" if thr == TAU_LOW else f"τ_H={thr:.2f}"
        draw.text((xx, plot_y0 - 12), label, font=fonts["tau"], fill=color, anchor="mb")

    points = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(points)
    for _, row in plotted.iterrows():
        x = _x_map(float(row["consistency_score"]), plot_x0, plot_x1)
        y = _y_map(float(row["y"]), plot_y0, plot_y1, vmin, vmax)
        branch = row["final_branch"]
        style = STYLES[branch]
        hollow = branch == "no_word_alignment" and not bool(row["speech_present"])
        draw_marker(
            pdraw,
            x,
            y,
            shape=style["shape"],
            color=style["color"],
            light=style["light"],
            hollow=hollow,
            scale=MARKER_SCALE,
        )
    base = Image.alpha_composite(base, points)

    # Y labels at 45° to save horizontal space
    for i, branch in enumerate(reversed(BRANCH_ORDER)):
        yy = _y_map(float(i), plot_y0, plot_y1, vmin, vmax)
        label = f"{BRANCH_LABELS[branch]} (n={int(counts[branch])})"
        _paste_text(
            base,
            label,
            fonts["tick"],
            (50, 55, 62),
            xy=(plot_x0 - 14, yy),
            angle=45,
            anchor="rm",
        )

    draw = ImageDraw.Draw(base)
    for s in np.linspace(0.0, 1.0, 6):
        xx = _x_map(float(s), plot_x0, plot_x1)
        draw.line([(xx, plot_y1), (xx, plot_y1 + 10)], fill=(80, 85, 92), width=2)
        draw.text((xx, plot_y1 + 14), f"{s:.1f}", font=fonts["num"], fill=(60, 65, 72), anchor="mt")

    draw.text(
        ((plot_x0 + plot_x1) // 2, plot_y1 + 56),
        "文本—语音一致性分数",
        font=fonts["axis"],
        fill=(45, 50, 58),
        anchor="mt",
    )

    # Legend card below axis title (aligned to shortened plot width)
    legend_items = [
        ("sphere", STYLES["full_mfa"], False, "完整 MFA"),
        ("cone", STYLES["full_mfa_windowed"], False, "窗口化 MFA"),
        ("diamond", STYLES["partial_asr_timestamp_fallback"], False, "高分但降级"),
        ("cube", STYLES["partial_asr_timestamp"], False, "直接局部对齐"),
        ("sphere", STYLES["no_word_alignment"], False, "无词级对齐（有人声）"),
        ("sphere", STYLES["no_word_alignment"], True, "无词级对齐（无人声）"),
    ]
    card_h = int(108 * DPI)
    card_y = H - int(18 * DPI) - card_h
    draw.rounded_rectangle(
        [plot_x0, card_y, plot_x1, card_y + card_h],
        radius=12,
        fill=(255, 255, 255, 245),
        outline=(216, 221, 227),
        width=2,
    )
    for idx, (shape, style, hollow, label) in enumerate(legend_items):
        row = idx // 3
        col = idx % 3
        lx = plot_x0 + 36 + col * ((plot_x1 - plot_x0) // 3)
        ly = card_y + 36 + row * 48
        draw_marker(
            draw,
            lx,
            ly,
            shape=shape,
            color=style["color"],
            light=style["light"],
            hollow=hollow,
            scale=LEGEND_MARKER_SCALE,
        )
        draw.text((lx + 30, ly), label, font=fonts["legend"], fill=(55, 60, 68), anchor="lm")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    base.convert("RGB").save(png_path, format="PNG", optimize=True)
    svg_path = output_stem.with_suffix(".svg")
    _write_svg(plotted, counts, svg_path)
    return [svg_path, png_path]


def _write_svg(plotted: pd.DataFrame, counts: pd.Series, path: Path) -> None:
    """Simple SVG export mirroring the PNG layout proportions."""
    pw, ph = 980, 600
    ml, mr, mt, mb = 175, 220, 48, 105
    x0, x1, y0, y1 = ml, pw - mr, mt, ph - mb
    vmin, vmax = -0.55, 4.45

    def xm(s: float) -> float:
        return x0 + float(s) * (x1 - x0)

    def ym(v: float) -> float:
        return y1 - (v - vmin) / (vmax - vmin) * (y1 - y0)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{pw}" height="{ph}" viewBox="0 0 {pw} {ph}">',
        "<style>text{font-family:'Microsoft YaHei','Segoe UI',sans-serif;font-size:15px;fill:#3A3F46}"
        ".num{font-family:Georgia,Cambria,serif;font-size:14px}"
        ".tau{font-family:Georgia,Cambria,serif;font-size:16px}</style>",
        f'<rect width="100%" height="100%" fill="#FFFFFF"/>',
        f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="#F7F8FA" stroke="#8A919A" stroke-width="1.5" rx="6"/>',
    ]
    for i in range(5):
        yy = ym(i)
        parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="#D5DAE0"/>')
        branch = list(reversed(BRANCH_ORDER))[i]
        parts.append(
            f'<text x="{x0-10}" y="{yy:.1f}" text-anchor="end" dominant-baseline="middle">'
            f"{BRANCH_LABELS[branch]} (n={int(counts[branch])})</text>"
        )
    for thr, color, lab in [(0.35, "#C47A3A", "τ_L=0.35"), (0.75, "#3F7CA8", "τ_H=0.75")]:
        xx = xm(thr)
        parts.append(
            f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y1}" stroke="{color}" '
            f'stroke-width="1.4" stroke-dasharray="5 3"/>'
        )
        parts.append(f'<text class="tau" x="{xx:.1f}" y="{y0-6}" text-anchor="middle" fill="{color}">{lab}</text>')

    shape_tag = {
        "sphere": "circle",
        "cone": "polygon",
        "diamond": "polygon",
        "cube": "rect",
    }
    for _, row in plotted.iterrows():
        x = xm(float(row["consistency_score"]))
        y = ym(float(row["y"]))
        branch = row["final_branch"]
        style = STYLES[branch]
        color = _hex(style["color"])
        hollow = branch == "no_word_alignment" and not bool(row["speech_present"])
        if hollow:
            parts.append(
                f'<g stroke="{color}" fill="none" stroke-width="1.8">'
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6"/>'
                f'<line x1="{x-4:.1f}" y1="{y-4:.1f}" x2="{x+4:.1f}" y2="{y+4:.1f}"/>'
                f'<line x1="{x-4:.1f}" y1="{y+4:.1f}" x2="{x+4:.1f}" y2="{y-4:.1f}"/></g>'
            )
        elif style["shape"] == "sphere":
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{color}" stroke="#2A2E33" stroke-width="0.7"/>'
            )
        elif style["shape"] == "cone":
            parts.append(
                f'<polygon points="{x:.1f},{y-7:.1f} {x+7:.1f},{y+6:.1f} {x-7:.1f},{y+6:.1f}" '
                f'fill="{color}" stroke="#2A2E33" stroke-width="0.7"/>'
            )
        elif style["shape"] == "diamond":
            parts.append(
                f'<polygon points="{x:.1f},{y-7:.1f} {x+7:.1f},{y:.1f} {x:.1f},{y+7:.1f} {x-7:.1f},{y:.1f}" '
                f'fill="{color}" stroke="#2A2E33" stroke-width="0.7"/>'
            )
        else:
            parts.append(
                f'<rect x="{x-5.5:.1f}" y="{y-5.5:.1f}" width="11" height="11" rx="1.5" '
                f'fill="{color}" stroke="#2A2E33" stroke-width="0.7"/>'
            )

    for s in np.linspace(0, 1, 6):
        xx = xm(float(s))
        parts.append(f'<line x1="{xx:.1f}" y1="{y1}" x2="{xx:.1f}" y2="{y1+5}" stroke="#555"/>')
        parts.append(f'<text class="num" x="{xx:.1f}" y="{y1+18}" text-anchor="middle">{s:.1f}</text>')
    parts.append(
        f'<text x="{(x0+x1)/2:.1f}" y="{y1+40}" text-anchor="middle" font-size="15">文本—语音一致性分数</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = load_and_validate(args.quality_audit, args.all_samples)
    counts = data["final_branch"].value_counts()
    no_speech = int((~data["speech_present"]).sum())
    high_score_fallback = int(
        (
            (data["consistency_score"] >= TAU_HIGH)
            & (data["final_branch"] == "partial_asr_timestamp_fallback")
        ).sum()
    )
    print(f"[ok] sample_id 一对一关联: {len(data)} 条")
    print("[ok] 分支计数: " + ", ".join(f"{branch}={int(counts[branch])}" for branch in BRANCH_ORDER))
    print(
        f"[ok] S_i 范围: [{data['consistency_score'].min():.6f}, "
        f"{data['consistency_score'].max():.6f}]"
    )
    print(f"[ok] 高分降级={high_score_fallback}，无人声={no_speech}")
    for path in draw_figure(data, args.output_stem):
        print(f"[saved] {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
