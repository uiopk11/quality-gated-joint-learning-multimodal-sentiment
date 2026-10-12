"""Draw fig:q3_evidence_trace — Attachment-4 real Top evidence traceback.

Three real windows side by side:
  full     — sample 01 vision classification rank1
  partial  — sample 02 vision classification rank1
  none     — sample 15 text classification rank1

Layer order: WordPiece → 父自然词 → 可靠词时间 → 正式播放时间轴 → 视频帧
Side stats: 460/564, Top 181 / 139 / 34.
Outputs under ../Figure/.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

CODE_DIR = Path(__file__).resolve().parent
FIG_DIR = CODE_DIR.parent / "Figure"
STEM = "q3_evidence_trace"

Q3_ROOT = CODE_DIR.parent.parent
TIMED_CSV = Q3_ROOT / "results" / "tables" / "attachment4_top_local_evidence_timed.csv"
TIMELINE_CSV = Q3_ROOT / "results" / "qg_hta" / "token_timeline.csv"
KEYFRAME_DIR = Q3_ROOT / "results" / "qg_hta" / "keyframes"

BODY_VERIFIED, BODY_TOTAL = 460, 564
TOP_FULL, TOP_PARTIAL, TOP_NONE = 181, 139, 34


def setup_fonts() -> str:
    preferred = ["Microsoft YaHei", "SimHei", "SimSun"]
    available = {f.name: f.fname for f in fm.fontManager.ttflist}
    cn = next((n for n in preferred if n in available), None)
    if cn is None:
        raise RuntimeError("No Chinese-capable font found.")
    plt.rcParams.update(
        {
            "font.family": cn,
            "font.size": 10,
            "axes.unicode_minus": False,
            "mathtext.fontset": "cm",
            "mathtext.default": "it",
        }
    )
    return cn


def clean_word(s: str) -> str:
    if not s:
        return ""
    return (
        s.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("â€™", "'")
        .replace("��", "'")
    )


def load_rows():
    timed = list(csv.DictReader(TIMED_CSV.open(encoding="utf-8-sig")))
    timeline = list(csv.DictReader(TIMELINE_CSV.open(encoding="utf-8-sig")))
    return timed, timeline


def pick(timed, sample_id, modality, target, rank):
    for r in timed:
        if (
            r["sample_id"] == sample_id
            and r["modality"] == modality
            and r["target"] == target
            and r["rank"] == str(rank)
        ):
            return r
    raise KeyError((sample_id, modality, target, rank))


def tokens_for_window(timeline, sample_id, body_start, body_end):
    rows = [
        r
        for r in timeline
        if r["sample_id"] == sample_id and body_start <= int(r["body_position"]) < body_end
    ]
    rows.sort(key=lambda r: int(r["body_position"]))
    return rows


def rounded(ax, x, y, w, h, *, fc, ec, lw=1.2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            transform=ax.transAxes,
            clip_on=False,
            zorder=1,
        )
    )


def fmt_wp(s: str) -> str:
    s = clean_word(s)
    if s in {".", ",", "!", "?", ";", ":"}:
        return f"[{s}]"
    if len(s) > 9:
        return s[:8] + "…"
    return s


def draw_token_chips(ax, tokens, x0, y0, w, h, *, mode: str):
    """mode: full | partial | none

    none: keep text visible (muted), do NOT stamp 留空 on tokens.
    partial: highlight only verified tokens; unverified dashed.
    full: all verified green.
    """
    n = max(len(tokens), 1)
    cw = w / n
    for i, t in enumerate(tokens):
        reliable = t["timestamp_status"] == "verified_word_interval"
        if mode == "none":
            fc, ec, tc, ls = "#F3F4F6", "#9CA3AF", "#4B5563", "-"
        elif reliable:
            fc, ec, tc, ls = "#DCFCE7", "#15803D", "#14532D", "-"
        else:
            fc, ec, tc, ls = "#FEE2E2", "#DC2626", "#991B1B", (0, (2, 1.2))
        x = x0 + i * cw
        ax.add_patch(
            Rectangle(
                (x + 0.003, y0 + 0.06),
                cw - 0.006,
                h - 0.12,
                facecolor=fc,
                edgecolor=ec,
                linewidth=1.15,
                linestyle=ls,
                transform=ax.transAxes,
                clip_on=False,
                zorder=3,
            )
        )
        ax.text(
            x + cw / 2,
            y0 + h / 2,
            fmt_wp(t["wordpiece"]),
            ha="center",
            va="center",
            fontsize=8.4,
            color=tc,
            fontweight="bold" if (mode == "partial" and not reliable) else "normal",
            transform=ax.transAxes,
            zorder=4,
        )
        if mode == "partial" and not reliable:
            ax.text(
                x + cw / 2,
                y0 + 0.10,
                "时间留空",
                ha="center",
                va="center",
                fontsize=6.2,
                color="#DC2626",
                transform=ax.transAxes,
                zorder=5,
            )


def draw_parent_row(ax, tokens, x0, y0, w, h, *, mode: str):
    groups = []
    for t in tokens:
        ow = clean_word(t["official_word"]) or "—"
        rel = t["timestamp_status"] == "verified_word_interval"
        if groups and groups[-1]["word"] == ow:
            groups[-1]["n"] += 1
            groups[-1]["reliable"] = groups[-1]["reliable"] and rel
        else:
            groups.append({"word": ow, "n": 1, "reliable": rel})
    total = sum(g["n"] for g in groups) or 1
    x = x0
    for g in groups:
        gw = w * g["n"] / total
        if mode == "none":
            fc, ec, tc, ls = "#EEF2FF", "#94A3B8", "#475569", "-"
        elif g["reliable"]:
            fc, ec, tc, ls = "#E0E7FF", "#4338CA", "#312E81", "-"
        else:
            fc, ec, tc, ls = "#F3F4F6", "#9CA3AF", "#6B7280", (0, (2, 1.2))
        ax.add_patch(
            Rectangle(
                (x + 0.003, y0 + 0.06),
                gw - 0.006,
                h - 0.12,
                facecolor=fc,
                edgecolor=ec,
                linewidth=1.1,
                linestyle=ls,
                transform=ax.transAxes,
                clip_on=False,
                zorder=3,
            )
        )
        label = g["word"]
        if label in {"—", ""}:
            label = "（无父词）"
        if len(label) > 12:
            label = label[:11] + "…"
        ax.text(
            x + gw / 2,
            y0 + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=8.4,
            color=tc,
            transform=ax.transAxes,
            zorder=4,
        )
        x += gw


def draw_timeline(ax, tokens, x0, y0, w, h, *, empty_av: bool):
    ax.add_patch(
        Rectangle(
            (x0, y0 + 0.30),
            w,
            0.16,
            facecolor="#E5E7EB",
            edgecolor="#9CA3AF",
            linewidth=0.85,
            transform=ax.transAxes,
            clip_on=False,
            zorder=2,
        )
    )
    reliable = [
        t
        for t in tokens
        if t["timestamp_status"] == "verified_word_interval" and t["time_start_seconds"]
    ]
    if empty_av or not reliable:
        ax.text(
            x0 + w / 2,
            y0 + h * 0.55,
            "音频秒数：留空（不伪映射）",
            ha="center",
            va="center",
            fontsize=10.0,
            color="#9CA3AF",
            transform=ax.transAxes,
            zorder=4,
        )
        return

    a_min = min(float(t["time_start_seconds"]) for t in reliable)
    b_max = max(float(t["time_end_seconds"]) for t in reliable)
    pad = max(0.05, 0.08 * (b_max - a_min))
    lo, hi = a_min - pad, b_max + pad
    span = hi - lo
    for t in reliable:
        a = float(t["time_start_seconds"])
        b = float(t["time_end_seconds"])
        xa = x0 + (a - lo) / span * w
        xb = x0 + (b - lo) / span * w
        ax.add_patch(
            Rectangle(
                (xa, y0 + 0.20),
                max(xb - xa, 0.012),
                0.36,
                facecolor="#93C5FD",
                edgecolor="#1D4ED8",
                linewidth=0.95,
                transform=ax.transAxes,
                clip_on=False,
                zorder=3,
            )
        )
    ax.text(x0, y0 + 0.02, f"{a_min:.2f}s", ha="left", va="bottom", fontsize=8.0, color="#1D4ED8", fontweight="bold", transform=ax.transAxes)
    ax.text(x0 + w, y0 + 0.02, f"{b_max:.2f}s", ha="right", va="bottom", fontsize=8.0, color="#1D4ED8", fontweight="bold", transform=ax.transAxes)
    ax.text(
        x0 + w / 2,
        y0 + 0.02,
        f"QG-HTA 核验播放 [{a_min:.2f}, {b_max:.2f}] s",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color="#4B5563",
        transform=ax.transAxes,
    )


def draw_column(ax, *, title, ec, fc, row, tokens, keyframe_path, mode: str):
    empty_av = mode == "none"
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rounded(ax, 0.025, 0.015, 0.95, 0.97, fc=fc, ec=ec, lw=1.4)
    ax.text(0.50, 0.955, title, ha="center", va="top", fontsize=13.0, fontweight="bold", color=ec, transform=ax.transAxes)

    meta = (
        f"样本 {row['sample_id']} · {row['modality']} · {row['target']} · rank {row['rank']}\n"
        f"正文位置 [{row['body_start']}, {row['body_end_exclusive']}) · "
        f"覆盖率 {float(row['qg_hta_token_coverage']):.0%}"
    )
    ax.text(0.50, 0.888, meta, ha="center", va="top", fontsize=8.5, color="#374151", transform=ax.transAxes)

    # Layer labels
    for y, lab in [
        (0.795, "① WordPiece"),
        (0.675, "② 父自然词"),
        (0.555, "③ 可靠词时间"),
        (0.420, "④ 正式播放时间轴"),
        (0.075, "⑤ 实际视频帧"),
    ]:
        ax.text(0.055, y + 0.048, lab, ha="left", va="center", fontsize=9.2, fontweight="bold", color="#1F2937", transform=ax.transAxes)

    draw_token_chips(ax, tokens, 0.055, 0.755, 0.89, 0.095, mode=mode)
    draw_parent_row(ax, tokens, 0.055, 0.635, 0.89, 0.095, mode=mode)

    # Layer ③ time note
    if empty_av:
        ax.text(
            0.50,
            0.585,
            "无可靠区间 → 时间戳留空（不按 50 槽位均分伪映射）",
            ha="center",
            va="center",
            fontsize=9.2,
            color="#9CA3AF",
            transform=ax.transAxes,
        )
    else:
        reliable = [t for t in tokens if t["timestamp_status"] == "verified_word_interval" and t["time_start_seconds"]]
        if reliable:
            t0 = min(float(t["time_start_seconds"]) for t in reliable)
            t1 = max(float(t["time_end_seconds"]) for t in reliable)
            note = f"核验区间 [{t0:.2f}, {t1:.2f}] s"
            bad = [t for t in tokens if t["timestamp_status"] != "verified_word_interval"]
            if bad:
                ups = "、".join(fmt_wp(t["wordpiece"]) for t in bad)
                note += f"；未核验 {ups}（仅这些词元时间留空）"
            ax.text(0.50, 0.585, note, ha="center", va="center", fontsize=8.3, color="#1E3A8A", transform=ax.transAxes)
        else:
            ax.text(0.50, 0.585, "时间戳留空", ha="center", va="center", fontsize=9.2, color="#9CA3AF", transform=ax.transAxes)

    draw_timeline(ax, tokens, 0.055, 0.430, 0.89, 0.12, empty_av=empty_av)

    # retained explanation (always keep)
    ax.text(
        0.50,
        0.365,
        f"局部解释保留：$\\Delta m$={float(row['classification_margin_delta']):+.3f}"
        f"  ·  片段「{clean_word(row['original_text_fragment'])}」",
        ha="center",
        va="center",
        fontsize=8.6,
        color="#374151",
        transform=ax.transAxes,
    )

    # Layer ⑤ frame
    fb = (0.16, 0.04, 0.68, 0.28)
    ax.add_patch(
        Rectangle(
            (fb[0], fb[1]),
            fb[2],
            fb[3],
            facecolor="#111827",
            edgecolor=ec,
            linewidth=1.25,
            transform=ax.transAxes,
            clip_on=False,
            zorder=2,
        )
    )
    if empty_av or not keyframe_path or not Path(keyframe_path).is_file():
        ax.text(
            fb[0] + fb[2] / 2,
            fb[1] + fb[3] / 2,
            "视频帧：留空",
            ha="center",
            va="center",
            fontsize=13,
            color="#D1D5DB",
            transform=ax.transAxes,
            zorder=5,
        )
    else:
        img = np.asarray(Image.open(keyframe_path).convert("RGB"))
        inset = ax.inset_axes([fb[0] + 0.012, fb[1] + 0.025, fb[2] - 0.024, fb[3] - 0.05])
        inset.imshow(img)
        inset.axis("off")
        ax.text(
            fb[0] + fb[2] / 2,
            fb[1] - 0.012,
            f"key frame = {row.get('video_key_frame','')}  ·  "
            f"frames [{row.get('video_frame_start','')}, {row.get('video_frame_end_exclusive','')})",
            ha="center",
            va="top",
            fontsize=7.8,
            color="#374151",
            transform=ax.transAxes,
        )


def draw_side_stats(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rounded(ax, 0.04, 0.04, 0.92, 0.92, fc="#F8FAFC", ec="#64748B", lw=1.2)
    ax.text(0.50, 0.93, "附件4实际统计", ha="center", va="center", fontsize=12.0, fontweight="bold", color="#334155", transform=ax.transAxes)

    ax.text(0.50, 0.84, "正文词元时间核验", ha="center", va="center", fontsize=9.6, color="#374151", transform=ax.transAxes)
    ax.add_patch(Rectangle((0.12, 0.74), 0.76, 0.055, facecolor="#E5E7EB", transform=ax.transAxes, clip_on=False))
    ax.add_patch(
        Rectangle(
            (0.12, 0.74),
            0.76 * BODY_VERIFIED / BODY_TOTAL,
            0.055,
            facecolor="#34D399",
            transform=ax.transAxes,
            clip_on=False,
        )
    )
    ax.text(0.50, 0.695, f"{BODY_VERIFIED} / {BODY_TOTAL}", ha="center", va="center", fontsize=13, fontweight="bold", color="#065F46", transform=ax.transAxes)

    ax.text(0.50, 0.60, "Top 证据窗口覆盖", ha="center", va="center", fontsize=9.6, color="#374151", transform=ax.transAxes)
    items = [
        ("完整覆盖", TOP_FULL, "#15803D", "#DCFCE7"),
        ("部分覆盖", TOP_PARTIAL, "#B45309", "#FEF3C7"),
        ("无可靠区间", TOP_NONE, "#6B7280", "#F3F4F6"),
    ]
    y = 0.50
    for name, n, ec, fc in items:
        ax.add_patch(
            FancyBboxPatch(
                (0.12, y - 0.015),
                0.76,
                0.085,
                boxstyle="round,pad=0.004,rounding_size=0.01",
                facecolor=fc,
                edgecolor=ec,
                linewidth=1.1,
                transform=ax.transAxes,
                clip_on=False,
            )
        )
        ax.text(0.18, y + 0.028, name, ha="left", va="center", fontsize=10.2, color=ec, transform=ax.transAxes)
        ax.text(0.82, y + 0.028, str(n), ha="right", va="center", fontsize=12.5, fontweight="bold", color=ec, transform=ax.transAxes)
        y -= 0.115

    total = TOP_FULL + TOP_PARTIAL + TOP_NONE
    ax.text(
        0.50,
        0.12,
        f"合计 {total} 条有效 Top 证据\n不均分 50 槽位伪造时间轴\n不虚构关键帧\n时间=QG-HTA 重建",
        ha="center",
        va="center",
        fontsize=8.4,
        color="#6B7280",
        transform=ax.transAxes,
        linespacing=1.35,
    )


def draw():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_fonts()
    timed, timeline = load_rows()

    cases = [
        dict(
            title="完整覆盖",
            ec="#15803D",
            fc="#F0FDF4",
            row=pick(timed, "01", "vision", "classification", 1),
            mode="full",
            keyframe=KEYFRAME_DIR / "01_classification_rank1.jpg",
        ),
        dict(
            title="部分覆盖",
            ec="#B45309",
            fc="#FFFBEB",
            row=pick(timed, "02", "vision", "classification", 1),
            mode="partial",
            keyframe=KEYFRAME_DIR / "02_classification_rank1.jpg",
        ),
        dict(
            title="无可靠区间",
            ec="#6B7280",
            fc="#F9FAFB",
            row=pick(timed, "15", "text", "classification", 1),
            mode="none",
            keyframe=None,
        ),
    ]

    fig = plt.figure(figsize=(16.4, 8.8), dpi=220)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.42], wspace=0.075, left=0.015, right=0.988, top=0.90, bottom=0.06)

    fig.suptitle(
        "局部证据的音视频回溯（附件4真实 Top 窗口 · QG-HTA 质量门控）",
        fontsize=14.8,
        fontweight="bold",
        color="#1F2937",
        y=0.97,
    )

    for i, case in enumerate(cases):
        ax = fig.add_subplot(gs[0, i])
        row = case["row"]
        tokens = tokens_for_window(timeline, row["sample_id"], int(row["body_start"]), int(row["body_end_exclusive"]))
        draw_column(
            ax,
            title=case["title"],
            ec=case["ec"],
            fc=case["fc"],
            row=row,
            tokens=tokens,
            keyframe_path=str(case["keyframe"]) if case["keyframe"] else "",
            mode=case["mode"],
        )

    draw_side_stats(fig.add_subplot(gs[0, 3]))

    fig.text(
        0.50,
        0.018,
        "部分覆盖仅高亮有可靠时间的词元；无可靠区间保留文本/特征位置与解释值，音频秒数及视频帧明确留空。"
        " 时间均为 QG-HTA 重建，非附件4官方逐行时间戳。",
        ha="center",
        va="center",
        fontsize=9.0,
        color="#6B7280",
    )

    for ext in ("pdf", "png", "svg"):
        out = FIG_DIR / f"{STEM}.{ext}"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.06, facecolor="white")
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    draw()
