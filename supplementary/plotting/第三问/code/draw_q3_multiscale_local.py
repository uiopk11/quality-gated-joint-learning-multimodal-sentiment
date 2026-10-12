"""Draw fig:q3_multiscale_local from Attachment-4 saved window results.

Uses real sample 01, text modality: window deltas, position aggregation, Top windows.
Outputs under ../Figure/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Rectangle

CODE_DIR = Path(__file__).resolve().parent
FIG_DIR = CODE_DIR.parent / "Figure"
STEM = "q3_multiscale_local"

Q3_ROOT = CODE_DIR.parent.parent
SAMPLE_JSON = Q3_ROOT / "results" / "attachment4" / "samples" / "01.json"
MODALITY = "text"
SAMPLE_ID = "01"


def setup_fonts() -> str:
    preferred = ["Microsoft YaHei", "SimHei", "SimSun"]
    available = {f.name: f.fname for f in fm.fontManager.ttflist}
    cn = next((n for n in preferred if n in available), None)
    if cn is None:
        raise RuntimeError("No Chinese-capable font found.")
    plt.rcParams.update(
        {
            "font.family": cn,
            "font.size": 10.5,
            "axes.unicode_minus": False,
            "mathtext.fontset": "cm",
            "mathtext.default": "it",
        }
    )
    return cn


def load_local():
    data = json.loads(SAMPLE_JSON.read_text(encoding="utf-8"))
    le = data["local_explanation"][MODALITY]
    sample = data["sample"]
    return sample, le


def window_matrix(windows, n_pos, size, field):
    """Build (n_win, n_pos) coverage with delta values on covered cells."""
    wins = [w for w in windows if w["window_size"] == size]
    mat = np.full((len(wins), n_pos), np.nan)
    for i, w in enumerate(wins):
        a, b = w["body_start"], w["body_end"]
        mat[i, a:b] = w[field]
    return wins, mat


def draw():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_fonts()
    sample, le = load_local()
    positions = le["positions"]
    windows = le["windows"]
    tops = le["top_classification_windows"]
    n_pos = len(positions)
    xs = np.arange(n_pos)
    wordpieces = [p["wordpiece"] for p in positions]
    cls_share = np.array([p["classification_share"] for p in positions])
    cls_signed = np.array([p["classification_signed"] for p in positions])
    int_share = np.array([p["intensity_share"] for p in positions])
    int_signed = np.array([p["intensity_signed"] for p in positions])

    fig = plt.figure(figsize=(13.8, 10.4), dpi=220)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        5,
        1,
        height_ratios=[0.48, 1.05, 1.05, 1.20, 1.45],
        hspace=0.36,
        left=0.08,
        right=0.97,
        top=0.945,
        bottom=0.17,
    )

    # Banner
    ax0 = fig.add_subplot(gs[0])
    ax0.axis("off")
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.add_patch(FancyBboxPatch((0.01, 0.15), 0.98, 0.70, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor="#FFF8E6", edgecolor="#B8860B", linewidth=1.2))
    ax0.text(
        0.50,
        0.55,
        f"附件4样本 {SAMPLE_ID} · 文本模态 · 真实逐窗口结果（横轴=模型可见正文位置，非视频秒数）",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#5C4300",
    )

    cmap = plt.get_cmap("RdBu_r")

    # ---- Panel A: multiscale window scans (classification margin delta) ----
    axA = fig.add_subplot(gs[1])
    axA.set_title("窗口扫描：宽度 1 / 3 / 5 的分类边际变化 $\\Delta m$（有符号）", fontsize=12, pad=6, loc="left")
    y_ticks = []
    y_labs = []
    y_cursor = 0
    for size in [1, 3, 5]:
        wins, mat = window_matrix(windows, n_pos, size, "classification_margin_delta")
        # collapse to one row per scale: for each position, mean of covering window deltas
        row = np.nanmean(mat, axis=0)
        # also draw individual windows as thin segments above the aggregate? Use heatmap row
        vmax = np.nanmax(np.abs([w["classification_margin_delta"] for w in windows]))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        for j in range(n_pos):
            if np.isnan(row[j]):
                continue
            axA.add_patch(
                Rectangle(
                    (j - 0.5, y_cursor - 0.4),
                    1.0,
                    0.8,
                    facecolor=cmap(norm(row[j])),
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=2,
                )
            )
        # overlay a few example windows as outlines for this scale
        # pick first, mid, last window
        picks = [wins[0], wins[len(wins) // 2], wins[-1]] if wins else []
        for w in picks:
            axA.add_patch(
                Rectangle(
                    (w["body_start"] - 0.5, y_cursor - 0.45),
                    w["body_end"] - w["body_start"],
                    0.9,
                    fill=False,
                    edgecolor="#111827",
                    linewidth=1.0,
                    linestyle="--",
                    zorder=3,
                )
            )
        y_ticks.append(y_cursor)
        y_labs.append(f"$w={size}$")
        y_cursor += 1.2

    axA.set_xlim(-0.6, n_pos - 0.4)
    axA.set_ylim(-0.8, y_cursor - 0.4)
    axA.set_yticks(y_ticks)
    axA.set_yticklabels(y_labs, fontsize=11)
    axA.set_xticks(xs)
    axA.set_xticklabels([])
    axA.tick_params(axis="x", length=0)
    for spine in ("top", "right"):
        axA.spines[spine].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axA, fraction=0.02, pad=0.01)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("$\\Delta m$", fontsize=9)

    # ---- Panel B: intensity window scan ----
    axB = fig.add_subplot(gs[2])
    axB.set_title("窗口扫描：宽度 1 / 3 / 5 的强度变化 $\\Delta s$（有符号）", fontsize=12, pad=6, loc="left")
    y_cursor = 0
    y_ticks = []
    vmax_s = np.nanmax(np.abs([w["intensity_delta"] for w in windows]))
    norm_s = TwoSlopeNorm(vmin=-vmax_s, vcenter=0.0, vmax=vmax_s)
    for size in [1, 3, 5]:
        wins, mat = window_matrix(windows, n_pos, size, "intensity_delta")
        row = np.nanmean(mat, axis=0)
        for j in range(n_pos):
            if np.isnan(row[j]):
                continue
            axB.add_patch(
                Rectangle(
                    (j - 0.5, y_cursor - 0.4),
                    1.0,
                    0.8,
                    facecolor=cmap(norm_s(row[j])),
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=2,
                )
            )
        y_ticks.append(y_cursor)
        y_cursor += 1.2
    axB.set_xlim(-0.6, n_pos - 0.4)
    axB.set_ylim(-0.8, y_cursor - 0.4)
    axB.set_yticks(y_ticks)
    axB.set_yticklabels([f"$w={s}$" for s in [1, 3, 5]], fontsize=11)
    axB.set_xticks(xs)
    axB.set_xticklabels([])
    axB.tick_params(axis="x", length=0)
    for spine in ("top", "right"):
        axB.spines[spine].set_visible(False)
    sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm_s)
    sm2.set_array([])
    cbar2 = fig.colorbar(sm2, ax=axB, fraction=0.02, pad=0.01)
    cbar2.ax.tick_params(labelsize=8)
    cbar2.set_label("$\\Delta s$", fontsize=9)

    # ---- Panel C: position aggregation ----
    axC = fig.add_subplot(gs[3])
    axC.set_title("位置汇聚：模态内归一化分布（实心=有符号；空心描边=绝对值份额）", fontsize=12, pad=6, loc="left")
    width = 0.38
    axC.bar(xs - width / 2, cls_signed, width=width, color="#2563EB", alpha=0.85, label="分类有符号")
    axC.bar(xs + width / 2, int_signed, width=width, color="#D97706", alpha=0.85, label="强度有符号")
    axC.plot(xs, cls_share, "o-", color="#1E3A8A", markersize=3.5, linewidth=1.0, label="分类绝对份额")
    axC.plot(xs, int_share, "s--", color="#9A3412", markersize=3.5, linewidth=1.0, label="强度绝对份额")
    axC.axhline(0, color="#9CA3AF", lw=0.8)
    axC.set_xlim(-0.6, n_pos - 0.4)
    axC.set_xticks(xs)
    axC.set_xticklabels([])
    axC.legend(loc="upper right", fontsize=8.5, frameon=False, ncol=2)
    for spine in ("top", "right"):
        axC.spines[spine].set_visible(False)
    axC.set_ylabel("汇聚值", fontsize=10)

    # ---- Panel D: Top non-overlapping windows ----
    axD = fig.add_subplot(gs[4])
    axD.set_title("不重叠 Top 证据窗口（分类，最多 3 个）", fontsize=12, pad=6, loc="left")

    # Token cells — shifted up with the Top strip
    cell_y0, cell_h = 0.18, 0.40
    for j, wp in enumerate(wordpieces):
        axD.add_patch(
            Rectangle(
                (j - 0.5, cell_y0),
                1.0,
                cell_h,
                facecolor="#F3F4F6",
                edgecolor="#E5E7EB",
                linewidth=0.6,
                zorder=1,
            )
        )
        show = wp if len(wp) <= 8 else wp[:7] + "…"
        axD.text(
            j,
            cell_y0 + cell_h / 2,
            show,
            ha="center",
            va="center",
            fontsize=8.5,
            color="#1F2937",
            zorder=2,
            clip_on=True,
        )

    # Colored spans + larger formula tightly above (shifted up)
    bar_y0, bar_h = 0.64, 0.18
    colors = ["#DC2626", "#2563EB", "#059669"]
    for rank, (w, col) in enumerate(zip(tops[:3], colors), start=1):
        a, b = int(w["body_start"]), int(w["body_end"])
        span = max(b - a, 1)
        x0 = a - 0.45
        bw = span - 0.10
        axD.add_patch(
            Rectangle(
                (x0, bar_y0),
                bw,
                bar_h,
                facecolor=col,
                alpha=0.28,
                edgecolor=col,
                linewidth=1.55,
                zorder=3,
            )
        )
        axD.text(
            x0 + bw / 2,
            bar_y0 + bar_h / 2,
            f"Top{rank}",
            ha="center",
            va="center",
            fontsize=8.5,
            color=col,
            fontweight="bold",
            zorder=4,
            clip_on=True,
        )
        frag = w["original_text_fragment"]
        if len(frag) > 26:
            frag = frag[:24] + "…"
        axD.text(
            (a + b) / 2,
            bar_y0 + bar_h + 0.085,
            f"w={w['window_size']}  $\\Delta m$={w['classification_margin_delta']:+.3f}  {frag}",
            ha="center",
            va="center",
            fontsize=10.8,
            color=col,
            fontweight="bold",
            zorder=5,
            clip_on=False,
        )

    axD.set_xlim(-0.6, n_pos - 0.4)
    axD.set_ylim(-0.12, 1.08)
    axD.set_yticks([])
    axD.set_xticks(xs)
    axD.set_xticklabels([str(i) for i in xs], fontsize=8)
    # Raise bottom axis to sit just under the token strip
    axD.spines["bottom"].set_position(("data", cell_y0 - 0.04))
    axD.tick_params(axis="x", pad=3)
    axD.set_xlabel("正文位置编号（body position）", fontsize=10.2, labelpad=8)
    for spine in ("top", "right", "left"):
        axD.spines[spine].set_visible(False)

    fig.text(
        0.525,
        0.022,
        f"样本正文长度 $L_i$={sample['body_length']}；聚合规则：覆盖窗口的均值（有符号或绝对值）再按窗宽归一。"
        " 窗口端点是模型输入正文位置，不是原视频秒数。",
        ha="center",
        va="center",
        fontsize=8.0,
        color="#6B7280",
    )

    out = FIG_DIR / f"{STEM}.png"
    fig.savefig(out, dpi=220, facecolor="white")
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    draw()
