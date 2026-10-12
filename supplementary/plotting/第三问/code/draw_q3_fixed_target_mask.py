"""Draw fig:q3_fixed_target_mask — fixed explanation target + input masking.

Rule schematic only: no invented logits, intensities, or contribution numbers.
Outputs under ../Figure/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

CODE_DIR = Path(__file__).resolve().parent
FIG_DIR = CODE_DIR.parent / "Figure"
STEM = "q3_fixed_target_mask"


def setup_fonts() -> str:
    preferred = ["Microsoft YaHei", "SimHei", "SimSun"]
    available = {f.name: f.fname for f in fm.fontManager.ttflist}
    cn = next((n for n in preferred if n in available), None)
    if cn is None:
        raise RuntimeError("No Chinese-capable font found.")
    plt.rcParams.update(
        {
            "font.family": cn,
            "font.size": 11,
            "axes.unicode_minus": False,
            "mathtext.fontset": "cm",
            "mathtext.default": "it",
        }
    )
    return cn


def rounded(ax, xy, w, h, *, fc, ec="#333333", lw=1.2, r=0.012, z=2):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        mutation_aspect=0.5,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, p0, p1, *, color="#444444", lw=1.4, rad=0.0, z=3):
    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=z,
        shrinkA=1,
        shrinkB=1,
    )
    ax.add_patch(arr)
    return arr


def draw_rank_panel(ax, x0, y0, w, h, *, title, ranks, note, flip_note=None):
    """ranks: list of (label, role) role in {'target','competitor','other'}."""
    rounded(ax, (x0, y0), w, h, fc="#FAFBFC", ec="#6B7280", lw=1.15)
    ax.text(x0 + w / 2, y0 + h - 0.028, title, ha="center", va="top", fontsize=11.5, fontweight="bold", color="#1F2937")
    row_h = 0.048
    top = y0 + h - 0.055
    for i, (lab, role) in enumerate(ranks):
        y = top - i * (row_h + 0.012)
        if role == "target":
            fc, ec, tc = "#FEF3C7", "#B45309", "#92400E"
            badge = r"$k_i^{\ast}$"
        elif role == "competitor":
            fc, ec, tc = "#E0E7FF", "#4338CA", "#3730A3"
            badge = "竞争类"
        else:
            fc, ec, tc = "#F3F4F6", "#9CA3AF", "#4B5563"
            badge = ""
        rounded(ax, (x0 + 0.025, y - row_h), w - 0.05, row_h, fc=fc, ec=ec, lw=1.05, r=0.01)
        ax.text(x0 + 0.045, y - row_h / 2, f"{i+1}.", ha="left", va="center", fontsize=11, color=tc)
        ax.text(x0 + 0.085, y - row_h / 2, lab, ha="left", va="center", fontsize=11.5, color=tc, fontweight="bold" if role == "target" else "normal")
        if badge:
            ax.text(x0 + w - 0.04, y - row_h / 2, badge, ha="right", va="center", fontsize=10.5, color=tc)
    ax.text(x0 + w / 2, y0 + 0.055, note, ha="center", va="center", fontsize=9.8, color="#374151", wrap=True)
    if flip_note:
        ax.text(x0 + w / 2, y0 + 0.022, flip_note, ha="center", va="center", fontsize=9.2, color="#B45309")


def draw():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_fonts()

    fig, ax = plt.subplots(figsize=(13.6, 7.4), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Title banner
    rounded(ax, (0.03, 0.915), 0.94, 0.062, fc="#FFF8E6", ec="#B8860B", lw=1.25, r=0.01)
    ax.text(
        0.50,
        0.946,
        r"固定解释目标与输入遮蔽（冻结 M1 上的解释扰动，非第五章双视图缺失构造）",
        ha="center",
        va="center",
        fontsize=13.0,
        fontweight="bold",
        color="#5C4300",
    )

    # Left / right panel frames
    rounded(ax, (0.025, 0.075), 0.455, 0.820, fc="#FFFFFF", ec="#9CA3AF", lw=1.1, r=0.012)
    rounded(ax, (0.520, 0.075), 0.455, 0.820, fc="#FFFFFF", ec="#9CA3AF", lw=1.1, r=0.012)

    ax.text(0.252, 0.860, "左侧：原输入确定目标；遮蔽后仍追踪原目标", ha="center", va="center", fontsize=12.2, fontweight="bold", color="#1F2937")
    ax.text(0.747, 0.860, "右侧：三模态同位置轴上的替换方式", ha="center", va="center", fontsize=12.2, fontweight="bold", color="#1F2937")

    # ---- Left: original vs masked ranking (schematic, no numbers) ----
    draw_rank_panel(
        ax,
        0.045,
        0.520,
        0.195,
        0.300,
        title="原输入 · 类别排序（示意）",
        ranks=[
            ("中性 Neutral", "target"),
            ("正向 Positive", "competitor"),
            ("负向 Negative", "other"),
        ],
        note=r"最大 logit 类别记为 $k_i^{\ast}$",
    )
    draw_rank_panel(
        ax,
        0.265,
        0.520,
        0.195,
        0.300,
        title="遮蔽输入 · 类别排序（示意）",
        ranks=[
            ("正向 Positive", "competitor"),
            ("中性 Neutral", "target"),
            ("负向 Negative", "other"),
        ],
        note=r"最大 logit 可改属其他类",
        flip_note=r"仍固定读取 $k_i^{\ast}$ 的边际",
    )
    arrow(ax, (0.245, 0.670), (0.260, 0.670), color="#6B7280", lw=1.5)

    # Margin / intensity readout boxes
    rounded(ax, (0.045, 0.300), 0.415, 0.185, fc="#F8FAFC", ec="#64748B", lw=1.1, r=0.01)
    ax.text(0.252, 0.455, "遮蔽后读出（相对原目标，无数值示意）", ha="center", va="center", fontsize=11.2, fontweight="bold", color="#334155")

    rounded(ax, (0.060, 0.330), 0.185, 0.095, fc="#FEF3C7", ec="#B45309", lw=1.15, r=0.01)
    ax.text(0.152, 0.395, "分类", ha="center", va="center", fontsize=10.5, color="#92400E", fontweight="bold")
    ax.text(
        0.152,
        0.355,
        r"$m_i^{\ast}=$ logit($k_i^{\ast}$)" + "\n" + r"$-\;\max_{k\neq k_i^{\ast}}$ logit$(k)$",
        ha="center",
        va="center",
        fontsize=9.8,
        color="#78350F",
    )

    rounded(ax, (0.260, 0.330), 0.185, 0.095, fc="#ECFDF5", ec="#059669", lw=1.15, r=0.01)
    ax.text(0.352, 0.395, "回归", ha="center", va="center", fontsize=10.5, color="#065F46", fontweight="bold")
    ax.text(
        0.352,
        0.355,
        r"目标强度取原输入" + "\n" + r"预测值 $s_i^{\ast}$（不改写）",
        ha="center",
        va="center",
        fontsize=9.8,
        color="#064E3B",
    )

    ax.text(
        0.252,
        0.250,
        "即使遮蔽后类别翻转，分类解释仍追踪原类别 "
        + r"$k_i^{\ast}$"
        + " 相对当前最强竞争类的边际；\n"
        "不使用附件4情感标签，不更新 M1 参数。",
        ha="center",
        va="center",
        fontsize=9.6,
        color="#4B5563",
    )

    # ---- Right: three-modality replacement ----
    ax.text(0.747, 0.800, "相同位置轴 · 长度 / 特殊词元 / 注意力掩码不变", ha="center", va="center", fontsize=11.0, color="#374151")

    # position axis
    x_ax, y_ax, w_ax = 0.560, 0.720, 0.375
    ax.plot([x_ax, x_ax + w_ax], [y_ax, y_ax], color="#6B7280", lw=1.2, zorder=2)
    for i, lab in enumerate(["0", "…", r"$L_i$", "…", "49"]):
        xx = x_ax + (i + 0.5) * (w_ax / 5)
        ax.plot([xx, xx], [y_ax - 0.008, y_ax + 0.008], color="#6B7280", lw=1.0)
        ax.text(xx, y_ax + 0.022, lab, ha="center", va="bottom", fontsize=9.5, color="#4B5563")
    ax.text(x_ax + w_ax / 2, y_ax + 0.048, "模型可见位置（示意槽位）", ha="center", va="bottom", fontsize=9.5, color="#6B7280")

    # Before / After comparison for one body span
    cells_before_t = [
        {"kind": "cls"},
        {"kind": "body", "text": r"$w_1$"},
        {"kind": "body", "text": r"$w_2$"},
        {"kind": "body", "text": r"$w_3$"},
        {"kind": "sep"},
        {"kind": "pad"},
    ]
    cells_after_t = [
        {"kind": "cls"},
        {"kind": "unk"},
        {"kind": "unk"},
        {"kind": "unk"},
        {"kind": "sep"},
        {"kind": "pad"},
    ]
    cells_before_a = [
        {"kind": "cls"},
        {"kind": "body", "text": "a"},
        {"kind": "body", "text": "a"},
        {"kind": "body", "text": "a"},
        {"kind": "sep"},
        {"kind": "pad"},
    ]
    cells_after_a = [
        {"kind": "cls"},
        {"kind": "zero"},
        {"kind": "zero"},
        {"kind": "zero"},
        {"kind": "sep"},
        {"kind": "pad"},
    ]
    cells_before_v = [
        {"kind": "cls"},
        {"kind": "body", "text": "v"},
        {"kind": "body", "text": "v"},
        {"kind": "body", "text": "v"},
        {"kind": "sep"},
        {"kind": "pad"},
    ]
    cells_after_v = [
        {"kind": "cls"},
        {"kind": "zero"},
        {"kind": "zero"},
        {"kind": "zero"},
        {"kind": "sep"},
        {"kind": "pad"},
    ]

    # two columns: 遮蔽前 / 遮蔽后
    ax.text(0.655, 0.685, "遮蔽前", ha="center", va="center", fontsize=11.0, fontweight="bold", color="#065F46")
    ax.text(0.860, 0.685, "遮蔽后", ha="center", va="center", fontsize=11.0, fontweight="bold", color="#991B1B")

    row_specs = [
        (0.600, "文本", "#1D4ED8", "#DBEAFE", cells_before_t, cells_after_t, "正文→[UNK]"),
        (0.500, "语音", "#047857", "#D1FAE5", cells_before_a, cells_after_a, "特征行置零"),
        (0.400, "视觉", "#C2410C", "#FFEDD5", cells_before_v, cells_after_v, "特征行置零"),
    ]
    rw, rh = 0.175, 0.070
    for y, name, ec, fc, before, after, tip in row_specs:
        ax.text(0.545, y + rh / 2, name, ha="center", va="center", fontsize=11.5, fontweight="bold", color=ec)
        # before
        n = len(before)
        cw = rw / n
        for i, c in enumerate(before):
            draw_cell(ax, 0.655 - rw / 2 + i * cw, y, cw, rh, c)
        # after
        for i, c in enumerate(after):
            draw_cell(ax, 0.860 - rw / 2 + i * cw, y, cw, rh, c)
        arrow(ax, (0.655 + rw / 2 + 0.008, y + rh / 2), (0.860 - rw / 2 - 0.008, y + rh / 2), color=ec, lw=1.35)
        ax.text(0.758, y + rh + 0.012, tip, ha="center", va="bottom", fontsize=9.0, color=ec)

    # invariant box
    rounded(ax, (0.545, 0.200), 0.405, 0.145, fc="#EFF6FF", ec="#2563EB", lw=1.15, r=0.01)
    ax.text(0.747, 0.310, "保持不变", ha="center", va="center", fontsize=11.2, fontweight="bold", color="#1E40AF")
    ax.text(
        0.747,
        0.250,
        "三模态位置对齐 · 正文长度 $L_i$ · [CLS]/[SEP]/[PAD]\n"
        "注意力掩码 attention_mask（仅屏蔽 PAD，不删时间步）",
        ha="center",
        va="center",
        fontsize=10.0,
        color="#1E3A8A",
    )

    ax.text(
        0.50,
        0.035,
        "规则示意图：不标注虚构 logit / 强度 / 贡献数值；度量的是冻结模型对 [UNK] 与零行基线的敏感性，不作现实世界因果解释。",
        ha="center",
        va="center",
        fontsize=9.2,
        color="#6B7280",
    )

    for ext in ("pdf", "png", "svg"):
        out = FIG_DIR / f"{STEM}.{ext}"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.08, facecolor="white")
        print(f"wrote {out}")
    plt.close(fig)


def draw_cell(ax, x, y, w, h, c):
    kind = c["kind"]
    if kind == "cls":
        face, edge, txt, tc = "#DBEAFE", "#1D4ED8", "[CLS]", "#1E3A8A"
    elif kind == "sep":
        face, edge, txt, tc = "#DBEAFE", "#1D4ED8", "[SEP]", "#1E3A8A"
    elif kind == "pad":
        face, edge, txt, tc = "#F3F4F6", "#D1D5DB", "[PAD]", "#9CA3AF"
    elif kind == "unk":
        face, edge, txt, tc = "#FEE2E2", "#DC2626", "[UNK]", "#991B1B"
    elif kind == "zero":
        face, edge, txt, tc = "#FCE7F3", "#BE185D", "0", "#9D174D"
    else:
        face, edge, txt, tc = "#ECFDF5", "#059669", c.get("text", "tok"), "#065F46"
    rect = Rectangle((x + 0.0015, y + 0.004), w - 0.003, h - 0.008, facecolor=face, edgecolor=edge, linewidth=0.95, zorder=3)
    ax.add_patch(rect)
    fs = 7.8 if kind in {"cls", "sep", "pad", "unk"} else 10.0
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=tc, zorder=4)


if __name__ == "__main__":
    draw()
