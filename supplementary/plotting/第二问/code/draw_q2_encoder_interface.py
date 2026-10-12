"""Draw fig:q2_encoder_interface — larger CN + CM (LaTeX) math fonts.

Outputs to ../Figure/ only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent
FIG_DIR = ROOT_DIR / "Figure"
PNG_PATH = FIG_DIR / "q2_encoder_interface.png"
SVG_PATH = FIG_DIR / "q2_encoder_interface.svg"
PDF_PATH = FIG_DIR / "q2_encoder_interface.pdf"

# Visual scale
FS_TITLE = 15.5
FS_COL = 13.5
FS_BANNER = 12.0
FS_BOX_CN = 12.0
FS_BOX_MATH = 13.5
FS_BOX_MATH_LG = 15.0
FS_NOTE = 11.0
FS_LEGEND = 11.0
ARROW_LW = 1.55
ARROW_MS = 14


def setup_fonts() -> str:
    preferred = ["Microsoft YaHei", "SimHei", "SimSun"]
    available = {f.name: f.fname for f in fm.fontManager.ttflist}
    cn = None
    for name in preferred:
        if name in available:
            cn = name
            break
    if cn is None:
        raise RuntimeError("No Chinese-capable font found.")

    plt.rcParams.update(
        {
            "font.family": cn,
            "font.size": 12,
            "axes.unicode_minus": False,
            # Academic LaTeX look for all $...$ math
            "mathtext.fontset": "cm",
            "mathtext.default": "it",
        }
    )
    return cn


def rounded(ax, xy, w, h, *, fc, ec="#333333", lw=1.35, r=0.018, z=2):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        mutation_aspect=0.5,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, p0, p1, *, color="#444444", lw=ARROW_LW, rad=0.0, z=3):
    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=ARROW_MS,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=z,
        shrinkA=1,
        shrinkB=1,
    )
    ax.add_patch(arr)
    return arr


def cn(ax, x, y, s, *, size=FS_BOX_CN, weight="normal", color="#1a1a1a", ha="center", va="center", **kw):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=5, **kw)


def math(ax, x, y, s, *, size=FS_BOX_MATH, color="#1a1a1a", ha="center", va="center", **kw):
    # Always wrap as mathtext if caller forgot $
    if not (s.startswith("$") or s.startswith(r"$")):
        s = f"${s}$"
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, zorder=5, **kw)


def draw():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_fonts()

    # Wider canvas so larger type still breathes
    fig, ax = plt.subplots(figsize=(14.5, 8.2), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Columns
    x_in, x_enc, x_h, x_proj, x_out = 0.028, 0.165, 0.340, 0.555, 0.800
    w_in, w_enc, w_h, w_proj = 0.112, 0.130, 0.168, 0.175
    h = 0.195
    gap = 0.040
    y0 = 0.700

    mods = [
        dict(
            title="文本",
            subtitle="Text",
            fc="#E8F1FA",
            ec="#245A86",
            y=y0,
            inp_math=r"$\mathrm{text\_bert}$" + "\n" + r"$[3\times 50]$",
            enc_math=r"$\mathrm{BERT}$" + "\n" + r"$\theta_T$",
            enc_cn="微调",
            h_math=r"$\mathbf{H}_i^{T}$" + "\n" + r"$L_i\times d_T$" + "\n" + r"$d_T{=}768$",
            h_cn="局部：去 CLS 正文",
            u_math=r"$\mathbf{u}_i^{T}$" + "\n" + r"$\in\mathbb{R}^{768}$",
            u_cn="CLS 摘要",
            z_math=r"$\phi_T{+}\mathrm{PE}$" + "\n" + r"$\mathbf{Z}_i^{T,0}$" + "\n" + r"$L_i\times d$",
            g_math=r"$\phi_T$" + "\n" + r"$\mathbf{g}_i^{T,0}$" + "\n" + r"$\in\mathbb{R}^{d}$",
        ),
        dict(
            title="音频",
            subtitle="Audio",
            fc="#E9F6EE",
            ec="#276B42",
            y=y0 - (h + gap),
            inp_math=r"$\bar{\mathbf{X}}_i^{A}$" + "\n" + r"$50\times 74$",
            enc_math=r"$\mathrm{LSTM}_{A}$" + "\n" + r"$h{=}16$",
            enc_cn=r"按 $L_i$ pack",
            h_math=r"$\mathbf{H}_i^{A}$" + "\n" + r"$L_i\times d_A$" + "\n" + r"$d_A{=}16$",
            h_cn="逐位置隐状态",
            u_math=r"$\mathbf{u}_i^{A}$" + "\n" + r"$\in\mathbb{R}^{16}$",
            u_cn="最后有效状态",
            z_math=r"$\phi_A{+}\mathrm{PE}$" + "\n" + r"$\mathbf{Z}_i^{A,0}$" + "\n" + r"$L_i\times d$",
            g_math=r"$\phi_A$" + "\n" + r"$\mathbf{g}_i^{A,0}$" + "\n" + r"$\in\mathbb{R}^{d}$",
        ),
        dict(
            title="视觉",
            subtitle="Vision",
            fc="#F8EEE6",
            ec="#8A4A2C",
            y=y0 - 2 * (h + gap),
            inp_math=r"$\bar{\mathbf{X}}_i^{V}$" + "\n" + r"$50\times 35$",
            enc_math=r"$\mathrm{LSTM}_{V}$" + "\n" + r"$h{=}32$",
            enc_cn=r"按 $L_i$ pack",
            h_math=r"$\mathbf{H}_i^{V}$" + "\n" + r"$L_i\times d_V$" + "\n" + r"$d_V{=}32$",
            h_cn="逐位置隐状态",
            u_math=r"$\mathbf{u}_i^{V}$" + "\n" + r"$\in\mathbb{R}^{32}$",
            u_cn="最后有效状态",
            z_math=r"$\phi_V{+}\mathrm{PE}$" + "\n" + r"$\mathbf{Z}_i^{V,0}$" + "\n" + r"$L_i\times d$",
            g_math=r"$\phi_V$" + "\n" + r"$\mathbf{g}_i^{V,0}$" + "\n" + r"$\in\mathbb{R}^{d}$",
        ),
    ]

    # Column headers
    for x, lab in [
        (x_in + w_in / 2, "① 输入"),
        (x_enc + w_enc / 2, "② 模态编码"),
        (x_h + w_h / 2, "③ 局部序列 ＋ 整体摘要"),
        (x_proj + w_proj / 2, r"④ 投影 $\phi_m$（$d{=}128$）"),
        (x_out + 0.085, "⑤ 接口输出"),
    ]:
        cn(ax, x, 0.965, lab, size=FS_COL, weight="bold", color="#1F2A37")

    # Banner
    rounded(ax, (0.04, 0.880), 0.92, 0.060, fc="#FFF8E6", ec="#B8860B", lw=1.4, r=0.012)
    cn(
        ax,
        0.50,
        0.920,
        r"有效长度  $L_i=\sum_{t}\mathrm{attention}_{it}-2$"
        "（正文 WordPiece）：含内部零行/UNK 原位；不含 SEP/PAD。"
        "  补齐位置仅由 mask 屏蔽，不删除时间步。",
        size=FS_BANNER,
        color="#5C4300",
    )

    z_pts, g_pts, u_pts = [], [], []

    for m in mods:
        y, ec, fc = m["y"], m["ec"], m["fc"]
        yc = y + h / 2

        # ① input
        rounded(ax, (x_in, y), w_in, h, fc="#F3F4F6", ec="#6B7280", lw=1.3)
        cn(ax, x_in + w_in / 2, y + h * 0.82, f"{m['title']}  {m['subtitle']}", size=11.5, weight="bold", color=ec)
        ax.text(x_in + w_in / 2, y + h * 0.38, m["inp_math"], fontsize=FS_BOX_MATH, color="#222", ha="center", va="center", zorder=5)

        # ② encoder
        rounded(ax, (x_enc, y), w_enc, h, fc=fc, ec=ec, lw=1.55)
        ax.text(x_enc + w_enc / 2, y + h * 0.58, m["enc_math"], fontsize=FS_BOX_MATH_LG, color=ec, ha="center", va="center", fontweight="bold", zorder=5)
        ax.text(x_enc + w_enc / 2, y + h * 0.18, m["enc_cn"], fontsize=FS_BOX_CN, color=ec, ha="center", va="center", zorder=5)
        arrow(ax, (x_in + w_in + 0.004, yc), (x_enc - 0.004, yc), color=ec)

        # ③ split
        h_loc, h_sum = h * 0.58, h * 0.34
        y_loc, y_sum = y + h - h_loc, y
        rounded(ax, (x_h, y_loc), w_h, h_loc, fc="#FFFFFF", ec=ec, lw=1.4)
        ax.text(x_h + w_h / 2, y_loc + h_loc * 0.58, m["h_math"], fontsize=FS_BOX_MATH, color=ec, ha="center", va="center", zorder=5)
        cn(ax, x_h + w_h / 2, y_loc + h_loc * 0.14, m["h_cn"], size=10.5, color="#4B5563")

        rounded(ax, (x_h, y_sum), w_h, h_sum, fc="#F8FAFC", ec=ec, lw=1.25)
        ax.text(x_h + w_h / 2, y_sum + h_sum * 0.62, m["u_math"], fontsize=FS_BOX_MATH, color=ec, ha="center", va="center", zorder=5)
        cn(ax, x_h + w_h / 2, y_sum + h_sum * 0.18, m["u_cn"], size=10.5, color="#4B5563")

        arrow(ax, (x_enc + w_enc + 0.004, yc + 0.030), (x_h - 0.004, y_loc + h_loc * 0.55), color=ec, lw=1.35)
        arrow(ax, (x_enc + w_enc + 0.004, yc - 0.030), (x_h - 0.004, y_sum + h_sum * 0.5), color=ec, lw=1.35)

        # ④ projection
        rounded(ax, (x_proj, y_loc), w_proj, h_loc, fc=fc, ec=ec, lw=1.45)
        ax.text(x_proj + w_proj / 2, y_loc + h_loc * 0.50, m["z_math"], fontsize=FS_BOX_MATH, color=ec, ha="center", va="center", zorder=5)
        rounded(ax, (x_proj, y_sum), w_proj, h_sum, fc="#FFFFFF", ec=ec, lw=1.25)
        ax.text(x_proj + w_proj / 2, y_sum + h_sum * 0.50, m["g_math"], fontsize=FS_BOX_MATH, color=ec, ha="center", va="center", zorder=5)

        arrow(ax, (x_h + w_h + 0.004, y_loc + h_loc * 0.55), (x_proj - 0.004, y_loc + h_loc * 0.55), color=ec)
        arrow(ax, (x_h + w_h + 0.004, y_sum + h_sum * 0.5), (x_proj - 0.004, y_sum + h_sum * 0.5), color=ec)

        z_pts.append((x_proj + w_proj, y_loc + h_loc * 0.55))
        g_pts.append((x_proj + w_proj, y_sum + h_sum * 0.5))
        u_pts.append((x_h + w_h / 2, y_sum + 0.008, ec))

    # ⑤ EMT
    rounded(ax, (x_out, 0.470), 0.172, 0.400, fc="#EAF0F8", ec="#1E3A5F", lw=1.7, r=0.016)
    cn(ax, x_out + 0.086, 0.825, "供后续 EMT 交互", size=13.0, weight="bold", color="#1E3A5F")
    ax.text(
        x_out + 0.086,
        0.700,
        r"$\mathbf{Z}_i^{T,0}$" + "\n" + r"$\mathbf{Z}_i^{A,0}$" + "\n" + r"$\mathbf{Z}_i^{V,0}$",
        fontsize=FS_BOX_MATH_LG,
        color="#1E3A5F",
        ha="center",
        va="center",
        zorder=5,
    )
    cn(ax, x_out + 0.086, 0.545, r"各  $L_i\times d$" + "\n有效位用 mask\n内部零行占原位", size=11.0, color="#334155")

    for zx, zy in z_pts:
        arrow(ax, (zx + 0.004, zy), (x_out - 0.004, max(0.55, min(0.78, zy))), color="#1E3A5F", lw=1.4)

    # ⑤ G^0
    rounded(ax, (x_out, 0.215), 0.172, 0.225, fc="#F3EAF8", ec="#5A3878", lw=1.7, r=0.016)
    cn(ax, x_out + 0.086, 0.400, "初始全局上下文", size=12.5, weight="bold", color="#5A3878")
    ax.text(
        x_out + 0.086,
        0.300,
        r"$\mathbf{G}_i^{0}=\mathrm{stack}$" + "\n"
        + r"$(\mathbf{g}_i^{T,0},\mathbf{g}_i^{A,0},\mathbf{g}_i^{V,0})$" + "\n"
        + r"$\in\mathbb{R}^{3\times d}$",
        fontsize=12.5,
        color="#5A3878",
        ha="center",
        va="center",
        zorder=5,
    )

    # stack node
    sx, sy = x_out - 0.040, 0.315
    rounded(ax, (sx - 0.028, sy - 0.028), 0.056, 0.056, fc="#5A3878", ec="#5A3878", lw=1.0, r=0.01)
    cn(ax, sx, sy, "stack", size=10.5, weight="bold", color="white")
    for gx, gy in g_pts:
        arrow(ax, (gx + 0.004, gy), (sx - 0.030, sy), color="#5A3878", lw=1.25, rad=0.05)
    arrow(ax, (sx + 0.030, sy), (x_out - 0.004, sy), color="#5A3878", lw=1.5)

    # Bypass band
    rounded(ax, (0.330, 0.028), 0.642, 0.135, fc="#FFF8F0", ec="#8A5728", lw=1.5, r=0.014)
    cn(ax, 0.300, 0.155, "bypass", size=12.0, weight="bold", color="#8A5728", ha="right")
    cn(
        ax,
        0.651,
        0.125,
        r"投影前摘要旁路（不进入  $\mathbf{G}_i^{0}$）：保留  "
        r"$\mathbf{u}_i^{T},\ \mathbf{u}_i^{A},\ \mathbf{u}_i^{V}$  至预测拼接",
        size=FS_NOTE + 0.5,
        color="#6A3D12",
    )
    ax.text(
        0.651,
        0.065,
        r"$\mathbf{f}_i=[\mathbf{u}_i^{T};\mathbf{u}_i^{A};\mathbf{u}_i^{V};\mathrm{vec}(\mathbf{G}_i^{K})]$"
        r"$\qquad 768+16+32+3\times 128=1200$",
        fontsize=FS_BOX_MATH,
        color="#6A3D12",
        ha="center",
        va="center",
        zorder=5,
    )

    for ux, uy, ec in u_pts:
        ax.add_line(
            Line2D(
                [ux, ux, 0.38],
                [uy, 0.165, 0.165],
                linestyle=(0, (3.2, 2.2)),
                color=ec,
                linewidth=1.35,
                zorder=1,
                alpha=0.85,
            )
        )

    # Legend
    cn(ax, 0.028, 0.115, "线型说明", size=FS_LEGEND + 1, weight="bold", color="#333", ha="left")
    cn(ax, 0.028, 0.070, r"实线：投影后进入 EMT / $\mathbf{G}_i^{0}$", size=FS_LEGEND, color="#444", ha="left")
    cn(ax, 0.028, 0.032, r"虚线：投影前 $\mathbf{u}$ 旁路至拼接", size=FS_LEGEND, color="#444", ha="left")

    fig.tight_layout(pad=0.15)
    fig.savefig(PNG_PATH, dpi=340, bbox_inches="tight", facecolor="white")
    fig.savefig(SVG_PATH, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {PNG_PATH}")
    print(f"wrote {SVG_PATH}")
    print(f"wrote {PDF_PATH}")


if __name__ == "__main__":
    draw()
