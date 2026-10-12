"""Draw fig:q3_framework — Q3 overall pipeline schematic.

Highlights frozen M1 predictor vs parameter-free explanation layer.
QG-HTA is post-hoc traceback only, not inside M1 training.
Outputs under ../Figure/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

CODE_DIR = Path(__file__).resolve().parent
FIG_DIR = CODE_DIR.parent / "Figure"
STEM = "q3_framework"


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


def box(ax, xy, w, h, *, fc, ec, title, body="", title_size=11.5, body_size=9.5, tw="bold"):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.35,
        zorder=2,
    )
    ax.add_patch(patch)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    if body:
        ax.text(cx, cy + h * 0.22, title, ha="center", va="center", fontsize=title_size, fontweight=tw, color=ec, zorder=4)
        ax.text(cx, cy - h * 0.22, body, ha="center", va="center", fontsize=body_size, color="#374151", zorder=4)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=title_size, fontweight=tw, color=ec, zorder=4)
    return patch


def arrow(ax, p0, p1, *, color="#4B5563", lw=1.35):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=lw,
            color=color,
            zorder=3,
            shrinkA=1,
            shrinkB=1,
        )
    )


def draw():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_fonts()
    fig, ax = plt.subplots(figsize=(13.2, 8.6), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.50, 0.965, "问题三总体框架：冻结 M1 预测器 ＋ 无参数解释层", ha="center", va="center", fontsize=14.5, fontweight="bold", color="#1F2937")

    # Zone backgrounds
    ax.add_patch(FancyBboxPatch((0.03, 0.55), 0.94, 0.38, boxstyle="round,pad=0.008,rounding_size=0.015", facecolor="#EEF5FB", edgecolor="#2F5F86", linewidth=1.4, zorder=1))
    ax.add_patch(FancyBboxPatch((0.03, 0.04), 0.94, 0.48, boxstyle="round,pad=0.008,rounding_size=0.015", facecolor="#FBF6F1", edgecolor="#8A5530", linewidth=1.4, zorder=1))

    ax.text(0.05, 0.905, "冻结 M1 预测器（问题二完整输入联合模型）", ha="left", va="center", fontsize=12, fontweight="bold", color="#2F5F86", zorder=5)
    ax.text(0.05, 0.490, "无参数解释层（问题三新增；不更新 M1 参数）", ha="left", va="center", fontsize=12, fontweight="bold", color="#8A5530", zorder=5)

    # M1 row
    box(ax, (0.06, 0.78), 0.16, 0.09, fc="#FFFFFF", ec="#5B6770", title="正式特征", body="文本/语音/视觉\n＋长度协议")
    box(ax, (0.27, 0.78), 0.20, 0.09, fc="#FFFFFF", ec="#2F5F86", title="三模态编码", body="BERT / LSTM$_A$ / LSTM$_V$")
    box(ax, (0.52, 0.78), 0.18, 0.09, fc="#FFFFFF", ec="#2F5F86", title="EMT 融合", body="跨模态交互")
    box(ax, (0.75, 0.72), 0.18, 0.07, fc="#E7F0F8", ec="#2F5F86", title="分类头", body="极性 / 概率 / 边际")
    box(ax, (0.75, 0.62), 0.18, 0.07, fc="#E7F0F8", ec="#2F5F86", title="回归头", body="情感强度")

    arrow(ax, (0.22, 0.825), (0.27, 0.825), color="#2F5F86")
    arrow(ax, (0.47, 0.825), (0.52, 0.825), color="#2F5F86")
    arrow(ax, (0.70, 0.825), (0.75, 0.755), color="#2F5F86")
    arrow(ax, (0.70, 0.825), (0.75, 0.655), color="#2F5F86")

    ax.text(0.50, 0.585, "参数冻结 · 附件4推理时反复前向，仅改变输入遮蔽", ha="center", va="center", fontsize=10, color="#2F5F86", zorder=5)

    # Explanation layer
    box(ax, (0.06, 0.36), 0.26, 0.10, fc="#FFF8E6", ec="#B8860B", title="固定解释目标", body=r"原输入确定 $k_i^{\ast}$ / $s_i^{\ast}$" + "\n遮蔽后仍追踪原目标")
    box(ax, (0.37, 0.36), 0.28, 0.10, fc="#F8EFE7", ec="#8A5530", title="模态联盟精确 Shapley", body="枚举 8 个联盟\n有符号贡献 + 绝对占比")
    box(ax, (0.70, 0.36), 0.23, 0.10, fc="#F8EFE7", ec="#8A5530", title="多尺度局部遮蔽", body="窗口 $w\\in\\{1,3,5\\}$\n逐位置分布 / Top≤3")

    arrow(ax, (0.84, 0.62), (0.84, 0.50), color="#8A5530")
    arrow(ax, (0.19, 0.36), (0.19, 0.30), color="#8A5530")
    # from heads down to fixed target conceptually
    arrow(ax, (0.32, 0.41), (0.37, 0.41), color="#8A5530")
    arrow(ax, (0.65, 0.41), (0.70, 0.41), color="#8A5530")

    box(ax, (0.06, 0.18), 0.40, 0.10, fc="#E8F4EE", ec="#2F6B4F", title="文本 offset 回溯", body="WordPiece → 原始字符片段")
    box(ax, (0.52, 0.18), 0.41, 0.10, fc="#E8F4EE", ec="#2F6B4F", title="QG-HTA 音视频回溯（预测后）", body="可靠秒数 / 视频帧；部分覆盖与无可靠区间显式留空\n不是 M1 训练内部模块，也不均分 50 槽位")

    arrow(ax, (0.19, 0.36), (0.26, 0.28), color="#2F6B4F")
    arrow(ax, (0.51, 0.41), (0.72, 0.28), color="#2F6B4F")
    arrow(ax, (0.81, 0.36), (0.72, 0.28), color="#2F6B4F")

    box(ax, (0.18, 0.055), 0.64, 0.085, fc="#FFFFFF", ec="#6B7280", title="交付物", body="解释卡 · 全量预测解释表 · Top 证据时间/帧表 · 忠实度与效率审计")

    arrow(ax, (0.26, 0.18), (0.40, 0.14), color="#6B7280")
    arrow(ax, (0.72, 0.18), (0.60, 0.14), color="#6B7280")

    ax.text(
        0.50,
        0.015,
        "蓝色区＝问题二复用的冻结预测器；橙色区＝问题三解释层；绿色区＝原素材回溯。附图不填虚构数值。",
        ha="center",
        va="center",
        fontsize=9.2,
        color="#6B7280",
    )

    for ext in ("pdf", "png", "svg"):
        out = FIG_DIR / f"{STEM}.{ext}"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.06, facecolor="white")
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    draw()
