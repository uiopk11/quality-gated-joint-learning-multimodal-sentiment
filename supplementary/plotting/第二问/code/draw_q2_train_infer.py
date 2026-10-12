"""Draw fig:q2_train_infer — training vs inference computation boundary (v2 layout)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent
FIG_DIR = ROOT_DIR / "Figure"
PNG_PATH = FIG_DIR / "q2_train_infer.png"
SVG_PATH = FIG_DIR / "q2_train_infer.svg"
PDF_PATH = FIG_DIR / "q2_train_infer.pdf"


def setup_fonts() -> str:
    preferred = ["Microsoft YaHei", "SimHei", "SimSun"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams.update(
                {
                    "font.family": name,
                    "font.size": 12,
                    "axes.unicode_minus": False,
                    "mathtext.fontset": "cm",
                    "mathtext.default": "it",
                }
            )
            return name
    raise RuntimeError("No Chinese-capable font found.")


def box(ax, xy, w, h, *, fc, ec, lw=1.5, r=0.014, alpha=1.0, ls="-"):
    p = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        alpha=alpha,
        mutation_aspect=0.5,
        zorder=2,
    )
    ax.add_patch(p)
    return p


def arrow(ax, p0, p1, *, color="#334155", lw=1.55, rad=0.0, ls="-", alpha=1.0, ms=13):
    a = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        linestyle=ls,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        zorder=3,
        shrinkA=1,
        shrinkB=1,
    )
    ax.add_patch(a)
    return a


def txt(ax, x, y, s, *, size=12, weight="normal", color="#1a1a1a", ha="center", va="center", alpha=1.0):
    return ax.text(
        x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, alpha=alpha, zorder=5
    )


def backbone(ax, x, y, w, h, *, title):
    box(ax, (x, y), w, h, fc="#E8EEF7", ec="#1E3A5F", lw=1.8)
    txt(ax, x + w / 2, y + h * 0.78, title, size=13.0, weight="bold", color="#1E3A5F")
    txt(
        ax,
        x + w / 2,
        y + h * 0.42,
        r"BERT / $\mathrm{LSTM}_{A,V}$" + "\n" + r"$\rightarrow$ EMT $\rightarrow$ 共享表示 $h$",
        size=12.0,
        color="#1E3A5F",
    )
    txt(ax, x + w / 2, y + h * 0.12, r"共享参数 $\theta$", size=11.5, color="#475569")


def draw():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_fonts()

    fig, ax = plt.subplots(figsize=(15.6, 9.4), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    txt(ax, 0.50, 0.978, "训练与推理计算边界（JPI-DLFR / M3）", size=17.0, weight="bold", color="#0F172A")

    # Left stats strip
    box(ax, (0.012, 0.04), 0.148, 0.910, fc="#FFFBEB", ec="#B45309", lw=1.45, r=0.012)
    txt(ax, 0.086, 0.90, "统计依赖", size=13.0, weight="bold", color="#92400E")
    txt(ax, 0.086, 0.78, "训练集拟合\n" + r"$\mu,\ \sigma$" + "\n仅有效非零\nA/V 通道", size=11.5, color="#92400E")
    arrow(ax, (0.086, 0.62), (0.086, 0.48), color="#B45309", lw=1.7)
    txt(ax, 0.086, 0.40, "冻结后复用", size=12.0, weight="bold", color="#92400E")
    txt(ax, 0.086, 0.22, "valid / test\n附件三推理\n不再重估", size=11.0, color="#A16207")

    # ===== TRAIN =====
    box(ax, (0.175, 0.52), 0.810, 0.430, fc="#F8FAFC", ec="#CBD5E1", lw=1.25, r=0.01)
    txt(ax, 0.195, 0.925, "上：训练路径", size=14.5, weight="bold", color="#1D4ED8", ha="left")
    txt(
        ax,
        0.195,
        0.888,
        "同一样本双视图配对  ·  双任务损失  ·  高低层辅助目标  ·  验证选模",
        size=11.2,
        color="#475569",
        ha="left",
    )

    # dual views
    box(ax, (0.195, 0.78), 0.150, 0.085, fc="#DBEAFE", ec="#1D4ED8", lw=1.5)
    txt(ax, 0.270, 0.835, "完整视图 clean", size=12.0, weight="bold", color="#1D4ED8")
    txt(ax, 0.270, 0.800, r"$v{=}c$（无新增遮蔽）", size=11.0, color="#1E40AF")

    box(ax, (0.195, 0.655), 0.150, 0.085, fc="#FEE2E2", ec="#B91C1C", lw=1.5)
    txt(ax, 0.270, 0.710, "缺失视图 damaged", size=12.0, weight="bold", color="#B91C1C")
    txt(ax, 0.270, 0.675, r"$v{=}d$（人工掩码 $M$）", size=11.0, color="#991B1B")
    txt(ax, 0.270, 0.625, "同一样本配对", size=11.0, weight="bold", color="#64748B")

    backbone(ax, 0.380, 0.655, 0.215, 0.210, title="共享主干（训练）")
    arrow(ax, (0.345, 0.822), (0.380, 0.800), color="#1D4ED8", lw=1.5)
    arrow(ax, (0.345, 0.698), (0.380, 0.720), color="#B91C1C", lw=1.5)

    # task
    box(ax, (0.625, 0.790), 0.160, 0.075, fc="#ECFDF5", ec="#047857", lw=1.5)
    txt(ax, 0.705, 0.840, "双任务头", size=12.0, weight="bold", color="#047857")
    txt(ax, 0.705, 0.808, r"$\hat c,\ \hat y$（两视图）", size=11.2, color="#065F46")

    box(ax, (0.805, 0.790), 0.160, 0.075, fc="#ECFDF5", ec="#047857", lw=1.5)
    txt(ax, 0.885, 0.840, "任务损失", size=12.0, weight="bold", color="#047857")
    txt(ax, 0.885, 0.808, r"$\mathcal{L}_{\mathrm{JPI}}^{c}+\mathcal{L}_{\mathrm{JPI}}^{d}$", size=12.0, color="#065F46")

    arrow(ax, (0.595, 0.820), (0.625, 0.828), color="#047857", lw=1.45)
    arrow(ax, (0.785, 0.828), (0.805, 0.828), color="#047857", lw=1.45)

    # aux train-only
    box(ax, (0.625, 0.655), 0.160, 0.105, fc="#FEF3C7", ec="#D97706", lw=1.65)
    txt(ax, 0.705, 0.730, "高层辅助（仅训练）", size=11.2, weight="bold", color="#B45309")
    txt(ax, 0.705, 0.685, "projector / predictor\n" + r"$\mathcal{L}_{\mathrm{high}}$（G,T,A,V）", size=10.8, color="#92400E")

    box(ax, (0.805, 0.655), 0.160, 0.105, fc="#FEF3C7", ec="#D97706", lw=1.65)
    txt(ax, 0.885, 0.730, "低层恢复（仅训练）", size=11.2, weight="bold", color="#B45309")
    txt(
        ax,
        0.885,
        0.685,
        r"$\widehat{R}^{m}=D_m(Z^{m,K,d})$" + "\n" + r"仅在 $S{=}P{\cdot}M{\cdot}O$",
        size=10.5,
        color="#92400E",
    )

    arrow(ax, (0.545, 0.720), (0.625, 0.710), color="#D97706", lw=1.4, rad=-0.04)
    arrow(ax, (0.545, 0.700), (0.625, 0.690), color="#D97706", lw=1.25, rad=0.06)

    # total loss + no refill on separate rows
    box(ax, (0.380, 0.585), 0.585, 0.050, fc="#FFFFFF", ec="#1E3A5F", lw=1.25)
    txt(
        ax,
        0.672,
        0.610,
        r"$\mathcal{L}_{\mathrm{JPI\text{-}DLFR}}"
        r"=\mathcal{L}_{\mathrm{JPI}}^{c}+\mathcal{L}_{\mathrm{JPI}}^{d}"
        r"+\mathcal{L}_{\mathrm{high}}+\mathcal{L}_{\mathrm{low}}$"
        "   ｜    恢复结果不回填输入、不二阶段重预测",
        size=11.5,
        color="#1E3A5F",
    )

    box(ax, (0.195, 0.540), 0.420, 0.035, fc="#EEF2FF", ec="#6366F1", lw=1.15)
    txt(
        ax,
        0.405,
        0.557,
        r"监督：真值 $y,c$  ·  掩码 $M$  ·  支持 $S$  ·  clean 参考（目标侧停梯度）",
        size=10.5,
        color="#3730A3",
    )
    box(ax, (0.630, 0.540), 0.335, 0.035, fc="#EEF2FF", ec="#6366F1", lw=1.15)
    txt(ax, 0.797, 0.557, r"验证选模：Acc $\rightarrow$ Macro-F1 $\rightarrow$ MAE", size=10.5, color="#3730A3")

    # ===== INFER =====
    box(ax, (0.175, 0.04), 0.810, 0.455, fc="#FAFAF9", ec="#D6D3D1", lw=1.25, r=0.01)
    txt(ax, 0.195, 0.465, "下：推理路径", size=14.5, weight="bold", color="#0F766E", ha="left")
    txt(
        ax,
        0.195,
        0.430,
        "单视图当前输入  ·  冻结统计与权重  ·  仅编码—融合—双头预测",
        size=11.2,
        color="#57534E",
        ha="left",
    )

    box(ax, (0.195, 0.300), 0.150, 0.100, fc="#CCFBF1", ec="#0F766E", lw=1.55)
    txt(ax, 0.270, 0.370, "当前可用输入", size=12.0, weight="bold", color="#0F766E")
    txt(ax, 0.270, 0.325, "已有缺失保持原位\n不再追加 " + r"$M$", size=10.8, color="#115E59")

    box(ax, (0.365, 0.320), 0.095, 0.070, fc="#FFF7ED", ec="#C2410C", lw=1.4)
    txt(ax, 0.412, 0.365, "冻结", size=11.5, weight="bold", color="#C2410C")
    txt(ax, 0.412, 0.335, r"$\mu,\sigma$", size=12.5, color="#9A3412")

    arrow(ax, (0.345, 0.350), (0.365, 0.355), color="#C2410C", lw=1.35)
    arrow(ax, (0.460, 0.355), (0.485, 0.355), color="#0F766E", lw=1.4)

    backbone(ax, 0.485, 0.280, 0.215, 0.150, title="共享主干（推理）")

    box(ax, (0.730, 0.350), 0.110, 0.075, fc="#ECFDF5", ec="#047857", lw=1.5)
    txt(ax, 0.785, 0.400, "极性", size=12.0, weight="bold", color="#047857")
    txt(ax, 0.785, 0.368, r"$\hat c=\arg\max$", size=11.0, color="#065F46")

    box(ax, (0.855, 0.350), 0.110, 0.075, fc="#ECFDF5", ec="#047857", lw=1.5)
    txt(ax, 0.910, 0.400, "强度", size=12.0, weight="bold", color="#047857")
    txt(ax, 0.910, 0.368, r"$\hat y$（原值）", size=11.0, color="#065F46")

    arrow(ax, (0.700, 0.390), (0.730, 0.390), color="#047857", lw=1.45)
    arrow(ax, (0.700, 0.340), (0.855, 0.375), color="#047857", lw=1.3, rad=0.06)
    txt(ax, 0.855, 0.310, "两头独立保存，不互改、不用真值事后修正", size=10.5, color="#047857")

    # not executed
    box(ax, (0.195, 0.085), 0.770, 0.195, fc="#F5F5F4", ec="#A8A29E", lw=1.4, ls="--")
    txt(ax, 0.215, 0.250, "推理时不执行（淡化 / 虚线框）", size=12.0, weight="bold", color="#78716C", ha="left")

    faded = [
        (0.215, "完整参考\nclean 视图"),
        (0.345, "真值标签\n" + r"$y,\ c$"),
        (0.475, "人工掩码 " + r"$M$" + "\n恢复支持 " + r"$S$"),
        (0.615, "projector\n/ predictor"),
        (0.745, "恢复头 " + r"$D_m$" + "\n" + r"$\widehat R$"),
        (0.860, "恢复回填\n再预测"),
    ]
    for x, lab in faded:
        box(ax, (x, 0.105), 0.110, 0.115, fc="#E7E5E4", ec="#A8A29E", lw=1.15, ls="--", alpha=0.95)
        txt(ax, x + 0.055, 0.162, lab, size=10.2, color="#78716C")

    txt(
        ax,
        0.58,
        0.055,
        "计算边界：配对监督与辅助模块只进入训练目标；正式推理仅走「当前输入 → 编码/融合 → 双头预测」。",
        size=11.2,
        color="#44403C",
    )

    fig.tight_layout(pad=0.18)
    fig.savefig(PNG_PATH, dpi=340, bbox_inches="tight", facecolor="white")
    fig.savefig(SVG_PATH, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {PNG_PATH}")
    print(f"wrote {SVG_PATH}")
    print(f"wrote {PDF_PATH}")


if __name__ == "__main__":
    draw()
