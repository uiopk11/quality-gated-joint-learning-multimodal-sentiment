"""Draw fig:q3_shapley_exact — exact 8-coalition Shapley schematic.

Uses Attachment-4 sample 01 real coalition outputs (classification margin).
Outputs under ../Figure/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

CODE_DIR = Path(__file__).resolve().parent
FIG_DIR = CODE_DIR.parent / "Figure"
STEM = "q3_shapley_exact"

Q3_ROOT = CODE_DIR.parent.parent
SAMPLE_JSON = Q3_ROOT / "results" / "attachment4" / "samples" / "01.json"


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


def mod_label(mods):
    if not mods:
        return r"\varnothing"
    mp = {"text": "T", "audio": "A", "vision": "V"}
    return r"\{" + ",".join(mp[m] for m in mods) + r"\}"


def draw():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_fonts()
    data = json.loads(SAMPLE_JSON.read_text(encoding="utf-8"))
    me = data["modality_explanation"]
    coals = me["coalitions"]
    cls = me["classification"]
    eff = me["efficiency_error"]["classification"]
    target = me["target_class"]
    label_map = {0: "负向", 1: "中性", 2: "正向"}

    # order coalitions by size then name
    order_key = {
        frozenset(): 0,
        frozenset(["text"]): 1,
        frozenset(["audio"]): 2,
        frozenset(["vision"]): 3,
        frozenset(["text", "audio"]): 4,
        frozenset(["text", "vision"]): 5,
        frozenset(["audio", "vision"]): 6,
        frozenset(["text", "audio", "vision"]): 7,
    }
    coals_sorted = sorted(coals, key=lambda c: order_key[frozenset(c["retained_modalities"])])

    fig = plt.figure(figsize=(13.8, 8.4), dpi=220)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 1, height_ratios=[0.7, 1.55, 1.35], hspace=0.35, left=0.06, right=0.97, top=0.92, bottom=0.06)

    # Banner
    ax0 = fig.add_subplot(gs[0])
    ax0.axis("off")
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.add_patch(
        FancyBboxPatch(
            (0.01, 0.12),
            0.98,
            0.76,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor="#FFF8E6",
            edgecolor="#B8860B",
            linewidth=1.2,
        )
    )
    ax0.text(
        0.50,
        0.55,
        f"精确三模态 Shapley（附件4样本 01 · 分类边际 · 原目标 $k_i^{{\\ast}}$={label_map[target]}）\n"
        "枚举全部 8 个联盟实际前向；无 Monte Carlo 采样。数值来自已保存结果，非示意虚构。",
        ha="center",
        va="center",
        fontsize=12.2,
        fontweight="bold",
        color="#5C4300",
    )

    # Coalition grid
    ax1 = fig.add_subplot(gs[1])
    ax1.axis("off")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.text(0.50, 0.96, "八个联盟的分类边际 $v_c(S)=s_c(X_S)$", ha="center", va="top", fontsize=12.5, fontweight="bold", color="#1F2937")

    # 2 rows x 4
    positions = [
        (0.03, 0.52),
        (0.27, 0.52),
        (0.51, 0.52),
        (0.75, 0.52),
        (0.03, 0.08),
        (0.27, 0.08),
        (0.51, 0.08),
        (0.75, 0.08),
    ]
    colors = {
        0: ("#F3F4F6", "#6B7280"),
        1: ("#DBEAFE", "#1D4ED8"),
        2: ("#DBEAFE", "#1D4ED8"),
        3: ("#DBEAFE", "#1D4ED8"),
        4: ("#DCFCE7", "#15803D"),
        5: ("#DCFCE7", "#15803D"),
        6: ("#DCFCE7", "#15803D"),
        7: ("#FEF3C7", "#B45309"),
    }
    for i, (c, (x, y)) in enumerate(zip(coals_sorted, positions)):
        fc, ec = colors[i]
        w, h = 0.21, 0.38
        ax1.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.006,rounding_size=0.012",
                facecolor=fc,
                edgecolor=ec,
                linewidth=1.25,
            )
        )
        mods = c["retained_modalities"]
        ax1.text(x + w / 2, y + h * 0.78, f"$S={mod_label(mods)}$", ha="center", va="center", fontsize=12, color=ec, fontweight="bold")
        # modality chips
        chip_y = y + h * 0.48
        for j, name, key in [(0, "T", "text"), (1, "A", "audio"), (2, "V", "vision")]:
            on = key in mods
            cx = x + w * (0.22 + j * 0.28)
            ax1.add_patch(
                Rectangle(
                    (cx - 0.018, chip_y - 0.035),
                    0.036,
                    0.07,
                    facecolor="#FFFFFF" if on else "#E5E7EB",
                    edgecolor=ec if on else "#9CA3AF",
                    linewidth=1.0,
                    linestyle="-" if on else "--",
                )
            )
            ax1.text(cx, chip_y, name if on else "—", ha="center", va="center", fontsize=9, color=ec if on else "#9CA3AF")
        ax1.text(
            x + w / 2,
            y + h * 0.18,
            f"$v_c={c['classification_margin']:+.3f}$\n$s_r={c['intensity']:+.3f}$",
            ha="center",
            va="center",
            fontsize=10,
            color="#1F2937",
        )

    # Bottom: Shapley result bars + formula note
    ax2 = fig.add_subplot(gs[2])
    ax2.axis("off")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    ax2.add_patch(
        FancyBboxPatch(
            (0.02, 0.08),
            0.46,
            0.84,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            facecolor="#F8FAFC",
            edgecolor="#64748B",
            linewidth=1.2,
        )
    )
    ax2.text(0.25, 0.84, "有符号 Shapley $\\phi_m^{(c)}$ 与绝对占比", ha="center", va="center", fontsize=11.5, fontweight="bold", color="#334155")

    signed = cls["signed_shapley"]
    share = cls["absolute_share"]
    names = [("text", "文本 T", "#1D4ED8"), ("audio", "语音 A", "#BE185D"), ("vision", "视觉 V", "#15803D")]
    # simple bars in axes coordinates
    max_abs = max(abs(signed[k]) for k, _, _ in names) * 1.15
    base_x, bar_w, bar_h0 = 0.10, 0.28, 0.12
    for i, (key, lab, col) in enumerate(names):
        y = 0.62 - i * 0.18
        ax2.text(0.05, y + 0.04, lab, ha="left", va="center", fontsize=10.5, color=col, fontweight="bold")
        # zero line
        z = base_x + bar_w * 0.5
        ax2.plot([z, z], [y - 0.02, y + 0.10], color="#9CA3AF", lw=0.8)
        val = signed[key]
        if val >= 0:
            x0 = z
            ww = bar_w * 0.5 * (val / max_abs)
        else:
            ww = bar_w * 0.5 * (abs(val) / max_abs)
            x0 = z - ww
        ax2.add_patch(Rectangle((x0, y), ww, bar_h0, facecolor=col, alpha=0.75, edgecolor=col, linewidth=1.0))
        ax2.text(
            0.42,
            y + 0.04,
            f"$\\phi={val:+.3f}$  ·  $p={share[key]*100:.1f}\\%$",
            ha="left",
            va="center",
            fontsize=10,
            color="#1F2937",
        )

    ax2.add_patch(
        FancyBboxPatch(
            (0.52, 0.08),
            0.46,
            0.84,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            facecolor="#EFF6FF",
            edgecolor="#2563EB",
            linewidth=1.2,
        )
    )
    ax2.text(0.75, 0.84, "解释读法（本样本）", ha="center", va="center", fontsize=11.5, fontweight="bold", color="#1E40AF")
    ax2.text(
        0.75,
        0.55,
        f"主要参考（$|\\phi|$最大）：{ {'text':'文本','audio':'语音','vision':'视觉'}[cls['most_influential_modality']] }\n"
        f"主要支持（$\\phi>0$最大）：{ {'text':'文本','audio':'语音','vision':'视觉'}[cls['primary_supporting_modality']] }\n"
        f"主要反对（$\\phi<0$最小）：{ {'text':'文本','audio':'语音','vision':'视觉'}[cls['primary_opposing_modality']] }\n\n"
        f"效率误差 $\\sum\\phi-[v(M)-v(\\varnothing)]={eff:.1e}$\n"
        "（分类目标；与审计浮点精度一致）",
        ha="center",
        va="center",
        fontsize=10.2,
        color="#1E3A8A",
    )

    fig.text(
        0.50,
        0.015,
        r"遮蔽基线：正文$\to$[UNK]；语音/视觉有效行置零；长度与 attention_mask 不变。"
        " 有符号值看方向，绝对占比看作用程度；二者不可混用。",
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
