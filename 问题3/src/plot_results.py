from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


Q3_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = Q3_ROOT / "results" / "attachment4" / "samples"
FIGURE_ROOT = Q3_ROOT / "results" / "figures"
MODALITIES = ("text", "audio", "vision")
LABELS = ("Text", "Audio", "Vision")
COLORS = ("#5A76A8", "#C96A6A", "#4FA38F")


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.8,
        "savefig.dpi": 300,
        "svg.fonttype": "none",
    })


def load_records() -> list[dict]:
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(SAMPLE_ROOT.glob("*.json"))]
    if len(records) != 20:
        raise ValueError("Expected 20 attachment4 explanation records")
    return records


def save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_modality_overview(records: list[dict]) -> None:
    class_values = np.array([[r["modality_explanation"]["classification"]["signed_shapley"][m] for m in MODALITIES] for r in records])
    reg_values = np.array([[r["modality_explanation"]["regression"]["signed_shapley"][m] for m in MODALITIES] for r in records])
    class_shares = np.array([[r["modality_explanation"]["classification"]["absolute_share"][m] for m in MODALITIES] for r in records])
    reg_shares = np.array([[r["modality_explanation"]["regression"]["absolute_share"][m] for m in MODALITIES] for r in records])
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 6.2), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.05, 1.0]})
    for ax, values, title in ((axes[0, 0], class_values, "Classification margin"),
                              (axes[1, 0], reg_values, "Sentiment intensity")):
        bound = max(float(np.abs(values).max()), 1e-8)
        image = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
        ax.set_xticks(range(3), LABELS)
        ax.set_yticks(range(20), [f"{i:02d}" for i in range(1, 21)])
        ax.set_ylabel("Attachment-4 sample")
        ax.set_title(title, loc="left", fontweight="bold")
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
        colorbar.set_label("Signed Shapley sensitivity")
    rng = np.random.default_rng(20260925)
    for ax, values, title in ((axes[0, 1], class_shares, "Classification effect shares"),
                              (axes[1, 1], reg_shares, "Intensity effect shares")):
        parts = ax.boxplot([values[:, i] for i in range(3)], positions=np.arange(3), widths=0.48,
                           patch_artist=True, showfliers=False,
                           medianprops={"color": "#222222", "linewidth": 1.2},
                           whiskerprops={"color": "#555555"}, capprops={"color": "#555555"})
        for patch, color in zip(parts["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.38)
            patch.set_edgecolor(color)
        for index, color in enumerate(COLORS):
            jitter = rng.normal(0, 0.045, len(values))
            ax.scatter(index + jitter, values[:, index], s=13, color=color, alpha=0.85,
                       edgecolors="white", linewidths=0.25, zorder=3)
        ax.set_xticks(range(3), LABELS)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("Absolute sensitivity share")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].text(-0.23, 1.03, "A", transform=axes[0, 0].transAxes, fontweight="bold", fontsize=11)
    axes[0, 1].text(-0.18, 1.03, "B", transform=axes[0, 1].transAxes, fontweight="bold", fontsize=11)
    axes[1, 0].text(-0.23, 1.03, "C", transform=axes[1, 0].transAxes, fontweight="bold", fontsize=11)
    axes[1, 1].text(-0.18, 1.03, "D", transform=axes[1, 1].transAxes, fontweight="bold", fontsize=11)
    save(fig, FIGURE_ROOT / "modality_effects_overview")


def plot_local_record(record: dict) -> None:
    sample = record["sample"]
    length = int(sample["body_length"])
    tokens = sample["wordpieces"]
    cls = np.array([[p["classification_signed"] for p in record["local_explanation"][m]["positions"]] for m in MODALITIES])
    reg = np.array([[p["intensity_signed"] for p in record["local_explanation"][m]["positions"]] for m in MODALITIES])
    fig, axes = plt.subplots(2, 1, figsize=(max(6.3, length * 0.24), 3.25), constrained_layout=True)
    for row, (ax, values, label) in enumerate(((axes[0], cls, "Classification margin"), (axes[1], reg, "Sentiment intensity"))):
        bound = max(float(np.abs(values).max()), 1e-8)
        image = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound,
                          interpolation="nearest")
        ax.set_yticks(range(3), LABELS)
        ax.set_xticks(range(length), tokens, rotation=60, ha="right", rotation_mode="anchor")
        ax.set_ylabel("Modality")
        ax.set_title(label, loc="left", fontweight="bold")
        colorbar = fig.colorbar(image, ax=ax, fraction=0.018, pad=0.018)
        colorbar.set_label("Signed local sensitivity")
        ax.text(-0.075, 1.05, "A" if row == 0 else "B", transform=ax.transAxes,
                fontweight="bold", fontsize=10)
    axes[1].set_xlabel("BERT WordPiece / aligned body position")
    save(fig, FIGURE_ROOT / "local_evidence" / f"{sample['sample_id']}_local_evidence")


def plot_fidelity() -> None:
    validation_root = Q3_ROOT / "results" / "validation" / "samples"
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(validation_root.glob("*.json"))]
    if len(records) != 20:
        raise ValueError("Expected 20 fidelity records")
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.6), sharex=True, constrained_layout=True)
    for row, target in enumerate(("classification", "regression")):
        for column, modality in enumerate(MODALITIES):
            ax = axes[row, column]
            entries = [record["fidelity_validation"]["results"][modality][target] for record in records]
            entries = [entry for entry in entries if entry]
            if not entries:
                ax.text(0.5, 0.5, "No available observations", ha="center", va="center")
                continue
            fractions = np.array([item["deleted_fraction"] for item in entries[0]], dtype=float)
            top = np.array([[item["top_absolute_output_change"] for item in sample] for sample in entries])
            random = np.array([[item["random_mean_absolute_output_change"] for item in sample] for sample in entries])
            ax.plot(fractions, top.mean(axis=0), color=COLORS[column], marker="o", linewidth=1.8,
                    markersize=3.5, label="Top-ranked")
            ax.fill_between(fractions, top.mean(axis=0) - top.std(axis=0),
                            top.mean(axis=0) + top.std(axis=0), color=COLORS[column], alpha=0.16, linewidth=0)
            ax.plot(fractions, random.mean(axis=0), color="#555555", marker="o", linewidth=1.3,
                    markersize=3.0, linestyle="--", label="Random")
            ax.set_title(LABELS[column], fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xticks(fractions, [f"{int(value * 100)}%" for value in fractions])
            if column == 0:
                ax.set_ylabel("Absolute margin change" if row == 0 else "Absolute intensity change")
            if row == 1:
                ax.set_xlabel("Deleted positions")
            if row == 0 and column == 2:
                ax.legend(frameon=False, fontsize=7, loc="best")
            panel = chr(ord("A") + row * 3 + column)
            ax.text(-0.22, 1.04, panel, transform=ax.transAxes, fontweight="bold", fontsize=10)
    save(fig, FIGURE_ROOT / "fidelity_top_vs_random")


def main() -> None:
    configure_style()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    (FIGURE_ROOT / "local_evidence").mkdir(parents=True, exist_ok=True)
    records = load_records()
    plot_modality_overview(records)
    for record in records:
        plot_local_record(record)
    plot_fidelity()
    print(json.dumps({
        "status": "completed",
        "overview_svg": str(FIGURE_ROOT / "modality_effects_overview.svg"),
        "overview_png": str(FIGURE_ROOT / "modality_effects_overview.png"),
        "local_figure_pairs": len(records),
        "fidelity_svg": str(FIGURE_ROOT / "fidelity_top_vs_random.svg"),
        "palette": {name: color for name, color in zip(LABELS, COLORS)},
        "smoothing": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
