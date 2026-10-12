from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from transformers import AutoTokenizer


Q2_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Q2_ROOT.parents[1]
FIGURE_DIR = Q2_ROOT / "cursor_draw" / "Figure"
LATEX_FIGURE_DIR = Q2_ROOT / "figures"

SOURCE_NPZ = Q2_ROOT / "resources" / "missing_data" / "mix_v1" / "source" / "valid.npz"
MASK_NPZ = (
    Q2_ROOT
    / "resources"
    / "missing_data"
    / "mix_v1"
    / "masks"
    / "main_mixed"
    / "valid"
    / "mixed_v1_s0.npz"
)
MASK_METADATA = (
    Q2_ROOT
    / "resources"
    / "missing_data"
    / "mix_v1"
    / "metadata"
    / "main_mixed"
    / "valid"
    / "mixed_v1_s0.jsonl"
)
PREDICTION_DIR = (
    Q2_ROOT
    / "reproducibility"
    / "reporting_revision"
    / "prior_outputs"
    / "fm_v1_m3_s1111"
    / "details"
)
CLEAN_CSV = PREDICTION_DIR / "valid__clean.csv"
MIXED_CSV = PREDICTION_DIR / "valid__main_mixed__mixed_v1_s0.csv"
TOKENIZER_DIR = Q2_ROOT / "resources" / "pretrained" / "bert-base-uncased"

CLASS_NAMES = {0: "NEG", 1: "NEU", 2: "POS"}
CASES = [
    {
        "panel": "A",
        "index": 15,
        "sample_id": "102408$_$9",
        "text": "Maybe it's just because I don't have little boys so I'm not really into the little (uhh) characters like that",
        "expected_intervals": [[5, 10], [13, 20]],
        "subtitle": "pronounced degradation",
    },
    {
        "panel": "C",
        "index": 13,
        "sample_id": "102408$_$2",
        "text": "(umm) not necessarily by choice",
        "expected_intervals": [[5, 6]],
        "subtitle": "opposite-direction single-sample change",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_predictions(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {int(row["index"]): row for row in rows}


def content_missing_positions(mask: np.ndarray, index: int, length: int) -> list[list[int]]:
    # Mask coordinates retain [CLS] at slot 0; content coordinate j maps to tensor slot j+1.
    positions_by_modality = []
    for modality_index in range(3):
        positions = np.flatnonzero(mask[index, modality_index, 1 : length + 1]).tolist()
        positions_by_modality.append(positions)
    if not (positions_by_modality[0] == positions_by_modality[1] == positions_by_modality[2]):
        raise ValueError(f"Case {index} is not T/A/V synchronous")
    return positions_by_modality


def intervals_from_positions(positions: list[int]) -> list[list[int]]:
    if not positions:
        return []
    intervals = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position != previous + 1:
            intervals.append([start, previous + 1])
            start = position
        previous = position
    intervals.append([start, previous + 1])
    return intervals


def prediction_payload(row: dict[str, str]) -> dict[str, object]:
    return {
        "class": int(row["pred_class"]),
        "class_name": CLASS_NAMES[int(row["pred_class"])],
        "intensity": float(row["y_pred"]),
        "probabilities": {
            "negative": float(row["p_negative"]),
            "neutral": float(row["p_neutral"]),
            "positive": float(row["p_positive"]),
        },
    }


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def draw_mask_panel(ax, case: dict[str, object], tokens: list[str], missing: list[int]) -> None:
    length = len(tokens)
    state = np.zeros((3, length), dtype=int)
    state[:, missing] = 1
    cmap = ListedColormap(["#EAF0F4", "#D86A68"])
    ax.imshow(state, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")

    # Cell boundaries keep every model-visible coordinate explicit.
    ax.set_xticks(np.arange(-0.5, length, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlim(-0.5, length - 0.5)
    ax.set_ylim(2.5, -0.5)
    ax.set_yticks([0, 1, 2], ["Text -> [UNK]", "Audio -> zero", "Vision -> zero"])

    ax.set_xticks(np.arange(length))
    ax.set_xticklabels([str(i) for i in range(length)])
    ax.tick_params(axis="x", bottom=True, labelbottom=True, top=False, pad=2)
    ax.set_xlabel("Zero-based content / numerical-feature position", labelpad=4)

    token_axis = ax.secondary_xaxis("top")
    token_axis.set_xticks(np.arange(length))
    token_axis.set_xticklabels(tokens, rotation=58, ha="left", rotation_mode="anchor")
    token_axis.tick_params(axis="x", length=0, pad=2, labelsize=7.0 if length > 12 else 8.3)
    token_axis.set_xlabel("Model-visible WordPiece slots", labelpad=9)

    intervals = case["expected_intervals"]
    interval_text = ", ".join(f"[{start},{end})" for start, end in intervals)
    display_sample_id = str(case["sample_id"]).replace("$", r"\$")
    ax.set_title(
        f"{case['panel']}  {case['subtitle']}   |   valid[{case['index']}]  {display_sample_id}\n"
        f"Synchronous T/A/V artificial gaps: {interval_text}",
        loc="left",
        pad=29,
        fontweight="bold",
    )


def draw_prediction_panel(ax, truth: float, clean: dict[str, object], mixed: dict[str, object]) -> None:
    y_positions = [2, 1, 0]
    values = [truth, clean["intensity"], mixed["intensity"]]
    colors = ["#333333", "#4F76A8", "#D45F5F"]
    labels = ["Truth", "Clean view", "Missing view"]
    classes = ["NEG", clean["class_name"], mixed["class_name"]]

    ax.axvline(0, color="#9B9B9B", linewidth=1.0, linestyle=(0, (3, 3)), zorder=0)
    ax.annotate(
        "",
        xy=(mixed["intensity"], 0),
        xytext=(clean["intensity"], 1),
        arrowprops={"arrowstyle": "-|>", "color": "#777777", "lw": 1.3},
        zorder=1,
    )
    for y, value, color, label, class_name in zip(y_positions, values, colors, labels, classes):
        ax.hlines(y, min(0, value), max(0, value), color=color, linewidth=2.0, alpha=0.75)
        ax.scatter(value, y, s=58, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(
            value,
            y + 0.22,
            f"{class_name}  {value:+.3f}",
            color=color,
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold" if label == "Missing view" else "normal",
        )

    delta = mixed["intensity"] - clean["intensity"]
    ax.text(
        0.98,
        0.04,
        f"Delta = {delta:+.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#444444",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F6F6F6", "edgecolor": "#D0D0D0"},
    )
    ax.set_yticks(y_positions, labels)
    ax.set_ylim(-0.55, 2.65)
    ax.set_xlim(-2.05, 0.55)
    ax.set_xlabel("Emotion intensity")
    ax.set_title("Frozen M3 predictions", loc="left", pad=8, fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E3E3E3", linewidth=0.7, zorder=0)


def main() -> None:
    configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    source = np.load(SOURCE_NPZ, allow_pickle=False)
    mask_data = np.load(MASK_NPZ, allow_pickle=False)
    missing_mask = mask_data["missing"]
    metadata_rows = [json.loads(line) for line in MASK_METADATA.read_text(encoding="utf-8").splitlines()]
    clean_rows = load_predictions(CLEAN_CSV)
    mixed_rows = load_predictions(MIXED_CSV)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True, use_fast=True)

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13.6, 7.7),
        gridspec_kw={"width_ratios": [4.9, 1.45], "hspace": 1.18, "wspace": 0.22},
    )
    audit_cases = []

    for row_index, case in enumerate(CASES):
        index = int(case["index"])
        sample_id = str(case["sample_id"])
        length = int(source["lengths"][index])
        if str(source["ids"][index]) != sample_id:
            raise ValueError(f"Source ID mismatch at valid[{index}]")

        encoded = tokenizer(str(case["text"]), add_special_tokens=True, truncation=True, max_length=50)
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])[1:-1]
        if len(tokens) != length:
            raise ValueError(f"Token count mismatch for {sample_id}: tokenizer={len(tokens)}, source={length}")

        positions_by_modality = content_missing_positions(missing_mask, index, length)
        missing_positions = positions_by_modality[0]
        intervals = intervals_from_positions(missing_positions)
        if intervals != case["expected_intervals"]:
            raise ValueError(f"Mask intervals mismatch for {sample_id}: {intervals}")
        metadata = metadata_rows[index]
        for modality in ("T", "A", "V"):
            if metadata["modalities"][modality]["intervals"] != intervals:
                raise ValueError(f"Metadata mismatch for {sample_id}/{modality}")

        clean_row = clean_rows[index]
        mixed_row = mixed_rows[index]
        if clean_row["id"] != sample_id or mixed_row["id"] != sample_id:
            raise ValueError(f"Prediction ID mismatch at valid[{index}]")
        clean = prediction_payload(clean_row)
        mixed = prediction_payload(mixed_row)
        truth = float(clean_row["y_true"])

        draw_mask_panel(axes[row_index, 0], case, tokens, missing_positions)
        draw_prediction_panel(axes[row_index, 1], truth, clean, mixed)

        audit_cases.append(
            {
                "panel": case["panel"],
                "valid_index": index,
                "sample_id": sample_id,
                "raw_text_from_official_label_xlsx": case["text"],
                "content_length": length,
                "wordpieces_zero_based": tokens,
                "synchronous_missing_positions_zero_based": missing_positions,
                "synchronous_missing_intervals_zero_based_half_open": intervals,
                "masked_text_replacement": "[UNK]",
                "masked_audio_replacement": "all-zero numerical row",
                "masked_vision_replacement": "all-zero numerical row",
                "truth": {"class": 0, "class_name": "NEG", "intensity": truth},
                "clean_prediction": clean,
                "missing_prediction": mixed,
                "intensity_change_missing_minus_clean": mixed["intensity"] - clean["intensity"],
            }
        )

    legend = [
        Patch(facecolor="#EAF0F4", edgecolor="#B7C4CD", label="Observed model input"),
        Patch(facecolor="#D86A68", edgecolor="#B24B49", label="Artificially missing"),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.39, 0.012))
    figure.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.095)

    outputs = []
    for directory in (FIGURE_DIR, LATEX_FIGURE_DIR):
        for suffix in (".png", ".svg", ".pdf"):
            path = directory / f"q2_validation_cases{suffix}"
            if suffix == ".png":
                figure.savefig(path, dpi=320, bbox_inches="tight", pad_inches=0.06)
            else:
                figure.savefig(path, bbox_inches="tight", pad_inches=0.06)
            outputs.append(str(path))
    plt.close(figure)

    audit = {
        "status": "passed",
        "figure_purpose": "Frozen-valid comparison for Problem 2 cases A and C",
        "model": "M3 / JPI-DLFR",
        "coordinate_rule": "zero-based content coordinates; content j maps to tensor slot j+1 because slot 0 is [CLS]",
        "claim_boundary": "No raw waveform, video frame, or timestamp is inferred; only model-visible WordPieces and aligned numerical-feature positions are shown.",
        "source_files": {
            str(path): sha256(path)
            for path in (SOURCE_NPZ, MASK_NPZ, MASK_METADATA, CLEAN_CSV, MIXED_CSV, TOKENIZER_DIR / "vocab.txt")
        },
        "cases": audit_cases,
        "outputs": outputs,
    }
    audit_path = FIGURE_DIR / "q2_validation_cases_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "audit": str(audit_path), "outputs": outputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
