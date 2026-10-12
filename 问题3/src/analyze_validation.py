from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Q3_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Q3_ROOT / "q3_handoff" / "checkpoint"
INPUT_CSV = MODEL_ROOT / "valid__clean.csv"
EVALUATION_JSON = MODEL_ROOT / "evaluation.json"
HISTORY_JSON = MODEL_ROOT / "history.json"
FIGURE_ROOT = Q3_ROOT / "results" / "figures" / "validation"
TABLE_ROOT = Q3_ROOT / "results" / "tables"
OUTPUT_JSON = Q3_ROOT / "results" / "validation" / "m1_validation_analysis.json"

CLASS_NAMES = ["Negative", "Neutral", "Positive"]
COLORS = ["#5A76A8", "#747474", "#C96A6A"]


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.8,
        "savefig.dpi": 600,
        "svg.fonttype": "none",
    })


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def safe_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def class_metrics(cm: np.ndarray) -> list[dict]:
    result = []
    for index, name in enumerate(CLASS_NAMES):
        tp = float(cm[index, index])
        support = float(cm[index].sum())
        predicted = float(cm[:, index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        result.append({
            "class_id": index,
            "class_name": name.lower(),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
            "predicted_count": int(predicted),
        })
    return result


def plot_confusion(cm: np.ndarray) -> None:
    normalized = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), constrained_layout=True)
    for panel, (ax, values, title, fmt) in enumerate((
        (axes[0], cm, "Counts", "d"),
        (axes[1], normalized, "Row-normalized", ".1%"),
    )):
        image = ax.imshow(values, cmap="Blues", vmin=0, vmax=(cm.max() if panel == 0 else 1.0))
        for row in range(3):
            for column in range(3):
                value = values[row, column]
                label = f"{int(value)}" if fmt == "d" else f"{value:.1%}"
                color = "white" if value > (values.max() * 0.52) else "#222222"
                ax.text(column, row, label, ha="center", va="center", color=color, fontsize=8)
        ax.set_xticks(range(3), CLASS_NAMES)
        ax.set_yticks(range(3), CLASS_NAMES)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.text(-0.16, 1.05, chr(ord("A") + panel), transform=ax.transAxes,
                fontweight="bold", fontsize=10)
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.035)
    save(fig, FIGURE_ROOT / "m1_validation_confusion_matrix")


def plot_class_metrics(metrics: list[dict]) -> None:
    x = np.arange(3)
    width = 0.23
    fig, ax = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)
    for offset, key, label, color in (
        (-width, "precision", "Precision", "#5A76A8"),
        (0.0, "recall", "Recall", "#4FA38F"),
        (width, "f1", "F1", "#C96A6A"),
    ):
        values = [entry[key] for entry in metrics]
        bars = ax.bar(x + offset, values, width=width, color=color, alpha=0.82, label=label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_xticks(x, CLASS_NAMES)
    ax.set_ylim(0, 0.86)
    ax.set_ylabel("Score")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, FIGURE_ROOT / "m1_validation_class_metrics")


def plot_regression(df: pd.DataFrame, mae_by_class: list[float], pearson: float) -> None:
    residual = df["y_pred"].to_numpy() - df["y_true"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.8), constrained_layout=True)

    ax = axes[0, 0]
    for class_id, (name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
        part = df[df["true_class"] == class_id]
        ax.scatter(part["y_true"], part["y_pred"], s=11, alpha=0.62, color=color,
                   edgecolors="none", label=f"{name} (n={len(part)})")
    low = float(min(df["y_true"].min(), df["y_pred"].min()))
    high = float(max(df["y_true"].max(), df["y_pred"].max()))
    ax.plot([low, high], [low, high], color="#222222", linewidth=1.0, linestyle="--")
    ax.set_xlabel("True intensity")
    ax.set_ylabel("Predicted intensity")
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.98, 0.04, f"Pearson r = {pearson:.3f}", transform=ax.transAxes,
            ha="right", va="bottom")

    ax = axes[0, 1]
    ax.hist(residual, bins=22, color="#8573B5", alpha=0.62, edgecolor="#574778", linewidth=0.7)
    ax.axvline(0, color="#222222", linewidth=1.0, linestyle="--")
    ax.axvline(float(np.mean(residual)), color="#C96A6A", linewidth=1.2,
               label=f"Mean = {np.mean(residual):.3f}")
    ax.set_xlabel("Residual (prediction - truth)")
    ax.set_ylabel("Samples")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.scatter(df["y_true"], residual, s=10, alpha=0.48, color="#5A76A8", edgecolors="none")
    ax.axhline(0, color="#222222", linewidth=1.0, linestyle="--")
    ax.set_xlabel("True intensity")
    ax.set_ylabel("Residual")

    ax = axes[1, 1]
    bars = ax.bar(CLASS_NAMES, mae_by_class, color=COLORS, alpha=0.82)
    for bar, value in zip(bars, mae_by_class):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}",
                ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, max(mae_by_class) * 1.18)
    ax.set_ylabel("Mean absolute error")

    for index, ax in enumerate(axes.ravel()):
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.16, 1.04, chr(ord("A") + index), transform=ax.transAxes,
                fontweight="bold", fontsize=10)
    save(fig, FIGURE_ROOT / "m1_validation_regression_diagnostics")


def plot_error_diagnostics(df: pd.DataFrame) -> None:
    correct = df["correct"].to_numpy(dtype=bool)
    confidence = df["confidence"].to_numpy()
    abs_error = df["abs_error"].to_numpy()
    head_agree = df["head_agree"].to_numpy(dtype=bool)
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.3), constrained_layout=True)

    axes[0, 0].boxplot([confidence[correct], confidence[~correct]], tick_labels=["Correct", "Incorrect"],
                       patch_artist=True,
                       boxprops={"facecolor": "#74A8C6", "alpha": 0.65},
                       medianprops={"color": "#222222"})
    axes[0, 0].set_ylabel("Maximum class probability")

    axes[0, 1].boxplot([abs_error[correct], abs_error[~correct]], tick_labels=["Correct", "Incorrect"],
                       patch_artist=True,
                       boxprops={"facecolor": "#D1A14B", "alpha": 0.65},
                       medianprops={"color": "#222222"})
    axes[0, 1].set_ylabel("Absolute intensity error")

    groups = [head_agree, ~head_agree]
    values = [correct[group].mean() for group in groups]
    bars = axes[1, 0].bar(["Heads agree", "Heads disagree"], values,
                          color=["#4FA38F", "#C96A6A"], alpha=0.82)
    for bar, value in zip(bars, values):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.1%}",
                        ha="center", va="bottom")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_ylabel("Classification accuracy")

    error_counts = np.zeros((3, 3), dtype=int)
    for true, pred in zip(df.loc[~df["correct"], "true_class"], df.loc[~df["correct"], "pred_class"]):
        error_counts[int(true), int(pred)] += 1
    labels, counts = [], []
    for true in range(3):
        for pred in range(3):
            if true != pred and error_counts[true, pred] > 0:
                short_names = ["Neg", "Neu", "Pos"]
                labels.append(f"{short_names[true]}→{short_names[pred]}")
                counts.append(error_counts[true, pred])
    order = np.argsort(counts)[::-1]
    labels = [labels[index] for index in order]
    counts = [counts[index] for index in order]
    bars = axes[1, 1].bar(labels, counts, color="#8573B5", alpha=0.78)
    for bar, value in zip(bars, counts):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, value + 1.2, str(value),
                        ha="center", va="bottom", fontsize=7)
    axes[1, 1].set_ylabel("Misclassified samples")
    axes[1, 1].set_xlabel("True → predicted class")
    for label in axes[1, 1].get_xticklabels():
        label.set_rotation(28)
        label.set_ha("right")
        label.set_fontsize(6.8)

    for index, ax in enumerate(axes.ravel()):
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.16, 1.04, chr(ord("A") + index), transform=ax.transAxes,
                fontweight="bold", fontsize=10)
    save(fig, FIGURE_ROOT / "m1_validation_error_diagnostics")


def plot_training(history: list[dict]) -> None:
    epochs = np.array([entry["epoch"] for entry in history])
    accuracy = np.array([entry["valid"]["clean"]["Accuracy"] for entry in history])
    macro_f1 = np.array([entry["valid"]["clean"]["Macro_F1"] for entry in history])
    mae = np.array([entry["valid"]["clean"]["MAE"] for entry in history])
    pearson = np.array([entry["valid"]["clean"]["Pearson"] for entry in history])
    best_epoch = int(history[-1]["best_epoch"])
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    axes[0].plot(epochs, accuracy, marker="o", markersize=3, color="#5A76A8", label="Accuracy")
    axes[0].plot(epochs, macro_f1, marker="o", markersize=3, color="#C96A6A", label="Macro-F1")
    axes[1].plot(epochs, mae, marker="o", markersize=3, color="#D1A14B", label="MAE")
    axes[1].plot(epochs, pearson, marker="o", markersize=3, color="#4FA38F", label="Pearson")
    for index, ax in enumerate(axes):
        ax.axvline(best_epoch, color="#222222", linestyle="--", linewidth=1.0,
                   label=f"Selected epoch {best_epoch}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric value")
        ax.legend(frameon=False, ncol=1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.13, 1.04, chr(ord("A") + index), transform=ax.transAxes,
                fontweight="bold", fontsize=10)
    save(fig, FIGURE_ROOT / "m1_training_selection_curves")


def main() -> None:
    configure_style()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    expected = {"split", "index", "id", "y_true", "y_pred", "true_class", "pred_class",
                "p_negative", "p_neutral", "p_positive"}
    if set(df.columns) != expected or len(df) != 728:
        raise ValueError(f"Unexpected validation schema or row count: {df.shape}, {list(df.columns)}")
    if df["id"].duplicated().any():
        raise ValueError("Validation IDs are not unique")

    probability_columns = ["p_negative", "p_neutral", "p_positive"]
    probabilities = df[probability_columns].to_numpy(dtype=float)
    if not np.all(np.isfinite(df[["y_true", "y_pred"]].to_numpy(dtype=float))):
        raise ValueError("Non-finite regression values")
    probability_error = float(np.max(np.abs(probabilities.sum(axis=1) - 1.0)))
    if probability_error > 1e-6:
        raise ValueError(f"Probability rows do not sum to one: {probability_error}")

    true_class = df["true_class"].to_numpy(dtype=int)
    pred_class = df["pred_class"].to_numpy(dtype=int)
    if not np.array_equal(probabilities.argmax(axis=1), pred_class):
        raise ValueError("pred_class differs from probability argmax")

    cm = np.zeros((3, 3), dtype=int)
    for true, pred in zip(true_class, pred_class):
        cm[true, pred] += 1
    metrics = class_metrics(cm)
    correct = true_class == pred_class
    residual = df["y_pred"].to_numpy(dtype=float) - df["y_true"].to_numpy(dtype=float)
    abs_error = np.abs(residual)
    confidence = probabilities.max(axis=1)
    pred_sign_class = np.where(df["y_pred"].to_numpy() < 0, 0,
                               np.where(df["y_pred"].to_numpy() > 0, 2, 1))
    head_agree = pred_sign_class == pred_class

    df["correct"] = correct
    df["confidence"] = confidence
    df["residual"] = residual
    df["abs_error"] = abs_error
    df["regression_sign_class"] = pred_sign_class
    df["head_agree"] = head_agree
    df["error_transition"] = ["correct" if c else f"{CLASS_NAMES[t].lower()}_to_{CLASS_NAMES[p].lower()}"
                              for t, p, c in zip(true_class, pred_class, correct)]

    evaluation = json.loads(EVALUATION_JSON.read_text(encoding="utf-8"))["valid"]["clean"]
    accuracy = float(correct.mean())
    macro_f1 = float(np.mean([entry["f1"] for entry in metrics]))
    weighted_f1 = float(sum(entry["f1"] * entry["support"] for entry in metrics) / len(df))
    mae = float(abs_error.mean())
    pearson = float(np.corrcoef(df["y_true"], df["y_pred"])[0, 1])
    for key, value in (("Accuracy", accuracy), ("Macro_F1", macro_f1),
                       ("Weighted_F1", weighted_f1), ("MAE", mae), ("Pearson", pearson)):
        if abs(value - float(evaluation[key])) > 1e-12:
            raise ValueError(f"Recomputed {key} differs from evaluation.json")

    mae_by_class = [float(abs_error[true_class == class_id].mean()) for class_id in range(3)]
    median_confidence_correct = float(np.median(confidence[correct]))
    median_confidence_incorrect = float(np.median(confidence[~correct]))
    mean_abs_error_correct = float(abs_error[correct].mean())
    mean_abs_error_incorrect = float(abs_error[~correct].mean())
    accuracy_heads_agree = float(correct[head_agree].mean())
    accuracy_heads_disagree = float(correct[~head_agree].mean())
    high_confidence_errors = df.loc[(~df["correct"]) & (df["confidence"] >= 0.8)].copy()
    high_confidence_errors = high_confidence_errors.sort_values(
        ["confidence", "abs_error"], ascending=[False, False]
    )

    sample_output = TABLE_ROOT / "m1_validation_sample_analysis.csv"
    error_output = TABLE_ROOT / "m1_validation_error_cases.csv"
    df.to_csv(sample_output, index=False, encoding="utf-8-sig")
    df.loc[~df["correct"]].sort_values(["confidence", "abs_error"], ascending=[False, False]).to_csv(
        error_output, index=False, encoding="utf-8-sig"
    )

    history = json.loads(HISTORY_JSON.read_text(encoding="utf-8"))
    plot_confusion(cm)
    plot_class_metrics(metrics)
    plot_regression(df, mae_by_class, pearson)
    plot_error_diagnostics(df)
    plot_training(history)

    report = {
        "status": "passed",
        "model_role": "M1 complete-input joint polarity-intensity predictor used by Problem 3",
        "no_retraining": True,
        "source_prediction_file": str(INPUT_CSV),
        "sample_count": int(len(df)),
        "metrics": {
            "Accuracy": accuracy,
            "Macro_F1": macro_f1,
            "Weighted_F1": weighted_f1,
            "MAE": mae,
            "Pearson": pearson,
        },
        "per_class": metrics,
        "confusion_matrix": cm.tolist(),
        "confusion_orientation": "rows=true, columns=predicted; order=negative, neutral, positive",
        "regression": {
            "residual_mean": float(residual.mean()),
            "residual_median": float(np.median(residual)),
            "residual_std": float(residual.std(ddof=0)),
            "mae_by_true_class": {name.lower(): value for name, value in zip(CLASS_NAMES, mae_by_class)},
        },
        "error_diagnostics": {
            "classification_error_count": int((~correct).sum()),
            "classification_error_rate": float((~correct).mean()),
            "neutral_to_positive_count": int(cm[1, 2]),
            "neutral_to_positive_rate_among_true_neutral": float(cm[1, 2] / cm[1].sum()),
            "neutral_to_negative_count": int(cm[1, 0]),
            "head_disagreement_count": int((~head_agree).sum()),
            "head_disagreement_rate": float((~head_agree).mean()),
            "accuracy_when_heads_agree": accuracy_heads_agree,
            "accuracy_when_heads_disagree": accuracy_heads_disagree,
            "median_confidence_correct": median_confidence_correct,
            "median_confidence_incorrect": median_confidence_incorrect,
            "mean_abs_intensity_error_correct_classification": mean_abs_error_correct,
            "mean_abs_intensity_error_incorrect_classification": mean_abs_error_incorrect,
            "high_confidence_error_threshold": 0.8,
            "high_confidence_error_count": int(len(high_confidence_errors)),
            "top_high_confidence_error_ids": high_confidence_errors["id"].head(10).tolist(),
        },
        "integrity": {
            "probability_sum_max_error": probability_error,
            "metrics_match_evaluation_json_tolerance": 1e-12,
            "unique_ids": True,
        },
        "figures": [
            str(FIGURE_ROOT / "m1_validation_confusion_matrix.svg"),
            str(FIGURE_ROOT / "m1_validation_class_metrics.svg"),
            str(FIGURE_ROOT / "m1_validation_regression_diagnostics.svg"),
            str(FIGURE_ROOT / "m1_validation_error_diagnostics.svg"),
            str(FIGURE_ROOT / "m1_training_selection_curves.svg"),
        ],
        "tables": [str(sample_output), str(error_output)],
        "claim_boundary": (
            "This is a descriptive validation-error diagnosis of frozen M1 outputs. "
            "Associations between confidence, head agreement and errors are not causal attributions."
        ),
    }
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
