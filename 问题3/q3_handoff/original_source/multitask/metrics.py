"""Explicit three-class metrics and independent, unclipped regression metrics."""
from __future__ import annotations

import numpy as np


CLASS_NAMES = ("negative", "neutral", "positive")
DECISION_RULE = "logits_argmax"


def _finite_vector(values, name):
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite array")
    return result


def labels_from_intensity(values):
    """Only exact zero is neutral; no prediction threshold redefines truth."""
    values = _finite_vector(values, "intensity")
    return np.where(values < 0, 0, np.where(values > 0, 2, 1)).astype(np.int64)


def _classes(values, name):
    values = _finite_vector(values, name)
    if not np.isin(values, (0, 1, 2)).all():
        raise ValueError(f"{name} must contain explicit class IDs 0, 1 or 2")
    return values.astype(np.int64)


def classification_metrics(true_class, pred_class):
    """Compare explicit class IDs, retaining absent classes in macro averaging."""
    truth, prediction = _classes(true_class, "true_class"), _classes(pred_class, "pred_class")
    if truth.shape != prediction.shape:
        raise ValueError("Class arrays must have equal lengths")
    matrix = np.bincount(truth * 3 + prediction, minlength=9).reshape(3, 3)
    support, predicted_count = matrix.sum(axis=1), matrix.sum(axis=0)
    per_class = {}
    for index, name in enumerate(CLASS_NAMES):
        tp = int(matrix[index, index])
        precision = tp / int(predicted_count[index]) if predicted_count[index] else 0.0
        recall = tp / int(support[index]) if support[index] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"class_id": index, "Precision": precision, "Recall": recall,
                           "F1": f1, "support": int(support[index]),
                           "predicted_count": int(predicted_count[index])}
    normalized = np.divide(matrix, support[:, None], out=np.zeros((3, 3), dtype=float),
                           where=support[:, None] != 0)
    neutral_n = int(support[1])
    return {
        "n": int(truth.size),
        "Accuracy": float(np.trace(matrix) / truth.size),
        "Macro_F1": float(np.mean([item["F1"] for item in per_class.values()])),
        "Weighted_F1": float(sum(item["F1"] * item["support"] for item in per_class.values()) / truth.size),
        "decision_rule": DECISION_RULE, "neutral_threshold": None,
        "class_order": list(CLASS_NAMES), "per_class": per_class,
        "true_class_counts": {name: int(support[i]) for i, name in enumerate(CLASS_NAMES)},
        "predicted_class_counts": {name: int(predicted_count[i]) for i, name in enumerate(CLASS_NAMES)},
        "confusion_matrix": matrix.tolist(), "confusion_matrix_normalized": normalized.tolist(),
        "confusion_orientation": "rows=true, columns=predicted", "zero_division": 0,
        "neutral_misdirection": {
            "true_neutral_count": neutral_n, "to_negative_count": int(matrix[1, 0]),
            "to_positive_count": int(matrix[1, 2]),
            "to_negative_rate": float(matrix[1, 0] / neutral_n) if neutral_n else None,
            "to_positive_rate": float(matrix[1, 2] / neutral_n) if neutral_n else None,
            "rate_denominator": "true_neutral_count",
        },
    }


def regression_metrics(y_true, intensity):
    """Use original intensity values, including predictions outside label range."""
    truth = _finite_vector(y_true, "y_true")
    prediction = _finite_vector(intensity, "intensity")
    if truth.shape != prediction.shape:
        raise ValueError("Regression arrays must have equal lengths")
    with np.errstate(over="ignore", invalid="ignore"):
        residual = prediction - truth
        mae = float(np.abs(residual).mean())
    if not np.isfinite(mae):
        raise ValueError("Nonfinite regression residual")
    correlation = None
    if truth.size > 1 and np.any(truth != truth[0]) and np.any(prediction != prediction[0]):
        # Scaling before centering avoids overflow for finite large predictions.
        centered_truth = truth / max(float(np.max(np.abs(truth))), 1.0)
        centered_prediction = prediction / max(float(np.max(np.abs(prediction))), 1.0)
        centered_truth -= centered_truth.mean()
        centered_prediction -= centered_prediction.mean()
        denominator = np.linalg.norm(centered_truth) * np.linalg.norm(centered_prediction)
        if denominator > 0:
            correlation = float(np.clip(np.dot(centered_truth, centered_prediction) / denominator, -1, 1))
    return {"n": int(truth.size), "MAE": mae, "Pearson": correlation,
            "Pearson_status": "defined" if correlation is not None else "undefined_constant_or_singleton",
            "prediction_clipping": False}


def evaluation_metrics(y_true, intensity, pred_class):
    """Class decisions come from logits; regression is never forced to match them."""
    truth = _finite_vector(y_true, "y_true")
    prediction = _finite_vector(intensity, "intensity")
    predicted_class = _classes(pred_class, "pred_class")
    if not (truth.shape == prediction.shape == predicted_class.shape):
        raise ValueError("Intensity and class arrays must have equal lengths")
    disagreement = predicted_class != labels_from_intensity(prediction)
    return {**regression_metrics(truth, prediction),
            **classification_metrics(labels_from_intensity(truth), predicted_class),
            "head_sign_disagreement_count": int(disagreement.sum()),
            "head_sign_disagreement_rate": float(disagreement.mean()),
            "head_sign_disagreement_definition": "pred_class != sign_class(raw_intensity); exact zero is neutral"}
