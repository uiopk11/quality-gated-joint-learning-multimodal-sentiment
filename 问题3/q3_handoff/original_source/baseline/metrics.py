"""Whole-split regression and three-class metrics on raw predictions.

Classification evaluation implemented with AI assistance (OpenAI Codex,
2026-09-24). No threshold changes the continuous labels or predictions.
"""
import numpy as np


CLASS_NAMES = ("negative", "neutral", "positive")


def _arrays(y_true, y_pred):
    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    prediction = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if truth.size == 0 or truth.shape != prediction.shape:
        raise ValueError("Expected equally sized, non-empty regression arrays")
    if not (np.isfinite(truth).all() and np.isfinite(prediction).all()):
        raise ValueError("Non-finite label or prediction")
    return truth, prediction


def regression_metrics(y_true, y_pred):
    truth, prediction = _arrays(y_true, y_pred)
    residual = prediction - truth
    correlation = None
    if truth.size > 1 and np.std(truth) > 0 and np.std(prediction) > 0:
        correlation = float(np.corrcoef(truth, prediction)[0, 1])
    return {
        "n": int(truth.size),
        "MAE": float(np.abs(residual).mean()),
        "RMSE": float(np.sqrt(np.square(residual).mean())),
        "Pearson": correlation,
        "prediction_clipping": False,
    }


def class_labels(values, neutral_threshold=0.0):
    """Map to 0=negative, 1=neutral, 2=positive, inclusive at +/-threshold.

Call with zero for true labels: only exact zero is a neutral target.
"""
    if (isinstance(neutral_threshold, bool) or not np.isscalar(neutral_threshold)
            or not np.isfinite(neutral_threshold) or neutral_threshold < 0):
        raise ValueError("neutral_threshold must be finite and nonnegative")
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError("Non-finite label or prediction")
    return np.where(array < -neutral_threshold, 0, np.where(array > neutral_threshold, 2, 1))


def classification_metrics(y_true, y_pred, neutral_threshold):
    """Keep all three classes in macro averaging, including absent classes."""
    truth, prediction = _arrays(y_true, y_pred)
    actual = class_labels(truth)
    predicted = class_labels(prediction, neutral_threshold)
    matrix = np.bincount(actual * 3 + predicted, minlength=9).reshape(3, 3)
    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    per_class = {}
    for index, name in enumerate(CLASS_NAMES):
        true_positive = int(matrix[index, index])
        precision = true_positive / int(predicted_count[index]) if predicted_count[index] else 0.0
        recall = true_positive / int(support[index]) if support[index] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"class_id": index, "Precision": precision, "Recall": recall,
                           "F1": f1, "support": int(support[index]),
                           "predicted_count": int(predicted_count[index])}
    normalized = np.divide(matrix, support[:, None], out=np.zeros((3, 3), dtype=float),
                           where=support[:, None] != 0)
    return {
        "Accuracy": float(np.trace(matrix) / truth.size),
        "Macro_F1": float(np.mean([item["F1"] for item in per_class.values()])),
        "Weighted_F1": float(sum(item["F1"] * item["support"] for item in per_class.values()) / truth.size),
        "neutral_threshold": float(neutral_threshold), "class_order": list(CLASS_NAMES),
        "per_class": per_class, "confusion_matrix": matrix.tolist(),
        "confusion_matrix_normalized": normalized.tolist(),
        "confusion_orientation": "rows=true, columns=predicted", "zero_division": 0,
    }


def evaluation_metrics(y_true, y_pred, neutral_threshold):
    return {**regression_metrics(y_true, y_pred),
            **classification_metrics(y_true, y_pred, neutral_threshold)}
