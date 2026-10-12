"""Accuracy-first checkpoint ordering and its matching early stopping state."""
from __future__ import annotations

from copy import deepcopy
import math

from multitask.config import PROTOCOL_VERSION
from multitask.metrics import CLASS_NAMES, DECISION_RULE


TIE_TOLERANCE = 1e-12
MIXED_CONDITION = "main_mixed/mixed_v1_s0"


def selection_protocol(method):
    if method not in ("M0", "M1", "M2"):
        raise ValueError("Expected method M0, M1 or M2")
    conditions = ["clean"] if method == "M0" else ["clean", MIXED_CONDITION]
    return {"protocol_version": PROTOCOL_VERSION, "selection_split": "valid",
            "conditions": conditions, "condition_weights": [1 / len(conditions)] * len(conditions),
            "ranking": ["Accuracy:max", "Macro_F1:max", "MAE:min", "epoch:min"],
            "tie_tolerance": TIE_TOLERANCE, "decision_rule": DECISION_RULE,
            "class_order": list(CLASS_NAMES), "neutral_threshold": None}


def _metric(value, name):
    if isinstance(value, bool):
        raise ValueError(f"Invalid selection metric {name}")
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"Invalid selection metric {name}") from error
    if not math.isfinite(value) or value < 0 or (name != "MAE" and value > 1):
        raise ValueError(f"Invalid selection metric {name}")
    return value


def selection_score(method, conditions):
    """Equal condition means; no pooled predictions or sample-count weighting."""
    protocol = selection_protocol(method)
    names, weights = protocol["conditions"], protocol["condition_weights"]
    if not isinstance(conditions, dict) or set(conditions) != set(names):
        raise ValueError(f"Selection requires exactly these conditions: {names}")
    sizes = [conditions[name].get("n") for name in names]
    if any(size is not None for size in sizes):
        if any(type(size) is not int or size <= 0 for size in sizes) or len(set(sizes)) != 1:
            raise ValueError("Selection conditions must cover the same nonempty cohort size")
    score = {key: sum(weight * _metric(conditions[name].get(key), key)
                      for name, weight in zip(names, weights))
             for key in ("Accuracy", "Macro_F1", "MAE")}
    return {**score, "conditions": list(names), "condition_weights": list(weights)}


def is_better(candidate, best, tolerance=TIE_TOLERANCE):
    """Lexicographic improvement; all ties retain the earlier checkpoint."""
    if isinstance(tolerance, bool) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    candidate_values = [_metric(candidate.get(key), key) for key in ("Accuracy", "Macro_F1", "MAE")]
    if best is None:
        return True
    best_values = [_metric(best.get(key), key) for key in ("Accuracy", "Macro_F1", "MAE")]
    for current, previous, direction in zip(candidate_values, best_values, (1, 1, -1)):
        delta = direction * (current - previous)
        if delta > tolerance:
            return True
        if delta < -tolerance:
            return False
    return False


class EarlyStopping:
    """Patience counts epochs without an improvement in the complete ordering."""

    def __init__(self, patience=8):
        if type(patience) is not int or patience < 1:
            raise ValueError("patience must be a positive integer")
        self.patience = patience
        self.best = None
        self.best_epoch = None
        self.bad_epochs = 0
        self.last_epoch = 0

    def update(self, score, epoch):
        if type(epoch) is not int or epoch <= self.last_epoch:
            raise ValueError("Epochs must be positive and strictly increasing")
        improved = is_better(score, self.best)
        self.last_epoch = epoch
        if improved:
            self.best, self.best_epoch = deepcopy(score), epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self):
        return self.bad_epochs >= self.patience
