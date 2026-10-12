"""Common paired validation conditions, ranking and early stopping."""
from __future__ import annotations

from src.models.four_models import is_missing, method_name
from src.training.four_model_config import (
    CLASS_ORDER, PROTOCOL_VERSION, TIE_TOLERANCE, decision_rule,
)
from src.training.selection import EarlyStopping, _metric, is_better


MIXED_CONDITION = "main_mixed/mixed_v1_s0"


def selection_protocol(method):
    method = method_name(method)
    conditions = ["clean", MIXED_CONDITION] if is_missing(method) else ["clean"]
    return {"protocol_version": PROTOCOL_VERSION, "selection_split": "valid",
            "conditions": conditions, "condition_weights": [1 / len(conditions)] * len(conditions),
            "ranking": ["Accuracy:max", "Macro_F1:max", "MAE:min", "epoch:min"],
            "tie_tolerance": TIE_TOLERANCE, "decision_rule": decision_rule(method),
            "class_order": list(CLASS_ORDER), "neutral_threshold": None}


def selection_score(method, conditions):
    """Equal per-condition means over the same paired valid cohort."""
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
