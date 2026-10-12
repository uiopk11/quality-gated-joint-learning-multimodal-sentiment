"""Four-role predictions with one valid-fitted, frozen regression decision.

Every record retains its input condition and source identity. Regression
baselines do not manufacture class probabilities; their classes use the saved
symmetric neutral interval. Joint models use logits independently of intensity.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from src.evaluation.multitask import INPUT_FIELDS, PROBABILITY_FIELDS, _finite, _integer, _numpy, save_json
from src.evaluation.multitask_metrics import CLASS_NAMES, classification_metrics, evaluation_metrics, labels_from_intensity
from src.training.four_model_config import (CLASS_ORDER, JOINT_DECISION_RULE,
    REGRESSION_DECISION_RULE, decision_rule, is_joint)
from src.training.four_model_selection import selection_protocol


FIELDS = ("split", "condition", "method", "index", "id", "y_true", "y_pred",
          "true_class", "pred_class", *PROBABILITY_FIELDS, "decision")
_DECISION_FIELDS = {"method", "decision_rule", "class_order", "neutral_threshold",
    "calibration_split", "calibration_conditions", "calibration_objective",
    "threshold_candidates", "threshold_boundary", "calibration_identity_sha256"}
_OBJECTIVE = ["mean_condition_Accuracy:max", "mean_condition_Macro_F1:max", "threshold:min"]


def fingerprint(value):
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def joint_decision(method):
    if not is_joint(method):
        raise ValueError("Regression baselines require a valid-fitted decision")
    return {"method": method, "decision_rule": JOINT_DECISION_RULE,
            "class_order": list(CLASS_ORDER), "neutral_threshold": None,
            "calibration_split": None, "calibration_conditions": [],
            "calibration_objective": None, "threshold_candidates": None,
            "threshold_boundary": None, "calibration_identity_sha256": None}


def validate_decision(decision, method=None):
    if not isinstance(decision, dict) or set(decision) != _DECISION_FIELDS:
        raise ValueError("Missing or unexpected frozen decision fields")
    method = decision["method"] if method is None else method
    if method not in ("M0", "M1", "M2", "M3"):
        raise ValueError("Unknown four-model role")
    if (decision["method"] != method or decision["decision_rule"] != decision_rule(method)
            or decision["class_order"] != CLASS_ORDER):
        raise ValueError("Decision rule disagrees with the four-model role")
    if is_joint(method):
        if decision != joint_decision(method):
            raise ValueError("Joint prediction requires logits argmax without calibration")
    else:
        threshold = _finite(decision["neutral_threshold"], "neutral_threshold")
        digest = decision["calibration_identity_sha256"]
        if (threshold < 0 or decision["calibration_split"] != "valid"
                or decision["calibration_conditions"] != selection_protocol(method)["conditions"]
                or decision["calibration_objective"] != _OBJECTIVE
                or decision["threshold_candidates"] != "zero_and_unique_abs_valid_predictions"
                or decision["threshold_boundary"] != "abs(raw_intensity)<=threshold_is_neutral"
                or not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError("Regression decision must be fitted once on the prescribed valid conditions")
    fingerprint(decision)
    return copy.deepcopy(decision)


def _base_records(records, split=None, cohort=None, *, full=False):
    records = list(records)
    if not records or not isinstance(records[0], dict):
        raise ValueError("Expected nonempty four-model prediction records")
    split = records[0].get("split") if split is None else split
    condition, method = records[0].get("condition"), records[0].get("method")
    if split not in ("train", "valid", "test"):
        raise ValueError("Unknown prediction split")
    if method not in ("M0", "M1", "M2", "M3"):
        raise ValueError("Unknown four-model role")
    if not isinstance(condition, str) or not condition.strip():
        raise ValueError("Prediction condition must be explicit")
    if full and cohort is None:
        raise ValueError("Full coverage validation requires the source cohort")
    source = None
    if cohort is not None:
        source = {}
        for expected, row in enumerate(cohort):
            if _integer(row.get("index"), "cohort index") != expected:
                raise ValueError("Source cohort must retain original full index order")
            source[expected] = row
        if not source or (full and len(records) != len(source)):
            raise ValueError("Prediction records do not cover the full source cohort")
    result, seen_ids, previous = [], set(), -1
    for row in records:
        if (not isinstance(row, dict) or row.get("split") != split
                or row.get("condition") != condition or row.get("method") != method):
            raise ValueError("Prediction records must share split, condition and method")
        index = _integer(row.get("index"), "index")
        if index <= previous:
            raise ValueError("Prediction indices must be unique and in original order")
        previous = index
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in seen_ids:
            raise ValueError("Prediction IDs must be nonempty and unique")
        seen_ids.add(sample_id)
        truth, prediction = _finite(row.get("y_true"), "y_true"), _finite(row.get("y_pred"), "y_pred")
        true_class = _integer(row.get("true_class"), "true_class")
        if true_class != int(labels_from_intensity([truth])[0]):
            raise ValueError("Class target differs from label sign")
        if is_joint(method):
            probabilities = [_finite(row.get(key), key) for key in PROBABILITY_FIELDS]
            if (any(p < 0 or p > 1 for p in probabilities)
                    or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6, rel_tol=0)):
                raise ValueError("Class probabilities must be in [0,1] and sum to one")
        else:
            if any(row.get(key) not in (None, "") for key in PROBABILITY_FIELDS):
                raise ValueError("Regression baselines do not have class probabilities")
            probabilities = [None] * 3
        if source is not None:
            original = source.get(index)
            if (original is None or sample_id != original["id"]
                    or truth != _finite(original["y_true"], "cohort y_true")):
                raise ValueError("Prediction identity/label differs from the source cohort")
        result.append({"split": split, "condition": condition, "method": method,
                       "index": index, "id": sample_id, "y_true": truth,
                       "y_pred": prediction, "true_class": true_class,
                       **dict(zip(PROBABILITY_FIELDS, probabilities))})
    return result


def _predicted_classes(records, decision):
    if is_joint(decision["method"]):
        return np.argmax([[row[key] for key in PROBABILITY_FIELDS] for row in records], axis=1)
    intensity = np.asarray([row["y_pred"] for row in records], dtype=np.float64)
    threshold = decision["neutral_threshold"]
    return np.where(np.abs(intensity) <= threshold, 1, np.where(intensity < 0, 0, 2))


def apply_decision(records, decision):
    """Apply a frozen decision to any input condition without using its labels."""
    records = _base_records(records)
    decision = validate_decision(decision, records[0]["method"])
    predictions = _predicted_classes(records, decision)
    return [{**row, "pred_class": int(predicted), "decision": copy.deepcopy(decision)}
            for row, predicted in zip(records, predictions)]


def validate_records(records, split=None, cohort=None, *, full=False):
    records = list(records)
    base = _base_records(records, split, cohort, full=full)
    decisions = []
    for row in records:
        value = row.get("decision")
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, ValueError) as error:
                raise ValueError("Malformed saved decision") from error
        decisions.append(validate_decision(value, base[0]["method"]))
    if any(value != decisions[0] for value in decisions):
        raise ValueError("Prediction records must use one frozen decision")
    result = apply_decision(base, decisions[0])
    for original, checked in zip(records, result):
        if _integer(original.get("pred_class"), "pred_class") != checked["pred_class"]:
            raise ValueError("pred_class disagrees with the frozen decision")
    return result


def calibrate_decision(records_by_condition, method):
    """Fit only on valid; one threshold jointly covers the selected conditions.

    Searching zero and each unique absolute prediction enumerates every class
    assignment possible under an inclusive symmetric neutral interval. Ties
    retain the smallest threshold. Test/panel recalibration is rejected.
    """
    protocol = selection_protocol(method)
    names = protocol["conditions"]
    if not isinstance(records_by_condition, dict) or set(records_by_condition) != set(names):
        raise ValueError("Decision calibration requires exactly the selected valid conditions")
    groups, identity = {}, None
    for condition in names:
        records = _base_records(records_by_condition[condition], "valid")
        if records[0]["condition"] != condition or records[0]["method"] != method:
            raise ValueError("Calibration condition or model role mismatch")
        current = [(row["index"], row["id"], row["y_true"]) for row in records]
        if identity is not None and current != identity:
            raise ValueError("Calibration conditions require identical paired sample IDs and labels")
        identity = current
        groups[condition] = records
    if is_joint(method):
        return joint_decision(method)
    arrays = [(np.array([row["true_class"] for row in groups[name]]),
               np.array([row["y_pred"] for row in groups[name]])) for name in names]
    candidates = np.unique(np.concatenate(([0.0], *[np.abs(pred) for _, pred in arrays])))
    best_score, best_threshold = None, None
    tolerance = protocol["tie_tolerance"]
    for threshold in candidates:
        scores = []
        for truth, prediction in arrays:
            predicted = np.where(np.abs(prediction) <= threshold, 1, np.where(prediction < 0, 0, 2))
            metrics = classification_metrics(truth, predicted)
            scores.append((metrics["Accuracy"], metrics["Macro_F1"]))
        score = tuple(float(np.mean([value[i] for value in scores])) for i in (0, 1))
        better = best_score is None
        if best_score is not None:
            for current, previous in zip(score, best_score):
                if abs(current - previous) > tolerance:
                    better = current > previous
                    break
        if better:
            best_score, best_threshold = score, float(threshold)
    return validate_decision({"method": method, "decision_rule": REGRESSION_DECISION_RULE,
        "class_order": list(CLASS_ORDER), "neutral_threshold": best_threshold,
        "calibration_split": "valid", "calibration_conditions": list(names),
        "calibration_objective": list(_OBJECTIVE),
        "threshold_candidates": "zero_and_unique_abs_valid_predictions",
        "threshold_boundary": "abs(raw_intensity)<=threshold_is_neutral",
        "calibration_identity_sha256": fingerprint(groups)})


def metrics_from_records(records):
    records = validate_records(records)
    decision = records[0]["decision"]
    metrics = evaluation_metrics([row["y_true"] for row in records],
                                 [row["y_pred"] for row in records],
                                 [row["pred_class"] for row in records])
    metrics.update(split=records[0]["split"], condition=records[0]["condition"],
                   method=records[0]["method"], decision_rule=decision["decision_rule"],
                   neutral_threshold=decision["neutral_threshold"], class_order=list(CLASS_ORDER),
                   decision=copy.deepcopy(decision), decision_sha256=fingerprint(decision))
    return metrics


def collect_records(model, batches, device, split, condition="clean"):
    """Collect current-view outputs; labels never enter the prediction call."""
    import torch
    from src.models.four_models import predict_batch

    method = getattr(model, "four_model_method", getattr(model, "method", None))
    if method not in ("M0", "M1", "M2", "M3"):
        raise ValueError("Unknown four-model role")
    records, was_training = [], model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                output = predict_batch(model, {key: batch[key] for key in INPUT_FIELDS}, device)
                intensity = np.asarray(_numpy(output["intensity"]), dtype=np.float64).reshape(-1)
                truth = np.asarray(_numpy(batch["labels"]), dtype=np.float64).reshape(-1)
                indices, ids = _numpy(batch["index"]).reshape(-1), batch["id"]
                count = len(truth)
                if len(intensity) != count or len(indices) != count or len(ids) != count:
                    raise ValueError("Prediction output/batch shapes disagree")
                probabilities = [[None] * 3 for _ in range(count)]
                if is_joint(method):
                    logits = np.asarray(_numpy(output["logits"]), dtype=np.float64)
                    if logits.shape != (count, 3) or not np.isfinite(logits).all():
                        raise ValueError("Expected finite [batch,3] logits")
                    probabilities = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                    probabilities /= probabilities.sum(axis=1, keepdims=True)
                elif "logits" in output:
                    raise ValueError("Regression baseline unexpectedly returned a classification head")
                true_class = labels_from_intensity(truth)
                for i in range(count):
                    records.append({"split": split, "condition": condition, "method": method,
                        "index": indices[i], "id": ids[i], "y_true": truth[i], "y_pred": intensity[i],
                        "true_class": int(true_class[i]),
                        **dict(zip(PROBABILITY_FIELDS, probabilities[i]))})
    finally:
        model.train(was_training)
    records.sort(key=lambda row: _integer(row["index"], "index"))
    return _base_records(records, split)


def evaluate(model, batches, device, split, decision=None, condition="clean"):
    records = collect_records(model, batches, device, split, condition)
    if decision is None:
        decision = getattr(model, "decision", None)
    if decision is None and is_joint(records[0]["method"]):
        decision = joint_decision(records[0]["method"])
    records = apply_decision(records, decision)
    return metrics_from_records(records), records


def load_records(path, *, split=None, cohort=None, full=False):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Malformed prediction CSV header")
        return validate_records(reader, split, cohort, full=full)


def save_records(path, records, *, overwrite=False):
    path = Path(path)
    records = validate_records(records)
    if path.exists() and not overwrite:
        if load_records(path) != records:
            raise ValueError("Refusing to overwrite different prediction records")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in records:
            writer.writerow({**row, "decision": json.dumps(row["decision"], sort_keys=True, allow_nan=False)})
    temporary.replace(path)
