"""Valid/main-only calibration and authenticated missing-view evaluation.

No model is imported here. Canonical predictions are saved once per condition;
paired panels reuse those predictions on their declared common sample cohort.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

from src.data.aligned import EXPECTED_COUNTS
from src.evaluation.regression import _validate_protocol, load_protocol, select_threshold
from src.evaluation.regression_metrics import evaluation_metrics


MAIN_FAMILY = "main_mixed"
MAIN_CONDITION = "mixed_v1_s0"
FIELDS = ("split", "index", "id", "y_true", "y_pred", "absolute_error")
_HASH_KEYS = ("mask_manifest_sha256", "data_sha256", "training_config_sha256")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if Path(path).name == "evaluation.json" and value.get("reporting_schema_version") == 2:
        from src.reporting.records import read_original_evaluation
        value = read_original_evaluation(path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _protocol(value):
    result = load_protocol(value) if isinstance(value, (str, Path)) else _validate_protocol(value)
    if result["neutral_threshold_candidates"] != [index / 20 for index in range(21)] or result["tie_tolerance"] != 1e-12:
        raise ValueError("Missing-view runs require the frozen 21-threshold grid and 1e-12 tie tolerance")
    return result


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Unsafe evaluation family, condition or panel name")
    return value


def _finite(value, label):
    if isinstance(value, bool):
        raise ValueError(f"Invalid {label}")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"Invalid {label}") from error
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite {label}")
    return result


def _index(value):
    if type(value) not in (int, str):
        raise ValueError("Sample index must be an integer or canonical decimal string")
    try:
        result = int(value)
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid sample index") from error
    if str(result) != str(value) or result < 0:
        raise ValueError("Invalid sample index")
    return result


def _cohort(binding, split):
    if split not in EXPECTED_COUNTS or split == "train":
        raise ValueError("Evaluation requires valid or test")
    rows = binding.get("cohorts", {}).get(split)
    if not isinstance(rows, list) or len(rows) != EXPECTED_COUNTS[split]:
        raise ValueError(f"Binding requires the complete {split} identity/label cohort")
    normalized, seen = [], set()
    for index, row in enumerate(rows):
        if _index(row.get("index")) != index:
            raise ValueError("Source cohort indices must retain the original complete order")
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in seen:
            raise ValueError("Source cohort has missing/duplicate IDs")
        seen.add(sample_id)
        normalized.append({"index": index, "id": sample_id, "y_true": _finite(row.get("y_true"), "cohort label")})
    return normalized


def validate_records(records, split, binding, *, full=False):
    """Authenticate IDs/labels and original order; optionally require a full split."""
    cohort = _cohort(binding, split)
    records = list(records)
    if not records or (full and len(records) != len(cohort)):
        raise ValueError(f"Expected {'complete ' if full else 'nonempty '}{split} predictions")
    result, previous = [], -1
    for row in records:
        if not isinstance(row, dict) or any(key not in row for key in FIELDS[:-1]) or row["split"] != split:
            raise ValueError("Missing prediction fields or incorrect split")
        index = _index(row["index"])
        if index <= previous or index >= len(cohort):
            raise ValueError("Prediction indices must be unique and in original order")
        previous = index
        truth, prediction = _finite(row["y_true"], "label"), _finite(row["y_pred"], "prediction")
        if row["id"] != cohort[index]["id"] or truth != cohort[index]["y_true"]:
            raise ValueError("Prediction ID/label differs from authenticated source cohort")
        error = abs(prediction - truth)
        if not math.isfinite(error):
            raise ValueError("Prediction residual overflow")
        if "absolute_error" in row and not math.isclose(_finite(row["absolute_error"], "absolute_error"), error, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("absolute_error differs from prediction")
        result.append({"split": split, "index": index, "id": row["id"], "y_true": truth,
                       "y_pred": prediction, "absolute_error": error})
    if full and [row["index"] for row in result] != list(range(len(cohort))):
        raise ValueError("Full split prediction coverage is incomplete")
    return result


def initialize_run(output, config, protocol, provenance=None):
    """Initialize new records without touching existing runs or checkpoints."""
    root = Path(output)
    if (root / "config.json").exists() or (root / "evaluation.json").exists():
        raise FileExistsError(f"Run already initialized: {root}")
    protocol = _protocol(protocol)
    envelope = {"training": config, "evaluation_protocol": protocol}
    empty = {"summary": {split: {"status": "not_run", "n": None, "Accuracy": None, "Macro_F1": None, "MAE": None, "Pearson": None}
                         for split in ("valid", "test")},
             "main": {"valid": None, "test": None}, "paired": {"valid": {}, "test": {}},
             "exploratory": {"valid": {}, "test": {}}, "conditions": {"valid": {}, "test": {}},
             "decision_rule": None, "threshold_search": [], "training_summary": {},
             "metadata": {"format_version": "robust_evaluation_v1", "calibration_status": "not_calibrated",
                          "config_sha256": fingerprint(envelope), "provenance": provenance}}
    # Serialize before the first mutation, to catch nonfinite/invalid config.
    canonical(envelope)
    canonical(empty)
    (root / "details").mkdir(parents=True, exist_ok=True)
    _write(root / "config.json", envelope)
    _write(root / "evaluation.json", empty)


def _run(root):
    root = Path(root)
    envelope, evaluation = _read(root / "config.json"), _read(root / "evaluation.json")
    if evaluation.get("metadata", {}).get("format_version") != "robust_evaluation_v1":
        raise ValueError("Not a robust evaluation run")
    if fingerprint(envelope) != evaluation["metadata"].get("config_sha256"):
        raise ValueError("Saved training config or protocol changed")
    return envelope, evaluation, _protocol(envelope["evaluation_protocol"])


def set_training_summary(output, summary):
    _, evaluation, _ = _run(output)
    evaluation["training_summary"] = summary
    _write(Path(output) / "evaluation.json", evaluation)


def _binding(root, checkpoint_sha, binding, config):
    if not isinstance(binding, dict) or binding.get("purpose") != "train" or config.get("purpose", "train") != "train":
        raise ValueError("Formal calibration/scoring requires a train checkpoint; smoke is rejected")
    epoch = binding.get("selected_epoch")
    if type(epoch) is not int or epoch < 1:
        raise ValueError("A valid-MAE selected training epoch is required")
    if binding.get("selection_split", "valid") != "valid" or binding.get("selection_metric", "MAE") != "MAE":
        raise ValueError("Checkpoint selection must use valid MAE")
    if binding.get("selection_family", MAIN_FAMILY) != MAIN_FAMILY:
        raise ValueError("Checkpoint selection must use main_mixed valid")
    if not isinstance(checkpoint_sha, str) or not re.fullmatch(r"[a-f0-9]{64}", checkpoint_sha):
        raise ValueError("Invalid checkpoint SHA256")
    if sha256_file(Path(root) / "best.pt") != checkpoint_sha:
        raise ValueError("best.pt differs from supplied checkpoint SHA256")
    for key in _HASH_KEYS:
        if not isinstance(binding.get(key), str) or not re.fullmatch(r"[a-f0-9]{64}", binding[key]):
            raise ValueError(f"Missing source binding: {key}")
    if binding["training_config_sha256"] != fingerprint(config):
        raise ValueError("Checkpoint training config differs from run config")
    return {"purpose": "train", "selected_epoch": epoch, "selection_split": "valid", "selection_metric": "MAE",
            "selection_family": MAIN_FAMILY, "checkpoint_sha256": checkpoint_sha,
            **{key: binding[key] for key in _HASH_KEYS},
            "valid_cohort_sha256": fingerprint(_cohort(binding, "valid"))}


def _summary(metrics):
    return {**{key: metrics[key] for key in ("n", "Accuracy", "Macro_F1", "MAE", "Pearson")},
            "status": "complete", "Pearson_status": "undefined_constant_or_singleton" if metrics["Pearson"] is None else "defined"}


def _metrics(records, delta):
    if not records:
        return {"n": 0, "status": "not_applicable", "Accuracy": None, "Macro_F1": None, "MAE": None,
                "Pearson": None, "Pearson_status": "undefined_empty", "confusion_matrix": [[0] * 3 for _ in range(3)]}
    values = evaluation_metrics([row["y_true"] for row in records], [row["y_pred"] for row in records], delta)
    return {**values, "status": "complete", "Pearson_status": "undefined_constant_or_singleton" if values["Pearson"] is None else "defined"}


def _path(root, split, family, condition):
    _name(family)
    _name(condition)
    if family == MAIN_FAMILY and condition == MAIN_CONDITION:
        return Path(root) / "details" / f"{split}_predictions.csv"
    return Path(root) / "details" / "predictions" / family / split / f"{condition}.csv"


def _read_records(path, split, binding, full=False):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Malformed prediction CSV")
        return validate_records(reader, split, binding, full=full)


def _save_records_once(path, rows, split, binding, full=False):
    if path.exists():
        if _read_records(path, split, binding, full) != rows:
            raise ValueError("Refusing to overwrite canonical predictions with different content")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def calibrate(output, records, checkpoint_sha, binding, *, split="valid", family=MAIN_FAMILY, condition=MAIN_CONDITION):
    """Freeze only full main-mixed valid; repeat calls verify, never recalibrate."""
    if (split, family, condition) != ("valid", MAIN_FAMILY, MAIN_CONDITION):
        raise ValueError("Calibration is restricted to full valid/main_mixed/mixed_v1_s0")
    root = Path(output)
    envelope, evaluation, protocol = _run(root)
    sources = _binding(root, checkpoint_sha, binding, envelope["training"])
    records = validate_records(records, "valid", binding, full=True)
    sources["valid_records_sha256"] = fingerprint(records)
    if evaluation["metadata"]["calibration_status"] == "complete":
        rule = load_frozen_rule(root, checkpoint_sha, binding)
        if sources != rule["sources"]:
            raise ValueError("Frozen valid predictions differ; recalibration is forbidden")
        return rule
    if evaluation["metadata"]["calibration_status"] != "not_calibrated" or evaluation["decision_rule"] is not None:
        raise ValueError("Incomplete calibration cannot be silently retried")
    delta, search = select_threshold(records, protocol)
    rule = {"neutral_threshold": delta, "calibration_split": "valid", "calibration_family": MAIN_FAMILY,
            "calibration_condition": MAIN_CONDITION, "selection_metric": "Macro_F1",
            "protocol": protocol, "protocol_sha256": fingerprint(protocol), "sources": sources}
    metrics = _metrics(records, delta)
    evaluation["metadata"]["calibration_status"] = "writing"
    _write(root / "evaluation.json", evaluation)
    _save_records_once(_path(root, "valid", MAIN_FAMILY, MAIN_CONDITION), records, "valid", binding, full=True)
    evaluation["decision_rule"], evaluation["threshold_search"] = rule, search
    evaluation["main"]["valid"], evaluation["summary"]["valid"] = metrics, _summary(metrics)
    evaluation["conditions"]["valid"][f"{MAIN_FAMILY}/{MAIN_CONDITION}"] = {
        "path": "details/valid_predictions.csv", "n": len(records), "records_sha256": fingerprint(records)}
    evaluation["metadata"].update(calibration_status="complete", decision_rule_sha256=fingerprint(rule),
                                  threshold_search_sha256=fingerprint(search))
    _write(root / "evaluation.json", evaluation)
    return rule


def load_frozen_rule(output, checkpoint_sha, binding):
    """Authenticate saved policy and provenance without selecting any threshold."""
    root = Path(output)
    envelope, evaluation, protocol = _run(root)
    metadata, rule = evaluation["metadata"], evaluation["decision_rule"]
    if metadata.get("calibration_status") != "complete" or not isinstance(rule, dict):
        raise ValueError("No completed frozen main-valid rule")
    if metadata.get("decision_rule_sha256") != fingerprint(rule) or rule.get("protocol") != protocol or rule.get("protocol_sha256") != fingerprint(protocol):
        raise ValueError("Frozen decision rule/protocol changed")
    if (rule.get("calibration_split"), rule.get("calibration_family"), rule.get("calibration_condition")) != ("valid", MAIN_FAMILY, MAIN_CONDITION):
        raise ValueError("Invalid threshold selection domain")
    delta, search = rule.get("neutral_threshold"), evaluation["threshold_search"]
    if (delta not in protocol["neutral_threshold_candidates"] or fingerprint(search) != metadata.get("threshold_search_sha256")
            or [row.get("neutral_threshold") for row in search] != protocol["neutral_threshold_candidates"]
            or [row.get("neutral_threshold") for row in search if row.get("selected") is True] != [delta]):
        raise ValueError("Threshold search is inconsistent")
    sources = _binding(root, checkpoint_sha, binding, envelope["training"])
    records = _read_records(_path(root, "valid", MAIN_FAMILY, MAIN_CONDITION), "valid", binding, full=True)
    sources["valid_records_sha256"] = fingerprint(records)
    if sources != rule["sources"]:
        raise ValueError("Checkpoint, mask, source, config or valid predictions changed")
    metrics = _metrics(records, delta)
    if metrics != evaluation["main"]["valid"] or _summary(metrics) != evaluation["summary"]["valid"]:
        raise ValueError("Saved main-valid metrics differ from frozen predictions")
    return rule


def panel_records(records, split, binding, condition, panel):
    """Select exactly the declared panel intersection, retaining original IDs."""
    records = validate_records(records, split, binding)
    if not isinstance(panel, dict) or condition not in panel.get("condition_ids", []):
        raise ValueError("Condition is not in this paired panel")
    indices = [_index(value) for value in panel.get("indices", [])]
    if indices != sorted(set(indices)):
        raise ValueError("Panel indices must be unique and in original order")
    condition_indices = binding.get("condition_indices", {}).get(split, {})
    expected_sets = []
    for condition_id in panel["condition_ids"]:
        values = condition_indices.get(f"paired/{condition_id}")
        if values is None:
            raise ValueError("Binding lacks paired applicability required to authenticate the intersection")
        expected_sets.append(set(values))
    if not expected_sets or indices != sorted(set.intersection(*expected_sets)):
        raise ValueError("Panel is not the declared conditions' common applicable intersection")
    source = _cohort(binding, split)
    if any(index >= len(source) for index in indices):
        raise ValueError("Panel index outside source cohort")
    expected_ids = [source[index]["id"] for index in indices]
    if panel.get("ids") != expected_ids:
        raise ValueError("Panel IDs differ from its source indices")
    by_index = {row["index"]: row for row in records}
    if any(index not in by_index for index in indices):
        raise ValueError("Canonical predictions omit part of the paired common intersection")
    return [by_index[index] for index in indices]


def save_evaluation(output, records, split, family, condition, checkpoint_sha, binding, *, panel=None, rule=None):
    """Save canonical predictions once and score main, panel or exploratory view.

    panel is the saved panel object plus name (or panel_name). With paired and
    panel=None, only save canonical predictions (no panel score). Pass those
    same predictions for each panel using that condition.
    """
    if split not in ("valid", "test") or family not in (MAIN_FAMILY, "paired", "exploratory_a3like"):
        raise ValueError("Unsupported evaluation domain")
    if family == MAIN_FAMILY and condition != MAIN_CONDITION:
        raise ValueError("Unknown main condition")
    root = Path(output)
    verified = load_frozen_rule(root, checkpoint_sha, binding)
    if rule is not None and rule != verified:
        raise ValueError("Supplied rule differs from this run's frozen threshold")
    rows = validate_records(records, split, binding, full=family != "paired")
    condition_indices = binding.get("condition_indices", {}).get(split, {}).get(f"{family}/{condition}")
    if family == "paired" and condition_indices is None:
        raise ValueError("Paired predictions require frozen condition applicability")
    if condition_indices is not None and [row["index"] for row in rows] != list(condition_indices):
        raise ValueError("Canonical condition predictions do not cover its complete applicable cohort")
    canonical_only = family == "paired" and panel is None
    if family == "paired" and not canonical_only:
        name = _name(panel.get("name", panel.get("panel_name")) if isinstance(panel, dict) else None)
        selected = panel_records(rows, split, binding, condition, panel)
    else:
        if panel is not None:
            raise ValueError("Only paired evaluations use a panel")
        selected = rows
    metrics = ({"status": "predictions_saved", "n": len(rows)} if canonical_only else
               _metrics(selected, verified["neutral_threshold"]))
    metrics.update(split=split, evaluation_family=family, condition_id=condition,
                   checkpoint_sha256=checkpoint_sha, decision_rule_sha256=fingerprint(verified),
                   records_sha256=fingerprint(selected), canonical_records_sha256=fingerprint(rows))
    if family == "paired" and not canonical_only:
        metrics.update(panel=name, cohort_indices=[row["index"] for row in selected],
                       cohort_ids=[row["id"] for row in selected], panel_sha256=fingerprint(panel))
    _, evaluation, _ = _run(root)
    if split == "valid" and family == MAIN_FAMILY:
        if fingerprint(rows) != verified["sources"]["valid_records_sha256"]:
            raise ValueError("Main-valid predictions differ from frozen calibration")
        return evaluation["main"]["valid"]
    path = _path(root, split, family, condition)
    _save_records_once(path, rows, split, binding, full=family != "paired")
    entry = {"path": path.relative_to(root).as_posix(), "n": len(rows), "records_sha256": fingerprint(rows)}
    prior = evaluation["conditions"][split].get(f"{family}/{condition}")
    if prior is not None and prior != entry:
        raise ValueError("Canonical condition provenance changed")
    evaluation["conditions"][split][f"{family}/{condition}"] = entry
    if family == MAIN_FAMILY:
        evaluation["main"][split], evaluation["summary"][split] = metrics, _summary(metrics)
    elif family == "paired" and not canonical_only:
        previous = evaluation["paired"][split].setdefault(name, {}).get(condition)
        if previous is not None and previous != metrics:
            raise ValueError("Refusing to overwrite paired panel results with a different cohort")
        evaluation["paired"][split][name][condition] = metrics
    elif family == "exploratory_a3like":
        evaluation["exploratory"][split][condition] = metrics
    _write(root / "evaluation.json", evaluation)
    return metrics
