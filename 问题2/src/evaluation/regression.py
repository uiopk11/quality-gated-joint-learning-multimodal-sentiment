"""Valid-only calibration and frozen-rule scoring with one evaluation.json.

Storage format 2 preserves evaluation semantics version 1. Dataset bytes are
fingerprinted, never deserialized here; optional exports are not score inputs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from src.runtime import resolve_path, source_path

from src.data.aligned import EXPECTED_COUNTS
from src.evaluation.regression_metrics import CLASS_NAMES, class_labels, evaluation_metrics, regression_metrics


EVALUATION_FILENAME = "evaluation.json"
DETAILS_DIRECTORY = "details"
RECORD_FIELDS = ("split", "index", "id", "y_true", "y_pred", "absolute_error")
_FIXED_PROTOCOL = {
    "evaluation_version": 1,
    "calibration_split": "valid",
    "selection_metric": "Macro_F1",
    "tie_breakers": ["Accuracy", "smaller_neutral_threshold"],
    "class_order": list(CLASS_NAMES),
    "truth_rule": "y<0:negative; y==0:neutral; y>0:positive",
    "prediction_rule": "p<-delta:negative; -delta<=p<=delta:neutral; p>delta:positive",
    "zero_division": 0,
    "prediction_clipping": False,
}


def _read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _json_hash(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _validate_protocol(protocol):
    required = set(_FIXED_PROTOCOL) | {"neutral_threshold_candidates", "tie_tolerance"}
    if not isinstance(protocol, dict) or set(protocol) != required:
        raise ValueError(f"Evaluation protocol must contain exactly {sorted(required)}")
    for key, expected in _FIXED_PROTOCOL.items():
        if type(protocol[key]) is not type(expected) or protocol[key] != expected:
            raise ValueError(f"Unsupported evaluation protocol {key}: expected {expected!r}")
    tolerance = protocol["tie_tolerance"]
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tie_tolerance must be finite and nonnegative")
    candidates = protocol["neutral_threshold_candidates"]
    if (not isinstance(candidates, list) or not candidates
            or any(type(item) not in (int, float) or not math.isfinite(item) or item < 0 for item in candidates)
            or candidates != sorted(set(candidates))):
        raise ValueError("neutral_threshold_candidates must be a sorted, unique, nonnegative finite list")
    # JSON round-trip makes a private, serializable copy of the caller's input.
    result = json.loads(_canonical(protocol))
    result["tie_tolerance"] = float(tolerance)
    result["neutral_threshold_candidates"] = [float(value) for value in candidates]
    return result


def load_protocol(path):
    """Read and validate the documented evaluation-v1 configuration."""
    return _validate_protocol(_read_json(path))


def validate_records(records, split, expected_n):
    """Reject omitted, reordered, duplicate, nonfinite, or inconsistent rows."""
    if split not in ("valid", "test"):
        raise ValueError("Evaluation records must be from valid or test")
    records = list(records)
    if type(expected_n) is not int or expected_n < 1 or len(records) != expected_n:
        raise ValueError(f"{split}: expected {expected_n} records, got {len(records)}")
    normalized, ids = [], set()
    for index, row in enumerate(records):
        if not isinstance(row, dict) or any(field not in row for field in RECORD_FIELDS):
            raise ValueError(f"{split} row {index}: missing required prediction fields")
        if row["split"] != split:
            raise ValueError(f"{split} row {index}: incorrect split {row['split']!r}")
        # Accept CSV decimal integers and native integer records, never bool/float.
        raw_index = row["index"]
        if (type(raw_index) not in (int, str) or str(raw_index) != str(index)):
            raise ValueError(f"{split} row {index}: indices must be consecutive in original order")
        sample_id = row["id"]
        if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in ids:
            raise ValueError(f"{split} row {index}: empty or duplicate sample ID")
        ids.add(sample_id)
        values = {}
        for key in ("y_true", "y_pred", "absolute_error"):
            try:
                if isinstance(row[key], bool):
                    raise ValueError("bool is not a prediction")
                value = float(row[key])
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{split} row {index}: invalid {key}") from error
            if not math.isfinite(value):
                raise ValueError(f"{split} row {index}: non-finite {key}")
            values[key] = value
        absolute_error = abs(values["y_pred"] - values["y_true"])
        if not math.isclose(values["absolute_error"], absolute_error, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"{split} row {index}: absolute_error disagrees with prediction and label")
        normalized.append({"split": split, "index": index, "id": sample_id, **values})
    return normalized


def read_predictions(path, split, expected_n):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Prediction CSV needs unique column names")
        return validate_records(list(reader), split, expected_n)


def select_threshold(records, protocol):
    """Use complete valid only; test and partial valid are rejected explicitly."""
    protocol = _validate_protocol(protocol)
    records = validate_records(records, "valid", EXPECTED_COUNTS["valid"])
    truth, prediction = [r["y_true"] for r in records], [r["y_pred"] for r in records]
    rows = []
    for delta in protocol["neutral_threshold_candidates"]:
        metrics = evaluation_metrics(truth, prediction, delta)
        row = {"neutral_threshold": delta, "Accuracy": metrics["Accuracy"],
               "Macro_F1": metrics["Macro_F1"], "Weighted_F1": metrics["Weighted_F1"]}
        for name in CLASS_NAMES:
            row.update({f"{name}_{key}": value for key, value in metrics["per_class"][name].items()
                        if key != "class_id"})
        rows.append(row)
    tolerance = protocol["tie_tolerance"]
    maximum = max(row["Macro_F1"] for row in rows)
    finalists = [row for row in rows if maximum - row["Macro_F1"] <= tolerance]
    best_accuracy = max(row["Accuracy"] for row in finalists)
    finalists = [row for row in finalists if best_accuracy - row["Accuracy"] <= tolerance]
    selected = min(finalists, key=lambda row: row["neutral_threshold"])
    for row in rows:
        row["selected"] = row is selected
    return selected["neutral_threshold"], rows


def _check_regression(saved, expected, context):
    if not isinstance(saved, dict) or saved.get("n") != expected["n"] or saved.get("prediction_clipping") is not False:
        raise ValueError(f"{context}: invalid sample count or prediction clipping policy")
    for key in ("MAE", "RMSE", "Pearson"):
        actual = saved.get(key)
        reference = expected[key]
        if reference is None:
            equal = key in saved and actual is None
        else:
            equal = (type(actual) in (int, float) and math.isfinite(actual)
                     and math.isclose(actual, reference, rel_tol=1e-10, abs_tol=1e-10))
        if not equal:
            raise ValueError(f"{context}: {key} disagrees with complete saved predictions")


def _metrics(records, split, rule):
    result = evaluation_metrics([r["y_true"] for r in records], [r["y_pred"] for r in records],
                                rule["neutral_threshold"])
    return {**result, "split": split, "evaluation_version": 1,
            "selected_epoch": rule["sources"]["selected_epoch"],
            "checkpoint_sha256": rule["sources"]["checkpoint_sha256"],
            "data_sha256": rule["sources"]["data_sha256"],
            "protocol_sha256": rule["protocol_sha256"], "records_sha256": _json_hash(records),
            "threshold_selection_split": "valid", "threshold_selection_metric": "Macro_F1"}


def _empty_evaluation(provenance=None):
    return {
        "summary": {split: {"Accuracy": None, "Macro_F1": None, "MAE": None,
                             "Pearson": None, "n": None, "status": "not_run"}
                    for split in ("valid", "test")},
        "per_class": {"valid": None, "test": None},
        "confusion_matrix": {"valid": None, "test": None},
        "additional_metrics": {"valid": None, "test": None},
        "decision_rule": None,
        "threshold_search": [],
        "training_summary": {},
        "metadata": {"format_version": 2, "evaluation_version": 1,
                     "calibration_status": "not_calibrated", "sources": None,
                     "prediction_sha256": {"valid": None, "test": None},
                     "provenance": provenance,
                     "paths": {"checkpoint": "best.pt", "config": "config.json",
                               "valid_predictions": "details/valid_predictions.csv",
                               "test_predictions": "details/test_predictions.csv",
                               "history": "details/history.csv", "log": "details/run.log"}},
    }


def initialize_run(run, config, protocol=None, provenance=None):
    """Create a new run's actual training configuration and unfinished summary.

    This does not alter the configuration stored inside a checkpoint. A missing
    protocol is allowed for development smoke runs, which cannot be calibrated.
    """
    run = Path(run)
    if (run / EVALUATION_FILENAME).exists() or (run / "config.json").exists():
        raise FileExistsError(f"Run configuration already exists: {run}")
    protocol = _validate_protocol(protocol) if protocol is not None else None
    (run / DETAILS_DIRECTORY).mkdir(parents=True, exist_ok=True)
    _write_json(run / "config.json", {"training": config, "evaluation_protocol": protocol})
    _write_json(run / EVALUATION_FILENAME, _empty_evaluation(provenance))


def update_training_summary(run, summary):
    run = Path(run)
    evaluation = _read_evaluation(run)
    evaluation["training_summary"] = summary
    _write_json(run / EVALUATION_FILENAME, evaluation)


def _read_evaluation(run):
    evaluation = _read_json(Path(run) / EVALUATION_FILENAME)
    metadata = evaluation.get("metadata", {})
    if metadata.get("format_version") != 2 or metadata.get("evaluation_version") != 1:
        raise ValueError("Unsupported evaluation storage or semantic version")
    required = ("summary", "per_class", "confusion_matrix", "additional_metrics",
                "decision_rule", "threshold_search", "training_summary", "metadata")
    if evaluation.get("reporting_schema_version") == 2:
        # The report view adds navigation only. Return exactly the historical
        # storage fields to the unchanged calibration/scoring implementation.
        # Missing fields still fail below; no rule or metric is reconstructed.
        evaluation = {key: evaluation[key] for key in required if key in evaluation}
    if set(evaluation) != set(required):
        raise ValueError("Incomplete evaluation.json structure")
    return evaluation


def _run_protocol(root, run, protocol=None):
    config = _read_json(Path(run) / "config.json")
    saved = config.get("evaluation_protocol")
    if saved is None:
        raise ValueError("Run config.json has no saved evaluation protocol")
    saved = _validate_protocol(saved)
    if protocol is not None and _validate_protocol(protocol) != saved:
        raise ValueError("Requested protocol differs from this run's frozen evaluation_protocol")
    return saved


def _checkpoint_payload(root, checkpoint_path):
    """Load existing checkpoint format without changing or resaving its tensors."""
    import torch

    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1
            or checkpoint.get("purpose") != "train"
            or checkpoint.get("selection_split") != "valid"
            or checkpoint.get("selection_metric") != "MAE"):
        raise ValueError("Requires a training checkpoint selected only by valid MAE; smoke is rejected")
    if type(checkpoint.get("epoch")) is not int or checkpoint["epoch"] < 1:
        raise ValueError("Checkpoint has no valid selected epoch")
    config = checkpoint.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("data_path"), str):
        raise ValueError("Checkpoint has no saved data_path")
    data_hash = _sha256(resolve_path(config["data_path"], root))
    if data_hash != checkpoint.get("data_sha256"):
        raise ValueError("Dataset SHA256 differs from the training checkpoint")
    sources = {"checkpoint_sha256": _sha256(checkpoint_path), "data_sha256": data_hash,
               "training_config_sha256": _json_hash(config), "selected_epoch": checkpoint["epoch"],
               "source_commit": checkpoint.get("source_commit")}
    valid_metrics = checkpoint.get("valid_metrics")
    del checkpoint
    return sources, config, valid_metrics


def _bind_sources(sources, records, provenance):
    if provenance is not None:
        if (not isinstance(provenance, dict) or provenance.get("data_sha256") != sources["data_sha256"]
                or provenance.get("source_commit") != sources["source_commit"]):
            raise ValueError("Run provenance differs from checkpoint data/source")
    return {**sources, "valid_records_sha256": _json_hash(records),
            "training_provenance_sha256": _json_hash(provenance)}


def _checkpoint_sources(root, checkpoint_path):
    checkpoint_path = Path(checkpoint_path).resolve()
    run = checkpoint_path.parent
    sources, config, saved_metrics = _checkpoint_payload(root, checkpoint_path)
    envelope = _read_json(run / "config.json")
    if (set(envelope) != {"training", "evaluation_protocol"}
            or _json_hash(envelope["training"]) != sources["training_config_sha256"]):
        raise ValueError("Run config.json training configuration differs from saved checkpoint configuration")
    evaluation = _read_evaluation(run)
    records = read_predictions(run / "details/valid_predictions.csv", "valid", EXPECTED_COUNTS["valid"])
    recomputed = regression_metrics([r["y_true"] for r in records], [r["y_pred"] for r in records])
    _check_regression(saved_metrics, recomputed, "checkpoint valid_metrics")
    return _bind_sources(sources, records, evaluation["metadata"].get("provenance")), records


def _enhanced(records, delta):
    actual = class_labels([r["y_true"] for r in records])
    predicted = class_labels([r["y_pred"] for r in records], delta)
    return [{**row, "true_class": int(truth), "predicted_class": int(prediction),
             "true_class_name": CLASS_NAMES[truth], "predicted_class_name": CLASS_NAMES[prediction],
             "classification_correct": bool(truth == prediction)}
            for row, truth, prediction in zip(records, actual, predicted)]


def _check_enhanced(path, records, delta):
    """Derived class columns must match the raw predictions and frozen rule."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for index, (row, expected) in enumerate(zip(rows, _enhanced(records, delta))):
        for key in ("true_class", "predicted_class", "true_class_name",
                    "predicted_class_name", "classification_correct"):
            if row.get(key) != str(expected[key]):
                raise ValueError(f"Prediction class column {key} disagrees at row {index}")


def _score_sections(metrics):
    summary = {key: metrics[key] for key in ("Accuracy", "Macro_F1", "MAE", "Pearson", "n")}
    summary["status"] = "complete"
    summary["Pearson_status"] = "undefined_constant_or_singleton" if metrics["Pearson"] is None else "defined"
    matrix = {"counts": metrics["confusion_matrix"], "normalized": metrics["confusion_matrix_normalized"],
              "class_order": metrics["class_order"], "orientation": metrics["confusion_orientation"]}
    additional = {key: metrics[key] for key in ("Weighted_F1", "RMSE", "prediction_clipping", "zero_division")}
    return {"summary": summary, "per_class": metrics["per_class"],
            "confusion_matrix": matrix, "additional_metrics": additional}


def _set_scores(evaluation, metrics):
    split = metrics["split"]
    for key, value in _score_sections(metrics).items():
        evaluation[key][split] = value
    evaluation["metadata"]["prediction_sha256"][split] = metrics["records_sha256"]


def _verify_scores(evaluation, metrics):
    split = metrics["split"]
    if any(evaluation[key].get(split) != value for key, value in _score_sections(metrics).items()):
        raise ValueError(f"Frozen threshold/{split} metrics are inconsistent with saved predictions")
    if evaluation["metadata"]["prediction_sha256"].get(split) != metrics["records_sha256"]:
        raise ValueError(f"{split} prediction content fingerprints differ from saved evaluation")


def _freeze(evaluation, sources, protocol, delta, rows):
    decision = {"neutral_threshold": delta, "calibration_split": "valid", "selection_metric": "Macro_F1",
                "truth_rule": protocol["truth_rule"], "prediction_rule": protocol["prediction_rule"],
                "tie_breakers": protocol["tie_breakers"], "tie_tolerance": protocol["tie_tolerance"],
                "class_order": protocol["class_order"], "zero_division": protocol["zero_division"],
                "prediction_clipping": protocol["prediction_clipping"], "protocol_sha256": _json_hash(protocol)}
    evaluation["decision_rule"] = decision
    evaluation["threshold_search"] = rows
    evaluation["metadata"].update({"calibration_status": "complete", "sources": sources,
                                   "decision_rule_sha256": _json_hash(decision),
                                   "threshold_search_sha256": _json_hash(rows),
                                   "code_sha256": {source_path(f"baseline/{name}").relative_to(Path(__file__).resolve().parents[2]).as_posix(): _sha256(source_path(f"baseline/{name}"))
                                                   for name in ("evaluation.py", "metrics.py", "plots.py")}})
    return _public_rule(evaluation, protocol)


def _public_rule(evaluation, protocol):
    return {"format_version": 2, "evaluation_version": 1, **evaluation["decision_rule"],
            "protocol": protocol, "sources": evaluation["metadata"]["sources"]}


def calibrate_run(root, checkpoint_path, protocol=None):
    """Freeze complete valid once; a repeat verifies it without a new search."""
    checkpoint_path = Path(checkpoint_path).resolve()
    run = checkpoint_path.parent
    protocol = _run_protocol(root, run, protocol)
    evaluation = _read_evaluation(run)
    status = evaluation["metadata"].get("calibration_status")
    if status == "complete":
        rule = load_decision_rule(root, checkpoint_path)
        records = read_predictions(run / "details/valid_predictions.csv", "valid", EXPECTED_COUNTS["valid"])
        return _metrics(records, "valid", rule)
    if status != "not_calibrated" or evaluation["decision_rule"] is not None:
        raise ValueError("Incomplete calibration; review this run before calibrating again")
    sources, records = _checkpoint_sources(root, checkpoint_path)
    delta, rows = select_threshold(records, protocol)
    evaluation["metadata"]["calibration_status"] = "writing"
    _write_json(run / EVALUATION_FILENAME, evaluation)
    rule = _freeze(evaluation, sources, protocol, delta, rows)
    metrics = _metrics(records, "valid", rule)
    _set_scores(evaluation, metrics)
    _write_csv(run / "details/valid_predictions.csv", _enhanced(records, delta))
    # Write the complete rule last; partially written predictions cannot score.
    _write_json(run / EVALUATION_FILENAME, evaluation)
    return metrics


def load_decision_rule(root, checkpoint_path):
    """Verify content, checkpoint, data and policy; never search a threshold."""
    checkpoint_path = Path(checkpoint_path).resolve()
    run = checkpoint_path.parent
    if not (run / EVALUATION_FILENAME).is_file():
        raise FileNotFoundError(f"No frozen decision rule: {run / EVALUATION_FILENAME}. Run calibrate-valid first.")
    evaluation = _read_evaluation(run)
    metadata, decision = evaluation["metadata"], evaluation["decision_rule"]
    if metadata.get("calibration_status") != "complete" or not isinstance(decision, dict):
        raise ValueError("Incomplete calibration: no complete frozen rule; run calibrate-valid first")
    protocol = _run_protocol(root, run)
    if (decision.get("protocol_sha256") != _json_hash(protocol)
            or metadata.get("decision_rule_sha256") != _json_hash(decision)):
        raise ValueError("Frozen decision rule or protocol fingerprint mismatch (inconsistent rule)")
    for key in ("calibration_split", "selection_metric", "truth_rule", "prediction_rule", "tie_breakers",
                "tie_tolerance", "class_order", "zero_division", "prediction_clipping"):
        if decision.get(key) != protocol[key]:
            raise ValueError(f"Frozen decision rule differs from protocol: {key}")
    delta = decision.get("neutral_threshold")
    if type(delta) not in (int, float) or delta not in protocol["neutral_threshold_candidates"]:
        raise ValueError("Frozen threshold is outside the saved candidate grid")
    rows = evaluation["threshold_search"]
    if (not isinstance(rows, list) or metadata.get("threshold_search_sha256") != _json_hash(rows)
            or [row.get("neutral_threshold") for row in rows] != protocol["neutral_threshold_candidates"]
            or [row.get("neutral_threshold") for row in rows if row.get("selected") is True] != [delta]):
        raise ValueError("Frozen threshold search is incomplete or inconsistent")
    sources, records = _checkpoint_sources(root, checkpoint_path)
    if metadata.get("sources") != sources:
        raise ValueError("Frozen rule source fingerprints differ from checkpoint/data/valid records")
    rule = _public_rule(evaluation, protocol)
    _verify_scores(evaluation, _metrics(records, "valid", rule))
    _check_enhanced(run / "details/valid_predictions.csv", records, delta)
    return rule


def score_records(root, checkpoint_path, records, split, rule=None):
    """Score records from authenticated inference and save one enhanced CSV."""
    if split not in ("valid", "test"):
        raise ValueError("Only valid and test can be scored")
    verified = load_decision_rule(root, checkpoint_path)
    if rule is not None and rule != verified:
        raise ValueError("Supplied rule differs from this checkpoint's frozen rule")
    records = validate_records(records, split, EXPECTED_COUNTS[split])
    run = Path(checkpoint_path).resolve().parent
    evaluation = _read_evaluation(run)
    metrics = _metrics(records, split, verified)
    if split == "valid":
        _verify_scores(evaluation, metrics)
        return metrics
    path = run / "details/test_predictions.csv"
    if evaluation["summary"]["test"]["status"] == "complete":
        previous = read_predictions(path, "test", EXPECTED_COUNTS["test"])
        if [(r["index"], r["id"], r["y_true"]) for r in previous] != [
                (r["index"], r["id"], r["y_true"]) for r in records]:
            raise ValueError("Test sample identity, order or true labels differ from this run's saved test")
    _write_csv(path, _enhanced(records, verified["neutral_threshold"]))
    _set_scores(evaluation, metrics)
    _write_json(run / EVALUATION_FILENAME, evaluation)
    return metrics


def score_saved_run(root, checkpoint_path, split):
    """Recompute saved predictions with the frozen rule, without model inference."""
    if split not in ("valid", "test"):
        raise ValueError("Only valid and test can be scored")
    checkpoint_path = Path(checkpoint_path).resolve()
    rule = load_decision_rule(root, checkpoint_path)
    run = checkpoint_path.parent
    evaluation = _read_evaluation(run)
    if evaluation["summary"][split]["status"] != "complete":
        raise ValueError(f"{split} has not been run; no authenticated saved predictions to score")
    path = run / "details" / f"{split}_predictions.csv"
    records = read_predictions(path, split, EXPECTED_COUNTS[split])
    metrics = _metrics(records, split, rule)
    _verify_scores(evaluation, metrics)
    _check_enhanced(path, records, rule["neutral_threshold"])
    return metrics


def export_confusion(root, checkpoint_path, split):
    """Generate optional paper tables/figures, only on an explicit export."""
    from src.reporting.plots import write_confusion_svg

    metrics = score_saved_run(root, checkpoint_path, split)
    directory = Path(checkpoint_path).resolve().parent / "details/exports"
    directory.mkdir(parents=True, exist_ok=True)
    for suffix, key, normalized in (("counts", "confusion_matrix", False),
                                     ("normalized", "confusion_matrix_normalized", True)):
        matrix = metrics[key]
        rows = [{"true_class": name, **dict(zip(CLASS_NAMES, row))} for name, row in zip(CLASS_NAMES, matrix)]
        _write_csv(directory / f"{split}_confusion_{suffix}.csv", rows)
        write_confusion_svg(directory / f"{split}_confusion_{suffix}.svg", matrix,
                            normalized=normalized, title=f"{split}: confusion matrix ({suffix})")
    return directory


def _legacy_search(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = []
    for row in rows:
        converted = {}
        for key, value in row.items():
            if key == "selected":
                if value not in ("True", "False"):
                    raise ValueError("Invalid legacy selected marker")
                converted[key] = value == "True"
            elif key.endswith("support") or key.endswith("predicted_count"):
                converted[key] = int(value)
            else:
                converted[key] = float(value)
        result.append(converted)
    return result


def migrate_legacy_run(root, checkpoint_path, legacy_evidence):
    """Convert a previously verified legacy run; leave old files for final QA.

    The caller must first validate the complete original artifacts using the
    original evaluator and retain its file fingerprints. We check every one
    again before writing; no threshold selection or model inference is called.
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    run = checkpoint_path.parent
    if (run / EVALUATION_FILENAME).exists():
        raise FileExistsError(f"Already has a new-format evaluation: {run}")
    if legacy_evidence.get("legacy_rule_validated") is not True:
        raise ValueError("Original rule was not verified before migration")
    hashes = legacy_evidence.get("file_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Missing original file fingerprint evidence")
    for relative, digest in hashes.items():
        path = (run / relative).resolve()
        if not path.is_relative_to(run) or not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"Legacy file changed since original validation: {relative}")
    legacy_rule = _read_json(run / "evaluation_v1/decision_rule.json")
    if legacy_rule != legacy_evidence.get("rule"):
        raise ValueError("Legacy frozen rule differs from pre-migration evidence")
    protocol = load_protocol(run / "evaluation_protocol.json")
    if protocol != legacy_rule["protocol"]:
        raise ValueError("Legacy saved protocol differs from frozen rule")
    sources, training, saved_valid = _checkpoint_payload(root, checkpoint_path)
    if _json_hash(_read_json(run / "config.json")) != sources["training_config_sha256"]:
        raise ValueError("Legacy config differs from checkpoint training configuration")
    provenance = _read_json(run / "provenance.json")
    valid = read_predictions(run / "valid_predictions_best.csv", "valid", EXPECTED_COUNTS["valid"])
    enhanced_valid = read_predictions(run / "evaluation_v1/valid_predictions.csv", "valid", EXPECTED_COUNTS["valid"])
    if valid != enhanced_valid:
        raise ValueError("Legacy valid prediction copies differ")
    _check_regression(saved_valid, regression_metrics([r["y_true"] for r in valid],
                                                     [r["y_pred"] for r in valid]), "checkpoint valid_metrics")
    sources = _bind_sources(sources, valid, provenance)
    for key in ("checkpoint_sha256", "data_sha256", "training_config_sha256", "selected_epoch", "source_commit"):
        if sources[key] != legacy_rule["sources"][key]:
            raise ValueError(f"Legacy rule source changed: {key}")
    rows = _legacy_search(run / "evaluation_v1/threshold_search.csv")
    delta = legacy_rule["neutral_threshold"]
    if ([r["neutral_threshold"] for r in rows] != protocol["neutral_threshold_candidates"]
            or [r["neutral_threshold"] for r in rows if r["selected"]] != [delta]):
        raise ValueError("Legacy threshold grid or selected marker differs from frozen rule")
    evaluation = _empty_evaluation(provenance)
    evaluation["training_summary"] = _read_json(run / "training_summary.json")
    rule = _freeze(evaluation, sources, protocol, delta, rows)
    valid_metrics = _metrics(valid, "valid", rule)
    legacy_metrics = _read_json(run / "evaluation_v1/valid_metrics.json")
    if valid_metrics != legacy_metrics:
        raise ValueError("Migration changed valid metrics")
    _check_enhanced(run / "evaluation_v1/valid_predictions.csv", valid, delta)
    _set_scores(evaluation, valid_metrics)
    test = None
    reconstructed = False
    if (run / "test_predictions.csv").exists():
        test = read_predictions(run / "test_predictions.csv", "test", EXPECTED_COUNTS["test"])
        saved_test = _read_json(run / "test_metrics.json")
        if (saved_test.get("checkpoint_sha256") != sources["checkpoint_sha256"]
                or saved_test.get("selected_epoch") != sources["selected_epoch"]
                or saved_test.get("selection_split") != "valid"):
            raise ValueError("Legacy test metrics do not prove matching checkpoint")
        _check_regression(saved_test, regression_metrics([r["y_true"] for r in test],
                                                        [r["y_pred"] for r in test]), "legacy test metrics")
        test_metrics = _metrics(test, "test", rule)
        if (run / "evaluation_v1/test_metrics.json").exists():
            if test_metrics != _read_json(run / "evaluation_v1/test_metrics.json"):
                raise ValueError("Migration changed test metrics")
            if read_predictions(run / "evaluation_v1/test_predictions.csv", "test", EXPECTED_COUNTS["test"]) != test:
                raise ValueError("Legacy test prediction copies differ")
            _check_enhanced(run / "evaluation_v1/test_predictions.csv", test, delta)
        else:
            reconstructed = True
        _set_scores(evaluation, test_metrics)
    evaluation["metadata"]["migration"] = {
        "from_storage": "evaluation_v1 with separate regression files",
        "legacy_rule_validated_before_changes": True,
        "legacy_file_sha256": hashes, "legacy_sources": legacy_rule["sources"],
        "legacy_evaluation_code_sha256": legacy_rule["code_sha256"],
        "prediction_hash_policy": "SHA256 of canonical validated raw row content; derived class columns checked separately",
        "threshold_action": "retained frozen threshold and original candidate scores; no search",
        "test_action": ("classification recomputed offline from existing predictions and frozen threshold"
                        if reconstructed else "retained verified existing full evaluation" if test is not None else "not_run"),
    }
    # All validation completes before the first mutation. Commit summary last.
    details = run / DETAILS_DIRECTORY
    details.mkdir(exist_ok=True)
    _write_csv(details / "valid_predictions.csv", _enhanced(valid, delta))
    if test is not None:
        _write_csv(details / "test_predictions.csv", _enhanced(test, delta))
    for name in ("history.csv", "run.log"):
        if (run / name).is_file():
            shutil.copyfile(run / name, details / name)
    _write_json(run / "config.json", {"training": training, "evaluation_protocol": protocol})
    _write_json(run / EVALUATION_FILENAME, evaluation)
    return evaluation
