"""Single-view multitask prediction, authenticated records and lossless exports.

This module does not use the legacy regression-threshold evaluation pipeline.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from multitask.metrics import CLASS_NAMES, evaluation_metrics, labels_from_intensity


PROBABILITY_FIELDS = tuple(f"p_{name}" for name in CLASS_NAMES)
FIELDS = ("split", "index", "id", "y_true", "y_pred", "true_class", "pred_class", *PROBABILITY_FIELDS)
INPUT_FIELDS = ("text", "audio", "vision", "audio_lengths", "vision_lengths")


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Invalid {name}")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Invalid {name}") from error
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite {name}")
    return result


def _integer(value, name):
    if not isinstance(value, (str, int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Invalid {name}")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {name}") from error
    if str(result) != str(value) or result < 0:
        raise ValueError(f"Invalid {name}")
    return result


def validate_records(records, split=None, cohort=None, *, full=False):
    """Check saved fields and optionally bind original sample identities/labels.

    ``cohort`` is the split-local output of MixDataStore.cohort. No split is
    loaded here, so validation of valid never triggers a train/test read.
    """
    records = list(records)
    if not records:
        raise ValueError("Expected nonempty prediction records")
    if split is None:
        if not isinstance(records[0], dict):
            raise ValueError("Expected prediction record objects")
        split = records[0].get("split")
    if split not in ("train", "valid", "test"):
        raise ValueError("Unknown prediction split")
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
        if not isinstance(row, dict) or any(key not in row for key in FIELDS) or row["split"] != split:
            raise ValueError("Missing multitask prediction fields or incorrect split")
        index = _integer(row["index"], "index")
        if index <= previous:
            raise ValueError("Prediction indices must be unique and in original order")
        previous = index
        sample_id = row["id"]
        if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in seen_ids:
            raise ValueError("Prediction IDs must be nonempty and unique")
        seen_ids.add(sample_id)
        truth, prediction = _finite(row["y_true"], "y_true"), _finite(row["y_pred"], "y_pred")
        true_class = _integer(row["true_class"], "true_class")
        pred_class = _integer(row["pred_class"], "pred_class")
        if true_class != int(labels_from_intensity([truth])[0]) or pred_class not in (0, 1, 2):
            raise ValueError("Invalid explicit classes or class target differs from label sign")
        probabilities = [_finite(row[key], key) for key in PROBABILITY_FIELDS]
        if any(p < 0 or p > 1 for p in probabilities) or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6, rel_tol=0):
            raise ValueError("Class probabilities must be in [0,1] and sum to one")
        if pred_class != int(np.argmax(probabilities)):
            raise ValueError("pred_class differs from probability argmax")
        if source is not None:
            original = source.get(index)
            if original is None or sample_id != original["id"] or truth != _finite(original["y_true"], "cohort y_true"):
                raise ValueError("Prediction identity/label differs from the source cohort")
        result.append({"split": split, "index": index, "id": sample_id, "y_true": truth,
                       "y_pred": prediction, "true_class": true_class, "pred_class": pred_class,
                       **dict(zip(PROBABILITY_FIELDS, probabilities))})
    return result


def metrics_from_records(records):
    records = validate_records(records)
    return evaluation_metrics([row["y_true"] for row in records], [row["y_pred"] for row in records],
                              [row["pred_class"] for row in records])


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def evaluate(model, batches, device, split):
    """Evaluate current input only; restore the caller's train/eval mode.

    Labels, references and reconstruction support are not passed to prediction.
    Raw logits are converted to probabilities solely for exported records.
    """
    import torch
    from multitask.model import predict_batch

    was_training = model.training
    records = []
    model.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                output = predict_batch(model, {key: batch[key] for key in INPUT_FIELDS}, device)
                logits = np.asarray(_numpy(output["logits"]), dtype=np.float64)
                intensity = np.asarray(_numpy(output["intensity"]), dtype=np.float64).reshape(-1)
                truth = np.asarray(_numpy(batch["labels"]), dtype=np.float64).reshape(-1)
                indices = _numpy(batch["index"]).reshape(-1)
                ids = batch["id"]
                count = len(truth)
                if logits.shape != (count, 3) or len(intensity) != count or len(indices) != count or len(ids) != count:
                    raise ValueError("Prediction output/batch shapes disagree")
                if not np.isfinite(logits).all():
                    raise ValueError("Nonfinite logits")
                probabilities = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                predicted_class, true_class = np.argmax(logits, axis=1), labels_from_intensity(truth)
                for i in range(count):
                    records.append({"split": split, "index": indices[i], "id": ids[i],
                                    "y_true": truth[i], "y_pred": intensity[i],
                                    "true_class": int(true_class[i]), "pred_class": int(predicted_class[i]),
                                    **dict(zip(PROBABILITY_FIELDS, probabilities[i]))})
    finally:
        model.train(was_training)
    records.sort(key=lambda row: _integer(row["index"], "index"))
    records = validate_records(records, split)
    return metrics_from_records(records), records


def load_records(path, *, split=None, cohort=None, full=False):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Malformed prediction CSV header")
        return validate_records(reader, split, cohort, full=full)


def save_records(path, records, *, overwrite=False):
    """Preserve all class/intensity fields and refuse differing canonical writes."""
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
        writer.writerows(records)
    temporary.replace(path)


def save_json(path, value):
    """Atomic JSON export; retain the caller's full condition/protocol metadata."""
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
