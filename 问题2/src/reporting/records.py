"""Lossless, offline presentation of existing Problem 2 run records.

This module never imports a model, reads a checkpoint with torch, scores
predictions, calibrates a threshold, or selects an epoch. JSON metric values
come exclusively from the original evaluation records. Original bytes are
preserved before either public summary is replaced.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
DISPLAY_KEYS = ("valid", "test", "robustness", "ablation", "attachment3", "diagnostics", "protocol", "artifacts")


def _read(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=128)
def _checkpoint_sha_cached(path: str, size: int, mtime_ns: int) -> str:
    return _sha(Path(path))


def _verify_checkpoint(run: Path, raw: dict) -> None:
    path = run / "best.pt"
    expected = _checkpoint_hash(raw)
    if path.is_file() and expected:
        stat = path.stat()
        actual = _checkpoint_sha_cached(str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        if actual != expected:
            raise ValueError(f"Checkpoint bytes do not match the original evaluation record: {path}")


def _relative(path: Path, run: Path) -> str:
    return Path(os.path.relpath(path.resolve(), run.resolve())).as_posix()


def _ref(path: Path, run: Path, pointer: str | None = None) -> dict:
    result = {"path": _relative(path, run), "sha256": _sha(path)}
    if pointer is not None:
        result["pointer"] = pointer
    return result


def _pointer(*parts: str) -> str:
    return "/" + "/".join(str(p).replace("~", "~0").replace("/", "~1") for p in parts)


def _write(path: Path, value: Any) -> None:
    content = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if path.is_file() and path.read_bytes() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".reporting.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _snapshot(path: Path, run: Path) -> dict:
    digest = _sha(path)
    target = run / "details" / "reporting_originals" / f"{path.stem}.{digest}{path.suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _sha(target) != digest:
            raise ValueError(f"Original snapshot has changed: {target}")
    else:
        shutil.copyfile(path, target)
    return {"path": _relative(target, run), "sha256": digest, "original_path": _relative(path, run)}


def _original(run: Path, name: str, *, preserve: bool = False) -> tuple[dict, Path | None]:
    path = run / name
    document = _read(path)
    if document.get("reporting_schema_version") == SCHEMA_VERSION:
        record = document.get("_source_records", {}).get(name)
        if record is None:
            return {}, None
        original = (run / record["path"]).resolve()
        if not original.is_relative_to(run.resolve()) or not original.is_file():
            raise ValueError(f"Missing or out-of-run original report: {original}")
        if _sha(original) != record["sha256"]:
            raise ValueError(f"Original report fingerprint mismatch: {original}")
        return _read(original), original
    if not path.is_file():
        return {}, None
    if preserve:
        reference = _snapshot(path, run)
        return document, run / reference["path"]
    return document, path


def read_original_evaluation(path: Path) -> dict:
    """Read the authoritative legacy-compatible record, verifying its snapshot.

    Evaluators must update this returned record, rather than modifying the
    derived presentation envelope, before writing their newly evaluated result.
    """
    path = Path(path).resolve()
    if path.name != "evaluation.json":
        raise ValueError("Expected a run's evaluation.json")
    return _original(path.parent, path.name)[0]


def _training_config(config: dict) -> dict:
    return config.get("training", config)


def _metadata_sources(raw: dict) -> dict:
    return raw.get("metadata", {}).get("sources", {})


def _checkpoint_hash(raw: dict) -> str | None:
    return (raw.get("checkpoint_sha256") or _metadata_sources(raw).get("checkpoint_sha256")
            or (raw.get("decision_rule", {}) if isinstance(raw.get("decision_rule"), dict) else {}).get("sources", {}).get("checkpoint_sha256"))


def _binding(raw: dict, run: Path) -> dict:
    return (raw.get("data_binding") or raw.get("metadata", {}).get("provenance", {}).get("data_binding")
            or _read(run / "provenance.json").get("data_binding") or {})


def _binding_summary(raw: dict, run: Path) -> dict | None:
    binding = _binding(raw, run)
    if not binding:
        return None
    identity = {key: deepcopy(binding[key]) for key in
                ("dataset_version", "mask_manifest_sha256", "manifest_fingerprint", "data_sha256",
                 "recipe_sha256", "exploratory_recipe_sha256", "tokenizer_vocab_sha256", "label_fingerprints")
                if key in binding}
    _, original = _original(run, "evaluation.json")
    if raw.get("data_binding") and original is not None:
        identity["full_record"] = _ref(original, run, "/data_binding")
    elif raw.get("metadata", {}).get("provenance", {}).get("data_binding") and original is not None:
        identity["full_record"] = _ref(original, run, "/metadata/provenance/data_binding")
    elif (run / "provenance.json").is_file():
        identity["full_record"] = _ref(run / "provenance.json", run, "/data_binding")
    return identity


def _purpose(raw: dict) -> str | None:
    rule = raw.get("decision_rule", {})
    return (raw.get("purpose") or raw.get("metadata", {}).get("provenance", {}).get("purpose")
            or (rule.get("sources", {}).get("purpose") if isinstance(rule, dict) else None))


def _outputs_root(run: Path) -> Path:
    for parent in (run, *run.parents):
        if parent.name == "outputs":
            return parent
    return run.parent


def _same_model(a: Path, a_raw: dict, b: Path, b_raw: dict) -> bool:
    a_hash, b_hash = _checkpoint_hash(a_raw), _checkpoint_hash(b_raw)
    if not a_hash or a_hash != b_hash:
        return False
    _verify_checkpoint(a, a_raw)
    _verify_checkpoint(b, b_raw)
    ac, bc = _read(a / "config.json"), _read(b / "config.json")
    if not ac or not bc or _training_config(ac) != _training_config(bc):
        return False
    for key in ("code_version", "protocol_version", "method"):
        if a_raw.get(key) is not None and b_raw.get(key) is not None and a_raw[key] != b_raw[key]:
            return False
    for key in ("data_sha256", "mask_manifest_sha256", "manifest_fingerprint"):
        av, bv = _binding(a_raw, a).get(key), _binding(b_raw, b).get(key)
        if av and bv and av != bv:
            return False
    return True


def _resolve_source(run: Path, raw: dict, requested: Path | None, candidates: list[Path]) -> tuple[Path | None, dict]:
    prior_relation = _read(run / "evaluation.json").get("relationships", {})
    explicitly_external = requested is not None or prior_relation.get("source_scope") == "explicit_external"
    evidence = requested if requested is not None else raw.get("source_run")
    if evidence is None:
        prior = _read(run / "evaluation.json")
        evidence = prior.get("relationships", {}).get("source_run")
    result = {"source_run": None, "status": "not_applicable", "match_policy": "checkpoint_sha256_and_effective_training_config"}
    evaluation = bool(evidence) or _purpose(raw) in {"evaluation", "evaluate", "eval", "test"} or "evaluations" in run.parts
    if not evaluation:
        return None, result
    result["status"] = "unresolved"
    possible = []
    if evidence:
        entered = Path(evidence)
        direct = entered if entered.is_absolute() else run / entered
        if (direct.is_dir() and direct.resolve() != run.resolve()
                and (direct.resolve().is_relative_to(_outputs_root(run)) or explicitly_external)):
            possible = [direct.resolve()]
        else:
            # Relocated packages can retain a historical absolute source path.
            basename = str(evidence).replace("\\", "/").rstrip("/").split("/")[-1]
            possible = [p for p in candidates if p.name == basename and p != run]
    else:
        possible = [p for p in candidates if p != run and (p / "best.pt").is_file()]
    matches = []
    for candidate in possible:
        if not candidate.is_relative_to(_outputs_root(run)) and not explicitly_external:
            continue
        other, _ = _original(candidate, "evaluation.json")
        if _same_model(run, raw, candidate, other):
            matches.append(candidate)
    if len(matches) == 1:
        result.update(status="matched", source_run=_relative(matches[0], run))
        if not matches[0].is_relative_to(_outputs_root(run)):
            result["source_scope"] = "explicit_external"
        return matches[0], result
    if requested is not None and not raw:
        entered = requested if requested.is_absolute() else run / requested
        if entered.resolve().is_relative_to(_outputs_root(run)):
            result["requested_source_run"] = _relative(entered, run)
        result["reason"] = "evaluation_record_not_created; no checkpoint or metric identity is asserted"
        return None, result
    if requested is not None:
        raise ValueError("Explicit source_run must match both checkpoint identity and effective training configuration")
    result["reason"] = "ambiguous_model_identity" if len(matches) > 1 else "source_identity_unverified"
    result["candidates"] = [_relative(p, run) for p in matches]
    return None, result


def _metric(value: Any) -> bool:
    return isinstance(value, dict) and any(value.get(key) is not None for key in ("Accuracy", "Macro_F1", "MAE", "Pearson"))


def _split_metrics(raw: dict, split: str) -> dict[str, tuple[dict, str]]:
    direct = raw.get(split)
    if isinstance(direct, dict):
        result = {key: (deepcopy(value), _pointer(split, key)) for key, value in direct.items() if _metric(value)}
        if result:
            return result
    main = raw.get("main", {}).get(split)
    result = {}
    if _metric(main):
        result["main_mixed/mixed_v1_s0"] = (deepcopy(main), _pointer("main", split))
        # 'complete' is the only paired cohort known to be the full clean split.
        complete = raw.get("paired", {}).get(split, {}).get("complete", {})
        clean = complete.get("conditions", complete).get("complete")
        if _metric(clean):
            result["clean"] = (deepcopy(clean), _pointer("paired", split, "complete", "complete"))
    elif _metric(raw.get("summary", {}).get(split)):
        value = deepcopy(raw["summary"][split])
        for key in ("per_class", "confusion_matrix", "additional_metrics"):
            if split in raw.get(key, {}):
                if key == "additional_metrics":
                    value.update(deepcopy(raw[key][split]))
                else:
                    value[key] = deepcopy(raw[key][split])
        result["clean"] = (value, _pointer("summary", split))
    return result


def _prediction_candidates(run: Path, raw: dict, *, known_copies: set[Path] | None = None) -> list[tuple[str, str, Path]]:
    result: dict[tuple[str, str, Path], None] = {}
    managed_copies = {(run / item["path"]).resolve()
                     for item in _read(run / "details" / "reporting_manifest.json").get("predictions", [])
                     if item.get("path") and item.get("source") and item["path"] != item["source"]}
    managed_copies.update(known_copies or ())
    for split in ("valid", "test"):
        conditions = raw.get("conditions", {}).get(split, {})
        for condition, item in conditions.items():
            if isinstance(item, dict) and item.get("path"):
                path = (run / item["path"]).resolve()
                if path.is_relative_to(run) and path.is_file():
                    result[(split, condition, path)] = None
    for path in sorted((run / "details").rglob("*.csv")) if (run / "details").exists() else []:
        if "reporting_originals" in path.parts:
            continue
        name = path.stem
        match = re.match(r"^(valid|test)__(.+)$", name)
        if match:
            result[(match[1], match[2].replace("__", "/"), path)] = None
        elif name in {"valid_predictions", "test_predictions"}:
            split = name.split("_")[0]
            if not any(p == path for _, _, p in result):
                condition = "main_mixed/mixed_v1_s0" if "main" in raw else "clean"
                result[(split, condition, path)] = None
        else:
            rel = path.relative_to(run / "details").parts
            if len(rel) == 4 and rel[0] == "predictions" and rel[2] in {"valid", "test"}:
                result[(rel[2], f"{rel[1]}/{path.stem}", path)] = None
    # Support files already exported directly under the new layout.
    for split in ("valid", "test"):
        folder = run / "predictions" / split
        for path in sorted(folder.rglob("*.csv")) if folder.exists() else []:
            if path.resolve() in managed_copies:
                continue
            condition = path.relative_to(folder).with_suffix("").as_posix()
            if not any(s == split and c == condition for s, c, _ in result):
                result[(split, condition, path)] = None
    return list(result)


def _prediction_leaf(condition: str) -> Path:
    parts = condition.replace("\\", "/").split("/")
    if any(part in {"", ".", ".."} or re.search(r'[<>:"|?*]', part) for part in parts):
        raise ValueError(f"Unsafe prediction condition name: {condition}")
    return Path(*parts).with_suffix(".csv")


def _matched_evaluation_copies(run: Path, sources: list[tuple[Path, dict, Path | None]]) -> set[Path]:
    """Recognize byte-exact child copies even if a prior manifest write stopped.

    Directory names alone never establish ownership: a candidate must match
    a prediction from a checkpoint/configuration-matched evaluation source.
    """
    result = set()
    for child, raw, _ in sources:
        for split, condition, source in _prediction_candidates(child, raw):
            target = run / "predictions" / split / child.name / _prediction_leaf(condition)
            digest = _sha(source)
            alternatives = (target, target.with_name(target.stem + "." + digest[:16] + target.suffix))
            for candidate in alternatives:
                if candidate.is_file() and _sha(candidate) == digest:
                    result.add(candidate.resolve())
    return result


def _predictions(run: Path, raw: dict, *, copy_files: bool = True, relative_to: Path | None = None,
                 target_run: Path | None = None, evaluation_id: str | None = None,
                 known_copies: set[Path] | None = None) -> list[dict]:
    base = relative_to or run
    destination = target_run or run
    records = []
    for split, condition, path in _prediction_candidates(run, raw, known_copies=known_copies):
        folder = destination / "predictions" / split
        if evaluation_id is not None:
            folder /= evaluation_id
        target = folder / _prediction_leaf(condition)
        digest = _sha(path)
        if copy_files:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and _sha(target) != digest:
                # A changed source gets its own name; never replace old evidence.
                target = target.with_name(target.stem + "." + digest[:16] + target.suffix)
                if target.exists() and _sha(target) != digest:
                    raise ValueError(f"Conflicting content-addressed prediction copy: {target}")
            if not target.exists():
                shutil.copyfile(path, target)
        else:
            # An independent evaluation owns its original CSV, not another
            # normalized copy. Its matched training run is the reading view.
            target = path
        records.append({"split": split, "condition": condition, "path": _relative(target, base),
                        "source": _relative(path, base), "sha256": digest,
                        "byte_preserving_copy": target.resolve() != path.resolve()})
    return records


def _merge_split(split: str, sources: list[tuple[Path, dict, Path | None]], run: Path) -> dict:
    values: dict[str, list[tuple[dict, dict]]] = {}
    for source_run, raw, source_path in sources:
        if source_path is None:
            continue
        for condition, (metrics, pointer) in _split_metrics(raw, split).items():
            values.setdefault(condition, []).append((metrics, _ref(source_path, run, pointer)))
    if not values:
        return {"status": "not_run", "dataset": "attachment2", "split": split, "metrics": None}
    result, condition_sources, conflicts = {}, {}, []
    for condition, entries in values.items():
        unique: list[dict] = []
        for metrics, _ in entries:
            if metrics not in unique:
                unique.append(metrics)
        condition_sources[condition] = [ref for _, ref in entries]
        if len(unique) == 1:
            result[condition] = unique[0]
        else:
            # Conflicting independent evaluations require an explicit choice.
            result[condition] = {"status": "conflicting_records", "metrics": None,
                                 "records": [{"metrics": metrics, "source": ref} for metrics, ref in entries]}
            conflicts.append(condition)
    result["_reporting"] = {"status": "completed" if not conflicts else "conflicting_records",
                             "dataset": "attachment2", "split": split,
                             "condition_sources": condition_sources, "conflicts": conflicts,
                             "metric_policy": "original_record_values_without_rescoring"}
    return result


def _frozen_resources(run: Path, raw: dict) -> tuple[Path | None, dict]:
    binding = _binding(raw, run)
    expected = binding.get("mask_manifest_sha256")
    if not expected:
        return None, {}
    project = next((p for p in run.parents if (p / "resources" / "missing_data").is_dir()), None)
    if project is None:
        return None, {}
    for path in sorted((project / "resources" / "missing_data").glob("*/manifest.json")):
        if _sha(path) == expected:
            return path.parent, _read(path)
    return None, {}


def _missingness(resource: Path | None, manifest: dict, run: Path, split: str, condition: str) -> dict:
    result = {"status": "not_recorded", "target": None, "actual": None,
              "rate_denominator": "modalities.*.rate = modality k / content length; actual.slot_rate = sum of T/A/V k / (3 * content length). union_k / content length is a distinct temporal union rate.",
              "native_missingness": "native unobserved positions are recorded separately from artificial masks",
              "duration_unit": "sequence positions, not seconds"}
    if resource is None:
        return result
    metadata = resource / "metadata" / "paired" / split / f"{condition}.jsonl"
    frozen = manifest.get("files", {}).get(metadata.relative_to(resource).as_posix(), {})
    if not metadata.is_file() or frozen.get("sha256") != _sha(metadata):
        return result
    reference = _ref(metadata, run)
    result.update(status="recorded_per_sample", target={"source": reference, "fields": ["plan.rate", "plan.modality_set", "plan.position", "plan.shape", "plan.synchrony"]},
                  actual={"source": reference, "fields": ["actual.slot_rate", "modalities.*.rate", "modalities.*.k", "modalities.*.segment_lengths", "modalities.*.native_unobserved", "modalities.*.overlap_native_unobserved"],
                          "scope": "join per-sample metadata to this panel's cohort; no new aggregate is computed"})
    statistics = resource / "statistics.json"
    entry = manifest.get("files", {}).get("statistics.json", {})
    if statistics.is_file() and entry.get("sha256") == _sha(statistics):
        for index, view in enumerate(_read(statistics).get(split, {}).get("views", [])):
            if view.get("evaluation_family") == "paired" and view.get("condition_id") == condition:
                result["condition_statistics"] = {"source": _ref(statistics, run, _pointer(split, "views", str(index))),
                                                  "scope": "all applicable samples of this condition; may differ from panel cohort",
                                                  **{key: deepcopy(view[key]) for key in ("applicable_samples", "excluded_samples", "target_rate_counts", "slot_micro_rate", "slot_macro_rate") if key in view}}
                break
    return result


def _robustness(sources: list[tuple[Path, dict, Path | None]], run: Path, predictions: list[dict]) -> dict:
    panels: dict = {"valid": {}, "test": {}}
    conflicts = []
    for source_run, raw, original in sources:
        if original is None:
            continue
        resource, manifest = _frozen_resources(source_run, raw)
        paired = raw.get("paired", {})
        for split in ("valid", "test"):
            for panel, data in paired.get(split, {}).items():
                if not isinstance(data, dict):
                    continue
                conditions = data.get("conditions", data)
                metrics = {key: value for key, value in conditions.items() if _metric(value)}
                if not metrics:
                    continue
                counts = {value.get("n") for value in metrics.values() if value.get("n") is not None}
                common_n = next(iter(counts)) if len(counts) == 1 else None
                cohort = None
                if isinstance(data.get("cohort_ids"), list):
                    cohort = _ref(original, run, _pointer("paired", split, panel, "cohort_ids"))
                    cohort["n"] = len(data["cohort_ids"])
                elif resource is not None:
                    panel_file = resource / "panels" / f"{split}.json"
                    recorded = _read(panel_file).get(panel, {})
                    identity = manifest.get("files", {}).get(f"panels/{split}.json", {}).get("sha256")
                    if (panel_file.is_file() and identity == _sha(panel_file)
                            and len(recorded.get("ids", [])) == common_n
                            and set(metrics) == set(recorded.get("condition_ids", []))):
                        cohort = _ref(panel_file, run, _pointer(panel, "ids"))
                        cohort["n"] = common_n
                record = {"status": "completed", "common_n": common_n, "cohort": cohort,
                          "source": _ref(original, run, _pointer("paired", split, panel)), "conditions": {}}
                for condition, metric in metrics.items():
                    paths = [item for item in predictions if item["split"] == split and item["condition"] == "paired/" + condition
                             and (run / item["source"]).resolve().is_relative_to(source_run)]
                    suffix = ("conditions", condition) if "conditions" in data else (condition,)
                    record["conditions"][condition] = {"metrics": deepcopy(metric),
                        "source": _ref(original, run, _pointer("paired", split, panel, *suffix)),
                        "prediction": paths[0] if len(paths) == 1 else None,
                        "prediction_scope": "condition applicable samples; use panel cohort to reproduce these metrics",
                        "missingness": _missingness(resource, manifest, run, split, condition)}
                previous = panels[split].get(panel)
                if previous is None:
                    panels[split][panel] = record
                else:
                    if previous.get("status") != "multiple_records":
                        panels[split][panel] = {"status": "multiple_records", "records": [previous]}
                    panels[split][panel]["records"].append(record)
                    conflicts.append(f"{split}/{panel}")
    for split in panels:
        for panel in ("modality", "rate", "position", "shape", "synchrony"):
            panels[split].setdefault(panel, {"status": "not_run", "metrics": None, "common_n": None, "cohort": None, "conditions": {}})
    complete = any(item.get("status") != "not_run" for split in panels.values() for item in split.values())
    missing = any(item.get("status") == "not_run" for split in panels.values() for item in split.values())
    status = "partial" if complete and missing else "completed" if complete else "not_run"
    return {"status": status, "panels": panels,
            "multiple_records": sorted(set(conflicts)),
            "interpretation": "Compare conditions within the same panel cohort; never substitute full-split clean for a paired clean baseline."}


def _protocol(raw: dict, config: dict, run: Path) -> dict:
    cfg = _training_config(config)
    rule = raw.get("decision_rule", cfg.get("decision_rule"))
    legacy_protocol = config.get("evaluation_protocol", {})
    return {"code_version": raw.get("code_version", cfg.get("code_version", "legacy_regression")),
            "protocol_version": raw.get("protocol_version", cfg.get("protocol_version")),
            "evaluation_version": raw.get("metadata", {}).get("evaluation_version", legacy_protocol.get("evaluation_version")),
            "decision_rule": deepcopy(rule), "decision": deepcopy(raw.get("decision")),
            "class_order": deepcopy(raw.get("class_order", cfg.get("class_order", legacy_protocol.get("class_order")))),
            "F1_definition": "unweighted mean of class F1 in the recorded class_order (Macro_F1); retain the recorded zero_division rule",
            "selection_protocol": deepcopy(raw.get("selection_protocol")),
            "legacy_evaluation_protocol": deepcopy(legacy_protocol) or None,
            "data_binding": _binding_summary(raw, run),
            "data_identity": {"dataset": "attachment2", "data_path_as_recorded": cfg.get("data_path"),
                              "data_sha256": _binding(raw, run).get("data_sha256", _metadata_sources(raw).get("data_sha256"))},
            "metric_policy": "No metric recalculation, calibration, checkpoint selection, or prediction clipping during organization; null Pearson and its recorded reason are retained."}


def _history_summary(run: Path) -> dict:
    history_path = run / "history.json"
    history = _read(history_path, [])
    if isinstance(history, dict):
        history = history.get("history", history.get("epochs", []))
    if not isinstance(history, list):
        history = []
    evidence = history_path if history_path.exists() else run / "details" / "history.csv"
    if not history and evidence.is_file() and evidence.suffix == ".csv":
        with evidence.open(encoding="utf-8-sig", newline="") as stream:
            history = list(csv.DictReader(stream))
    if not history:
        return {"epochs_completed": None, "best_epoch": None, "source": None}
    epochs = []
    for row in history:
        try:
            epochs.append(int(row["epoch"]))
        except (KeyError, TypeError, ValueError):
            pass
    last_best = history[-1].get("best_epoch") if isinstance(history[-1], dict) else None
    return {"epochs_completed": max(epochs) if epochs else None, "best_epoch": last_best,
            "source": _ref(evidence, run), "record_count": len(history),
            "limitation": "History proves recorded epochs, not completion or the reason execution stopped."}


def _training(run: Path, raw: dict, config: dict, old_training: dict, sources: dict, relationship: dict, runtime: dict | None, prior: dict) -> dict:
    cfg = _training_config(config)
    is_eval = relationship["status"] != "not_applicable" or _purpose(raw) in {"evaluation", "evaluate", "eval", "test"}
    summary = deepcopy(raw.get("training_summary", {}))
    history = _history_summary(run)
    active_runtime = runtime if runtime and runtime.get("command") == "train" else prior.get("runtime")
    if is_eval:
        state, stop_reason, completed, best = "not_applicable", None, None, None
        summary = {}
        active_runtime = None
    else:
        stop_reason = summary.get("stop_reason")
        completed = summary.get("epochs_completed", history["epochs_completed"])
        best = summary.get("best_epoch", history["best_epoch"])
        state = "completed" if stop_reason in {"early_stopping", "max_epochs", "completed"} else "unknown"
        if active_runtime:
            state = active_runtime.get("status", state)
            if state in {"failed", "interrupted"}:
                stop_reason = active_runtime.get("stop_reason", state)
        if completed is None and not raw and not history.get("record_count") and state == "unknown":
            state = "not_run"
    records = {name: _ref(run / name, run) for name in ("config.json", "history.json", "details/history.csv", "preprocessing.json", "provenance.json") if (run / name).is_file()}
    record = {"reporting_schema_version": SCHEMA_VERSION, "format_version": SCHEMA_VERSION,
              "run_id": run.name, "kind": "evaluation" if is_eval else "training", "status": state,
              "code_version": raw.get("code_version", cfg.get("code_version", old_training.get("code_version"))),
              "method": raw.get("method", cfg.get("method", old_training.get("method"))), "seed": cfg.get("seed"),
              "configuration": records.get("config.json"), "initialization": cfg.get("initialization"),
              "data_protocol": {"data_path_as_recorded": cfg.get("data_path"), "artifact_path_as_recorded": cfg.get("artifact_path"), "data_binding": _binding_summary(raw, run)},
              "selection_protocol": deepcopy(raw.get("selection_protocol")) if not is_eval else None,
              "early_stopping": {"patience": summary.get("early_stop_patience", cfg.get("early_stop_patience")), "min_delta": summary.get("early_stop_min_delta", cfg.get("early_stop_min_delta"))} if not is_eval else None,
              "max_epochs": summary.get("max_epochs", cfg.get("epochs")) if not is_eval else None,
              "epochs_completed": completed, "best_epoch": best, "stop_reason": stop_reason,
              "training_summary": summary, "history_evidence": history if not is_eval else None,
              "runtime": active_runtime, "source_run": relationship.get("source_run"),
              "same_model_runs": deepcopy(relationship.get("same_model_runs", [])),
              "records": records, "_source_records": sources,
              "note": "Derived summary, not a resume checkpoint. Missing facts stay null. Evaluation directories do not represent additional training runs. Directories with the same checkpoint and effective configuration may be copied records; their count is not evidence of independent training repetitions."}
    # Every old field remains accessible; overlapping legacy meanings are kept
    # together rather than silently replaced (v1 selection_protocol sometimes
    # described threshold calibration rather than training model selection).
    if old_training:
        record["legacy_training"] = deepcopy(old_training)
    for key in ("training_seconds", "elapsed_seconds", "duration_seconds", "total_seconds"):
        if key in raw.get("training_summary", {}) and not is_eval:
            record[key] = raw["training_summary"][key]
    if not is_eval and record["selection_protocol"] is None:
        selection = {key: summary[key] for key in ("selection_split", "selection_metric", "selection_family", "selection_condition") if key in summary}
        record["selection_protocol"] = selection or None
    return record


def _training_inputs(run: Path, runtime: dict | None, *, preserve: bool) -> dict:
    raw, original = _original(run, "evaluation.json", preserve=preserve)
    old, old_path = _original(run, "training.json", preserve=preserve)
    records = {}
    for name, path in (("evaluation.json", original), ("training.json", old_path)):
        # A read-only preview must not advertise an overwritten public file as
        # a preserved original. The writer always snapshots before reaching here.
        if path is not None and "reporting_originals" in path.parts:
            records[name] = {**_ref(path, run), "original_path": name}
    candidates = sorted({p.parent.resolve() for p in _outputs_root(run).rglob("evaluation.json") if "reporting_originals" not in p.parts})
    _, relationship = _resolve_source(run, raw, None, candidates)
    relationship["same_model_runs"] = deepcopy(_read(run / "evaluation.json").get("relationships", {}).get("same_model_runs", []))
    return _training(run, raw, _read(run / "config.json"), old, records, relationship,
                     runtime, _read(run / "training.json"))


def summarize_training(run: Path, *, runtime: dict | None = None) -> dict:
    """Read-only training summary preview. Use the writer to persist safely."""
    return _training_inputs(Path(run).resolve(), runtime, preserve=False)


def write_training_summary(run: Path, *, runtime: dict | None = None) -> dict:
    """Write a training run's summary, never a synthetic evaluation training run."""
    run = Path(run).resolve()
    summary = _training_inputs(run, runtime, preserve=True)
    if summary["kind"] == "evaluation":
        return summary
    _write(run / "training.json", summary)
    manifest_path = run / "details" / "reporting_manifest.json"
    manifest = _read(manifest_path)
    originals = {item["path"]: item for item in manifest.get("originals", [])}
    for path in sorted((run / "details" / "reporting_originals").glob("*.json")):
        originals[_relative(path, run)] = {"path": _relative(path, run), "sha256": _sha(path)}
    manifest.update(reporting_schema_version=SCHEMA_VERSION,
                    originals=[originals[key] for key in sorted(originals)])
    _write(manifest_path, manifest)
    return summary


def _retained_ablation(run: Path, previous: dict) -> dict | None:
    value = previous.get("ablation")
    if not isinstance(value, dict) or value.get("generator") != "problem2.reporting.comparisons":
        return None
    references = value.get("comparisons", [])
    for item in references:
        path = (run / item.get("path", "")).resolve()
        if not path.is_relative_to(_outputs_root(run)) or not path.is_file():
            return None
        comparison = _read(path)
        manifest = _read(path.parent / "manifest.json")
        if (comparison.get("generator", {}).get("id") != "problem2.reporting.comparisons"
                or manifest.get("files", {}).get(path.name) != _sha(path)):
            return None
    return deepcopy(value) if references else None


def organize_run(run: Path, *, source_run: Path | None = None, runtime: dict | None = None) -> dict:
    """Create presentation records from frozen existing evidence, idempotently.

    ``runtime`` is optional evidence from the invoking command, never an
    inferred timing. Calling this function alone performs no experiment.
    Returns the written evaluation document. Related evaluations are included
    only after both checkpoint and effective configuration identities match.
    """
    run = Path(run).resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    previous = _read(run / "evaluation.json")
    prior_training = _read(run / "training.json")
    raw, original = _original(run, "evaluation.json", preserve=True)
    _verify_checkpoint(run, raw)
    old_training, training_original = _original(run, "training.json", preserve=True)
    source_records = {}
    for name, path in (("evaluation.json", original), ("training.json", training_original)):
        if path is not None:
            source_records[name] = {**_ref(path, run), "original_path": name}
    config = _read(run / "config.json")
    root = _outputs_root(run)
    candidates = sorted({p.parent.resolve() for p in root.rglob("evaluation.json") if "reporting_originals" not in p.parts})
    parent, relationships = _resolve_source(run, raw, Path(source_run) if source_run is not None else None, candidates)
    is_evaluation = (relationships["status"] != "not_applicable"
                     or _purpose(raw) in {"evaluation", "evaluate", "eval", "test"})
    relationships["same_model_runs"] = []
    for candidate in candidates:
        if candidate == run:
            continue
        candidate_raw, _ = _original(candidate, "evaluation.json")
        if _same_model(run, raw, candidate, candidate_raw):
            relationships["same_model_runs"].append({"run": _relative(candidate, run),
                "checkpoint_sha256": _checkpoint_hash(candidate_raw),
                "identity": "same_checkpoint_and_effective_training_configuration",
                "interpretation": "Does not establish an additional training repetition or select a parent run."})
    sources = [(run, raw, original)]
    related = []
    if parent is None and relationships["status"] == "not_applicable":
        for candidate in candidates:
            if candidate == run:
                continue
            candidate_raw, _ = _original(candidate, "evaluation.json")
            owner, relation = _resolve_source(candidate, candidate_raw, None, candidates)
            if owner == run and relation["status"] == "matched":
                candidate_raw, source_path = _original(candidate, "evaluation.json", preserve=True)
                sources.append((candidate, candidate_raw, source_path))
                related.append({"run": _relative(candidate, run), "evaluation": _ref(source_path, run), "identity_status": "matched"})
    relationships["evaluations"] = related
    predictions = _predictions(run, raw, copy_files=not is_evaluation,
                               known_copies=_matched_evaluation_copies(run, sources[1:]))
    for candidate, candidate_raw, _ in sources[1:]:
        # Keep each evaluation's identity in the parent's reading view, so
        # repeated evaluations never replace each other's saved predictions.
        predictions.extend(_predictions(candidate, candidate_raw, relative_to=run,
                                        target_run=run, evaluation_id=candidate.name))
    valid = _merge_split("valid", sources, run)
    test = _merge_split("test", sources, run)
    checkpoint_run = parent or run
    checkpoint = checkpoint_run / "best.pt"
    checkpoint_ref = {"path": _relative(checkpoint, run), "sha256": _checkpoint_hash(raw), "fingerprint_source": "original_evaluation_record"} if checkpoint.is_file() else None
    training_reference = (_relative(parent / "training.json", run) if parent is not None
                          else None if is_evaluation else "training.json")
    artifacts = {"checkpoint": checkpoint_ref, "configuration": _ref(run / "config.json", run) if (run / "config.json").is_file() else None,
                 "predictions": predictions, "training": training_reference, "original_records": source_records,
                 "manifest": "details/reporting_manifest.json"}
    report = {"valid": valid, "test": test, "robustness": _robustness(sources, run, predictions),
              "ablation": _retained_ablation(run, previous) or {"status": "not_run", "comparisons": [], "note": "No comparison is inferred from method names or from a different code/protocol version."},
              "attachment3": {"status": "not_run", "scoring_status": "not_applicable_unlabeled", "metrics": None, "coverage_count": None, "csv": None, "prediction_distribution": None},
              "diagnostics": {"status": "not_run"}, "protocol": _protocol(raw, config, run), "artifacts": artifacts}
    # Older readers still expect their original fields inline. Four-model
    # evaluation readers use the verified original record, so large duplicated
    # evidence can stay in its immutable snapshot with explicit pointers.
    referenced_fields = {"paired", "data_binding"} if (
        is_evaluation and _training_config(config).get("code_version", raw.get("code_version"))
        == "four_models_avstd_v1" and original is not None) else set()
    legacy_fields_refs = {}
    for key, value in raw.items():
        if key in referenced_fields:
            legacy_fields_refs[key] = _ref(original, run, _pointer(key))
        elif key not in DISPLAY_KEYS:
            report[key] = deepcopy(value)
    if legacy_fields_refs:
        report["legacy_fields_refs"] = legacy_fields_refs
    report.update(reporting_schema_version=SCHEMA_VERSION, relationships=relationships, _source_records=source_records)
    if relationships.get("source_run") is not None:
        report["source_run"] = relationships["source_run"]
    if _checkpoint_hash(raw) is not None:
        report["checkpoint_sha256"] = _checkpoint_hash(raw)
    evaluation_runtime = runtime if runtime and runtime.get("command") == "evaluate" else previous.get("evaluation_runtime")
    if evaluation_runtime:
        report["evaluation_runtime"] = evaluation_runtime
    if not is_evaluation:
        training = _training(run, raw, config, old_training, source_records, relationships, runtime, prior_training)
        _write(run / "training.json", training)
        from .artifacts import build_analysis
        additions = build_analysis(run, report)
        for key in ("diagnostics", "artifacts"):
            if isinstance(additions.get(key), dict):
                report[key].update(additions[key])
    else:
        report["diagnostics"] = {
            "status": "refer_to_source_run" if parent is not None else "not_generated",
            "source_run": relationships.get("source_run"),
            "report": _relative(parent / "analysis" / "report.md", run) if parent is not None else None,
            "note": "Independent evaluation evidence is analyzed once in the matched training run's analysis directory.",
        }
    manifest_path = run / "details" / "reporting_manifest.json"
    manifest = _read(manifest_path)
    originals = {item["path"]: item for item in manifest.get("originals", [])}
    for path in sorted((run / "details" / "reporting_originals").glob("*.json")):
        originals[_relative(path, run)] = {"path": _relative(path, run), "sha256": _sha(path)}
    _write(manifest_path, {"reporting_schema_version": SCHEMA_VERSION,
                           "originals": [originals[key] for key in sorted(originals)], "predictions": predictions,
                           "policy": "Original records, prediction CSVs and weights are retained. Public summaries are derived; no model execution or metric recalculation."})
    _write(run / "evaluation.json", report)
    return report
