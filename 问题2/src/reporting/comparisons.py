"""Compare existing four-model reports without inference, rescoring or run selection."""
from __future__ import annotations

import csv
import hashlib
from html import escape
import io
import json
import math
import os
from pathlib import Path
import tempfile


OWNER = "problem2.reporting.comparisons"
VERSION = 1
METRICS = ("n", "Accuracy", "Macro_F1", "MAE", "Pearson", "Pearson_status")
PAIRS = (("M0", "M1"), ("M2", "M3"))


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pointer(parts):
    return "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in parts)


def _ref(path, base, pointer=None):
    value = {"path": Path(os.path.relpath(path, base)).as_posix(), "sha256": _hash(path)}
    if pointer is not None:
        value["json_pointer"] = pointer
    return value


def _metric_leaves(value, parts=(), cohort=None):
    """Read metric objects, carrying explicit paired cohort evidence downward."""
    if not isinstance(value, dict):
        return
    for name in ("cohort_ids", "common_ids", "ids"):
        if isinstance(value.get(name), list):
            cohort = value[name]
            break
    if all(name in value for name in ("n", "Accuracy", "Macro_F1", "MAE", "Pearson")):
        if (type(value["n"]) is int and value["n"] > 0
                and value.get("status") not in ("not_run", "not_applicable", "not_applicable_unlabeled")
                and all(type(value[name]) in (int, float) and math.isfinite(value[name])
                        for name in ("Accuracy", "Macro_F1", "MAE"))
                and (value["Pearson"] is None or
                     type(value["Pearson"]) in (int, float) and math.isfinite(value["Pearson"]))):
            yield parts, {key: value[key] for key in METRICS if key in value}, cohort
        return
    for name, nested in value.items():
        if name.startswith("_") or name in ("sources", "artifacts", "diagnostics"):
            continue
        yield from _metric_leaves(nested, (*parts, name), cohort)


def _extract(run, report, destination):
    results = {}
    for split in ("valid", "test"):
        section = report.get(split, {})
        sources = section.get("_reporting", {}).get("condition_sources", {})
        for condition, metrics in section.items():
            if not isinstance(metrics, dict) or not list(_metric_leaves(metrics)):
                continue
            references = sources.get(condition, [])
            if not references:
                continue
            verified = [_verified_reference(reference, run, destination, expected_metrics=metrics)
                        for reference in references]
            results[(split, condition)] = {"metrics": {key: metrics[key] for key in METRICS if key in metrics},
                "evaluation": verified[0], "sources": verified, "cohort_ids": None}
    for split, panels in report.get("robustness", {}).get("panels", {}).items():
        for name, panel in panels.items():
            if panel.get("status") != "completed":
                continue
            cohort = None
            cohort_reference = None
            indices = indices_reference = None
            if panel.get("cohort"):
                cohort_reference, cohort = _verified_reference(panel["cohort"], run, destination, return_value=True)
                if not isinstance(cohort, list):
                    raise ValueError("Paired cohort reference must identify the saved ID list")
            if panel.get("source"):
                source_reference, panel_record = _verified_reference(panel["source"], run, destination, return_value=True)
                if isinstance(panel_record, dict) and isinstance(panel_record.get("cohort_indices"), list):
                    indices = panel_record["cohort_indices"]
                    indices_reference = {**source_reference,
                        "json_pointer": source_reference.get("json_pointer", "") + "/cohort_indices"}
            for condition, value in panel.get("conditions", {}).items():
                metrics = value.get("metrics")
                if not isinstance(metrics, dict) or not list(_metric_leaves(metrics)) or not value.get("source"):
                    continue
                reference = _verified_reference(value["source"], run, destination, expected_metrics=metrics)
                results[("robustness", split, name, condition)] = {
                    "metrics": {key: metrics[key] for key in METRICS if key in metrics},
                    "evaluation": reference, "sources": [reference], "cohort_ids": cohort,
                    "_cohort_reference": cohort_reference, "_cohort_indices": indices,
                    "_cohort_indices_reference": indices_reference}
    return results


def _verified_reference(reference, run, destination, expected_metrics=None, return_value=False):
    path = (run / reference["path"]).resolve()
    package_outputs = next((ancestor for ancestor in (run, *run.parents) if ancestor.name == "outputs"), run.parent)
    project = package_outputs.parent
    if not path.is_relative_to(project) or _hash(path) != reference["sha256"]:
        raise ValueError("Saved comparison source escapes the project or has changed: " + str(path))
    value = _read(path)
    pointer = reference.get("pointer", reference.get("json_pointer", ""))
    if pointer:
        for token in pointer.split("/")[1:]:
            token = token.replace("~1", "/").replace("~0", "~")
            value = value[int(token)] if isinstance(value, list) else value[token]
    if expected_metrics is not None and value != expected_metrics:
        raise ValueError("Saved comparison source does not equal the displayed metrics")
    converted = _ref(path, destination, pointer)
    return (converted, value) if return_value else converted


def _scan(outputs):
    groups, skipped = {}, []
    # Evaluation directories and historical subfolders are inspected too, but
    # only explicit four-model training reports can become comparison members.
    for report_path in sorted(outputs.rglob("evaluation.json")):
        if "comparisons" in report_path.relative_to(outputs).parts:
            continue
        try:
            report = _read(report_path)
            if not isinstance(report, dict) or report.get("purpose") != "train":
                continue
            if report.get("code_version") != "four_models_avstd_v1":
                continue
            run = report_path.parent
            configuration = _read(run / "config.json")
            configuration = configuration.get("training", configuration)
            method, seed = report.get("method"), configuration.get("seed")
            protocol = report.get("protocol_version")
            if protocol is None:
                protocol = report.get("selection_protocol", {}).get("protocol_version")
            if protocol != "four_models_accuracy_v1" or method not in ("M0", "M1", "M2", "M3"):
                raise ValueError("Unsupported or unspecified four-model protocol/method")
            if configuration.get("method") != method or type(seed) is not int:
                raise ValueError("Configuration method/seed does not identify this training run")
            if report.get("reporting_schema_version") != 2:
                raise ValueError("Organize this training report before building comparisons")
            groups.setdefault((protocol, seed), {}).setdefault(method, []).append((run, report, configuration))
        except (ValueError, OSError, TypeError, AttributeError) as error:
            skipped.append({"path": report_path.relative_to(outputs).as_posix(), "reason": str(error)})
    return groups, skipped


def _source(member, destination):
    run, report, config = member
    original = report.get("_source_records", {}).get("evaluation.json")
    if original is None:
        raise ValueError("Organized training run has no original evaluation evidence")
    return {"run_id": run.name, "run": Path(os.path.relpath(run, destination)).as_posix(),
            "method": report["method"], "seed": config["seed"],
            "checkpoint_sha256": report.get("checkpoint_sha256"),
            "evaluation": _verified_reference(original, run, destination),
            "configuration": _ref(run / "config.json", destination)}


def _data_identity(member):
    run, report, config = member
    binding = report.get("data_binding")
    if not isinstance(binding, dict) and (run / "provenance.json").is_file():
        provenance = _read(run / "provenance.json")
        binding = provenance.get("data_binding")
    if isinstance(binding, dict):
        return {"kind": "frozen_data_binding", "value": binding}
    return {"kind": "unverified_configuration_paths", "value": {
        key: config.get(key) for key in ("data_path", "artifact_path")}}


def _data_identity_summary(member, identity, destination):
    """Keep complete equality checks in memory, referencing long saved bindings."""
    if identity["kind"] != "frozen_data_binding":
        return identity
    run, report, _ = member
    binding = identity["value"]
    provenance = run / "provenance.json"
    if provenance.is_file() and _read(provenance).get("data_binding") == binding:
        reference = _ref(provenance, destination, "/data_binding")
    else:
        original = report.get("_source_records", {}).get("evaluation.json")
        if original is None:
            raise ValueError("Saved data binding has no authoritative record")
        reference, recorded = _verified_reference(
            {**original, "pointer": "/data_binding"}, run, destination, return_value=True)
        if recorded != binding:
            raise ValueError("Saved data binding differs from its original record")
    canonical = json.dumps(binding, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {"kind": identity["kind"], "binding_sha256": hashlib.sha256(canonical).hexdigest(),
            "hash_encoding": "UTF-8 canonical JSON; sorted keys; compact separators; ensure_ascii=false",
            "summary": {key: binding[key] for key in ("dataset_version", "data_sha256",
                         "mask_manifest_sha256", "manifest_fingerprint", "recipe_sha256",
                         "tokenizer_vocab_sha256") if key in binding},
            "full_record": reference,
            "comparison_policy": "Full saved bindings are compared for equality before summarization."}


def _compact_cohort(member):
    """Serialize cohort evidence only after the original comparability checks."""
    result = {key: value for key, value in member.items()
              if key != "cohort_ids" and not key.startswith("_cohort_")}
    ids = member.get("cohort_ids")
    if ids is None:
        result["cohort"] = None
        return result
    reference = member.get("_cohort_reference")
    if reference is None:
        raise ValueError("Saved cohort IDs have no verified source reference")
    def list_reference(values, source):
        canonical = json.dumps(values, ensure_ascii=False, separators=(",", ":"),
                               sort_keys=True, allow_nan=False).encode("utf-8")
        return {"n": len(values), "content_sha256": hashlib.sha256(canonical).hexdigest(),
                "hash_encoding": "UTF-8 canonical JSON; sorted keys; compact separators; ensure_ascii=false",
                "full_record": source}
    result["cohort"] = {"common_n": len(ids), "ids": list_reference(ids, reference)}
    indices = member.get("_cohort_indices")
    if indices is not None:
        result["cohort"]["indices"] = list_reference(indices, member["_cohort_indices_reference"])
    result["cohort"]["comparison_policy"] = (
        "Sample counts and full saved ID lists are compared before summarization; "
        "indices are retained as source evidence, not an additional comparison rule.")
    return result


def _comparison(protocol, seed, pair, candidates, destination):
    members = {method: [_source(member, destination) for member in candidates.get(method, [])]
               for method in pair}
    report = {"generator": {"id": OWNER, "version": VERSION}, "comparison_id": destination.name,
              "code_version": "four_models_avstd_v1", "protocol_version": protocol,
              "seed": seed, "methods": list(pair), "status": "not_run", "main_results_status": "not_run",
              "robustness_status": "not_run", "candidates": members,
              "status_scope": "Main valid/test condition comparisons only; robustness_status is independent.",
              "comparisons": [], "artifacts": {"csv": "comparison.csv", "figure": "figures/accuracy.svg"},
              "note": "Only saved metric values are compared. No training, inference, recalibration, "
                      "rescoring, best-run selection, deltas or statistical significance tests are performed."}
    if any(len(members[method]) > 1 for method in pair):
        report.update(status="ambiguous", main_results_status="ambiguous",
                      reason="Multiple training runs share a method and seed; select explicitly outside this report.")
        return report
    if any(not members[method] for method in pair):
        report["reason"] = "Both methods require an existing training run at the same seed."
        return report
    left, right = (candidates[method][0] for method in pair)
    data = [_data_identity(member) for member in (left, right)]
    report["data_identity"] = [_data_identity_summary(member, identity, destination)
                               for member, identity in zip((left, right), data)]
    if data[0] != data[1]:
        report.update(status="incompatible", main_results_status="incompatible",
                      reason="Saved data identities differ; metrics are not paired.")
        return report
    if data[0]["kind"] != "frozen_data_binding":
        report.update(status="not_run", reason="Matching source data cannot be verified from saved data bindings.")
        return report
    extracted = [_extract(member[0], member[1], destination) for member in (left, right)]
    conditions = ("clean",) if pair[0] == "M0" else ("clean", "main_mixed/mixed_v1_s0")
    expected = {(split, condition) for split in ("valid", "test") for condition in conditions}
    identities = sorted(set(extracted[0]) | set(extracted[1]) | expected)
    for identity in identities:
        item = {"section": identity[0], "condition": "/".join(identity[1:]),
                "status": "not_run", "members": {}}
        for method, values in zip(pair, extracted):
            existing = values.get(identity)
            item["members"][method] = (existing if existing is not None else
                                          {"metrics": None, "evaluation": None, "cohort_ids": None})
        a, b = (item["members"][method] for method in pair)
        if a["metrics"] is not None and b["metrics"] is not None:
            if a["metrics"]["n"] != b["metrics"]["n"]:
                item.update(status="not_comparable", reason="Saved sample counts differ.")
            elif a["cohort_ids"] != b["cohort_ids"]:
                item.update(status="not_comparable", reason="Saved paired cohorts differ or one is missing.")
            elif identity[0] == "robustness" and a["cohort_ids"] is None:
                item.update(status="not_comparable", reason="Paired cohort IDs are not present in saved evidence.")
            else:
                item["status"] = "completed"
        report["comparisons"].append(item)
    completed = [item for item in report["comparisons"] if item["status"] == "completed"]
    coverage = {split: {panel: {method: member[1].get("robustness", {}).get("panels", {}).get(split, {})
        .get(panel, {}).get("status", "not_run") for method, member in zip(pair, (left, right))}
        for panel in ("modality", "rate", "position", "shape", "synchrony")} for split in ("valid", "test")}
    report["robustness_coverage"] = coverage
    robust_complete = [item for item in completed if item["section"] == "robustness"]
    all_panels = all(status == "completed" for panels in coverage.values()
                     for methods in panels.values() for status in methods.values())
    report["robustness_status"] = ("completed" if robust_complete and all_panels
        and all(item["status"] == "completed" for item in report["comparisons"] if item["section"] == "robustness")
        else "partial" if robust_complete else "not_run")
    main_results = [item for item in report["comparisons"] if item["section"] in ("valid", "test")]
    main_complete = [item for item in main_results if item["status"] == "completed"]
    report["main_results_status"] = ("completed" if main_complete and len(main_complete) == len(main_results)
                                      else "partial" if main_complete else "not_run")
    report["status"] = report["main_results_status"]
    for item in report["comparisons"]:
        item["members"] = {method: _compact_cohort(member) for method, member in item["members"].items()}
    return report


def _csv(report):
    stream = io.StringIO(newline="")
    fields = ["section", "condition", "comparison_status", "method", "seed", "run_id",
              *METRICS, "evaluation_path", "evaluation_sha256", "json_pointer"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for item in report["comparisons"]:
        for method, value in item["members"].items():
            reference = value["evaluation"] or {}
            metrics = value["metrics"] or {}
            writer.writerow({"section": item["section"], "condition": item["condition"],
                "comparison_status": item["status"], "method": method, "seed": report["seed"],
                "run_id": report["candidates"][method][0]["run_id"], **metrics,
                "evaluation_path": reference.get("path"), "evaluation_sha256": reference.get("sha256"),
                "json_pointer": reference.get("json_pointer")})
    return stream.getvalue().encode("utf-8-sig")


def _figure(report):
    complete = [item for item in report["comparisons"] if item["status"] == "completed"]
    height = max(180, 115 + len(complete) * 72)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}">',
             f'<title>{escape(report["comparison_id"])}: saved Accuracy</title>',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="Arial,sans-serif" fill="#182b49">',
             '<text x="24" y="30" font-size="21">Saved Accuracy on matching conditions and cohorts</text>',
             '<text x="24" y="57" font-size="13">Display percentages only; full precision values and source references are in comparison.json.</text>']
    colors = ("#2267a7", "#ba5b21")
    for index, item in enumerate(complete):
        y = 92 + 72 * index
        label = item["section"] + "/" + item["condition"]
        parts.append(f'<text x="24" y="{y + 18}" font-size="12">{escape(label)}</text>')
        for offset, (method, color) in enumerate(zip(report["methods"], colors)):
            metrics = item["members"][method]["metrics"]
            value, top = metrics["Accuracy"], y + 24 * offset
            parts.append(f'<rect x="450" y="{top}" width="{500 * value:.8f}" height="18" fill="{color}"><title>{method}: {value!r}; N={metrics["n"]}</title></rect>')
            parts.append(f'<text x="{960}" y="{top + 14}" font-size="12">{method} {value:.2%} (N={metrics["n"]})</text>')
    if not complete:
        parts.append(f'<text x="24" y="105" font-size="17">Status: {escape(report["status"])}. No completed paired comparison.</text>')
    parts.extend(['</g>', '</svg>'])
    return ("\n".join(parts) + "\n").encode("utf-8")


def _atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_bytes() == content:
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".reporting-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _write_owned(destination, report):
    manifest_path = destination / "manifest.json"
    if destination.exists() and any(destination.iterdir()):
        if not manifest_path.is_file():
            raise FileExistsError("Comparison directory is not owned by reporting: " + str(destination))
        previous = _read(manifest_path)
        if previous.get("generator") != {"id": OWNER, "version": VERSION}:
            raise FileExistsError("Comparison belongs to another generator: " + str(destination))
        if set(previous.get("files", {})) != {"comparison.json", "comparison.csv", "figures/accuracy.svg"}:
            raise ValueError("Comparison ownership manifest has an unexpected file set")
        for relative, recorded in previous.get("files", {}).items():
            path = (destination / relative).resolve()
            if not path.is_relative_to(destination.resolve()) or not path.is_file() or _hash(path) != recorded:
                raise ValueError("Owned comparison was edited or removed: " + str(path))
    payloads = {"comparison.json": (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"),
                "comparison.csv": _csv(report), "figures/accuracy.svg": _figure(report)}
    for relative, content in payloads.items():
        _atomic(destination / relative, content)
    manifest = {"generator": {"id": OWNER, "version": VERSION},
                "files": {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()}}
    _atomic(manifest_path, (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))


def build_comparisons(outputs):
    """Write owned same-seed comparisons; ambiguous runs are listed, never chosen."""
    outputs = Path(outputs).resolve()
    groups, skipped = _scan(outputs)
    result = {"comparisons": [], "skipped": skipped}
    for (protocol, seed), candidates in sorted(groups.items()):
        for pair in PAIRS:
            if not any(method in candidates for method in pair):
                continue
            destination = outputs / "comparisons" / f"{protocol}_{pair[0]}_vs_{pair[1]}_s{seed}"
            report = _comparison(protocol, seed, pair, candidates, destination)
            _write_owned(destination, report)
            for method in pair:
                for run, existing, _ in candidates.get(method, []):
                    # Navigation has no digest back-edge to this mutable summary.
                    # Every comparison metric refers to immutable raw evidence.
                    existing["ablation"] = {"generator": OWNER, "status": report["status"],
                        "comparisons": [{"path": Path(os.path.relpath(destination / "comparison.json", run)).as_posix(),
                                         "status": report["status"], "comparison_id": destination.name}],
                        "note": "Existing same-seed records only; no run selection or new scoring."}
                    from .artifacts import build_analysis
                    additions = build_analysis(run, existing)
                    for field in ("diagnostics", "artifacts"):
                        existing[field].update(additions.get(field, {}))
                    _atomic(run / "evaluation.json", (json.dumps(existing, ensure_ascii=False, indent=2, allow_nan=False)
                                                      + "\n").encode("utf-8"))
            result["comparisons"].append({"id": destination.name, "status": report["status"],
                "path": (destination / "comparison.json").relative_to(outputs).as_posix()})
    return result
