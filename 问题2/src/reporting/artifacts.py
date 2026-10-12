"""Offline, evidence-only analysis of saved reports and prediction CSV files.

This module intentionally imports only the standard library: reporting must not
initialize a model, load a checkpoint, calibrate a threshold, or affect an RNG.
SVG figures remain usable without a plotting runtime. Headline metrics always
come from the saved evaluation; CSV statistics below are labelled diagnostics.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from typing import Any


_OWNER = "problem2.reporting.artifacts.v1"
_COLORS = ("#2563eb", "#d97706", "#059669", "#7c3aed", "#db2777")
_MISSING = "未保存对应证据；未运行或未记录，不能据此补造结果。"


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _relative(path: Path, run: Path) -> str:
    return Path(os.path.relpath(path, run)).as_posix()


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_")[:110] or "unknown"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _Writer:
    """Update only files previously generated here and still unmodified."""

    def __init__(self, run: Path):
        self.run = run
        self.manifest = run / "analysis" / ".reporting_artifacts.json"
        self.previous: dict[str, str] = {}
        if self.manifest.is_file():
            saved = _read_json(self.manifest)
            if saved.get("generator") != _OWNER:
                raise ValueError(f"Refusing to replace unrelated artifact manifest: {self.manifest}")
            self.previous = saved.get("files", {})
        self.files: dict[str, str] = {}

    def write(self, name: str, data: str) -> str:
        payload = data.encode("utf-8")
        digest = _sha(payload)
        target = self.run / name
        if target.is_file():
            old = _sha(target.read_bytes())
            if old != digest and self.previous.get(name) != old:
                target = target.with_name(f"{target.stem}.{digest[:12]}{target.suffix}")
                name = _relative(target, self.run)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != payload:
            old = _sha(target.read_bytes())
            if self.previous.get(name) != old:
                raise ValueError(f"Refusing to replace an unrelated analysis artifact: {target}")
        if not target.exists() or target.read_bytes() != payload:
            target.write_bytes(payload)
        self.files[name] = digest
        return name

    def finish(self) -> str:
        # Retain ownership of old generated figures without removing them.
        files = dict(self.previous)
        files.update(self.files)
        text = json.dumps({"generator": _OWNER, "files": files}, ensure_ascii=False, indent=2) + "\n"
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        if not self.manifest.exists() or self.manifest.read_text(encoding="utf-8") != text:
            self.manifest.write_text(text, encoding="utf-8", newline="\n")
        return _relative(self.manifest, self.run)


def _svg(title: str, body: list[str], width: int = 900, height: int = 470) -> str:
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{escape(title)}</title>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#172b4d">',
        f'<text x="28" y="34" font-size="19">{escape(title)}</text>',
        *body, "</g></svg>", "",
    ])


def _text(x: float, y: float, value: Any, **attrs: Any) -> str:
    properties = " ".join(f'{k.replace("_", "-")}="{escape(str(v), quote=True)}"' for k, v in attrs.items())
    return f'<text x="{x:.2f}" y="{y:.2f}" {properties}>{escape(str(value))}</text>'


def _matrix_svg(matrix: Any, names: list[str], title: str, normalized: bool) -> str | None:
    if not isinstance(matrix, list) or not matrix or not all(isinstance(row, list) and len(row) == len(matrix) for row in matrix):
        return None
    if any(_number(value) is None or float(value) < 0 for row in matrix for value in row):
        return None
    n = len(matrix)
    if len(names) != n:
        names = [str(i) for i in range(n)]
    maximum = 1.0 if normalized else max(1.0, max(float(v) for row in matrix for v in row))
    side = 300 / n
    body = [_text(340, 76, "Predicted class", text_anchor="middle", font_size=15),
            _text(25, 259, "True class", transform="rotate(-90 25 259)", text_anchor="middle", font_size=15)]
    for j, name in enumerate(names):
        body.append(_text(190 + (j + .5) * side, 108, name, text_anchor="middle", font_size=13))
    for i, row in enumerate(matrix):
        body.append(_text(174, 120 + (i + .55) * side, names[i], text_anchor="end", font_size=13))
        for j, value in enumerate(row):
            strength = float(value) / maximum
            rgb = tuple(round(a + strength * (b - a)) for a, b in zip((239, 246, 255), (30, 64, 175)))
            body.append(f'<rect x="{190 + j * side:.2f}" y="{120 + i * side:.2f}" width="{side:.2f}" height="{side:.2f}" fill="#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}" stroke="white"/>')
            label = f"{float(value):.1%}" if normalized else f"{float(value):g}"
            body.append(_text(190 + (j + .5) * side, 120 + (i + .56) * side, label, text_anchor="middle", fill="white" if strength > .55 else "#172b4d", font_size=21))
    body.append(_text(28, 452, "Saved row-normalized values; rows=true, columns=predicted." if normalized else "Saved counts; rows=true, columns=predicted.", font_size=12))
    return _svg(title, body, width=660)


def _bar_svg(classes: dict[str, Any], title: str) -> str | None:
    names = list(classes)
    if not names:
        return None
    body: list[str] = []
    series = ("Precision", "Recall", "F1")
    for tick in range(6):
        y = 376 - tick * 56
        body.extend([f'<path d="M70 {y}H850" stroke="#e2e8f0"/>', _text(55, y + 4, f"{tick / 5:.1f}", text_anchor="end", font_size=12)])
    group_width = 720 / len(names)
    for i, name in enumerate(names):
        for j, metric in enumerate(series):
            value = _number(classes[name].get(metric)) if isinstance(classes[name], dict) else None
            if value is None:
                continue
            x, h = 90 + i * group_width + j * group_width * .23, value * 280
            body.append(f'<rect x="{x:.2f}" y="{376-h:.2f}" width="{group_width*.20:.2f}" height="{h:.2f}" fill="{_COLORS[j]}"/>')
        body.append(_text(90 + i * group_width + group_width * .34, 401, name, text_anchor="middle", font_size=13))
        support = classes[name].get("support") if isinstance(classes[name], dict) else None
        body.append(_text(90 + i * group_width + group_width * .34, 419, f"support={support if support is not None else 'unknown'}", text_anchor="middle", font_size=11))
    for j, metric in enumerate(series):
        body.append(_text(100 + j * 180, 67, metric, fill=_COLORS[j], font_size=14))
    body.append(_text(28, 454, "Saved per-class metrics; no scores recomputed.", font_size=12))
    return _svg(title, body)


def _line_svg(series: dict[str, list[tuple[float, float]]], title: str, x_label: str) -> str | None:
    values = [point for points in series.values() for point in points]
    if not values:
        return None
    xmin, xmax = min(x for x, _ in values), max(x for x, _ in values)
    ymin, ymax = min(y for _, y in values), max(y for _, y in values)
    if xmax == xmin:
        xmax = xmin + 1
    if ymax == ymin:
        ymax = ymin + 1
    padding = (ymax - ymin) * .08
    ymin, ymax = ymin - padding, ymax + padding
    body: list[str] = []
    for tick in range(6):
        y = 358 - tick * 52
        body.extend([f'<path d="M80 {y}H850" stroke="#e2e8f0"/>', _text(68, y + 4, f"{ymin + tick * (ymax-ymin)/5:.3g}", text_anchor="end", font_size=12)])
        x = 80 + tick * 154
        body.append(_text(x, 382, f"{xmin + tick * (xmax-xmin)/5:.3g}", text_anchor="middle", font_size=12))
    for i, (name, points) in enumerate(series.items()):
        color = _COLORS[i % len(_COLORS)]
        coords = [(80 + (x-xmin)/(xmax-xmin)*770, 358 - (y-ymin)/(ymax-ymin)*260) for x, y in sorted(points)]
        body.append(f'<polyline points="{" ".join(f"{x:.3f},{y:.3f}" for x,y in coords)}" fill="none" stroke="{color}" stroke-width="2"/>')
        body.extend(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="2.4" fill="{color}"/>' for x, y in coords)
        body.append(_text(85 + (i % 3) * 260, 416 + (i // 3) * 19, name, fill=color, font_size=12))
    body.append(_text(465, 399, x_label, text_anchor="middle", font_size=12))
    return _svg(title, body, height=max(470, 450 + (len(series)//3)*19))


def _intensity_svg(rows: list[dict[str, str]], title: str) -> str | None:
    pairs = [(_number(row.get("y_true")), _number(row.get("y_pred"))) for row in rows]
    points = [(truth, pred) for truth, pred in pairs if truth is not None and pred is not None]
    if not points:
        return None
    low = min(min(pair) for pair in points)
    high = max(max(pair) for pair in points)
    if low == high:
        high = low + 1
    scale = lambda value: (value-low)/(high-low)*285
    body = [_text(230, 71, f"Raw intensity (n={len(points)})", text_anchor="middle", font_size=14),
            _text(680, 71, "Residual = prediction - truth", text_anchor="middle", font_size=14),
            '<path d="M80 90V375H365 M80 375L365 90" fill="none" stroke="#94a3b8"/>']
    body.extend(f'<circle cx="{80+scale(truth):.3f}" cy="{375-scale(pred):.3f}" r="2.1" fill="#2563eb" fill-opacity="0.38"/>' for truth, pred in points)
    for i in range(4):
        value = low + (high-low)*i/3
        body.extend([_text(80+i*95, 393, f"{value:.2g}", text_anchor="middle", font_size=11), _text(69, 379-i*95, f"{value:.2g}", text_anchor="end", font_size=11)])
    body.extend([_text(223, 417, "True intensity", text_anchor="middle", font_size=13),
                 _text(26, 238, "Predicted intensity", transform="rotate(-90 26 238)", text_anchor="middle", font_size=13)])
    residuals = [pred-truth for truth, pred in points]
    bound = max(max(abs(value) for value in residuals), 1e-12)
    bins = [0] * 24
    for residual in residuals:
        bins[min(23, int((residual+bound)/(2*bound)*24))] += 1
    peak = max(bins)
    body.append('<path d="M515 90V375H850" fill="none" stroke="#94a3b8"/>')
    for i, count in enumerate(bins):
        h = count/peak*270
        body.append(f'<rect x="{516+i*13.9:.2f}" y="{375-h:.2f}" width="12.8" height="{h:.2f}" fill="#059669"/>')
    for value, x in ((-bound, 515), (0., 682), (bound, 850)):
        body.append(_text(x, 396, f"{value:.2g}", text_anchor="middle", font_size=11))
    body.extend([_text(510, 106, f"peak bin n={peak}", text_anchor="end", font_size=11),
                 _text(682, 417, "Residual (24 equal-width bins)", text_anchor="middle", font_size=13),
                 _text(28, 454, "All finite saved rows included; no clipping, resampling, or model execution.", font_size=12)])
    return _svg(title, body)


def _metric_records(report: dict[str, Any]) -> list[tuple[str, str, dict[str, Any], str]]:
    records: dict[tuple[str, str], tuple[dict[str, Any], str]] = {}
    for split in ("valid", "test"):
        for root in ("", "main", "summary"):
            block = report.get(split) if not root else report.get(root, {}).get(split)
            if not isinstance(block, dict):
                continue
            pointer = f"/{root}/{split}" if root else f"/{split}"
            candidates = [("main", block, pointer)] if any(key in block for key in ("Accuracy", "MAE", "per_class")) else [(key, value, f"{pointer}/{key.replace('~','~0').replace('/','~1')}") for key, value in block.items() if isinstance(value, dict) and not key.startswith("_")]
            for condition, metrics, location in candidates:
                if not any(key in metrics for key in ("Accuracy", "MAE", "per_class", "confusion_matrix")):
                    continue
                key = (split, condition)
                if key not in records or len(metrics) > len(records[key][0]):
                    records[key] = (metrics, location)
        # Old baseline reports store class diagnostics outside summary.
        old_classes = report.get("per_class", {}).get(split) if isinstance(report.get("per_class"), dict) else None
        old_matrix = report.get("confusion_matrix", {}).get(split) if isinstance(report.get("confusion_matrix"), dict) else None
        if old_classes or old_matrix:
            key = (split, "main")
            metrics = dict(records.get(key, ({}, ""))[0])
            if old_classes:
                metrics["per_class"] = old_classes
            if isinstance(old_matrix, dict):
                metrics.update(confusion_matrix=old_matrix.get("counts"), confusion_matrix_normalized=old_matrix.get("normalized"), class_order=old_matrix.get("class_order"))
            metrics["_analysis_sources"] = {"per_class": f"/per_class/{split}", "confusion_matrix": f"/confusion_matrix/{split}/counts", "confusion_matrix_normalized": f"/confusion_matrix/{split}/normalized"}
            records[key] = (metrics, f"/summary/{split}")
    return [(split, condition, metric, source) for (split, condition), (metric, source) in records.items()]


def _prediction_sources(run: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    supplied = report.get("artifacts", {}).get("predictions", [])
    if isinstance(supplied, list) and supplied:
        return [entry for entry in supplied if isinstance(entry, dict) and entry.get("path")]
    candidates = sorted((run / "predictions").rglob("*.csv")) if (run / "predictions").is_dir() else []
    if not candidates:
        candidates = sorted((run / "details").glob("*.csv"))
    result = []
    for path in candidates:
        stem = path.stem
        if not (stem.startswith(("valid", "test")) or path.parent.name in ("valid", "test")):
            continue
        split = path.parent.name if path.parent.name in ("valid", "test") else stem.split("_")[0]
        # Legacy `*_predictions.csv` does not by itself reveal clean vs mixed.
        condition = stem.split("__", 1)[1].replace("__", "/") if "__" in stem else None
        result.append({"split": split, "condition": condition, "path": _relative(path, run), "source": _relative(path, run)})
    return result


def _load_groups(run: Path, report: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    warnings: list[str] = []
    seen: set[Path] = set()
    for entry in _prediction_sources(run, report):
        path = (run / entry["path"]).resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            warnings.append(f"预测文件不存在：{entry['path']}")
            continue
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not any(name in reader.fieldnames for name in ("y_pred", "raw_intensity", "intensity")):
                warnings.append(f"无法识别预测列，未纳入分析：{entry['path']}")
                continue
            for row_index, raw in enumerate(reader):
                row = dict(raw)
                row["source_csv"] = _relative(path, run)
                row["source_row_index"] = str(row_index)
                row["source_record_number"] = str(row_index + 2)
                row["original_source_csv"] = str(entry.get("source") or "")
                row["split"] = row.get("split") or str(entry.get("split") or "unknown")
                row["condition"] = row.get("condition") or str(entry.get("condition") or "unknown")
                row["y_pred"] = row.get("y_pred") or row.get("raw_intensity") or row.get("intensity") or ""
                row["y_true"] = row.get("y_true") or ""
                row["true_class"] = row.get("true_class") or ""
                row["pred_class"] = row.get("pred_class") or row.get("predicted_class") or ""
                row["id"] = row.get("id") or row.get("sample_id") or ""
                grouped[(row["split"], row["condition"], row["source_csv"])].append(row)
    return [{"split": split, "condition": condition, "source_csv": source, "rows": rows} for (split, condition, source), rows in grouped.items()], warnings


def _class_error(row: dict[str, str]) -> bool | None:
    truth, predicted = row.get("true_class"), row.get("pred_class")
    if truth in (None, "") or predicted in (None, ""):
        return None
    a, b = _number(truth), _number(predicted)
    return a != b if a is not None and b is not None else truth != predicted


def _group_diagnostic(group: dict[str, Any]) -> dict[str, Any]:
    rows = group["rows"]
    classified = [row for row in rows if _class_error(row) is not None]
    intensity = [row for row in rows if _number(row.get("y_true")) is not None and _number(row.get("y_pred")) is not None]
    mistakes = sum(bool(_class_error(row)) for row in classified)
    ids = [row["id"] for row in rows if row["id"]]
    return {
        "split": group["split"], "condition": group["condition"], "source_csv": group["source_csv"],
        "rows": len(rows), "unique_ids": len(set(ids)) if ids else None,
        "duplicate_id_rows": len(ids) - len(set(ids)),
        "classification": {"status": "completed" if classified else "not_run", "n": len(classified),
                           "errors": mistakes if classified else None,
                           "error_rate": mistakes / len(classified) if classified else None,
                           "denominator": "rows with both saved true_class and pred_class in this CSV and condition",
                           "missing_class_rows": len(rows)-len(classified),
                           "note": "未从连续值推导类别，未重新应用或校准阈值。"},
        "intensity": {"status": "completed" if intensity else "not_run", "finite_pairs": len(intensity),
                      "excluded_nonfinite_or_unlabeled_rows": len(rows)-len(intensity),
                      "prediction_clipping": False},
    }


def _paired_diagnostics(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for group in groups:
        if group["condition"] in ("clean", "paired/complete", "unknown"):
            continue
        # Prefer the paired complete cohort where present. Never pool sources.
        candidates = [candidate for candidate in groups if candidate["split"] == group["split"] and candidate["condition"] in ("clean", "paired/complete") and Path(candidate["source_csv"]).parent == Path(group["source_csv"]).parent]
        if not candidates:
            candidates = [candidate for candidate in groups if candidate["split"] == group["split"] and candidate["condition"] in ("clean", "paired/complete")]
        paired = [candidate for candidate in candidates if candidate["condition"] == "paired/complete"]
        candidates = paired or candidates
        if len(candidates) != 1:
            continue
        reference = candidates[0]
        def indexed(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
            counts = Counter(row["id"] for row in rows if row["id"])
            return {row["id"]: row for row in rows if row["id"] and counts[row["id"]] == 1 and _class_error(row) is not None}
        before, after = indexed(reference["rows"]), indexed(group["rows"])
        cohort = sorted(set(before) & set(after))
        coherent = [sample_id for sample_id in cohort if before[sample_id]["true_class"] == after[sample_id]["true_class"] and before[sample_id]["y_true"] == after[sample_id]["y_true"]]
        if not coherent:
            continue
        counts = Counter((bool(_class_error(before[sample_id])), bool(_class_error(after[sample_id]))) for sample_id in coherent)
        result.append({"split": group["split"], "condition": group["condition"], "source_csv": group["source_csv"],
                       "reference_condition": reference["condition"], "reference_csv": reference["source_csv"],
                       "n": len(coherent), "correct_to_wrong": counts[(False, True)], "wrong_to_correct": counts[(True, False)],
                       "always_wrong": counts[(True, True)], "always_correct": counts[(False, False)],
                       "excluded_label_mismatch": len(cohort)-len(coherent),
                       "cohort_id_sha256": _sha(json.dumps(coherent, ensure_ascii=False, separators=(",", ":")).encode("utf-8")),
                       "denominator": "unique shared IDs with saved class labels and exactly matching saved truth; separate for each condition",
                       "interpretation": "配对描述性统计，不作为独立样本合并；不据此声称缺失的因果机制。"})
    return result


def _training_figures(run: Path, writer: _Writer) -> list[dict[str, Any]]:
    path = run / "history.json"
    history = _read_json(path) if path.is_file() else None
    if isinstance(history, dict):
        history = history.get("epochs", history.get("history"))
    if not isinstance(history, list):
        path = run / "details" / "history.csv"
        if not path.is_file():
            return []
        with path.open(encoding="utf-8-sig", newline="") as stream:
            history = list(csv.DictReader(stream))
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for item in history:
        if not isinstance(item, dict) or _number(item.get("epoch")) is None:
            continue
        epoch = float(item["epoch"])
        total = item.get("train", {}).get("total") if isinstance(item.get("train"), dict) else item.get("train_total", item.get("train_loss"))
        if _number(total) is not None:
            series["train total loss"].append((epoch, float(total)))
        valid = item.get("valid")
        blocks = valid.items() if isinstance(valid, dict) and not any(key in valid for key in ("MAE", "Accuracy")) else [("main", valid or {"MAE": item.get("valid_MAE"), "Accuracy": item.get("valid_Accuracy")})]
        for condition, metrics in blocks:
            if not isinstance(metrics, dict):
                continue
            for key in ("Accuracy", "MAE"):
                if _number(metrics.get(key)) is not None:
                    series[f"valid {condition} {key}"].append((epoch, float(metrics[key])))
    figures = []
    for category, selected in (("loss", {k: v for k, v in series.items() if k.startswith("train")}), ("accuracy", {k: v for k, v in series.items() if k.endswith("Accuracy")}), ("mae", {k: v for k, v in series.items() if k.endswith("MAE")})):
        svg = _line_svg(selected, f"Recorded training history: {category}", "Epoch")
        if svg:
            output = writer.write(f"analysis/figures/history_{category}.svg", svg)
            figures.append({"kind": f"history_{category}", "path": output, "source": _relative(path, run), "note": "仅绘制已保存轮次；不推断训练结束原因。"})
    return figures


def _robustness_figures(report: dict[str, Any], writer: _Writer) -> list[dict[str, Any]]:
    """Plot only panel records already present; each figure keeps its own N."""
    robustness = report.get("robustness")
    panels = robustness.get("panels", {}) if isinstance(robustness, dict) else {}
    figures = []
    for split, split_panels in panels.items():
        if not isinstance(split_panels, dict):
            continue
        for panel_name, panel in split_panels.items():
            if not isinstance(panel, dict) or not isinstance(panel.get("conditions"), dict):
                continue
            conditions = panel["conditions"]
            rows = []
            for name, value in conditions.items():
                metrics = value.get("metrics", {}) if isinstance(value, dict) else {}
                if any(_number(metrics.get(key)) is not None for key in ("Accuracy", "Macro_F1", "MAE", "Pearson")):
                    rows.append((name, metrics))
            if not rows:
                continue
            height = 160 + len(rows) * 39
            common_n = panel.get("common_n")
            body = [_text(28, 61, f"Saved panel results; common N={common_n if common_n is not None else 'not recorded'}; each condition is a repeated cohort.", font_size=12)]
            metrics_order = ("Accuracy", "Macro_F1", "MAE", "Pearson")
            for col, key in enumerate(metrics_order):
                body.append(_text(330 + col * 146, 96, key, font_size=13))
            body.append(_text(28, 96, "Condition (saved n)", font_size=13))
            for i, (name, metrics) in enumerate(rows):
                y = 122 + i * 39
                body.append(_text(28, y + 10, f"{name} (n={metrics.get('n', 'unknown')})", font_size=11))
                for col, key in enumerate(metrics_order):
                    value = _number(metrics.get(key))
                    x = 330 + col * 146
                    if value is None:
                        body.append(_text(x, y + 10, "null / unavailable", font_size=10))
                        continue
                    body.append(_text(x, y + 9, f"{value:.5f}", font_size=12))
                    # Bars are explicitly scaled by metric, not new statistics.
                    maximum = max(1., max((_number(record.get(key)) or 0.) for _, record in rows)) if key == "MAE" else 1.
                    width = max(0., value / maximum) * 110
                    body.append(f'<rect x="{x}" y="{y+14}" width="{width:.3f}" height="5" fill="{_COLORS[col]}"/>')
            body.append(_text(28, height-16, "Display rounding only. Bars: Accuracy/F1/Pearson scale 0..1; MAE scale 0..max(1, panel max). Negative Pearson has no bar.", font_size=11))
            svg = _svg(f"{split} / {panel_name}: robustness panel", body, width=940, height=height)
            output = writer.write(f"analysis/figures/{_slug(split)}__robustness_{_slug(panel_name)}.svg", svg)
            pointer = f"/robustness/panels/{str(split).replace('~','~0').replace('/','~1')}/{str(panel_name).replace('~','~0').replace('/','~1')}"
            figures.append({"kind": "robustness_panel", "split": split, "condition": panel_name, "path": output, "source": f"evaluation.json#{pointer}", "common_n": common_n})
    return figures


def build_analysis(run: Path, report: dict[str, Any]) -> dict[str, Any]:
    """Create evidence-linked diagnostics and figures; return report additions.

    ``report`` is read-only. Returned paths are relative to ``run``; the caller
    merges the ``diagnostics`` and ``artifacts`` additions into its report.
    Unknown classes stay empty in error_cases.csv. Regression-only rows with a
    nonzero saved residual are included with their distinct case kind.
    """
    run = Path(run).resolve()
    writer = _Writer(run)
    figures: list[dict[str, Any]] = []
    class_sources = []
    metric_records = _metric_records(report)
    for split, condition, metrics, pointer in metric_records:
        stem = f"{_slug(split)}__{_slug(condition)}"
        names = metrics.get("class_order") or list(metrics.get("per_class", {}))
        class_sources.append({"split": split, "condition": condition, "report": "evaluation.json", "pointer": pointer})
        for key, normalized in (("confusion_matrix", False), ("confusion_matrix_normalized", True)):
            svg = _matrix_svg(metrics.get(key), names, f"{split} / {condition}: {key}", normalized)
            if svg:
                output = writer.write(f"analysis/figures/{stem}__{key}.svg", svg)
                source_pointer = metrics.get("_analysis_sources", {}).get(key, pointer)
                figures.append({"kind": key, "split": split, "condition": condition, "path": output, "source": f"evaluation.json#{source_pointer}"})
        if isinstance(metrics.get("per_class"), dict):
            svg = _bar_svg(metrics["per_class"], f"{split} / {condition}: per-class metrics")
            if svg:
                output = writer.write(f"analysis/figures/{stem}__per_class.svg", svg)
                source_pointer = metrics.get("_analysis_sources", {}).get("per_class", pointer)
                figures.append({"kind": "per_class", "split": split, "condition": condition, "path": output, "source": f"evaluation.json#{source_pointer}"})
    groups, warnings = _load_groups(run, report)
    diagnostics = [_group_diagnostic(group) for group in groups]
    columns = ["split", "condition", "source_csv", "original_source_csv", "source_row_index", "source_record_number", "index", "id", "method", "case_kind", "y_true", "y_pred", "true_class", "pred_class", "classification_error", "absolute_error", "residual", "decision"]
    buffer = io.StringIO(newline="")
    csv_writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
    csv_writer.writeheader()
    cases = 0
    for group in groups:
        for row in group["rows"]:
            wrong = _class_error(row)
            truth, predicted = _number(row.get("y_true")), _number(row.get("y_pred"))
            residual = predicted - truth if truth is not None and predicted is not None else None
            if wrong is not True and not (wrong is None and residual is not None and residual != 0):
                continue
            case = dict(row)
            case.update(case_kind="classification_error" if wrong is True else "regression_residual_without_saved_classes", classification_error="true" if wrong is True else "", absolute_error=abs(residual) if residual is not None else "", residual=residual if residual is not None else "")
            csv_writer.writerow(case)
            cases += 1
        # Main diagnostics are readable; paired panels are summarized separately.
        if not str(group["condition"]).startswith("paired/"):
            svg = _intensity_svg(group["rows"], f"{group['split']} / {group['condition']}: intensity diagnostics")
            if svg:
                suffix = _sha(group["source_csv"].encode("utf-8"))[:8]
                output = writer.write(f"analysis/figures/{_slug(group['split'])}__{_slug(group['condition'])}__intensity_{suffix}.svg", svg)
                figures.append({"kind": "intensity", "split": group["split"], "condition": group["condition"], "path": output, "source": group["source_csv"]})
    cases_path = writer.write("analysis/error_cases.csv", buffer.getvalue())
    figures.extend(_training_figures(run, writer))
    figures.extend(_robustness_figures(report, writer))
    paired = _paired_diagnostics(groups)
    limitations = [
        "本报告只整理已保存的 JSON、CSV 和 history，不加载模型、不训练、不评估、不重选模型、不重新校准阈值。",
        "主指标、逐类指标和混淆矩阵引用原评价；CSV 错误计数和配对转变是补充诊断，不替代正式指标。",
        "每个 split、condition、来源 CSV 分别报告分母；不同缺失条件常复用同一批样本，不能将行数相加作为独立样本量。",
        "没有保存类别的历史 CSV 不从强度推导类别；相应分类错误统计为空。error_cases 中此类非零残差行单独标记。",
        "这里只描述统计现象，未验证讽刺、噪声、模态冲突等原因；未进行原始文本/音视频人工错误归因。",
        "附件三无标签，评分不适用；缺失实验和附件三预测状态以 evaluation.json 为准。",
    ]
    report_lines = [f"# {run.name}：已有产物分析", "", "<!-- " + _OWNER + " -->", "", "本报告由已保存结果离线生成。所有路径以本报告所在目录为基准；正式评价原始浮点数保留在 [evaluation.json](../evaluation.json)。", "", "## 已保存的主指标", "", "| 划分 | 条件 | n | Accuracy | Macro_F1 | MAE | Pearson |", "|---|---|---:|---:|---:|---:|---:|"]
    for split, condition, metrics, _ in metric_records:
        displayed = [format(float(metrics[key]), ".6g") if _number(metrics.get(key)) is not None else "null / 未保存" for key in ("Accuracy", "Macro_F1", "MAE", "Pearson")]
        report_lines.append(f"| {split} | {condition} | {metrics.get('n', '未保存')} | {' | '.join(displayed)} |")
    if not metric_records:
        report_lines.extend(["", _MISSING])
    report_lines.extend(["", "表格仅对原评价值做展示舍入；Pearson 的 null 原因以原评价 Pearson_status 为准。不同协议、条件和 cohort 的指标不能直接混合。", "", "## 已有逐样本证据", "", "| 划分 | 条件 | CSV 行数 | 有类别行数 | 分类错误数 | 错误率 | 来源 |", "|---|---|---:|---:|---:|---:|---|"])
    for group in diagnostics:
        classification = group["classification"]
        ratio = f"{classification['error_rate']:.2%}" if classification["error_rate"] is not None else "未知"
        source = "../" + group["source_csv"]
        report_lines.append(f"| {group['split']} | {group['condition']} | {group['rows']} | {classification['n']} | {classification['errors'] if classification['errors'] is not None else '未知'} | {ratio} | [CSV]({source}) |")
    if not diagnostics:
        report_lines.extend(["", _MISSING])
    report_lines.extend(["", "分类错误率的分母是同一来源、划分、条件中同时保存真类和预测类的行数。空字段表示未保存，不能解释为零。", "", f"[逐样本错误/残差案例]({Path(os.path.relpath(run / cases_path, run / 'analysis')).as_posix()})：{cases} 行；source_row_index 为 CSV 数据行零基索引，source_record_number 包含表头按记录计数，index 为原 CSV 的样本索引。原始强度和类别逐字保留；未知字段为空。", "", "## 同一样本的配对变化", ""])
    if paired:
        report_lines.extend(["| 划分 | 受损条件 | 参考条件 | 共同 N | 对→错 | 错→对 | 始终错 | 始终对 |", "|---|---|---|---:|---:|---:|---:|---:|"])
        for pair in paired:
            report_lines.append(f"| {pair['split']} | {pair['condition']} | {pair['reference_condition']} | {pair['n']} | {pair['correct_to_wrong']} | {pair['wrong_to_correct']} | {pair['always_wrong']} | {pair['always_correct']} |")
        report_lines.append("\n仅匹配唯一 ID 且保存真值完全一致的行；每个条件独立报告。cohort ID 的 SHA256 与来源路径见 evaluation.json diagnostics.paired_transitions。")
    else:
        report_lines.append("未找到可唯一匹配、同时保存类别的 clean/缺失预测证据，未生成配对变化统计。")
    report_lines.extend(["", "## 图表", ""])
    for figure in figures:
        path = Path(os.path.relpath(run / figure["path"], run / "analysis")).as_posix()
        report_lines.append(f"- [{figure.get('split', '')} {figure.get('condition', '')} {figure['kind']}]({path})；来源：`{figure['source']}`。")
    if not figures:
        report_lines.append(_MISSING)
    report_lines.extend(["", "## 实验状态与未完成事项", ""])
    for field in ("valid", "test", "robustness", "ablation", "attachment3"):
        value = report.get(field)
        state = value.get("status", value.get("_reporting", {}).get("status", "未显式记录；以原始字段为准")) if isinstance(value, dict) else "not_run"
        report_lines.append(f"- {field}：{state}。")
    ablation = report.get("ablation")
    comparisons = ablation.get("comparisons", []) if isinstance(ablation, dict) else []
    if comparisons:
        report_lines.extend(["", "### 模型比较与消融记录", ""])
        for comparison in comparisons:
            if not isinstance(comparison, dict) or not comparison.get("path"):
                continue
            destination = run / comparison["path"]
            link = Path(os.path.relpath(destination, run / "analysis")).as_posix()
            title = comparison.get("comparison_id") or destination.parent.name
            state = comparison.get("status", "未显式记录")
            report_lines.append(f"- [{title}]({link})：{state}。比较使用各运行已保存结果，缺项不补造。")
    report_lines.extend(["", *[f"- {limitation}" for limitation in limitations], *[f"- 数据提示：{warning}" for warning in warnings], ""])
    report_path = writer.write("analysis/report.md", "\n".join(report_lines))
    manifest_path = writer.finish()
    return {"diagnostics": {
                "status": "completed" if diagnostics or class_sources or figures else "not_run",
                "generator": _OWNER, "sources": class_sources, "groups": diagnostics,
                "paired_transitions": paired, "error_cases": {"status": "completed" if groups else "not_run", "path": cases_path, "rows": cases, "selection": "saved class mismatch; otherwise nonzero residual only if classes unavailable"},
                "figures": figures, "report": report_path, "limitations": limitations, "warnings": warnings,
            }, "artifacts": {"analysis": {"report": report_path, "error_cases": cases_path, "figures": [item["path"] for item in figures], "manifest": manifest_path}}}
