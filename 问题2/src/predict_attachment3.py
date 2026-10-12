"""Frozen M3 inference on all unlabeled aligned Attachment 3 inputs.

No training data, generated masks, reference views or labels enter this path.
The trained model applies its own checkpoint-bound A/V normalization.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import re
import warnings

import numpy as np

from src.data.aligned import FINITE_POLICY, LENGTH_POLICY
from src.runtime import ROOT, configure_environment, resolve_path


INPUT_FIELDS = ("text", "audio", "vision", "audio_lengths", "vision_lengths")
FIELDS = ("sample_id", "source_file", "sample_index", "polarity", "intensity")
POLARITIES = ("Negative", "Neutral", "Positive")


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _numeric(value, name):
    value = np.asarray(value)
    if value.dtype.kind not in "biuf":
        raise ValueError(f"{name}: expected a real numeric array")
    return value


class Attachment3Dataset:
    """Read trusted local competition PKLs without manufacturing labels/IDs.

    IDs are derived from source file stems and, for multi-row files, row indices.
    __getitem__ exposes only model inputs and an ordering index.
    """

    def __init__(self, directory, expected_files=30, expected_samples=30):
        self.directory = Path(directory).resolve()
        if not self.directory.is_dir():
            raise FileNotFoundError(f"Attachment 3 input directory missing: {self.directory}")
        paths = sorted(self.directory.glob("*.pkl"),
                       key=lambda p: [int(s) if s.isdigit() else s
                                      for s in re.split(r"(\d+)", p.name)])
        if not paths or len(paths) != expected_files:
            raise ValueError(f"Expected {expected_files} input PKL files, found {len(paths)}")
        self.files, self.identities = [], []
        texts, audios, visions, content_lengths = [], [], [], []
        for path in paths:
            # Only the bundled/trusted local competition inputs are supported.
            content_bytes = path.read_bytes()
            archive = pickle.loads(content_bytes)
            if not isinstance(archive, dict) or set(archive) != {"test"}:
                raise ValueError(f"{path.name}: expected only the unlabeled test split")
            fields = archive["test"]
            required = {"text_bert", "audio", "vision"}
            if not isinstance(fields, dict) or set(fields) != required:
                raise ValueError(f"{path.name}: expected unlabeled text_bert/audio/vision fields only")
            text = _numeric(fields["text_bert"], path.name + "/text_bert")
            if text.ndim != 3 or text.shape[1:] != (3, 50) or text.shape[0] < 1:
                raise ValueError(f"{path.name}: expected text_bert [N,3,50]")
            if not np.isfinite(text).all() or not np.equal(text, np.floor(text)).all():
                raise ValueError(f"{path.name}: text_bert must contain finite integers")
            count = text.shape[0]
            mask = text[:, 1]
            if not np.isin(mask, [0, 1]).all():
                raise ValueError(f"{path.name}: attention mask must be binary")
            lengths = mask.sum(axis=1).astype(np.int64)
            positions = np.arange(50)[None, :]
            if np.any(lengths < 3) or not np.equal(mask, positions < lengths[:, None]).all():
                raise ValueError(f"{path.name}: invalid attention mask or empty content")
            rows = np.arange(count)
            if (not np.all(text[:, 0, 0] == 101)
                    or not np.all(text[rows, 0, lengths - 1] == 102)
                    or not np.all(text[:, 0][positions >= lengths[:, None]] == 0)
                    or not np.all((text[:, 0] >= 0) & (text[:, 0] < 30522))
                    or not np.all(text[:, 2] == 0)):
                raise ValueError(f"{path.name}: invalid BERT CLS/SEP/PAD/token IDs")
            inside = (positions >= 1) & (positions < lengths[:, None] - 1)
            feature_audit, adapted = {}, {}
            for name, dimension in (("audio", 74), ("vision", 35)):
                source = _numeric(fields[name], path.name + "/" + name)
                if source.shape != (count, 50, dimension):
                    raise ValueError(f"{path.name}: unexpected {name} shape {source.shape}")
                finite = np.isfinite(source)
                nonzero = np.any(finite & (source != 0), axis=-1)
                if np.any(nonzero & ~inside):
                    raise ValueError(f"{path.name}: {name} has data in CLS/SEP/padding slots")
                nonfinite_count = int((~finite).sum())
                if nonfinite_count:
                    warnings.warn(f"{path.name}/{name}: replacing {nonfinite_count} nonfinite values with zero")
                source = np.nan_to_num(source, copy=True, nan=0., posinf=0., neginf=0.)
                shifted = np.zeros(source.shape, dtype=np.float32)
                with np.errstate(over="ignore"):
                    shifted[:, :-1] = source[:, 1:]
                if not np.isfinite(shifted).all():
                    raise ValueError(f"{path.name}/{name}: finite values overflowed float32")
                adapted[name] = shifted
                feature_audit[name] = {
                    "nonfinite_values_replaced": nonfinite_count,
                    "zero_content_rows_retained": int((~nonzero & inside).sum()),
                }
            texts.append(text.astype(np.int64))
            audios.append(adapted["audio"])
            visions.append(adapted["vision"])
            content_lengths.append(lengths - 2)
            self.files.append({"source_file": path.name,
                "sha256": hashlib.sha256(content_bytes).hexdigest(), "sample_count": count,
                "content_lengths": (lengths - 2).tolist(), "features": feature_audit})
            for index in range(count):
                sample_id = path.stem if count == 1 else f"{path.stem}:{index}"
                self.identities.append({"sample_id": sample_id, "source_file": path.name,
                                        "sample_index": index})
        if len(self.identities) != expected_samples:
            raise ValueError(f"Expected {expected_samples} samples, found {len(self.identities)}")
        if len({row["sample_id"] for row in self.identities}) != len(self.identities):
            raise ValueError("Duplicate generated sample IDs")
        self.text = np.concatenate(texts)
        self.audio = np.concatenate(audios)
        self.vision = np.concatenate(visions)
        self.lengths = np.concatenate(content_lengths)

    def __len__(self):
        return len(self.identities)

    def __getitem__(self, index):
        import torch

        return {"text": torch.from_numpy(self.text[index]),
                "audio": torch.from_numpy(self.audio[index]),
                "vision": torch.from_numpy(self.vision[index]),
                "audio_lengths": torch.tensor(self.lengths[index], dtype=torch.long),
                "vision_lengths": torch.tensor(self.lengths[index], dtype=torch.long),
                "index": index}


def validate_outputs(rows, dataset):
    if len(rows) != len(dataset):
        raise ValueError("Predictions do not cover every input sample")
    for row, identity in zip(rows, dataset.identities):
        if set(row) != set(FIELDS) or any(row.get(k) != v for k, v in identity.items()):
            raise ValueError("Prediction identity/order differs from the input; duplicate or missing sample")
        if row["polarity"] not in POLARITIES:
            raise ValueError("Unknown predicted polarity")
        if (isinstance(row["intensity"], bool)
                or not isinstance(row["intensity"], (float, int))
                or not math.isfinite(row["intensity"])):
            raise ValueError("Prediction intensity must be finite")
    return rows


def collect_predictions(model, dataset, device, batch_size):
    import torch
    from torch.utils.data import DataLoader
    from src.models.four_models import predict_batch

    if batch_size < 1:
        raise ValueError("Prediction batch size must be positive")
    batches = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                         num_workers=0)
    records, was_training = [], model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch in batches:
                indices = batch["index"].tolist()
                if indices != list(range(len(records), len(records) + len(indices))):
                    raise ValueError("Prediction loader changed sample order")
                output = predict_batch(model, {key: batch[key] for key in INPUT_FIELDS}, device)
                values = output["intensity"].detach().cpu().reshape(-1)
                logits = output["logits"].detach().cpu()
                if (values.shape != (len(indices),) or logits.shape != (len(indices), 3)
                        or not torch.isfinite(values).all() or not torch.isfinite(logits).all()):
                    raise ValueError("Expected finite intensity [B] and classification logits [B,3]")
                classes = logits.argmax(dim=1).tolist()
                for index, category, value in zip(indices, classes, values.tolist()):
                    records.append({**dataset.identities[index], "polarity": POLARITIES[category],
                                    "intensity": float(value)})
                print(f"Attachment 3: {len(records)}/{len(dataset)} samples predicted", flush=True)
    finally:
        model.train(was_training)
    return validate_outputs(records, dataset)


def export_results(destination, rows, dataset, metadata):
    """Publish only full, validated predictions into a new, never-overwritten folder."""
    validate_outputs(rows, dataset)
    counts = {name: sum(row["polarity"] == name for row in rows) for name in POLARITIES}
    values = np.array([row["intensity"] for row in rows], dtype=np.float64)
    coverage = [{**item, "predicted_samples": sum(row["source_file"] == item["source_file"]
                                                  for row in rows)} for item in dataset.files]
    if any(item["sample_count"] != item["predicted_samples"] for item in coverage):
        raise ValueError("Per-file prediction coverage mismatch")
    summary = {**metadata, "status": "completed", "purpose": "attachment3_prediction",
        "scoring_status": "not_applicable_unlabeled", "metrics": None,
        "sample_count": len(dataset), "prediction_count": len(rows), "coverage_fraction": 1.0,
        "input_directory": str(dataset.directory), "files": coverage,
        "prediction_distribution": {name: {"count": counts[name], "fraction": counts[name] / len(rows)}
                                    for name in POLARITIES},
        "predicted_intensity": {"mean": float(values.mean()), "median": float(np.median(values)),
            "minimum": float(values.min()), "maximum": float(values.max()),
            "outside_label_range_minus3_plus3_count": int((np.abs(values) > 3).sum()),
            "clipped": False},
        "length_policy": LENGTH_POLICY, "finite_policy": FINITE_POLICY,
        "prediction_csv": "predictions.csv", "paper_summary": "results.md"}
    # Validate serializability before creating any deliverable files.
    serialized = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    lines = ["# 附件三全量预测结果", "",
        f"- 模型：JPI-DLFR（{metadata['method']}）；训练 seed={metadata['seed']}；选中 epoch={metadata['selected_epoch']}。",
        f"- 完成覆盖：{len(rows)}/{len(dataset)} 条，覆盖率 100%。",
        "- 附件三无真实标签；以下为预测结果，不计算 Accuracy、F1、MAE 或 Pearson。",
        "", "## 预测分布", "", "| 预测极性 | 数量 | 占比 |", "|---|---:|---:|"]
    lines.extend(f"| {name} | {counts[name]} | {counts[name] / len(rows):.2%} |" for name in POLARITIES)
    lines.extend(["", f"预测强度：均值 {values.mean():.6f}，中位数 {np.median(values):.6f}，"
                  f"范围 [{values.min():.6f}, {values.max():.6f}]。",
                  "", "## 全量结果", "",
                  "CSV 保留原始预测精度；下表仅将显示值保留六位小数。文件内索引从 0 开始。", "",
                  "| 样本编号 | 源文件 | 文件内索引 | 预测极性 | 预测强度 |", "|---|---|---:|---|---:|"])
    escape = lambda value: str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    lines.extend(f"| {escape(row['sample_id'])} | {escape(row['source_file'])} | {row['sample_index']} | "
                 f"{row['polarity']} | {row['intensity']:.6f} |" for row in rows)
    lines.extend(["", "预测分布只描述模型输出，不能作为附件三预测准确性或鲁棒性优越的证据。", ""])
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "predictions.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (destination / "results.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    # Written last: a completed summary is the completion marker.
    with (destination / "summary.json").open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    return summary


def run_prediction(args):
    configure_environment()
    import torch
    from src.compat.provenance import verify_current_sources
    from src.four_model_checkpoint import load_checkpoint, restore_model_state
    from src.models.four_models import build_model

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        raise ValueError("run-name must name one existing outputs subdirectory")
    run = ROOT / "outputs" / args.run_name
    checkpoint_path = run / "best.pt"
    checkpoint_hash = sha256_file(checkpoint_path)
    print(f"Checking checkpoint: {args.run_name}", flush=True)
    checkpoint = load_checkpoint(checkpoint_path, ROOT)
    config = checkpoint["config"]
    if checkpoint["method"] != "M3":
        raise ValueError("Attachment 3 prediction requires the selected M3 checkpoint")
    if args.expected_seed is not None and config["seed"] != args.expected_seed:
        raise ValueError(f"Expected seed {args.expected_seed}, checkpoint uses {config['seed']}")
    pretrained = resolve_path(config["bert_path"])
    for name in ("config.json", "model.safetensors", "vocab.txt"):
        if not (pretrained / name).is_file():
            raise FileNotFoundError(f"Missing local BERT resource: {pretrained / name}")
    dataset = Attachment3Dataset(resolve_path(args.input_dir), args.expected_files, args.expected_samples)
    print(f"Checked {len(dataset.files)} files / {len(dataset)} samples; M3 seed={config['seed']}, "
          f"epoch={checkpoint['epoch']}", flush=True)
    if args.check_only:
        print(json.dumps({"status": "checked_not_predicted", "method": "M3", "seed": config["seed"],
                          "selected_epoch": checkpoint["epoch"], "input_files": len(dataset.files),
                          "samples": len(dataset), "checkpoint_sha256": checkpoint_hash}, ensure_ascii=False))
        return
    destination = (resolve_path(args.output_dir) if args.output_dir else
                   ROOT / "outputs" / "attachment3" /
                   (args.run_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    if destination.exists():
        raise FileExistsError(f"Output directory already exists; choose a new --output-dir: {destination}")
    requested_device = args.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use the training environment or explicitly choose --device cpu")
    device = torch.device(requested_device)
    torch.set_num_threads(args.cpu_threads or config["cpu_threads"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    metadata = {"created_at": datetime.now(timezone.utc).isoformat(), "model_name": "JPI-DLFR",
        "method": "M3", "seed": config["seed"], "selected_epoch": checkpoint["epoch"],
        "source_run": str(run), "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash, "decision": checkpoint["decision"],
        "preprocessing_sha256": checkpoint["preprocessing_sha256"],
        "source_code_sha256": verify_current_sources(), "torch_version": str(torch.__version__),
        "device": str(device), "batch_size": args.batch_size or config["eval_batch_size"]}
    print(f"Loading frozen model on {device}; no training or scoring", flush=True)
    model = build_model(config, ROOT, checkpoint["preprocessing"])
    restore_model_state(model, checkpoint)
    del checkpoint
    model.to(device).eval()
    records = collect_predictions(model, dataset, device, metadata["batch_size"])
    if sha256_file(checkpoint_path) != checkpoint_hash:
        raise ValueError("Checkpoint changed during inference; refusing to publish predictions")
    export_results(destination, records, dataset, metadata)
    print(json.dumps({"status": "completed", "predicted_samples": len(records),
                      "output_directory": str(destination),
                      "prediction_csv": str(destination / "predictions.csv"),
                      "paper_summary": str(destination / "results.md")}, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=["four-models"], default="four-models")
    parser.add_argument("--run-name", required=True, help="Existing M3 run under outputs/")
    parser.add_argument("--expected-seed", type=int, help="Reject a different training seed")
    parser.add_argument("--input-dir", default="resources/data/attachment3/aligned")
    parser.add_argument("--expected-files", type=_positive, default=30)
    parser.add_argument("--expected-samples", type=_positive, default=30)
    parser.add_argument("--output-dir", help="New output directory; default creates a timestamped directory")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=_positive)
    parser.add_argument("--cpu-threads", type=_positive)
    parser.add_argument("--check-only", action="store_true", help="Validate inputs/checkpoint without forward or output files")
    args = parser.parse_args(argv)
    try:
        run_prediction(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
