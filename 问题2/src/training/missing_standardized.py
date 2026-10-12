"""M1/M2 with fixed train-fitted A/V standardization and mix_v1 evaluation.

Version av_standardized_v1. The legacy run_missing.py stays unchanged for old
checkpoints. Both A/V inputs and M2 A/V targets use one train-fitted scaler.
Formal training is an
explicit `train` command; `smoke` is bounded and cannot publish an evaluation
threshold or a formal experiment checkpoint.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from src.runtime import resolve_path, source_path
import re
import time

from src.runtime import ROOT
from src.runtime import configure_environment
configure_environment()

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.training.baseline import configure, make_optimizer, save_csv
from src.evaluation.regression import load_protocol
from src.evaluation.regression_metrics import regression_metrics
from src.data.construction.common import canonical_json, json_fingerprint, sha256_file
from src.data.missing import MixDataStore
from src.models.standardized import (CODE_VERSION, build_robust_model as build_model,
                                forward_training, predict_batch, compute_losses, restore_model_state)
from src.data.normalization import fit_preprocessing, validate_preprocessing
from src.compat.standardized import (implementation_provenance as standardized_provenance,
                                     validate_implementation)
from src.evaluation.robust import (initialize_run, calibrate, load_frozen_rule, save_evaluation)

LOGGER = logging.getLogger("missing_standardized")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def path_in_project(value):
    return resolve_path(value, ROOT)


def default_config(method):
    config = read_json(ROOT / "configs/baseline.json")
    config.pop("missing_rate", None)
    config.update(method=method, epochs=20, early_stop_patience=5, batch_size=8, gradient_accumulation_steps=8,
                  eval_batch_size=8, artifact_path="dataset_construction/artifacts/mix_v1",
                  reconstruction_loss="SmoothL1Loss", reconstruction_weight=1.0,
                  consistency_weight=1.0, prediction_missing_weight=1.0,
                  prediction_reference_weight=1.0, regression_only=(method == "M1"),
                  initialization="same-seed local base BERT plus fresh EMT; no complete-run warm start",
                  singleton_batchnorm_policy="running statistics only for physical batch=1",
                  training_family="main_mixed", training_condition="mixed_v1_s0",
                  selection_family="main_mixed", selection_condition="mixed_v1_s0")
    return config


def validate_config(config):
    if config.get("method") not in ("M1", "M2"):
        raise ValueError("method must be M1 or M2")
    for key in ("epochs", "batch_size", "eval_batch_size", "gradient_accumulation_steps", "early_stop_patience", "cpu_threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["seed"]) is not int or not 0 <= config["seed"] <= 2**32 - 1:
        raise ValueError("seed must be in 0..2**32-1")
    if config["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    for key in ("early_stop_min_delta", "learning_rate_bert", "learning_rate_audio", "learning_rate_video", "learning_rate_other"):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    for key in ("reconstruction_weight", "consistency_weight"):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
        if config["method"] == "M2" and config[key] == 0:
            raise ValueError("M2 requires both DLFR losses; use M1 for no DLFR")
    if config["prediction_missing_weight"] != 1 or config["prediction_reference_weight"] != 1:
        raise ValueError("Fair M1/M2 comparison fixes both prediction loss weights to one")
    if config["reconstruction_loss"] not in ("SmoothL1Loss", "L1Loss", "MSELoss"):
        raise ValueError("Unsupported reconstruction loss")
    for key, expected in (("training_family", "main_mixed"), ("selection_family", "main_mixed"),
                          ("training_condition", "mixed_v1_s0"), ("selection_condition", "mixed_v1_s0")):
        if config[key] != expected:
            raise ValueError(f"{key} is fixed to {expected}")
    if config["regression_only"] != (config["method"] == "M1"):
        raise ValueError("regression_only disagrees with method")
    return config


def make_loader(dataset, config, train=False):
    return DataLoader(dataset, batch_size=config["batch_size"] if train else config["eval_batch_size"],
                      shuffle=train, drop_last=False, num_workers=0,
                      pin_memory=config["device"] == "cuda",
                      generator=torch.Generator().manual_seed(config["seed"]))


def _auxiliary_name(name):
    return "projector" in name or "predictor" in name or name.startswith("recon_")


def backbone_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if _auxiliary_name(name):
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(canonical_json(list(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def gradient_evidence(model, method):
    evidence = {}
    prefixes = ["text_model", "audio_model", "video_model", "fusion", "post_fusion_layer_3"]
    if method == "M2":
        prefixes.extend(["recon_text", "recon_audio", "recon_video", "gmc_tokens_predictor", "text_predictor", "audio_predictor", "video_predictor"])
    for prefix in prefixes:
        gradients = [param.grad for name, param in model.named_parameters() if name.startswith(prefix + ".") and param.grad is not None]
        if not gradients or any(not bool(torch.isfinite(value).all()) for value in gradients):
            raise RuntimeError(f"Missing/nonfinite gradients in {prefix}")
        norm = math.sqrt(sum(float(value.detach().float().square().sum()) for value in gradients))
        if norm <= 0:
            raise RuntimeError(f"Expected nonzero gradient in {prefix}")
        evidence[prefix] = norm
    if method == "M1" and any(param.grad is not None for name, param in model.named_parameters() if _auxiliary_name(name)):
        raise RuntimeError("M1 produced DLFR gradients")
    return evidence


def train_epoch(model, batches, optimizer, config, device, check_gradients=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    seen, steps, sums, evidence, batch_sizes = 0, 0, {}, None, []
    accumulation = config["gradient_accumulation_steps"]
    for batch_idx, batch in enumerate(tqdm(batches, desc=config["method"] + " train", mininterval=2, ascii=True)):
        group_start = (batch_idx // accumulation) * accumulation * config["batch_size"]
        group_samples = min(accumulation * config["batch_size"], len(batches.dataset) - group_start)
        outputs = forward_training(model, batch, device)
        losses = compute_losses(outputs, batch, config, device)
        n = int(batch["labels"].numel())
        if any(not bool(torch.isfinite(value).all()) for value in losses.values()):
            raise RuntimeError("Nonfinite M1/M2 loss")
        (losses["total"] * (n / group_samples)).backward()
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach()) * n
        batch_sizes.append(n)
        seen += n
        if (batch_idx + 1) % accumulation == 0 or batch_idx + 1 == len(batches):
            if check_gradients:
                evidence = gradient_evidence(model, config["method"])
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
        del outputs, losses
    if seen != len(batches.dataset) or steps < 1:
        raise RuntimeError("Training lost samples or failed to step")
    return {"n": seen, "optimizer_steps": steps, **{key: value / seen for key, value in sums.items()}}, evidence, batch_sizes


@torch.inference_mode()
def evaluate(model, batches, device, split, label=None):
    model.eval()
    records = []
    for batch in tqdm(batches, desc=label or split, mininterval=2, ascii=True):
        if "reference" in batch or "reconstruction" in batch:
            raise RuntimeError("Evaluation batch exposed a training reference or simulator support")
        predictions = predict_batch(model, batch, device).cpu().reshape(-1).tolist()
        for index, sample_id, truth, pred in zip(batch["index"].tolist(), batch["id"], batch["labels"].reshape(-1).tolist(), predictions):
            records.append({"split": split, "index": int(index), "id": str(sample_id), "y_true": truth,
                            "y_pred": pred, "absolute_error": abs(pred - truth)})
    if not records:
        raise ValueError("Empty evaluation cohort")
    return regression_metrics([row["y_true"] for row in records], [row["y_pred"] for row in records]), records


def experiment_binding(store, config, epoch, purpose="train", include_test=False):
    binding = {**store.binding(), "purpose": purpose, "selected_epoch": int(epoch),
               "training_config_sha256": json_fingerprint(config), "cohorts": {"valid": store.cohort("valid")}}
    if include_test:
        binding["cohorts"]["test"] = store.cohort("test")
    return binding


def implementation_provenance():
    return standardized_provenance(ROOT)


def validate_checkpoint_version(checkpoint):
    """Reject old artifacts before evaluation can write into their run folder."""
    if checkpoint.get("code_version") != CODE_VERSION:
        raise ValueError("This entry requires an av_standardized_v1 checkpoint. "
                         "Use run.py evaluate --version missing to reproduce legacy unstandardized M1/M2 runs.")
    validate_preprocessing(checkpoint.get("preprocessing"))


def require_new_output(output):
    """Protect historical runs even when train/smoke is called as a Python API."""
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite an existing nonempty run: {output}")


def save_checkpoint(path, model, config, epoch, valid_metrics, store, initial_hash, purpose="train"):
    payload = {"format_version": 1, "code_version": CODE_VERSION,
               "preprocessing": model.av_normalizer.metadata(),
               "purpose": purpose, "method": config["method"],
               "model_name": "EMT standardized A/V fixed missing augmentation" if config["method"] == "M1" else "EMT-DLFR standardized A/V fixed missing augmentation",
               "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
               "config": config, "epoch": int(epoch), "selection_split": "valid", "selection_family": "main_mixed",
               "selection_condition": "mixed_v1_s0", "selection_metric": "MAE", "valid_metrics": valid_metrics,
               "data_binding": store.binding(), "initial_backbone_sha256": initial_hash,
               "code_sha256": implementation_provenance()}
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def load_checkpoint(path, store, *, allow_smoke=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    validate_checkpoint_version(checkpoint)
    validate_preprocessing(checkpoint["preprocessing"], source_sha256=store.binding()["data_sha256"])
    if checkpoint.get("format_version") != 1 or checkpoint.get("method") not in ("M1", "M2"):
        raise ValueError("Not an M1/M2 checkpoint")
    if checkpoint.get("purpose") != "train" and not (allow_smoke and checkpoint.get("purpose") == "smoke"):
        raise ValueError("A smoke checkpoint cannot be used for formal evaluation")
    if (checkpoint.get("selection_split"), checkpoint.get("selection_family"), checkpoint.get("selection_condition"), checkpoint.get("selection_metric")) != ("valid", "main_mixed", "mixed_v1_s0", "MAE"):
        raise ValueError("Checkpoint was not selected by main mixed valid MAE")
    if checkpoint["data_binding"] != store.binding():
        raise ValueError("Checkpoint fixed-mask/source binding differs")
    validate_implementation(checkpoint["code_sha256"], implementation_provenance())
    validate_config(checkpoint["config"])
    return checkpoint


def train(config, output):
    require_new_output(output)
    device = configure(config, output)
    store = MixDataStore(path_in_project(config["artifact_path"]), path_in_project(config["data_path"]))
    preprocessing = fit_preprocessing(store.source_path)
    validate_preprocessing(preprocessing, source_sha256=store.binding()["data_sha256"])
    write_json(output / "preprocessing.json", preprocessing)
    protocol = load_protocol(ROOT / "configs/evaluation.json")
    provenance = {"method": config["method"], "code_version": CODE_VERSION, "preprocessing": preprocessing,
                  "data_binding": store.binding(), "code_sha256": implementation_provenance(),
                  "purpose": "train", "selection": "all main_mixed valid MAE; one frozen threshold across views"}
    initialize_run(output, config, protocol, provenance)
    datasets = {"train": store.dataset("train", training=True), "valid": store.dataset("valid")}
    batches = {split: make_loader(dataset, config, train=(split == "train")) for split, dataset in datasets.items()}
    model = build_model(config, ROOT, preprocessing)
    initial_hash = backbone_fingerprint(model)
    model.to(device)
    optimizer = make_optimizer(model, config)
    LOGGER.info("Method=%s; fixed train=%d; fixed valid=%d; no test dataset for fitting", config["method"], len(datasets["train"]), len(datasets["valid"]))
    LOGGER.info("Prediction supervision: original L1 + missing L1, both weight 1; aux enabled=%s", config["method"] == "M2")
    best, best_epoch, history, stop_reason = float("inf"), 0, [], "max_epochs"
    for epoch in range(1, config["epochs"] + 1):
        train_metrics, _, _ = train_epoch(model, batches["train"], optimizer, config, device)
        valid_metrics, records = evaluate(model, batches["valid"], device, "valid", "main mixed valid")
        history.append({"epoch": epoch, **{"train_" + key: value for key, value in train_metrics.items()},
                        "valid_MAE": valid_metrics["MAE"], "valid_Pearson": valid_metrics["Pearson"]})
        if best_epoch == 0 or valid_metrics["MAE"] <= best - config["early_stop_min_delta"]:
            best, best_epoch = valid_metrics["MAE"], epoch
            save_checkpoint(output / "best.pt", model, config, epoch, valid_metrics, store, initial_hash)
        save_csv(output / "details/history.csv", history)
        LOGGER.info("Epoch=%d; missing train MAE=%.6f; original train MAE=%.6f; valid MAE=%.6f; best=%d", epoch, train_metrics["pred_missing"], train_metrics["pred_reference"], valid_metrics["MAE"], best_epoch)
        if epoch - best_epoch >= config["early_stop_patience"]:
            stop_reason = "early_stopping"
            break
    del optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    checkpoint = load_checkpoint(output / "best.pt", store)
    model = build_model(config, ROOT, checkpoint["preprocessing"])
    restore_model_state(model, checkpoint)
    model.to(device)
    del checkpoint
    valid_metrics, records = evaluate(model, batches["valid"], device, "valid", "best main mixed valid")
    if not math.isclose(valid_metrics["MAE"], best, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError("Reloaded best checkpoint did not reproduce its validation MAE")
    binding = experiment_binding(store, config, best_epoch)
    rule = calibrate(output, records, sha256_file(output / "best.pt"), binding)
    report = read_json(output / "evaluation.json")
    report["training_summary"] = {"max_epochs": config["epochs"], "epochs_completed": len(history), "best_epoch": best_epoch,
                                  "best_valid_MAE": best, "early_stop_patience": config["early_stop_patience"],
                                  "early_stop_min_delta": config["early_stop_min_delta"], "stop_reason": stop_reason,
                                  "selection_split": "valid", "selection_family": "main_mixed", "selection_metric": "MAE",
                                  "initial_backbone_sha256": initial_hash, "test_run": False}
    write_json(output / "evaluation.json", report)
    print(json.dumps({"method": config["method"], "best_epoch": best_epoch, "valid_MAE": best,
                      "decision_rule": rule, "results": str(output / "evaluation.json")}, ensure_ascii=False), flush=True)


def test(output, device_override=None):
    checkpoint_path = output / "best.pt"
    # Read config only to locate the frozen data, then validate all bindings.
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_checkpoint_version(raw_checkpoint)
    config = dict(raw_checkpoint["config"])
    del raw_checkpoint
    store = MixDataStore(path_in_project(config["artifact_path"]), path_in_project(config["data_path"]))
    checkpoint = load_checkpoint(checkpoint_path, store)
    runtime = {**config, "device": device_override or config["device"]}
    device = configure(runtime, output)
    binding = experiment_binding(store, config, checkpoint["epoch"], include_test=True)
    checkpoint_hash = sha256_file(checkpoint_path)
    rule = load_frozen_rule(output, checkpoint_hash, binding)
    model = build_model(config, ROOT, checkpoint["preprocessing"])
    restore_model_state(model, checkpoint)
    model.to(device)
    del checkpoint
    main_test = store.dataset("test")
    _, records = evaluate(model, make_loader(main_test, runtime), device, "test", "main mixed test")
    save_evaluation(output, records, "test", "main_mixed", "mixed_v1_s0", checkpoint_hash, binding, rule=rule)
    for split in ("valid", "test"):
        panels = store.panels(split)
        conditions = list(dict.fromkeys(condition for panel in panels.values() for condition in panel["condition_ids"]))
        for condition in conditions:
            dataset = store.prediction_dataset(split, condition)
            _, records = evaluate(model, make_loader(dataset, runtime), device, split, f"{split} paired {condition}")
            save_evaluation(output, records, split, "paired", condition, checkpoint_hash, binding, rule=rule)
            for name, panel in panels.items():
                if condition in panel["condition_ids"]:
                    save_evaluation(output, records, split, "paired", condition, checkpoint_hash, binding,
                                    panel={"name": name, **panel}, rule=rule)
    report = read_json(output / "evaluation.json")
    report.setdefault("training_summary", {})["test_run"] = True
    write_json(output / "evaluation.json", report)
    print(json.dumps({"method": config["method"], "status": "test and paired evaluation complete", "results": str(output / "evaluation.json")}, ensure_ascii=False))


def _smoke_indices(artifact_root, batch_size):
    path = artifact_root / "metadata/main_mixed/train/mixed_v1_s0.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    predicates = [lambda row: row["modalities"]["V"]["original_all_unobserved"],
                  lambda row: row["actual"]["synchrony"] == "shared_plus_extra",
                  lambda row: row["length"] == 48,
                  lambda row: row["plan"]["modality_set"] == "NONE"]
    selected = []
    for predicate in predicates:
        candidates = [row["index"] for row in rows if predicate(row)]
        if candidates and candidates[0] not in selected:
            selected.append(candidates[0])
    for index in range(len(rows)):
        if len(selected) >= batch_size:
            break
        if index not in selected:
            selected.append(index)
    selected = selected[:batch_size]
    final = next(row["index"] for row in rows if row["length"] == 1 and row["index"] not in selected)
    return selected + [final]


def smoke(config, output):
    require_new_output(output)
    config = {**config, "epochs": 1, "gradient_accumulation_steps": 2, "eval_batch_size": 2}
    device = configure(config, output)
    artifact_root = path_in_project(config["artifact_path"])
    store = MixDataStore(artifact_root, path_in_project(config["data_path"]))
    # Fit the complete original train, not the bounded smoke subset.
    preprocessing = fit_preprocessing(store.source_path)
    validate_preprocessing(preprocessing, source_sha256=store.binding()["data_sha256"])
    write_json(output / "preprocessing.json", preprocessing)
    indices = _smoke_indices(artifact_root, config["batch_size"])
    train_dataset = Subset(store.dataset("train", training=True), indices)
    valid_dataset = Subset(store.dataset("valid"), [0, 1, 2])
    # Keep explicit order so the final singleton (L=1, no recovery support) is exercised.
    batches = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=False, num_workers=0)
    model = build_model(config, ROOT, preprocessing)
    initial_hash = backbone_fingerprint(model)
    initial_head = model.post_fusion_layer_3.weight.detach().clone()
    model.to(device)
    optimizer = make_optimizer(model, config)
    begin = time.perf_counter()
    train_metrics, gradients, sizes = train_epoch(model, batches, optimizer, config, device, check_gradients=True)
    if sizes != [config["batch_size"], 1] or train_metrics["optimizer_steps"] != 1:
        raise RuntimeError("Smoke did not exercise the intended two microbatches and one update")
    if torch.equal(initial_head, model.post_fusion_layer_3.weight.detach().cpu()):
        raise RuntimeError("Smoke optimizer did not update the prediction head")
    metrics, before = evaluate(model, make_loader(valid_dataset, config), device, "valid", "smoke valid (3 only)")
    temporary_checkpoint = output / "roundtrip.pt"
    save_checkpoint(temporary_checkpoint, model, config, 1, metrics, store, initial_hash, purpose="smoke")
    temporary_hash = sha256_file(temporary_checkpoint)
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None
    del optimizer, model, initial_head
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    checkpoint = load_checkpoint(temporary_checkpoint, store, allow_smoke=True)
    model = build_model(config, ROOT, checkpoint["preprocessing"])
    restore_model_state(model, checkpoint)
    model.to(device)
    del checkpoint
    if model.av_normalizer.metadata() != preprocessing:
        raise RuntimeError("Smoke reload changed the saved train-fitted preprocessing")
    _, after = evaluate(model, make_loader(valid_dataset, config), device, "valid", "smoke strict reload")
    differences = np.abs(np.asarray([row["y_pred"] for row in before]) - np.asarray([row["y_pred"] for row in after]))
    if not np.allclose([row["y_pred"] for row in before], [row["y_pred"] for row in after], rtol=1e-6, atol=1e-6):
        raise RuntimeError("Smoke checkpoint reload changed predictions")
    temporary_checkpoint.unlink()
    result = {"passed": True, "purpose": "smoke", "method": config["method"], "config": config,
              "code_version": CODE_VERSION, "preprocessing": preprocessing,
              "preprocessing_roundtrip_identical": True,
              "train_original_indices": indices, "microbatch_sizes": sizes, "optimizer_steps": 1,
              "train_loss_terms": train_metrics, "gradients": gradients, "initial_backbone_sha256": initial_hash,
              "strict_reload_max_abs_difference": float(differences.max()), "temporary_checkpoint_sha256": temporary_hash,
              "temporary_checkpoint_removed": True, "peak_cuda_allocated_mb": peak_mb,
              "seconds": time.perf_counter() - begin, "formal_training_started": False,
              "threshold_selected": False, "test_evaluated": False, "valid_probe_n": 3,
              "data_binding": store.binding()}
    write_json(output / "smoke_report.json", result)
    print(json.dumps({"passed": True, "method": config["method"], "smoke_report": str(output / "smoke_report.json"),
                      "peak_cuda_allocated_mb": peak_mb}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="M1/M2 with fixed standardized A/V on frozen mix_v1; smoke never starts formal training")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "smoke"):
        entry = sub.add_parser(command)
        entry.add_argument("--method", choices=("M1", "M2"), required=True)
        entry.add_argument("--run-name")
        entry.add_argument("--config", type=Path)
        entry.add_argument("--epochs", type=int)
        entry.add_argument("--batch-size", type=int)
        entry.add_argument("--gradient-accumulation-steps", type=int)
        entry.add_argument("--seed", type=int)
        entry.add_argument("--device", choices=("cpu", "cuda"))
    entry = sub.add_parser("test")
    entry.add_argument("--run-name", required=True)
    entry.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.run_name and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_name):
        parser.error("run-name must be a simple name, not a path")
    if args.command == "test":
        test(ROOT / "outputs" / args.run_name, args.device)
        return
    config = default_config(args.method)
    if args.config:
        config.update(read_json(args.config))
    config.update(method=args.method, regression_only=(args.method == "M1"))
    for key in ("epochs", "batch_size", "gradient_accumulation_steps", "seed", "device"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    validate_config(config)
    if args.command == "train":
        if not args.run_name:
            parser.error("Formal training requires --run-name")
        output = ROOT / "outputs" / args.run_name
    else:
        if args.run_name:
            parser.error("Smoke does not create a named formal run")
        output = ROOT / "_dev/records/standardized_smoke" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + args.method)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output}")
    if args.command == "train":
        train(config, output)
    else:
        smoke(config, output)


if __name__ == "__main__":
    main()
