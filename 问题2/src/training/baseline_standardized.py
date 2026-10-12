"""M0 with mandatory train-fitted A/V channel standardization.

Evaluation integration was implemented with AI assistance (OpenAI Codex,
2026-09-24); the original EMT architecture and MAE selection are unchanged.
"""
import argparse
import csv
from datetime import datetime
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from src.runtime import resolve_path, source_path, source_commit
import random
import time

from src.runtime import ROOT
# Local weights are mandatory: a run must not silently fetch another model.
from src.runtime import configure_environment
configure_environment()

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.aligned import load_splits
from src.evaluation.regression import (calibrate_run, load_decision_rule, load_protocol, score_records,
                                 score_saved_run, initialize_run, update_training_summary,
                                 export_confusion)
from src.evaluation.regression_metrics import regression_metrics
from src.models.standardized import (build_baseline_model as build_model, forward_batch,
                                restore_model_state, validate_checkpoint_preprocessing)
from src.data.normalization import fit_preprocessing, validate_preprocessing
from src.compat.standardized import (CODE_VERSION, implementation_provenance as source_provenance,
                                     validate_implementation)
from src.reporting.console import present_results

LOGGER = logging.getLogger("baseline")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def configure(config, output):
    (output / "details").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(output / "details/run.log", encoding="utf-8"), logging.StreamHandler()
    ], force=True)
    torch.set_num_threads(config["cpu_threads"])
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Check .venv; explicitly use --device cpu if intended.")
    device = torch.device(config["device"])
    LOGGER.info("Device=%s; torch=%s; seed=%s", device, torch.__version__, seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        LOGGER.info("GPU=%s", torch.cuda.get_device_name(0))
    return device


def loader(dataset, config, train=False):
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(dataset, batch_size=config["batch_size"] if train else config["eval_batch_size"],
                      shuffle=train, drop_last=False, num_workers=0,
                      pin_memory=config["device"] == "cuda", generator=generator)


def make_optimizer(model, config):
    groups = {key: [] for key in ["bert_decay", "bert_no_decay", "audio", "video", "other"]}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("text_model."):
            kind = "bert_no_decay" if any(s in name for s in ("bias", "LayerNorm.weight")) else "bert_decay"
        elif name.startswith("audio_model."):
            kind = "audio"
        elif name.startswith("video_model."):
            kind = "video"
        else:
            kind = "other"
        groups[kind].append(param)
    parameters = []
    for kind, params in groups.items():
        modality = "bert" if kind.startswith("bert") else kind
        parameters.append({"params": params, "lr": config[f"learning_rate_{modality}"],
                           "weight_decay": 0.0 if kind == "bert_no_decay" else config[f"weight_decay_{modality}"]})
    # Preserve the official optimizer family and parameter-group learning rates.
    return torch.optim.Adam(parameters, foreach=False)


def gradient_report(model):
    result = {}
    for prefix in ("text_model", "audio_model", "video_model", "fusion", "post_fusion_layer_3"):
        gradients = [p.grad for name, p in model.named_parameters()
                     if name.startswith(prefix + ".") and p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise RuntimeError(f"Missing or non-finite gradients in {prefix}")
        norm = sum(float(g.detach().float().square().sum()) for g in gradients) ** 0.5
        if norm <= 0:
            raise RuntimeError(f"Zero gradients in {prefix}")
        result[prefix] = norm
    return result


def train_epoch(model, batches, optimizer, config, device, check_gradients=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, seen, steps = 0.0, 0, 0
    gradient_evidence = None
    accumulation = config["gradient_accumulation_steps"]
    batch_size = config["batch_size"]
    for batch_idx, batch in enumerate(tqdm(batches, desc="train", mininterval=2, ascii=True)):
        # Normalize each accumulation group by its actual sample count, including
        # the final short group. This also ensures no final samples are discarded.
        group_start = (batch_idx // accumulation) * accumulation * batch_size
        group_samples = min(accumulation * batch_size, len(batches.dataset) - group_start)
        prediction = forward_batch(model, batch, device)
        target = batch["labels"].to(device).reshape(-1)
        loss_sum = torch.nn.functional.l1_loss(prediction, target, reduction="sum")
        if not bool(torch.isfinite(loss_sum)):
            raise RuntimeError("Non-finite training loss")
        (loss_sum / group_samples).backward()
        seen += target.numel()
        total_loss += loss_sum.item()
        if (batch_idx + 1) % accumulation == 0 or batch_idx + 1 == len(batches):
            if check_gradients:
                gradient_evidence = gradient_report(model)
            # Fail on invalid gradients without imposing a new clipping policy.
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
    return {"n": seen, "MAE": total_loss / seen, "optimizer_steps": steps}, gradient_evidence


@torch.inference_mode()
def evaluate(model, batches, device, split):
    model.eval()
    records = []
    for batch in tqdm(batches, desc=split, mininterval=2, ascii=True):
        prediction = forward_batch(model, batch, device).cpu().tolist()
        truth = batch["labels"].reshape(-1).tolist()
        for index, sample_id, target, pred in zip(batch["index"].tolist(), batch["id"], truth, prediction):
            records.append({"split": split, "index": index, "id": str(sample_id),
                            "y_true": target, "y_pred": pred, "absolute_error": abs(pred - target)})
    return regression_metrics([r["y_true"] for r in records], [r["y_pred"] for r in records]), records


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_checkpoint(path, model, config, epoch, metrics, data_hash, purpose):
    checkpoint = {"format_version": 1, "purpose": purpose, "method": "M0",
                  "code_version": CODE_VERSION, "preprocessing": model.av_normalizer.metadata(),
                  "code_sha256": source_provenance(ROOT),
                  "model_name": "EMT-DLFR complete-input regression backbone",
                  "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                  "config": config, "epoch": epoch, "selection_split": "valid", "selection_metric": "MAE",
                  "valid_metrics": metrics, "data_sha256": data_hash,
                  "source_commit": source_commit()}
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def provenance(config, output, preprocessing, protocol=None):
    data_hash = sha256(resolve_path(config["data_path"], ROOT))
    validate_preprocessing(preprocessing, source_sha256=data_hash)
    write_json(output / "preprocessing.json", preprocessing)
    initialize_run(output, config, protocol, {"data_sha256": data_hash,
               "source_commit": source_commit(),
               "code_sha256": source_provenance(ROOT), "code_version": CODE_VERSION,
               "preprocessing": preprocessing,
               "torch": str(torch.__version__), "numpy": np.__version__,
               "split_policy": "original train for fitting; valid MAE for selection; separate final test command"})
    return data_hash


def train(config, output, smoke=False):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output already exists: {output}; use a new run directory")
    # Fix the evaluation protocol before seeing this run's validation results.
    # Smoke uses only three validation examples and must never calibrate a rule.
    protocol = None if smoke else load_protocol(ROOT / "configs/evaluation.json")
    device = configure(config, output)
    preprocessing = fit_preprocessing(resolve_path(config["data_path"], ROOT))
    data_hash = provenance(config, output, preprocessing, protocol)
    # The pickle container physically contains all splits. Only requested splits
    # become datasets; test is never passed to training or validation.
    datasets = load_splits(resolve_path(config["data_path"], ROOT), ("train", "valid"))
    if smoke:
        datasets = {"train": Subset(datasets["train"], range(config["batch_size"] + 1)),
                    "valid": Subset(datasets["valid"], range(3))}
    train_batches = loader(datasets["train"], config, train=True)
    valid_batches = loader(datasets["valid"], config)
    model = build_model(config, ROOT, preprocessing).to(device)
    optimizer = make_optimizer(model, config)
    LOGGER.info("Train=%d; valid=%d; test not evaluated", len(datasets["train"]), len(datasets["valid"]))
    LOGGER.info("Parameters=%d; trainable=%d", sum(p.numel() for p in model.parameters()),
                sum(p.numel() for p in model.parameters() if p.requires_grad))
    patience = config["early_stop_patience"]
    min_delta = config["early_stop_min_delta"]
    LOGGER.info("Max epochs=%d; batch size=%d; accumulation=%d; effective batch size=%d",
                config["epochs"], config["batch_size"], config["gradient_accumulation_steps"],
                config["batch_size"] * config["gradient_accumulation_steps"])
    LOGGER.info("Early stopping: valid MAE; patience=%d; min_delta=%g", patience, min_delta)
    initial_head = model.post_fusion_layer_3.weight.detach().cpu().clone() if smoke else None
    best = float("inf")
    best_epoch = 0
    stop_reason = "max_epochs"
    history = []
    start = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        train_metrics, gradients = train_epoch(model, train_batches, optimizer, config, device, smoke)
        valid_metrics, predictions = evaluate(model, valid_batches, device, "valid")
        history.append({"epoch": epoch, "train_MAE": train_metrics["MAE"], "train_n": train_metrics["n"],
                        "optimizer_steps": train_metrics["optimizer_steps"], "valid_MAE": valid_metrics["MAE"],
                        "valid_RMSE": valid_metrics["RMSE"], "valid_Pearson": valid_metrics["Pearson"]})
        # Match the upstream improvement threshold while selecting only by valid MAE.
        if valid_metrics["MAE"] <= best - min_delta:
            best = valid_metrics["MAE"]
            best_epoch = epoch
            save_checkpoint(output / "best.pt", model, config, epoch, valid_metrics, data_hash, "smoke" if smoke else "train")
            save_csv(output / "details/valid_predictions.csv", predictions)
        save_csv(output / "details/history.csv", history)
        epochs_without_improvement = epoch - best_epoch
        LOGGER.info("Epoch %d/%d train MAE=%.6f valid MAE=%.6f; best epoch=%d; "
                    "no improvement=%d/%d; selected by valid only", epoch,
                    config["epochs"], train_metrics["MAE"], valid_metrics["MAE"], best_epoch,
                    epochs_without_improvement, patience)
        if epochs_without_improvement >= patience:
            stop_reason = "early_stopping"
            LOGGER.info("Early stopping at epoch %d: valid MAE has not improved by at least %g "
                        "for %d epochs. Keeping epoch %d (valid MAE=%.6f).",
                        epoch, min_delta, patience, best_epoch, best)
            break
    update_training_summary(output, {
        "max_epochs": config["epochs"], "epochs_completed": len(history),
        "best_epoch": best_epoch, "best_valid_MAE": best,
        "selection_split": "valid", "selection_metric": "MAE",
        "early_stop_patience": patience, "early_stop_min_delta": min_delta,
        "stop_reason": stop_reason, "stopped_early": len(history) < config["epochs"],
    })
    if smoke:
        if torch.equal(initial_head, model.post_fusion_layer_3.weight.detach().cpu()):
            raise RuntimeError("Optimizer did not update the prediction head")
        reference = torch.tensor([r["y_pred"] for r in predictions])
        del optimizer, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        checkpoint = load_checkpoint(output / "best.pt", allow_smoke=True)
        reloaded = build_model(config, ROOT, checkpoint["preprocessing"])
        restore_model_state(reloaded, checkpoint)
        reloaded.to(device)
        _, after = evaluate(reloaded, valid_batches, device, "valid_reload")
        actual = torch.tensor([r["y_pred"] for r in after])
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
        report = {"status": "passed", "purpose": "pipeline check, not a model-quality result",
                  "train_samples": len(datasets["train"]), "valid_samples": len(datasets["valid"]),
                  "test_samples_evaluated": 0, "train_batch_sizes": [config["batch_size"], 1],
                  "single_sample_training_and_evaluation": True, "finite_gradients_l2": gradients,
                  "optimizer_steps": train_metrics["optimizer_steps"], "head_parameters_changed": True,
                  "checkpoint_strict_reload": True, "reload_max_abs_difference": float((actual-reference).abs().max()),
                  "elapsed_seconds": time.perf_counter() - start,
                  "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None}
        report.update(code_version=CODE_VERSION, preprocessing=checkpoint["preprocessing"],
                      formal_training_started=False, threshold_selected=False,
                      preprocessing_restored_without_fitting=True,
                      temporary_checkpoint_sha256=sha256(output / "best.pt"))
        (output / "best.pt").unlink()
        report["temporary_checkpoint_removed"] = True
        write_json(output / "smoke_test.json", report)
        LOGGER.info("Smoke check PASSED: forward, backward, parameter update, checkpoint save/reload, batch size 1")
    else:
        # The saved CSV belongs to best.pt, not necessarily the final epoch.
        metrics = calibrate_run(ROOT, output / "best.pt", protocol=protocol)
        present_results(ROOT, output, metrics)
    LOGGER.info("Saved to %s", output)


def load_checkpoint(checkpoint_path, *, allow_smoke=False):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_checkpoint_preprocessing(checkpoint)
    if checkpoint.get("method") != "M0" or checkpoint.get("format_version") != 1:
        raise ValueError("Requires a standardized M0 checkpoint")
    if checkpoint.get("purpose") != "train" and not (allow_smoke and checkpoint.get("purpose") == "smoke"):
        raise ValueError("Smoke checkpoints cannot be used for formal evaluation")
    if (checkpoint.get("selection_split"), checkpoint.get("selection_metric")) != ("valid", "MAE"):
        raise ValueError("Requires a checkpoint selected by valid MAE")
    validate_implementation(checkpoint.get("code_sha256"), source_provenance(ROOT))
    return checkpoint


def final_test(checkpoint_path, device_override=None):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Train first; checkpoint does not exist: {checkpoint_path}")
    # Fail before constructing test data if calibration is missing or stale.
    # This path must never search for a threshold using test labels.
    checkpoint = load_checkpoint(checkpoint_path)
    rule = load_decision_rule(ROOT, checkpoint_path)
    if checkpoint.get("purpose") != "train" or checkpoint.get("selection_split") != "valid":
        raise ValueError("Final test requires a training checkpoint selected by valid; smoke checkpoints are rejected")
    config = checkpoint["config"].copy()
    if device_override:
        config["device"] = device_override
    output = checkpoint_path.parent
    device = configure(config, output)
    if sha256(resolve_path(config["data_path"], ROOT)) != checkpoint["data_sha256"]:
        raise ValueError("Dataset SHA256 differs from the training checkpoint")
    datasets = load_splits(resolve_path(config["data_path"], ROOT), ("test",))
    model = build_model(config, ROOT, checkpoint["preprocessing"])
    restore_model_state(model, checkpoint)
    model.to(device)
    metrics, predictions = evaluate(model, loader(datasets["test"], config), device, "test")
    full_metrics = score_records(ROOT, checkpoint_path, predictions, "test", rule=rule)
    present_results(ROOT, output, full_metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["smoke", "train", "test", "calibrate-valid", "score", "export"])
    parser.add_argument("--config", type=Path,
                        help="Training JSON config (default: configs/baseline.json)")
    parser.add_argument("--epochs", type=int, help="Maximum training epochs; early stopping may finish sooner")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int,
                        help="Training micro-batches per optimizer update; overrides the JSON config")
    parser.add_argument("--seed", type=int, help="Training random seed (0..4294967295); overrides the JSON config")
    parser.add_argument("--device", choices=["cuda", "cpu"])
    parser.add_argument("--run-name", default="complete_standardized_s1111")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=["valid", "test"], help="Saved split to score or export")
    parser.add_argument("--evaluation-config", type=Path, help="Protocol for calibrate-valid only")
    args = parser.parse_args()
    if (not args.run_name or Path(args.run_name).name != args.run_name
            or args.run_name in (".", "..") or ":" in args.run_name
            or "/" in args.run_name or "\\" in args.run_name):
        parser.error("--run-name must be a single folder name")
    if args.command in ("calibrate-valid", "score", "export"):
        if any(value is not None for value in (args.config, args.epochs, args.batch_size,
                                               args.gradient_accumulation_steps, args.seed, args.device)):
            parser.error("Saved-prediction commands do not accept training or device overrides")
        if args.command == "calibrate-valid" and args.split == "test":
            parser.error("calibrate-valid can only use valid predictions")
        if args.command in ("score", "export") and (args.split is None or args.evaluation_config is not None):
            parser.error("score/export require --split and use the frozen protocol; no --evaluation-config")
        checkpoint = (args.checkpoint or ROOT / "outputs" / args.run_name / "best.pt").resolve()
        try:
            checked = load_checkpoint(checkpoint)
            del checked
            if args.command == "calibrate-valid":
                protocol = load_protocol(args.evaluation_config) if args.evaluation_config else None
                metrics = calibrate_run(ROOT, checkpoint, protocol=protocol)
            elif args.command == "score":
                metrics = score_saved_run(ROOT, checkpoint, args.split)
            else:
                destination = export_confusion(ROOT, checkpoint, args.split)
                print(f"Exported to {destination}")
                return
            present_results(ROOT, checkpoint.parent, metrics)
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))
        return
    if args.split is not None or args.evaluation_config is not None:
        parser.error("--split/--evaluation-config apply only to saved-prediction commands")
    if args.command == "test":
        if any(value is not None for value in (args.config, args.epochs, args.batch_size,
                                               args.gradient_accumulation_steps, args.seed)):
            parser.error("test uses saved checkpoint configuration; no training overrides")
        final_test((args.checkpoint or ROOT / "outputs" / args.run_name / "best.pt").resolve(), args.device)
        return
    config = read_json(args.config or ROOT / "configs/baseline.json")
    # Saved run configs now wrap the unchanged training settings. Accept both
    # storage formats; a new run still freezes its own evaluation protocol.
    if set(config) == {"training", "evaluation_protocol"}:
        config = config["training"].copy()
    # Older configuration files inherit the new default training controls.
    config.setdefault("early_stop_patience", 8)
    config.setdefault("early_stop_min_delta", 1e-6)
    for key, value in (("epochs", args.epochs), ("batch_size", args.batch_size), ("device", args.device),
                       ("gradient_accumulation_steps", args.gradient_accumulation_steps), ("seed", args.seed)):
        if value is not None:
            config[key] = value
    for key in ("epochs", "batch_size", "eval_batch_size", "gradient_accumulation_steps", "early_stop_patience"):
        if type(config[key]) is not int or config[key] < 1:
            parser.error(f"{key} must be a positive integer")
    if type(config["seed"]) is not int or not 0 <= config["seed"] <= 2**32 - 1:
        parser.error("seed must be an integer between 0 and 4294967295")
    min_delta = config["early_stop_min_delta"]
    if type(min_delta) not in (int, float) or not math.isfinite(min_delta) or min_delta <= 0:
        parser.error("early_stop_min_delta must be a positive finite number")
    if config["missing_rate"] != [0.0, 0.0, 0.0] or config["regression_only"] is not True:
        parser.error("This first-step baseline supports complete-input regression only")
    smoke = args.command == "smoke"
    if smoke:
        config.update(epochs=1, eval_batch_size=2)
        if args.gradient_accumulation_steps is None:
            config["gradient_accumulation_steps"] = 2
    output = (ROOT / "_dev/records/standardized_smoke" / ("M0_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
              if smoke else ROOT / "outputs" / args.run_name)
    if not smoke and output.exists() and any(output.iterdir()):
        parser.error(f"Output already exists: {output}. Use --run-name with a new name to preserve the previous run.")
    train(config, output, smoke=smoke)


if __name__ == "__main__":
    main()
