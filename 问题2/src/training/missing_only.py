"""Single corrupted-view EMT: one L1 task loss, no reference or DLFR.

Separate entry so existing M1/M2 source fingerprints and checkpoints stay valid.
Implemented with OpenAI Codex assistance, 2026-09-25.
"""
from __future__ import annotations

import argparse
import gc
import logging
import math
from pathlib import Path
import re

# Import the existing runner first to set the same offline/runtime environment.
import src.training.missing as shared
import torch
from torch.nn import functional as F
from tqdm import tqdm

from src.evaluation.regression import load_protocol
from src.evaluation.regression_metrics import regression_metrics
from src.data.construction.common import sha256_file
from src.data.missing import MixDataStore
from src.models.robust import build_model as build_backbone, predict_batch
from src.evaluation.robust import initialize_run, calibrate, load_frozen_rule, save_evaluation

from src.runtime import ROOT
METHOD = "EMT_MISSING_ONLY"
LOGGER = logging.getLogger("missing_only")
configure = shared.configure
make_optimizer = shared.make_optimizer
make_loader = shared.make_loader
save_csv = shared.save_csv
read_json = shared.read_json
write_json = shared.write_json
path_in_project = shared.path_in_project
experiment_binding = shared.experiment_binding
backbone_fingerprint = shared.backbone_fingerprint


def default_config():
    config = shared.default_config("M1")
    config.update(method=METHOD, input_view="single_corrupted", regression_only=True,
                  prediction_reference_weight=0.0, prediction_missing_weight=1.0,
                  reconstruction_weight=0.0, consistency_weight=0.0)
    return config


def validate_config(config):
    fixed = {"method": METHOD, "input_view": "single_corrupted", "regression_only": True,
             "prediction_missing_weight": 1.0, "prediction_reference_weight": 0.0,
             "reconstruction_weight": 0.0, "consistency_weight": 0.0}
    for key, expected in fixed.items():
        if config.get(key) != expected:
            raise ValueError(f"Single-view EMT requires {key}={expected!r}")
    if config.get("need_data_aligned") is not True:
        raise ValueError("Single-view mix_v1 requires aligned inputs")
    # Reuse numeric/selection validation without altering single-view identity.
    shared.validate_config({**config, "method": "M1", "prediction_reference_weight": 1.0})
    return config


def build_model(config, root=ROOT):
    validate_config(config)
    # M1 selects the frozen auxiliary-head architecture; its two-view training
    # functions are never called. The experiment identity remains METHOD.
    return build_backbone({**config, "method": "M1"}, root)


def implementation_provenance():
    from src.compat.provenance import current_provenance
    return current_provenance("missing_only", ROOT)


def _single_view(batch):
    if "reference" in batch or "reconstruction" in batch:
        raise RuntimeError("Single-view EMT forbids reference/reconstruction in every batch")


def train_epoch(model, batches, optimizer, config, device, check_gradients=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    seen, steps, loss_sum, sizes, evidence = 0, 0, 0.0, [], None
    accumulation = config["gradient_accumulation_steps"]
    for batch_idx, batch in enumerate(tqdm(batches, desc="single-view EMT train", mininterval=2, ascii=True)):
        _single_view(batch)
        start = (batch_idx // accumulation) * accumulation * config["batch_size"]
        group_n = min(accumulation * config["batch_size"], len(batches.dataset) - start)
        prediction = predict_batch(model, batch, device)
        labels = batch["labels"].to(device=device, dtype=torch.float32).reshape(-1)
        if prediction.shape != labels.shape:
            raise ValueError("Prediction and label shapes disagree")
        loss = F.l1_loss(prediction, labels, reduction="sum")
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite single-view task loss")
        (loss / group_n).backward()
        n = labels.numel()
        seen += n
        sizes.append(n)
        loss_sum += float(loss.detach())
        if (batch_idx + 1) % accumulation == 0 or batch_idx + 1 == len(batches):
            if check_gradients:
                evidence = shared.gradient_evidence(model, "M1")
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
    if seen != len(batches.dataset) or seen == 0 or steps == 0:
        raise RuntimeError("Training omitted samples or optimizer updates")
    mae = loss_sum / seen
    return {"n": seen, "optimizer_steps": steps, "pred_missing": mae,
            "total": mae, "prediction_views_per_sample": 1}, evidence, sizes


@torch.inference_mode()
def evaluate(model, batches, device, split, label=None):
    model.eval()
    records = []
    for batch in tqdm(batches, desc=label or split, mininterval=2, ascii=True):
        _single_view(batch)
        predictions = predict_batch(model, batch, device).cpu().reshape(-1).tolist()
        for index, sample_id, truth, pred in zip(batch["index"].tolist(), batch["id"],
                                               batch["labels"].reshape(-1).tolist(), predictions):
            records.append({"split": split, "index": int(index), "id": str(sample_id),
                            "y_true": truth, "y_pred": pred, "absolute_error": abs(pred - truth)})
    if not records:
        raise ValueError("Empty evaluation cohort")
    return regression_metrics([r["y_true"] for r in records], [r["y_pred"] for r in records]), records


def save_checkpoint(path, model, config, epoch, valid_metrics, store, initial_hash, purpose="train"):
    validate_config(config)
    payload = {"format_version": 1, "purpose": purpose, "method": METHOD,
               "model_name": "EMT single corrupted-view supervision",
               "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "config": config, "epoch": int(epoch), "valid_metrics": valid_metrics,
               "selection_split": "valid", "selection_family": "main_mixed",
               "selection_condition": "mixed_v1_s0", "selection_metric": "MAE",
               "data_binding": store.binding(), "initial_backbone_sha256": initial_hash,
               "code_sha256": implementation_provenance()}
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, store, *, allow_smoke=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("method") != METHOD:
        raise ValueError("Not an EMT_MISSING_ONLY checkpoint; use this experiment's own test entry")
    if checkpoint.get("purpose") != "train" and not (allow_smoke and checkpoint.get("purpose") == "smoke"):
        raise ValueError("A smoke checkpoint cannot be used for formal testing")
    validate_config(checkpoint["config"])
    if tuple(checkpoint.get(k) for k in ("selection_split", "selection_family", "selection_condition", "selection_metric")) != (
            "valid", "main_mixed", "mixed_v1_s0", "MAE"):
        raise ValueError("Checkpoint was not selected by main mixed valid MAE")
    if checkpoint["data_binding"] != store.binding():
        raise ValueError("Checkpoint mask/source binding changed")
    from src.compat.provenance import validate_provenance
    validate_provenance(checkpoint["code_sha256"], implementation_provenance(), "missing_only", ROOT)
    return checkpoint


def train(config, output):
    validate_config(config)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output}")
    device = configure(config, output)
    store = MixDataStore(path_in_project(config["artifact_path"]), path_in_project(config["data_path"]))
    protocol = load_protocol(ROOT / "configs/evaluation.json")
    initialize_run(output, config, protocol, {"method": METHOD, "purpose": "train",
                   "data_binding": store.binding(), "code_sha256": implementation_provenance(),
                   "prediction_supervision": "one L1 loss on the fixed corrupted view only"})
    # training=False deliberately withholds complete references/support even on train.
    datasets = {split: store.dataset(split, training=False) for split in ("train", "valid")}
    batches = {s: make_loader(d, config, train=s == "train") for s, d in datasets.items()}
    model = build_model(config, ROOT)
    initial_hash = backbone_fingerprint(model)
    model.to(device)
    optimizer = make_optimizer(model, config)
    LOGGER.info("Single corrupted view, L1 only; train=%d valid=%d; mix_v1 clean quota retained",
                len(datasets["train"]), len(datasets["valid"]))
    best, best_epoch, history, stop_reason = float("inf"), 0, [], "max_epochs"
    for epoch in range(1, config["epochs"] + 1):
        metrics, _, _ = train_epoch(model, batches["train"], optimizer, config, device)
        valid, _ = evaluate(model, batches["valid"], device, "valid", "main mixed valid")
        history.append({"epoch": epoch, **{"train_" + k: v for k, v in metrics.items()},
                        "valid_MAE": valid["MAE"], "valid_Pearson": valid["Pearson"]})
        if best_epoch == 0 or valid["MAE"] <= best - config["early_stop_min_delta"]:
            best, best_epoch = valid["MAE"], epoch
            save_checkpoint(output / "best.pt", model, config, epoch, valid, store, initial_hash)
        save_csv(output / "details/history.csv", history)
        LOGGER.info("Epoch=%d; single-view train MAE=%.6f; valid MAE=%.6f; best=%d",
                    epoch, metrics["pred_missing"], valid["MAE"], best_epoch)
        if epoch - best_epoch >= config["early_stop_patience"]:
            stop_reason = "early_stopping"
            break
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    checkpoint = load_checkpoint(output / "best.pt", store)
    model = build_model(config, ROOT)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    del checkpoint
    valid, records = evaluate(model, batches["valid"], device, "valid", "best main mixed valid")
    if not math.isclose(valid["MAE"], best, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError("Reloaded checkpoint changed validation MAE")
    binding = experiment_binding(store, config, best_epoch)
    calibrate(output, records, sha256_file(output / "best.pt"), binding)
    report = read_json(output / "evaluation.json")
    report["training_summary"] = {"method": METHOD, "prediction_views_per_sample": 1,
        "max_epochs": config["epochs"], "epochs_completed": len(history), "best_epoch": best_epoch,
        "best_valid_MAE": best, "early_stop_patience": config["early_stop_patience"],
        "early_stop_min_delta": config["early_stop_min_delta"], "stop_reason": stop_reason,
        "selection_split": "valid", "selection_family": "main_mixed", "selection_metric": "MAE",
        "initial_backbone_sha256": initial_hash, "test_run": False}
    write_json(output / "evaluation.json", report)
    print(f"{METHOD}: best epoch {best_epoch}; valid {report['summary']['valid']}", flush=True)
    print(f"Results: {output / 'evaluation.json'}", flush=True)


def test(output, device_override=None):
    output = Path(output)
    raw = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    if raw.get("method") != METHOD:
        raise ValueError("Use run.py evaluate --version missing-only only with an EMT_MISSING_ONLY run")
    config = dict(raw["config"])
    del raw
    store = MixDataStore(path_in_project(config["artifact_path"]), path_in_project(config["data_path"]))
    checkpoint = load_checkpoint(output / "best.pt", store)
    runtime = {**config, "device": device_override or config["device"]}
    device = configure(runtime, output)
    binding = experiment_binding(store, config, checkpoint["epoch"], include_test=True)
    checkpoint_hash = sha256_file(output / "best.pt")
    rule = load_frozen_rule(output, checkpoint_hash, binding)
    model = build_model(config, ROOT)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    del checkpoint
    _, records = evaluate(model, make_loader(store.dataset("test"), runtime), device, "test", "main mixed test")
    save_evaluation(output, records, "test", "main_mixed", "mixed_v1_s0", checkpoint_hash, binding, rule=rule)
    for split in ("valid", "test"):
        panels = store.panels(split)
        conditions = list(dict.fromkeys(c for p in panels.values() for c in p["condition_ids"]))
        for condition in conditions:
            dataset = store.prediction_dataset(split, condition)
            _, records = evaluate(model, make_loader(dataset, runtime), device, split, f"{split} paired {condition}")
            save_evaluation(output, records, split, "paired", condition, checkpoint_hash, binding, rule=rule)
            for name, panel in panels.items():
                if condition in panel["condition_ids"]:
                    save_evaluation(output, records, split, "paired", condition, checkpoint_hash, binding,
                                    panel={"name": name, **panel}, rule=rule)
    report = read_json(output / "evaluation.json")
    report["training_summary"]["test_run"] = True
    write_json(output / "evaluation.json", report)
    print(f"{METHOD}: test {report['summary']['test']}", flush=True)
    print(f"Results: {output / 'evaluation.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Single-view EMT on existing mix_v1; no complete reference or DLFR")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--run-name", required=True)
    fit.add_argument("--config", type=Path)
    for flag in ("epochs", "batch-size", "eval-batch-size", "gradient-accumulation-steps", "seed", "early-stop-patience"):
        fit.add_argument("--" + flag, type=int)
    fit.add_argument("--device", choices=("cpu", "cuda"))
    check = sub.add_parser("test")
    check.add_argument("--run-name", required=True)
    check.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_name):
        parser.error("run-name must be a simple name, not a path")
    output = ROOT / "outputs" / args.run_name
    if args.command == "test":
        test(output, args.device)
        return
    config = default_config()
    if args.config:
        supplied = read_json(args.config)
        config.update(supplied.get("training", supplied))
    for key in ("epochs", "batch_size", "eval_batch_size", "gradient_accumulation_steps", "seed", "early_stop_patience", "device"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    train(validate_config(config), output)


if __name__ == "__main__":
    main()
