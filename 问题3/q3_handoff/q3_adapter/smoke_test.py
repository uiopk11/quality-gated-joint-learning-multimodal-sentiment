"""Validate the handoff copies only; never train or write formal Q2 results.

Run from the handoff with: python -B -m q3_adapter.smoke_test --data <copied pkl>
All output is restricted to q3_adapter. The supplied data must be inside this
handoff, and is not intended to remain in the distributed archive.
"""
from __future__ import annotations

from .runtime import configure_runtime

HANDOFF_ROOT = configure_runtime()

import argparse
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import torch

from multitask.model import predict_batch
from .data import load_complete_batch
from .hooks import RepresentationHooks, count_registered_hooks
from .masking import mask_text_span, mask_audio_span, mask_vision_span, mask_modality
from .model_adapter import load_frozen_m0
from .targets import compare_predictions, fixed_target_class, score_prediction


def _require(value, message):
    if not value:
        raise AssertionError(message)


def _same(left, right):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.shape == right.shape
                and left.dtype == right.dtype and left.device == right.device
                and torch.equal(left, right))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def _shapes(value):
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, dict):
        return {key: _shapes(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_shapes(item) for item in value]
    return value


def _json_tensors(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _json_tensors(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_tensors(item) for item in value]
    return value


def _comparison(expected, actual, atol):
    _require(_finite(actual), "Prediction or representation contains NaN/Inf")
    result = {}
    for key in ("logits", "intensity"):
        _require(expected[key].shape == actual[key].shape, key + " shape differs")
        error = float((expected[key] - actual[key]).abs().max().item())
        result[key + "_max_absolute_error"] = error
        _require(error <= atol, key + " exceeds absolute tolerance " + str(atol))
    reference_class = expected["logits"].argmax(dim=-1)
    actual_class = actual["logits"].argmax(dim=-1)
    result["pred_class_exact_match"] = bool(torch.equal(reference_class, actual_class))
    _require(result["pred_class_exact_match"], "Predicted classes differ")
    if "pred_class" in actual:
        _require(torch.equal(actual["pred_class"], actual_class), "Adapter pred_class differs from logits argmax")
    if "probabilities" in actual:
        _require(torch.allclose(actual["probabilities"], actual["logits"].softmax(-1), atol=atol, rtol=0),
                 "Adapter probabilities differ from softmax")
    return result


def _mask_contract(original, masked, modality, *, start=0, end=48, unk_id=100):
    _require(masked is not original, "Mask returned original dict")
    _require(masked.keys() == original.keys(), "Mask changed batch field names")
    for key, value in original.items():
        if isinstance(value, torch.Tensor):
            updated = masked[key]
            _require(value.shape == updated.shape and value.dtype == updated.dtype and value.device == updated.device,
                     "Mask changed shape/dtype/device: " + key)
        if key != modality:
            _require(_same(value, masked[key]), "Mask changed unrelated field or metadata: " + key)
    expected = original[modality].clone()
    for row, length in enumerate(original["audio_lengths"].detach().cpu().tolist()):
        stop = min(end, int(length))
        if start >= stop:
            continue
        if modality == "text":
            expected[row, 0, start + 1:stop + 1] = unk_id
        else:
            expected[row, start:stop, :] = 0
    _require(torch.equal(expected, masked[modality]), "Mask did not follow exact body-span protocol: " + modality)
    _require(_finite(masked), "Masked input contains NaN/Inf")
    return {"shape_dtype_device_preserved": True, "lengths_preserved": True,
            "padding_and_special_tokens_preserved": True, "metadata_preserved": True,
            "exact_expected_mask": True, "input_finite": True}


def _target_contract(original, masked, target, atol):
    comparison = compare_predictions(original, masked, target_class=target)
    _require(torch.equal(comparison["target_class"], target), "Target class changed after masking")
    for name, prediction in (("original", original), ("masked", masked)):
        logits = prediction["logits"]
        selected = logits.gather(1, target[:, None]).squeeze(1)
        others = logits.clone().scatter(1, target[:, None], -torch.inf)
        direct = {"class_logit": selected,
                  "class_probability": prediction["probabilities"].gather(1, target[:, None]).squeeze(1),
                  "class_margin": selected - others.max(dim=-1).values,
                  "intensity": prediction["intensity"]}
        scores = score_prediction(prediction, target)
        for key in direct:
            _require(torch.allclose(scores[key], direct[key], atol=atol, rtol=0), "Target score mismatch: " + key)
            _require(torch.allclose(comparison[name][key], direct[key], atol=atol, rtol=0),
                     "Comparison target mismatch: " + key)
    for key in comparison["original"]:
        delta = comparison["original"][key] - comparison["masked"][key]
        _require(torch.allclose(comparison["signed_delta"][key], delta, atol=atol, rtol=0), "Wrong signed delta")
        _require(torch.allclose(comparison["absolute_delta"][key], delta.abs(), atol=atol, rtol=0), "Wrong absolute delta")
    _require(torch.equal(comparison["delta_cls"], comparison["signed_delta"]["class_margin"]), "Wrong delta_cls")
    _require(torch.equal(comparison["delta_reg"], comparison["signed_delta"]["intensity"]), "Wrong delta_reg")
    return comparison


def _import_audit():
    source = (HANDOFF_ROOT / "original_source").resolve()
    preprocessing = json.loads((HANDOFF_ROOT / "checkpoint/preprocessing.json").read_text(encoding="utf-8"))
    # This is provenance metadata only. No path under the original root is opened.
    original_root = Path(preprocessing["source"]["path"]).parent.parent.resolve()
    project_prefixes = ("baseline", "robust", "standardized", "dataset_construction", "multitask", "models")
    project_imports, forbidden_imports = {}, {}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.is_relative_to(original_root):
            forbidden_imports[name] = str(path)
        if name.split(".", 1)[0] in project_prefixes:
            project_imports[name] = str(path)
            _require(path.is_relative_to(source), "Project import escaped original_source copy: " + name)
    _require(not forbidden_imports, "Imported code/dependencies from original project: " + str(forbidden_imports))
    _require(not Path(sys.executable).resolve().is_relative_to(original_root), "Python executable is inside original project")
    dependency_imports = {name: str(Path(sys.modules[name].__file__).resolve())
                          for name in ("torch", "numpy", "transformers", "tokenizers", "safetensors")
                          if name in sys.modules and getattr(sys.modules[name], "__file__", None)}
    return {"project_modules": project_imports, "dependency_modules": dependency_imports,
            "original_project_root_from_metadata": str(original_root), "forbidden_imports": forbidden_imports,
            "all_project_modules_from_handoff": True}


def _environment(device):
    versions = {}
    for name in ("torch", "transformers", "numpy", "tokenizers", "safetensors", "scipy", "scikit-learn", "einops"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    cache_names = ("TMP", "TEMP", "TMPDIR", "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE",
                   "XDG_CACHE_HOME", "TORCH_HOME", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR",
                   "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "MPLCONFIGDIR", "PYTHONPYCACHEPREFIX", "NUMBA_CACHE_DIR")
    cache_paths = {name: os.environ.get(name) for name in cache_names}
    _require(all(value and Path(value).resolve().is_relative_to(HANDOFF_ROOT) for value in cache_paths.values()),
             "Runtime writable state is not fully contained in handoff")
    return {"python": platform.python_version(), "python_executable": sys.executable,
            "platform": platform.platform(), "package_versions": versions,
            "torch_cuda_build": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(), "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else platform.processor(),
            "torch_num_threads": torch.get_num_threads(), "runtime_paths": cache_paths,
            "working_directory": str(Path.cwd()), "bytecode_disabled": sys.dont_write_bytecode,
            "huggingface_offline": os.environ.get("HF_HUB_OFFLINE") == "1"}


def run_smoke(data_path, *, device="cpu", atol=1e-6):
    torch.set_num_threads(4)
    start_time = time.perf_counter()
    report = {"passed": False, "run": "mt_m0_avstd_s1111_v1", "started_utc": datetime.now(timezone.utc).isoformat(),
              "absolute_tolerance": atol, "relative_tolerance": 0, "environment": _environment(device)}
    batch = load_complete_batch(data_path, device=device, require_inside_handoff=True)
    original = copy.deepcopy(batch)
    _require(len(set(batch["audio_lengths"].tolist())) >= 2, "Smoke requires two distinct body lengths")
    report["data"] = batch["data_metadata"]
    report["samples"] = [{"id": sample_id, "index": int(batch["index"][i]),
                           "split": batch["split"], "body_length": int(batch["audio_lengths"][i])}
                          for i, sample_id in enumerate(batch["id"])]
    report["input_tensors"] = {key: _shapes(value) for key, value in batch.items() if isinstance(value, torch.Tensor)}
    adapter = load_frozen_m0(device=device)
    _require(adapter.model.av_normalizer.metadata()["source"]["sha256"] == batch["data_metadata"]["sha256"],
             "Loaded checkpoint data binding differs from dataset")
    _require(all(not p.requires_grad for p in adapter.model.parameters()), "Parameters are not all frozen")
    _require(all(not module.training for module in adapter.model.modules()), "Model has non-eval modules")
    report["formal_loader"] = {"entry": "multitask.checkpoint.load_for_prediction", "allow_smoke": False,
                               "success": True, "checkpoint": str(adapter.checkpoint_path),
                               "method": adapter.model.method, "all_parameters_frozen": True, "all_modules_eval": True,
                               "class_order": list(adapter.class_order), "data_binding_matches": True}
    _require(tuple(adapter.class_order) == ("negative", "neutral", "positive"), "Wrong class order")
    hook_count = count_registered_hooks(adapter.model)
    with torch.no_grad():
        standard = predict_batch(adapter.model, batch, adapter.device)
    prediction = adapter.predict(batch)
    report["standard_vs_adapter"] = _comparison(standard, prediction, atol)
    report["original_prediction"] = _json_tensors(prediction)
    representation = adapter.extract_representations(batch)
    report["standard_vs_extraction"] = _comparison(standard, representation, atol)
    report["hook_metadata"] = representation["hook_metadata"]
    report["representation_shapes"] = {key: _shapes(value) for key, value in representation.items()
                                        if isinstance(value, torch.Tensor) or key in ("local_masks", "lengths")}
    _require(representation["shared_feature"].shape == (len(batch["id"]), 128), "Shared feature is not [B,128]")
    _require(count_registered_hooks(adapter.model) == hook_count, "Extraction left registered hooks")
    _comparison(standard, adapter.extract_representations(batch), atol)
    _require(count_registered_hooks(adapter.model) == hook_count, "Repeated extraction accumulated hooks")
    try:
        with RepresentationHooks(adapter.model):
            raise RuntimeError("intentional cleanup check")
    except RuntimeError as error:
        _require(str(error) == "intentional cleanup check", "Unexpected hook setup error")
    _require(count_registered_hooks(adapter.model) == hook_count, "Error path left registered hooks")
    report["standard_vs_after_hook_removal"] = _comparison(standard, adapter.predict(batch), atol)
    report["hooks"] = {"before": hook_count, "after_repeated_extraction": count_registered_hooks(adapter.model),
                       "exception_cleanup_passed": True, "no_duplicate_or_residual_hooks": True}
    fixed_class = fixed_target_class(prediction)
    report["masking"] = {}
    unk_id = int(adapter.tokenizer.unk_token_id)
    for modality, function in (("text", mask_text_span), ("audio", mask_audio_span), ("vision", mask_vision_span)):
        for mode in ("span", "whole_modality"):
            end = 2 if mode == "span" else 48
            masked = function(batch, 0, end) if mode == "span" else mask_modality(batch, modality)
            case = _mask_contract(batch, masked, modality, end=end, unk_id=unk_id)
            _require(_same(batch, original), "Mask mutated original batch")
            case["original_batch_unchanged"] = True
            masked_prediction = adapter.predict(masked)
            _require(_finite(masked_prediction), "Masked forward contains NaN/Inf")
            _require(masked_prediction["logits"].shape == prediction["logits"].shape, "Masked logits shape differs")
            _require(masked_prediction["intensity"].shape == prediction["intensity"].shape, "Masked intensity shape differs")
            case["forward_finite"] = True
            case["fixed_target_comparison"] = _json_tensors(_target_contract(prediction, masked_prediction, fixed_class, atol))
            report["masking"][modality + "_" + mode] = case
        _require(_same(function(batch, 0, 48), mask_modality(batch, modality)), "Full span differs from modality mask")
    if torch.device(device).type == "cpu":
        gradient_representation = adapter.extract_representations(batch, enable_grad=True)
        gradient_report = _comparison(standard, gradient_representation, atol)
        leaves = gradient_representation["gradient_inputs"]
        _require(set(leaves) == {"text_embedding_output", "audio_standardized", "vision_standardized"},
                 "Gradient extraction did not expose three expected continuous leaves")
        scores = score_prediction(gradient_representation, fixed_class)
        derivatives = torch.autograd.grad(scores["class_margin"].sum() + scores["intensity"].sum(), tuple(leaves.values()))
        _require(all(torch.isfinite(value).all() for value in derivatives), "Continuous derivatives are nonfinite")
        _require(all(p.grad is None and not p.requires_grad for p in adapter.model.parameters()), "Frozen parameter acquired gradient")
        _require(count_registered_hooks(adapter.model) == hook_count, "Gradient extraction left hooks")
        gradient_report.update(passed=True, target="sum of fixed-class margins plus intensities",
                               leaves={name: {**_shapes(leaf), "gradient_abs_sum": float(derivative.abs().sum())}
                                       for (name, leaf), derivative in zip(leaves.items(), derivatives)},
                               all_derivatives_finite=True, frozen_parameters_without_grad=True)
        report["continuous_gradient"] = gradient_report
    else:
        report["continuous_gradient"] = {"passed": None, "skipped": True,
                                          "reason": "CUDA cuDNN LSTM backward requires training mode; preserve M0 eval mode. Run this check on CPU."}
    _require(_same(batch, original), "Validation changed the original batch")
    report["final_standard_consistency"] = _comparison(standard, adapter.predict(batch), atol)
    report["import_audit"] = _import_audit()
    report["passed"] = True
    report["duration_seconds"] = time.perf_counter() - start_time
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    report["scope"] = "Two real valid samples; no retraining, full evaluation, CSV export, or writes to original project."
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(HANDOFF_ROOT / "q3_adapter/_runtime/input/aligned_50.pkl"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--report", default=str(HANDOFF_ROOT / "q3_adapter/audit/smoke_test.json"))
    args = parser.parse_args()
    report_path = Path(args.report).expanduser().resolve()
    if not report_path.is_relative_to((HANDOFF_ROOT / "q3_adapter").resolve()):
        raise ValueError("Smoke report must be inside q3_handoff/q3_adapter")
    try:
        if not 0 <= args.atol <= 1e-5:
            raise ValueError("atol must be between 0 and 1e-5")
        report = run_smoke(args.data, device=args.device, atol=args.atol)
    except Exception as error:
        report = {"passed": False, "error": repr(error), "traceback": traceback.format_exc(),
                  "device": args.device, "completed_utc": datetime.now(timezone.utc).isoformat()}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "report": str(report_path),
                      "standard_vs_adapter": report.get("standard_vs_adapter"), "error": report.get("error")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
