"""Fixed preprocessing around unchanged EMT prediction and DLFR objectives.

The train-fitted normalizer has buffers only. Original parameter names are kept
so the existing Adam parameter groups and initial random stream are unchanged.
Prediction reads only the current input; clean references are training-only.
"""

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

import baseline.model as baseline_model
import robust.model as robust_model
from robust.losses import compute_losses as legacy_compute_losses
from standardized.normalization import AVNormalizer, validate_preprocessing
from standardized.provenance import CODE_VERSION, implementation_provenance, validate_implementation


def _attach(model, preprocessing, method):
    model.av_normalizer = AVNormalizer(preprocessing)
    model.standardized_method = method
    return model


def build_baseline_model(config, root, preprocessing):
    validate_preprocessing(preprocessing)
    return _attach(baseline_model.build_model(config, root), preprocessing, "M0")


def build_robust_model(config, root, preprocessing):
    validate_preprocessing(preprocessing)
    return _attach(robust_model.build_model(config, root), preprocessing, robust_model.method_name(config))


def _input_view(model, batch: Mapping[str, Any], device):
    # Do not copy labels, references or simulator masks into a prediction view.
    # Observability is measured from these raw, already left-shifted inputs.
    result = {"text": batch["text"]}
    for modality in ("audio", "vision"):
        length_key = modality + "_lengths"
        lengths = batch[length_key].to(device=device, dtype=torch.long).reshape(-1)
        value = batch[modality].to(device=device, dtype=torch.float32)
        result[modality] = model.av_normalizer(value, lengths, modality)
        result[length_key] = lengths
    return result


def forward_batch(model: nn.Module, batch: Mapping[str, Any], device):
    """M0 training and inference, always with the fixed A/V preprocessing."""
    if model.standardized_method != "M0":
        raise ValueError("forward_batch requires the standardized M0 model")
    return baseline_model.forward_batch(model, _input_view(model, batch, device), device)


def forward_training(model: nn.Module, batch: Mapping[str, Any], device):
    if model.standardized_method not in ("M1", "M2") or "reference" not in batch:
        raise ValueError("Standardized M1/M2 training requires the clean reference view")
    damaged = _input_view(model, batch, device)
    original = _input_view(model, batch["reference"], device)
    outputs = robust_model.forward_training(model, {**damaged, "reference": original}, device)
    # These are exactly the clean tensors passed into the shared encoders.
    # Never derive reconstruction support from the damaged view's nonzero rows.
    outputs["audio_for_recon"] = original["audio"].detach()
    outputs["vision_for_recon"] = original["vision"].detach()
    return outputs


def compute_losses(outputs, batch, config, device):
    """Keep every loss formula, weight and support; change only A/V target units."""
    if robust_model.method_name(config) == "M1":
        return legacy_compute_losses(outputs, batch, config, device)
    reference = dict(batch["reference"])
    for modality in ("audio", "vision"):
        target = outputs[modality + "_for_recon"]
        if target.shape != batch["reference"][modality].shape:
            raise ValueError("Standardized clean reconstruction target has an invalid shape")
        reference[modality] = target
    return legacy_compute_losses(outputs, {**batch, "reference": reference}, config, device)


def predict_batch(model: nn.Module, batch: Mapping[str, Any], device):
    """Label-free M0/M1/M2 prediction from a raw aligned input batch."""
    if model.standardized_method == "M0":
        return forward_batch(model, batch, device)
    if model.standardized_method not in ("M1", "M2"):
        raise ValueError("Unknown standardized prediction model")
    return robust_model.predict_batch(model, _input_view(model, batch, device), device)


def validate_checkpoint_preprocessing(checkpoint):
    if checkpoint.get("code_version") != CODE_VERSION:
        raise ValueError("Not an av_standardized_v1 checkpoint; use the original runner for legacy models")
    source_hash = checkpoint.get("data_sha256")
    if source_hash is None:
        source_hash = checkpoint.get("data_binding", {}).get("data_sha256")
    validate_preprocessing(checkpoint.get("preprocessing"), source_sha256=source_hash)


def restore_model_state(model, checkpoint):
    """Restore only saved statistics; reject mismatched metadata and buffers."""
    validate_checkpoint_preprocessing(checkpoint)
    if model.av_normalizer.metadata() != checkpoint["preprocessing"]:
        raise ValueError("Model preprocessing metadata differs from the saved checkpoint")
    state = checkpoint["model_state_dict"]
    for name, expected in model.av_normalizer.state_dict().items():
        saved = state.get("av_normalizer." + name)
        if (not isinstance(saved, torch.Tensor) or saved.dtype != expected.dtype
                or not torch.equal(saved.detach().cpu(), expected.detach().cpu())):
            raise ValueError("Checkpoint preprocessing buffers differ from metadata: " + name)
    model.load_state_dict(state, strict=True)
    return model


def load_for_prediction(checkpoint_path, root, device="cpu"):
    """Restore a formal model without opening any train/valid/test input data.

    Feed raw, already aligned/left-shifted batches to predict_batch afterwards.
    Classification continues to use the run's saved frozen decision rule.
    """
    root = Path(root).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_checkpoint_preprocessing(checkpoint)
    if checkpoint.get("purpose") != "train" or checkpoint.get("selection_split") != "valid":
        raise ValueError("Final prediction requires a formal valid-selected checkpoint")
    validate_implementation(checkpoint.get("code_sha256"), implementation_provenance(root))
    method = checkpoint.get("method", "M0")
    if method not in ("M0", "M1", "M2") or checkpoint.get("selection_metric") != "MAE":
        raise ValueError("Unsupported standardized model or checkpoint selection rule")
    if method in ("M1", "M2") and (
        robust_model.method_name(checkpoint["config"]) != method
        or checkpoint.get("selection_family") != "main_mixed"
        or checkpoint.get("selection_condition") != "mixed_v1_s0"
    ):
        raise ValueError("M1/M2 checkpoint does not match the fixed selection protocol")
    builder = build_baseline_model if method == "M0" else build_robust_model
    model = builder(checkpoint["config"], root, checkpoint["preprocessing"])
    restore_model_state(model, checkpoint)
    return model.to(device).eval()
