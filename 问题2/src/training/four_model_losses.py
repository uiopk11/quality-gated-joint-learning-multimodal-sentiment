"""L1 baseline or CE + 0.5 L1 joint supervision, with full M2/M3 DLFR."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch.nn import functional as F

from src.models.four_models import is_joint, is_missing, method_name
from src.training.dlfr_losses import _graph_zero, _weight, compute_losses as _dlfr_losses
from src.training.multitask_losses import labels_to_classes


def compute_losses(outputs: Mapping[str, torch.Tensor], batch: Mapping[str, Any],
                   config: Mapping[str, Any], device) -> dict[str, torch.Tensor]:
    """Sum equally weighted views; reuse every masked DLFR auxiliary once."""
    method = method_name(config)
    joint, missing = is_joint(method), is_missing(method)
    labels = batch["labels"].to(device=device, dtype=torch.float32, non_blocking=True).reshape(-1)
    classes = labels_to_classes(labels)
    if labels.numel() == 0:
        raise ValueError("Loss requires a nonempty batch")
    expected_weights = {"classification_weight": 1.0 if joint else 0.0,
                        "regression_weight": 0.5 if joint else 1.0,
                        "consistency_weight": 1.0, "reconstruction_weight": 1.0,
                        "prediction_reference_weight": 1.0, "prediction_missing_weight": 1.0}
    for name, expected in expected_weights.items():
        if _weight(config, name, expected) != expected:
            raise ValueError(f"Four-model v1 fixes {name}={expected} for {method}")
    for name in ("pred_missing_weight", "pred_reference_weight"):
        if name in config and _weight(config, name, 1.0) != 1.0:
            raise ValueError("Four-model v1 requires equal unit weights on both views")
    zero = _graph_zero(outputs["intensity_reference"])
    result = {}
    views = ("reference", "missing") if missing else ("reference",)
    for view in views:
        intensity = outputs["intensity_" + view].reshape(-1)
        if intensity.shape != labels.shape:
            raise ValueError("Intensity and label shapes disagree")
        result["l1_" + view] = F.l1_loss(intensity, labels)
        result["ce_" + view] = zero
        if joint:
            logits = outputs["logits_" + view]
            if logits.shape != (labels.numel(), 3):
                raise ValueError("Logits and label shapes disagree")
            result["ce_" + view] = F.cross_entropy(logits, classes)
    result["ce"] = sum(result["ce_" + view] for view in views)
    result["l1"] = sum(result["l1_" + view] for view in views)
    auxiliary_names = ("consistency_gmc", "consistency_text", "consistency_audio", "consistency_video",
                       "reconstruction_text", "reconstruction_audio", "reconstruction_vision",
                       "consistency", "reconstruction")
    result.update({name: zero for name in auxiliary_names})
    if missing:
        reference = {}
        for modality in ("audio", "vision"):
            target = outputs[modality + "_for_recon"]
            if target.shape != batch["reference"][modality].shape:
                raise ValueError("Standardized clean reconstruction target has an invalid shape")
            reference[modality] = target.detach()
        auxiliary_outputs = {**outputs, "pred_reference": outputs["intensity_reference"],
                             "pred_missing": outputs["intensity_missing"]}
        # The reusable historic kernel names its full-DLFR branch M2; that
        # internal implementation detail does not reinterpret any old run.
        auxiliary = _dlfr_losses(auxiliary_outputs, {**batch, "reference": reference},
                                 {**config, "method": "M2"}, device)
        result.update({name: auxiliary[name] for name in auxiliary_names})
    result["total"] = (expected_weights["classification_weight"] * result["ce"]
                       + expected_weights["regression_weight"] * result["l1"]
                       + result["consistency"] + result["reconstruction"])
    return result
