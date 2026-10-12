"""CE plus L1 per view, with the unchanged M2 auxiliary objectives."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch.nn import functional as F

from robust.losses import _graph_zero, _weight, compute_losses as _legacy_losses
from multitask.model import method_name


def labels_to_classes(labels: torch.Tensor) -> torch.Tensor:
    """Exact original-label mapping: negative=0, neutral=1, positive=2."""
    if not bool(torch.isfinite(labels).all()):
        raise ValueError("Intensity labels must be finite")
    return (labels > 0).long() + (labels >= 0).long()


def compute_losses(outputs: Mapping[str, torch.Tensor], batch: Mapping[str, Any],
                   config: Mapping[str, Any], device) -> dict[str, torch.Tensor]:
    """M1/M2 sum two equally weighted joint views; never average their losses.

    ``ce`` and ``l1`` are unweighted view sums. M2 reuses the historical masked
    losses, discarding its regression total so L1 is counted exactly once.
    """
    method = method_name(config)
    labels = batch["labels"].to(device=device, dtype=torch.float32, non_blocking=True).reshape(-1)
    classes = labels_to_classes(labels)
    if labels.numel() == 0:
        raise ValueError("Loss requires a nonempty batch")
    weights = {name: _weight(config, name, default) for name, default in (
        ("classification_weight", 1.0), ("regression_weight", 0.5),
        ("consistency_weight", 1.0), ("reconstruction_weight", 1.0))}
    for name in ("pred_missing_weight", "pred_reference_weight",
                 "prediction_missing_weight", "prediction_reference_weight"):
        if name in config and _weight(config, name, 1.0) != 1.0:
            raise ValueError("Multitask v1 requires equal unit weights on both views")
    result = {}
    views = ("reference",) if method == "M0" else ("reference", "missing")
    for view in views:
        logits = outputs["logits_" + view]
        intensity = outputs["intensity_" + view].reshape(-1)
        if logits.shape != (labels.numel(), 3) or intensity.shape != labels.shape:
            raise ValueError("Logits, intensity and label shapes disagree")
        result["ce_" + view] = F.cross_entropy(logits, classes)
        result["l1_" + view] = F.l1_loss(intensity, labels)
    result["ce"] = sum(result["ce_" + view] for view in views)
    result["l1"] = sum(result["l1_" + view] for view in views)
    zero = _graph_zero(outputs["intensity_reference"])
    auxiliary_names = ("consistency_gmc", "consistency_text", "consistency_audio", "consistency_video",
                       "reconstruction_text", "reconstruction_audio", "reconstruction_vision",
                       "consistency", "reconstruction")
    result.update({name: zero for name in auxiliary_names})
    if method == "M2":
        reference = {}
        for modality in ("audio", "vision"):
            target = outputs[modality + "_for_recon"]
            if target.shape != batch["reference"][modality].shape:
                raise ValueError("Standardized clean reconstruction target has an invalid shape")
            reference[modality] = target.detach()
        auxiliary_outputs = {**outputs, "pred_reference": outputs["intensity_reference"],
                             "pred_missing": outputs["intensity_missing"]}
        auxiliary = _legacy_losses(auxiliary_outputs, {**batch, "reference": reference}, config, device)
        result.update({name: auxiliary[name] for name in auxiliary_names})
    result["total"] = (weights["classification_weight"] * result["ce"]
                       + weights["regression_weight"] * result["l1"]
                       + weights["consistency_weight"] * result["consistency"]
                       + weights["reconstruction_weight"] * result["reconstruction"])
    return result
