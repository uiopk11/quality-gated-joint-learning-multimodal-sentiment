"""Fair two-view prediction losses and explicitly masked DLFR auxiliaries."""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch.nn import functional as F


def _graph_zero(value: torch.Tensor) -> torch.Tensor:
    # Empty reduction is exactly zero with a valid gradient, even if excluded
    # values happen to be nonfinite. No multiplication of an unknown value by 0.
    return value.reshape(-1)[:0].sum()


def masked_reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor,
                               support: torch.Tensor, kind: str = "SmoothL1Loss") -> torch.Tensor:
    """Mean over selected positions × all feature dimensions; targets stop grad.

    support has already excluded original all-zero rows/native UNK. A numerical
    zero component in an otherwise observed feature row remains a valid target.
    """
    if prediction.ndim != 3 or prediction.shape != target.shape or support.shape != prediction.shape[:2]:
        raise ValueError("Reconstruction prediction/target/support shapes disagree")
    if support.dtype != torch.bool:
        raise ValueError("Reconstruction support must be boolean")
    if kind not in ("SmoothL1Loss", "L1Loss", "MSELoss"):
        raise ValueError(f"Unknown reconstruction loss: {kind}")
    if not bool(support.any()):
        return _graph_zero(prediction)
    selected_prediction = prediction[support]
    selected_target = target.detach()[support]
    criterion = {"SmoothL1Loss": F.smooth_l1_loss, "L1Loss": F.l1_loss, "MSELoss": F.mse_loss}[kind]
    # reduction='mean' is exactly sum/(selected positions * feature dimension).
    return criterion(selected_prediction, selected_target, reduction="mean")


def symmetric_negative_cosine(p_missing: torch.Tensor, z_reference: torch.Tensor,
                              p_reference: torch.Tensor, z_missing: torch.Tensor) -> torch.Tensor:
    """Author SimSiam objective, with explicit stop-gradient on both targets."""
    shapes = [value.shape for value in (p_missing, z_reference, p_reference, z_missing)]
    if any(shape != shapes[0] for shape in shapes) or len(shapes[0]) != 2:
        raise ValueError("Consistency tensors must share a [batch, feature] shape")
    return -0.5 * (F.cosine_similarity(p_missing, z_reference.detach(), dim=1).mean()
                   + F.cosine_similarity(p_reference, z_missing.detach(), dim=1).mean())


def _method(config: Mapping[str, Any]) -> str:
    method = str(config.get("method", config.get("model_variant", ""))).upper()
    if method not in ("M1", "M2"):
        raise ValueError("Robust method must be M1 or M2")
    return method


def _weight(config: Mapping[str, Any], name: str, default: float) -> float:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid loss weight: {name}")
    return float(value)


def compute_losses(outputs: Mapping[str, torch.Tensor], batch: Mapping[str, Any],
                   config: Mapping[str, Any], device: torch.device | str) -> dict[str, torch.Tensor]:
    """Both methods use L1_missing + L1_reference, each with fixed weight one.

    M1 never reads auxiliary outputs or supervision. M2 adds four symmetric
    negative cosine objectives and three modality reconstruction objectives.
    Selection must use damaged-view validation MAE, never this auxiliary total.
    """
    method = _method(config)
    labels = batch["labels"].to(device=device, dtype=torch.float32, non_blocking=True).reshape(-1)
    missing = outputs["pred_missing"].reshape(-1)
    reference = outputs["pred_reference"].reshape(-1)
    if labels.shape != missing.shape or labels.shape != reference.shape:
        raise ValueError("Prediction and label counts disagree")
    # Reject a misleading config that claims an unequal prediction comparison.
    for key in ("pred_missing_weight", "pred_reference_weight", "prediction_missing_weight", "prediction_reference_weight"):
        if key in config and _weight(config, key, 1.0) != 1.0:
            raise ValueError("M1 and M2 require unit weights on both prediction losses")
    result = {"pred_missing": F.l1_loss(missing, labels),
              "pred_reference": F.l1_loss(reference, labels)}
    zero = _graph_zero(missing)
    consistency_names = ("gmc", "text", "audio", "video")
    reconstruction_names = ("text", "audio", "vision")
    if method == "M1":
        result.update({f"consistency_{name}": zero for name in consistency_names})
        result.update({f"reconstruction_{name}": zero for name in reconstruction_names})
        result.update(consistency=zero, reconstruction=zero,
                      total=result["pred_missing"] + result["pred_reference"])
        return result
    for name in consistency_names:
        prefix = "gmc_tokens" if name == "gmc" else name
        result[f"consistency_{name}"] = symmetric_negative_cosine(
            outputs[f"p_{prefix}_m"], outputs[f"z_{prefix}"],
            outputs[f"p_{prefix}"], outputs[f"z_{prefix}_m"])
    kind = config.get("reconstruction_loss", config.get("recon_loss", "SmoothL1Loss"))
    for name in reconstruction_names:
        prediction = outputs[{"text": "text_recon", "audio": "audio_recon", "vision": "video_recon"}[name]]
        support = batch["reconstruction"][name].to(device=device, non_blocking=True)
        if support.ndim != 2 or support.shape[0] != prediction.shape[0] or support.shape[1] < prediction.shape[1]:
            raise ValueError(f"Invalid reconstruction support dimensions: {name}")
        # Packed A/V LSTM output ends at the current batch's maximum L; text has
        # the 49 post-CLS BERT positions. Reject truncating any true support.
        if bool(support[:, prediction.shape[1]:].any()):
            raise ValueError(f"Reconstruction output would omit supported positions: {name}")
        support = support[:, :prediction.shape[1]]
        if name == "text":
            target = outputs["text_for_recon"]
        else:
            target = batch["reference"][name].to(device=device, dtype=prediction.dtype, non_blocking=True)
            target = target[:, :prediction.shape[1]]
        result[f"reconstruction_{name}"] = masked_reconstruction_loss(prediction, target, support, kind)
    result["consistency"] = sum(result[f"consistency_{name}"] for name in consistency_names)
    result["reconstruction"] = sum(result[f"reconstruction_{name}"] for name in reconstruction_names)
    result["total"] = (result["pred_missing"] + result["pred_reference"]
                       + _weight(config, "consistency_weight", 1.0) * result["consistency"]
                       + _weight(config, "reconstruction_weight", 1.0) * result["reconstruction"])
    return result
