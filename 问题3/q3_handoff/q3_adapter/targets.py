"""Fixed-target classification and regression perturbation scores.

Class IDs are 0=negative, 1=neutral, 2=positive. For each original sample fix
k*=argmax(logits(X)) once, then keep k* for every perturbed input. The class
margin is logits[k*]-max(logits[j], j!=k*). A signed delta is ORIGINAL minus
MASKED. Softmax is uncalibrated; it must not be called calibrated confidence.
These scores quantify masking sensitivity, not identified causal effects.
"""
from __future__ import annotations

from .runtime import configure_runtime

configure_runtime()

from typing import Mapping

import torch


def _prediction(prediction: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    logits = prediction.get('logits')
    intensity = prediction.get('intensity')
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError("prediction['logits'] must be [B,3]")
    if not logits.is_floating_point() or logits.shape[0] < 1:
        raise ValueError('Logits must be floating point with a nonempty batch')
    if not isinstance(intensity, torch.Tensor) or intensity.shape != (logits.shape[0],):
        raise ValueError("prediction['intensity'] must be [B]")
    if not intensity.is_floating_point() or intensity.device != logits.device:
        raise ValueError('Intensity must be floating point on the logits device')
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(intensity).all()):
        raise ValueError('Prediction contains NaN or Inf')
    return logits, intensity


def fixed_target_class(original_prediction: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Create independent [B] integer targets from the ORIGINAL predictions."""
    logits, _ = _prediction(original_prediction)
    return logits.argmax(dim=-1).detach().clone()


def score_prediction(prediction: Mapping[str, torch.Tensor], target_class: torch.Tensor) -> dict:
    """Score supplied fixed targets without recomputing a predicted target class.

    Returned tensors preserve the computation graph when the caller enabled
    gradients. No model parameters are changed and no autograd mode is altered.
    """
    logits, intensity = _prediction(prediction)
    if not isinstance(target_class, torch.Tensor) or target_class.shape != (logits.shape[0],):
        raise ValueError('target_class must be an integer tensor [B]')
    if target_class.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError('target_class must have integer dtype')
    targets = target_class.to(device=logits.device, dtype=torch.long)
    if not bool(((targets >= 0) & (targets < 3)).all()):
        raise ValueError('target_class must contain only 0, 1, or 2')
    selected = targets[:, None]
    class_logit = logits.gather(1, selected).squeeze(1)
    probability = logits.softmax(dim=-1).gather(1, selected).squeeze(1)
    other_logits = logits.masked_fill(
        torch.arange(3, device=logits.device)[None] == selected, -torch.inf)
    margin = class_logit - other_logits.max(dim=-1).values
    return {'class_logit': class_logit, 'class_probability': probability,
            'class_margin': margin, 'intensity': intensity}


def compare_predictions(original_prediction: Mapping[str, torch.Tensor],
                        masked_prediction: Mapping[str, torch.Tensor],
                        target_class: torch.Tensor | None = None) -> dict:
    """Compare original and masked scores for original k*, never masked argmax.

    ``target_class`` may reuse a previously frozen target but must equal the
    argmax of the supplied original logits. Returns original/masked scores and
    signed/absolute deltas for logit, probability, margin and intensity. The
    convenience fields delta_cls and delta_reg are margin and intensity deltas.
    """
    original_logits, _ = _prediction(original_prediction)
    masked_logits, _ = _prediction(masked_prediction)
    if original_logits.shape != masked_logits.shape or original_logits.device != masked_logits.device:
        raise ValueError('Original and masked logits must have matching shape and device')
    original_targets = fixed_target_class(original_prediction)
    targets = original_targets if target_class is None else target_class
    original = score_prediction(original_prediction, targets)
    if not torch.equal(targets.to(device=original_targets.device, dtype=torch.long), original_targets):
        raise ValueError('target_class must remain the original argmax for this sample batch')
    masked = score_prediction(masked_prediction, targets)
    signed = {key: original[key] - masked[key] for key in original}
    absolute = {key: value.abs() for key, value in signed.items()}
    return {'target_class': targets.detach().clone(), 'original': original, 'masked': masked,
            'signed_delta': signed, 'absolute_delta': absolute,
            'delta_cls': signed['class_margin'], 'abs_delta_cls': absolute['class_margin'],
            'delta_reg': signed['intensity'], 'abs_delta_reg': absolute['intensity']}
