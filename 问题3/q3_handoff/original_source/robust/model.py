"""New M1/M2 adapter; the frozen baseline and vendored source stay unchanged.

Both methods construct exactly the same complete parameter layout and consume
the same initialization random stream. M1 freezes and skips the auxiliary heads.
Prediction uses one damaged view and never evaluates an auxiliary head.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from baseline.model import OFFICIAL_MOSEI_DEFAULTS


SINGLETON_BATCHNORM_POLICY = (
    "Auxiliary BatchNorm1d uses running mean/variance for a training singleton; "
    "running statistics and num_batches_tracked are unchanged for that forward. "
    "Physical batches with at least two observations per channel use ordinary "
    "BatchNorm. Gradient accumulation does not change this decision."
)


def method_name(config: Mapping[str, Any]) -> str:
    method = str(config.get("method", config.get("model_variant", ""))).upper()
    if method not in ("M1", "M2"):
        raise ValueError("Robust method must be M1 or M2")
    return method


class SafeBatchNorm1d(nn.BatchNorm1d):
    """State-dict-compatible BatchNorm with an explicit singleton policy."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(value)
        observations = value.numel() // value.shape[1]
        if self.training and observations == 1:
            if self.running_mean is None or self.running_var is None:
                raise ValueError("Singleton auxiliary BatchNorm requires tracked running statistics")
            return F.batch_norm(value, self.running_mean, self.running_var, self.weight, self.bias,
                                training=False, momentum=0.0, eps=self.eps)
        return super().forward(value)


def _replace_batchnorm(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm1d) and not isinstance(child, SafeBatchNorm1d):
            tensors = [*child.parameters(), *child.buffers()]
            exemplar = next((value for value in tensors if value.is_floating_point()), None)
            replacement = SafeBatchNorm1d(
                child.num_features, eps=child.eps, momentum=child.momentum,
                affine=child.affine, track_running_stats=child.track_running_stats,
                device=exemplar.device if exemplar is not None else None,
                dtype=exemplar.dtype if exemplar is not None else None,
            )
            replacement.load_state_dict(child.state_dict())
            replacement.train(child.training)
            for parameter_name, parameter in replacement.named_parameters():
                parameter.requires_grad_(dict(child.named_parameters())[parameter_name].requires_grad)
            setattr(module, name, replacement)
        else:
            _replace_batchnorm(child)


def _load_vendor_class(root: Path):
    vendor = root / "vendor" / "EMT-DLFR"
    if not (vendor / "models" / "missingTask" / "EMT_DLFR.py").is_file():
        raise FileNotFoundError(f"Vendored EMT-DLFR source missing: {vendor}")
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from models.missingTask.EMT_DLFR import EMT_DLFR
    return EMT_DLFR


class _PredictionViewMixin:
    def prediction_view(self, text, text_lengths, audio, audio_lengths, vision, vision_lengths):
        """The original EMT prediction trunk, with no original-view/auxiliary IO.

        This follows vendored forward_once through its prediction return. The
        separate method avoids toggling regression_only temporarily on M2 and
        ensures evaluation cannot accidentally execute its auxiliary heads.
        """
        text = self.text_model(text)
        text_utterance, text = text[:, 0], text[:, 1:]
        audio, audio_utterance = self.audio_model(audio, audio_lengths, return_temporal=True)
        vision, vision_utterance = self.video_model(vision, vision_lengths, return_temporal=True)
        global_context = torch.stack((self.proj_text(text_utterance), self.proj_audio(audio_utterance),
                                      self.proj_video(vision_utterance)), dim=1)
        modalities = [self.proj_text(text), self.proj_audio(audio), self.proj_video(vision)]
        valid = [torch.arange(value.shape[1], device=value.device)[None] < lengths[:, None]
                 for value, lengths in zip(modalities, (text_lengths, audio_lengths, vision_lengths))]
        global_context, _ = self.fusion(global_context, modalities, valid)
        global_context = global_context.reshape(global_context.shape[0], -1)
        fused = torch.cat((text_utterance, audio_utterance, vision_utterance, global_context), dim=-1)
        fused = self.post_fusion_dropout(fused)
        fused = F.relu(self.post_fusion_layer_1(fused), inplace=False)
        fused = F.relu(self.post_fusion_layer_2(fused), inplace=False)
        return self.post_fusion_layer_3(fused).reshape(-1)


def build_model(config: Mapping[str, Any], root: str | Path) -> nn.Module:
    """Initialize from local pretrained BERT plus identically sampled EMT heads.

    The caller seeds RNGs before construction. Neither method loads an existing
    complete-input checkpoint. State-dict names match the vendored architecture.
    """
    root = Path(root).resolve()
    method = method_name(config)
    values = {**OFFICIAL_MOSEI_DEFAULTS, **dict(config)}
    if values.get("need_data_aligned") is not True:
        raise ValueError("The fixed mix_v1 adapter requires aligned inputs")
    # Construct the same modules for both methods before choosing trainable heads.
    values["regression_only"] = False
    pretrained = Path(values["bert_path"])
    if not pretrained.is_absolute():
        pretrained = root / pretrained
    required = ("config.json", "model.safetensors", "vocab.txt")
    missing = [name for name in required if not (pretrained / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Local pretrained BERT files missing: {missing}")
    values["bert_pretrained_path"] = str(pretrained.resolve())
    values["bert_local_files_only"] = True
    vendor_class = _load_vendor_class(root)

    class RobustEMTDLFR(_PredictionViewMixin, vendor_class):
        pass

    model = RobustEMTDLFR(SimpleNamespace(**values))
    _replace_batchnorm(model)
    model.method = method
    model.regression_only = method == "M1"
    model.singleton_batchnorm_policy = SINGLETON_BATCHNORM_POLICY
    if method == "M1":
        for name, module in model.named_children():
            if "projector" in name or "predictor" in name or name.startswith("recon_"):
                module.requires_grad_(False)
    return model


def _input_view(batch: Mapping[str, Any], device: torch.device | str) -> tuple:
    # Deliberately access only these five input keys; no labels/reference/support.
    text = batch["text"].to(device=device, dtype=torch.long, non_blocking=True)
    audio = batch["audio"].to(device=device, dtype=torch.float32, non_blocking=True)
    vision = batch["vision"].to(device=device, dtype=torch.float32, non_blocking=True)
    audio_lengths = batch["audio_lengths"].to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)
    vision_lengths = batch["vision_lengths"].to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)
    if text.ndim != 3 or text.shape[1:] != (3, 50):
        raise ValueError("Expected text[B,3,50]")
    text_lengths = text[:, 1].sum(dim=1).long() - 2
    if not torch.equal(audio_lengths, text_lengths) or not torch.equal(vision_lengths, text_lengths):
        raise ValueError("Aligned A/V lengths must equal original content lengths")
    if bool(((text_lengths < 1) | (text_lengths > 48)).any()):
        raise ValueError("Invalid content length")
    return text, text_lengths, audio, audio_lengths, vision, vision_lengths


def forward_training(model: nn.Module, batch: Mapping[str, Any], device: torch.device | str) -> dict:
    """Compute the same original + damaged prediction supervision for M1/M2."""
    if "reference" not in batch:
        raise ValueError("Robust training requires a train-only original reference view")
    original = _input_view(batch["reference"], device)
    damaged = _input_view(batch, device)
    if not torch.equal(original[0][:, 1:], damaged[0][:, 1:]):
        raise ValueError("Missing construction must preserve attention and segment IDs")
    reference_outputs = model.forward_once(*original, missing=False)
    missing_outputs = model.forward_once(*damaged, missing=True)
    return {**reference_outputs, **missing_outputs,
            "pred_reference": reference_outputs["pred"].reshape(-1),
            "pred_missing": missing_outputs["pred_m"].reshape(-1)}


def predict_batch(model: nn.Module, batch: Mapping[str, Any], device: torch.device | str) -> torch.Tensor:
    """Return [B] predictions from the damaged view only; caller controls no_grad."""
    return model.prediction_view(*_input_view(batch, device))
