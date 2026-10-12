"""Shared EMT features for direct three-class polarity and auxiliary intensity.

This adapter leaves the historical and vendored implementations untouched. Both
training views and prediction call ``forward_view``; the classification head is
attached to the ReLU feature consumed by the original regression head. Prediction
never needs labels, clean references, simulator masks or auxiliary heads.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from baseline.model import OFFICIAL_MOSEI_DEFAULTS
from robust.model import (
    SINGLETON_BATCHNORM_POLICY, _input_view as _aligned_input_view,
    _load_vendor_class, _replace_batchnorm,
)
from standardized.model import _input_view as _standardized_input_view
from standardized.normalization import AVNormalizer, validate_preprocessing


def method_name(config: Mapping[str, Any]) -> str:
    method = str(config.get("method", config.get("model_variant", ""))).upper()
    if method not in ("M0", "M1", "M2"):
        raise ValueError("Multitask method must be M0, M1 or M2")
    return method


class _MultitaskViewMixin:
    def forward_view(self, text, text_lengths, audio, audio_lengths, vision, vision_lengths,
                     *, auxiliary=False, missing=False):
        """One common feature computation; auxiliary computation is explicit."""
        encoded_text = self.text_model(text)
        text_utterance, text = encoded_text[:, 0], encoded_text[:, 1:]
        text_for_recon = text.detach()
        audio, audio_utterance = self.audio_model(audio, audio_lengths, return_temporal=True)
        vision, vision_utterance = self.video_model(vision, vision_lengths, return_temporal=True)
        context = torch.stack((self.proj_text(text_utterance), self.proj_audio(audio_utterance),
                               self.proj_video(vision_utterance)), dim=1)
        modalities = [self.proj_text(text), self.proj_audio(audio), self.proj_video(vision)]
        masks = [torch.arange(value.shape[1], device=value.device)[None] < length[:, None]
                 for value, length in zip(modalities, (text_lengths, audio_lengths, vision_lengths))]
        context, temporal = self.fusion(context, modalities, masks)
        context = context.reshape(context.shape[0], -1)
        fused = torch.cat((text_utterance, audio_utterance, vision_utterance, context), dim=-1)
        fused = self.post_fusion_dropout(fused)
        fused = F.relu(self.post_fusion_layer_1(fused), inplace=False)
        shared = F.relu(self.post_fusion_layer_2(fused), inplace=False)
        result = {"logits": self.cls_head(shared),
                  "intensity": self.post_fusion_layer_3(shared).reshape(-1)}
        if not auxiliary:
            return result
        if self.method != "M2":
            raise ValueError("Only M2 may execute DLFR auxiliary heads")
        suffix = "_m" if missing else ""
        for name, feature in (("gmc_tokens", context), ("text", text_utterance),
                              ("audio", audio_utterance), ("video", vision_utterance)):
            projection = getattr(self, name + "_projector")(feature)
            result[f"z_{name}{suffix}"] = projection.detach()
            result[f"p_{name}{suffix}"] = getattr(self, name + "_predictor")(projection)
        if missing:
            result.update(text_recon=self.recon_text(temporal[0]),
                          audio_recon=self.recon_audio(temporal[1]),
                          video_recon=self.recon_video(temporal[2]))
        else:
            result["text_for_recon"] = text_for_recon
        return result

    def forward_once(self, text, text_lengths, audio, audio_lengths, video, video_lengths,
                     missing=False):
        """Keep the vendor signature, but route it through the new shared trunk."""
        result = self.forward_view(text, text_lengths, audio, audio_lengths, video, video_lengths,
                                   auxiliary=self.method == "M2", missing=missing)
        suffix = "_m" if missing else ""
        logits, intensity = result.pop("logits"), result.pop("intensity")
        return {**result, f"logits{suffix}": logits, f"intensity{suffix}": intensity,
                f"pred{suffix}": intensity.reshape(-1, 1)}

    def prediction_view(self, text, text_lengths, audio, audio_lengths, vision, vision_lengths):
        return self.forward_view(text, text_lengths, audio, audio_lengths, vision, vision_lengths)


def build_model(config: Mapping[str, Any], root: str | Path, preprocessing: dict) -> nn.Module:
    """Initialize local BERT and new EMT/heads; never warm-start a legacy run."""
    validate_preprocessing(preprocessing)
    root = Path(root).resolve()
    method = method_name(config)
    values = {**OFFICIAL_MOSEI_DEFAULTS, **dict(config)}
    if values.get("need_data_aligned") is not True:
        raise ValueError("Multitask v1 requires aligned inputs")
    # All three methods sample an identical layout before auxiliaries are frozen.
    values["regression_only"] = False
    pretrained = Path(values["bert_path"])
    if not pretrained.is_absolute():
        pretrained = root / pretrained
    required = ("config.json", "model.safetensors", "vocab.txt")
    missing_files = [name for name in required if not (pretrained / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Local pretrained BERT files missing: {missing_files}")
    values["bert_pretrained_path"] = str(pretrained.resolve())
    values["bert_local_files_only"] = True
    vendor_class = _load_vendor_class(root)

    class MultitaskEMTDLFR(_MultitaskViewMixin, vendor_class):
        pass

    model = MultitaskEMTDLFR(SimpleNamespace(**values))
    _replace_batchnorm(model)
    model.cls_head = nn.Linear(model.post_fusion_layer_3.in_features, 3)
    model.av_normalizer = AVNormalizer(preprocessing)
    model.method = method
    model.multitask_method = method
    model.regression_only = method in ("M0", "M1")
    model.singleton_batchnorm_policy = SINGLETON_BATCHNORM_POLICY
    if method != "M2":
        for name, module in model.named_children():
            if "projector" in name or "predictor" in name or name.startswith("recon_"):
                module.requires_grad_(False)
    if not all(parameter.requires_grad for parameter in model.cls_head.parameters()):
        raise RuntimeError("The classification head must remain trainable")
    return model


def _input_view(model, batch, device):
    # Reuse the fixed transform and input validation, accessing only five keys.
    return _aligned_input_view(_standardized_input_view(model, batch, device), device)


def forward_training(model: nn.Module, batch: Mapping[str, Any], device) -> dict:
    """M0 uses one clean view; M1/M2 use clean and damaged views equally."""
    if model.method == "M0":
        result = model.forward_view(*_input_view(model, batch, device))
        return {**result, "logits_reference": result["logits"],
                "intensity_reference": result["intensity"], "pred_reference": result["intensity"]}
    if model.method not in ("M1", "M2") or "reference" not in batch:
        raise ValueError("Multitask M1/M2 training requires a clean reference view")
    original = _input_view(model, batch["reference"], device)
    damaged = _input_view(model, batch, device)
    if not torch.equal(original[0][:, 1:], damaged[0][:, 1:]):
        raise ValueError("Missing construction must preserve attention and segment IDs")
    auxiliary = model.method == "M2"
    clean = model.forward_view(*original, auxiliary=auxiliary, missing=False)
    missing = model.forward_view(*damaged, auxiliary=auxiliary, missing=True)
    result = {}
    for view, output in (("reference", clean), ("missing", missing)):
        result["logits_" + view] = output.pop("logits")
        result["intensity_" + view] = output.pop("intensity")
        result["pred_" + view] = result["intensity_" + view]
        result.update(output)
    if auxiliary:
        result["audio_for_recon"] = original[2].detach()
        result["vision_for_recon"] = original[4].detach()
    return result


def predict_batch(model: nn.Module, batch: Mapping[str, Any], device) -> dict:
    """Return raw logits [B,3] and raw intensity [B] from current inputs only."""
    return model.prediction_view(*_input_view(model, batch, device))
