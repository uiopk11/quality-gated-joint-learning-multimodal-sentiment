"""Versioned M0/M1 clean and M2/M3 full-DLFR paired experiment models.

M0/M2 have only the original regression prediction head. M1/M3 add a
classification head to the same shared feature. All four initialize the same
EMT parameters and normalize inputs with statistics fitted on original train.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from src.data.normalization import AVNormalizer, validate_preprocessing
from src.models.baseline import OFFICIAL_MOSEI_DEFAULTS
from src.models.robust import (
    SINGLETON_BATCHNORM_POLICY, _input_view as _aligned_input_view,
    _load_vendor_class, _replace_batchnorm,
)
from src.models.standardized import _input_view as _standardized_input_view
from src.runtime import resolve_path


METHODS = ("M0", "M1", "M2", "M3")


def method_name(config: Mapping[str, Any] | str) -> str:
    method = config if isinstance(config, str) else config.get("method", "")
    method = str(method).upper()
    if method not in METHODS:
        raise ValueError("Four-model method must be M0, M1, M2 or M3")
    return method


def is_joint(method: Mapping[str, Any] | str) -> bool:
    return method_name(method) in ("M1", "M3")


def is_missing(method: Mapping[str, Any] | str) -> bool:
    return method_name(method) in ("M2", "M3")


class _FourModelViewMixin:
    def forward_view(self, text, text_lengths, audio, audio_lengths, vision, vision_lengths,
                     *, auxiliary=False, missing=False):
        """The original EMT trunk; recovery heads execute only on request."""
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
        result = {"intensity": self.post_fusion_layer_3(shared).reshape(-1)}
        if is_joint(self.method):
            result["logits"] = self.cls_head(shared)
        if not auxiliary:
            return result
        if not is_missing(self.method):
            raise ValueError("Only M2/M3 may execute DLFR auxiliary heads")
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
        """Maintain the vendor signature without changing the versioned roles."""
        result = self.forward_view(text, text_lengths, audio, audio_lengths, video, video_lengths,
                                   auxiliary=is_missing(self.method), missing=missing)
        suffix = "_m" if missing else ""
        intensity = result.pop("intensity")
        if "logits" in result:
            result["logits" + suffix] = result.pop("logits")
        return {**result, "intensity" + suffix: intensity,
                "pred" + suffix: intensity.reshape(-1, 1)}

    def prediction_view(self, text, text_lengths, audio, audio_lengths, vision, vision_lengths):
        return self.forward_view(text, text_lengths, audio, audio_lengths, vision, vision_lengths)


def build_model(config: Mapping[str, Any], root: str | Path, preprocessing: dict) -> nn.Module:
    """Build fresh local-BERT EMT; paired backbones and RNG states are equal.

    The optional classifier is initialized inside a forked CPU RNG context, so
    adding its parameters does not perturb the next training dropout or shuffle.
    DLFR modules retain the vendor layout but are frozen and skipped for clean
    roles; no synthetic missing/reference view is constructed for M0/M1.
    """
    validate_preprocessing(preprocessing)
    root = Path(root).resolve()
    method = method_name(config)
    values = {**OFFICIAL_MOSEI_DEFAULTS, **dict(config)}
    if values.get("need_data_aligned") is not True:
        raise ValueError("Four-model v1 requires aligned inputs")
    values["regression_only"] = False
    pretrained = resolve_path(values["bert_path"], root)
    missing_files = [name for name in ("config.json", "model.safetensors", "vocab.txt")
                     if not (pretrained / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Local pretrained BERT files missing: {missing_files}")
    values["bert_pretrained_path"] = str(pretrained.resolve())
    values["bert_local_files_only"] = True
    vendor_class = _load_vendor_class(root)

    class FourModelEMTDLFR(_FourModelViewMixin, vendor_class):
        pass

    model = FourModelEMTDLFR(SimpleNamespace(**values))
    _replace_batchnorm(model)
    if is_joint(method):
        with torch.random.fork_rng(devices=[]):
            model.cls_head = nn.Linear(model.post_fusion_layer_3.in_features, 3)
    model.av_normalizer = AVNormalizer(preprocessing)
    model.method = method
    model.four_model_method = method
    model.regression_only = not is_missing(method)
    model.singleton_batchnorm_policy = SINGLETON_BATCHNORM_POLICY
    if not is_missing(method):
        for name, module in model.named_children():
            if "projector" in name or "predictor" in name or name.startswith("recon_"):
                module.requires_grad_(False)
    return model


def _input_view(model, batch, device):
    # Prediction reads only the current raw input and original valid lengths.
    return _aligned_input_view(_standardized_input_view(model, batch, device), device)


def _training_output(output, view):
    result = {key: value for key, value in output.items() if key not in ("logits", "intensity")}
    result["intensity_" + view] = output["intensity"]
    result["pred_" + view] = output["intensity"]
    if "logits" in output:
        result["logits_" + view] = output["logits"]
    return result


def forward_training(model: nn.Module, batch: Mapping[str, Any], device) -> dict:
    """M0/M1: a single clean pass. M2/M3: two views with complete DLFR."""
    if not is_missing(model.method):
        output = model.forward_view(*_input_view(model, batch, device))
        return {**output, **_training_output(output, "reference")}
    if "reference" not in batch:
        raise ValueError("Four-model M2/M3 training requires a clean reference view")
    original = _input_view(model, batch["reference"], device)
    damaged = _input_view(model, batch, device)
    if not torch.equal(original[0][:, 1:], damaged[0][:, 1:]):
        raise ValueError("Missing construction must preserve attention and segment IDs")
    clean = model.forward_view(*original, auxiliary=True, missing=False)
    missing = model.forward_view(*damaged, auxiliary=True, missing=True)
    return {**_training_output(clean, "reference"), **_training_output(missing, "missing"),
            "audio_for_recon": original[2].detach(), "vision_for_recon": original[4].detach()}


def predict_batch(model: nn.Module, batch: Mapping[str, Any], device) -> dict:
    """Return raw intensity and, for M1/M3 only, logits from current inputs."""
    return model.prediction_view(*_input_view(model, batch, device))
