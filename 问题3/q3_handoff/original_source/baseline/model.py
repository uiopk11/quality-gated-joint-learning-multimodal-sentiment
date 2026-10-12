"""Thin adapter for the official EMT-DLFR complete-input regression path.

No new network blocks are introduced. MOSEI architecture defaults are copied from
the pinned upstream config/config_regression.py; run config can override them.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch import nn


OFFICIAL_MOSEI_DEFAULTS: dict[str, Any] = {
    "language": "en",
    "need_data_aligned": True,
    "use_finetune": True,
    "feature_dims": [768, 74, 35],
    "a_lstm_hidden_size": 16,
    "v_lstm_hidden_size": 32,
    "a_lstm_layers": 1,
    "v_lstm_layers": 1,
    "text_out": 768,
    "audio_out": 16,
    "video_out": 32,
    "a_lstm_dropout": 0.0,
    "v_lstm_dropout": 0.0,
    "t_bert_dropout": 0.1,
    "d_model": 128,
    "fusion_layers": 2,
    "heads": 4,
    "learnable_pos_emb": False,
    "emb_dropout": 0.0,
    "attn_dropout": 0.0,
    "ff_dropout": 0.0,
    "ff_expansion": 4,
    "mpu_share": True,
    "modality_share": True,
    "layer_share": True,
    "attn_act_fn": "tanh",
    "fusion_method": "emt",
    "gmc_tokens_pred_dim": 128,
    "text_pred_dim": 256,
    "audio_pred_dim": 8,
    "video_pred_dim": 16,
    "post_fusion_dim": 128,
    "post_fusion_dropout": 0.0,
    "gradient_checkpointing": True,
    "bert_path": "pretrained/bert-base-uncased",
    "regression_only": True,
}


def build_model(config: Mapping[str, Any], root: str | Path) -> nn.Module:
    """Build the vendored original network with local, fine-tuned BERT weights.

    The returned object is EMT_DLFR itself (no AMIO wrapper). Parameter groups are
    therefore named text_model.*, audio_model.*, video_model.*, and fusion.*.
    """
    root = Path(root).resolve()
    values = {**OFFICIAL_MOSEI_DEFAULTS, **dict(config)}
    if values.get("regression_only") is not True:
        raise ValueError("This first-step baseline requires regression_only=true.")
    pretrained = Path(values["bert_path"])
    if not pretrained.is_absolute():
        pretrained = root / pretrained
    required = ("config.json", "model.safetensors", "vocab.txt")
    missing = [name for name in required if not (pretrained / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Local BERT files missing in {pretrained}: {missing}")
    values["bert_pretrained_path"] = str(pretrained.resolve())
    values["bert_local_files_only"] = True
    vendor = root / "vendor" / "EMT-DLFR"
    if not (vendor / "models" / "missingTask" / "EMT_DLFR.py").is_file():
        raise FileNotFoundError(f"Vendored EMT-DLFR source missing: {vendor}")
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from models.missingTask.EMT_DLFR import EMT_DLFR

    return EMT_DLFR(SimpleNamespace(**values))


def forward_batch(model: nn.Module, batch: Mapping[str, Any], device: torch.device | str) -> torch.Tensor:
    """Return one unbounded sentiment-intensity prediction per sample: shape (B,).

    Data must already be right padded. Valid audio/video lengths are mandatory;
    alignment does not mean that padding is valid input. No missing view is made.
    """
    text = batch["text"].to(device=device, non_blocking=True)
    audio = batch["audio"].to(device=device, dtype=torch.float32, non_blocking=True)
    vision = batch["vision"].to(device=device, dtype=torch.float32, non_blocking=True)
    audio_lengths = batch["audio_lengths"].to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)
    vision_lengths = batch["vision_lengths"].to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)
    # None placeholders keep the official forward signature without copying or
    # perturbing any complete input. regression_only returns before using them.
    outputs = model((text, None), (audio, None, audio_lengths), (vision, None, vision_lengths))
    return outputs["pred"].reshape(-1)
