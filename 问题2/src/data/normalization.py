"""Fixed train-only A/V channel standardization for aligned features.

The legacy reader remains responsible for special-slot validation, nonfinite
replacement, float32 conversion and left alignment. This module fits the same
float32 features, using float64 population statistics, and preserves zero rows
without deleting time positions. There are deliberately no selectable modes.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn

from src.data.aligned import read_pickle, summarize_raw_split


PREPROCESSING_VERSION = "av-channel-standardization-v1"
DIMENSIONS = {"audio": 74, "vision": 35}
MINIMUM_STD = 1e-6
PREPROCESSING_RULE = {
    "fit_split": "train",
    "fit_rows": "valid_content_and_original_nonzero_row",
    "apply_rows": "valid_content_and_current_nonzero_row",
    "zero_row_detection": "before_standardization_after_nonfinite_to_zero_float32",
    "excluded_slots": "cls_sep_padding",
    "zero_coordinates": "include_in_observed_rows",
    "std_ddof": 0,
    "minimum_std": MINIMUM_STD,
    "small_std_scale": 1.0,
    "statistics_dtype": "float64",
    "input_dtype": "float32",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_integer(value: object, name: str, *, zero_allowed: bool = False) -> int:
    if type(value) is not int or value < (0 if zero_allowed else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if zero_allowed else 'positive'} integer")
    return value


def validate_preprocessing(metadata: dict, source_sha256: str | None = None) -> dict:
    """Validate fixed semantics and return an independent JSON-safe copy.

    This validates saved metadata; it never opens the source or fits new data.
    A caller verifying data provenance may supply its already-known source hash.
    """
    if not isinstance(metadata, dict) or set(metadata) != {"version", "rule", "source", "modalities"}:
        raise ValueError("Unexpected A/V preprocessing metadata fields")
    if metadata["version"] != PREPROCESSING_VERSION or metadata["rule"] != PREPROCESSING_RULE:
        raise ValueError("Unsupported A/V preprocessing version or fixed rule")
    source = metadata["source"]
    if not isinstance(source, dict) or set(source) != {"path", "sha256", "split", "sample_count", "content_rows"}:
        raise ValueError("Unexpected train source metadata fields")
    if source["split"] != "train":
        raise ValueError("A/V statistics must come only from original train")
    if not isinstance(source["path"], str) or not Path(source["path"]).is_absolute():
        raise ValueError("Train source path must be absolute")
    if not isinstance(source["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"]):
        raise ValueError("Invalid train source SHA-256")
    if source_sha256 is not None and source["sha256"] != source_sha256:
        raise ValueError("A/V preprocessing train source SHA-256 mismatch")
    count = _positive_integer(source["sample_count"], "train sample_count")
    content_rows = _positive_integer(source["content_rows"], "train content_rows")
    if not count <= content_rows <= count * 48:
        raise ValueError("Invalid train content row count")
    modalities = metadata["modalities"]
    if not isinstance(modalities, dict) or set(modalities) != set(DIMENSIONS):
        raise ValueError("A/V preprocessing requires both audio and vision")
    for name, dimension in DIMENSIONS.items():
        values = modalities[name]
        if not isinstance(values, dict) or set(values) != {
            "dimension", "observed_rows", "native_zero_rows", "mean", "std", "scale"
        }:
            raise ValueError(f"Unexpected {name} statistics fields")
        if type(values["dimension"]) is not int or values["dimension"] != dimension:
            raise ValueError(f"Invalid {name} channel dimension")
        observed = _positive_integer(values["observed_rows"], f"{name} observed_rows")
        zero = _positive_integer(values["native_zero_rows"], f"{name} native_zero_rows", zero_allowed=True)
        if observed + zero != content_rows:
            raise ValueError(f"{name} row counts do not cover original train content")
        arrays = {}
        for key in ("mean", "std", "scale"):
            raw = values[key]
            if not isinstance(raw, list) or any(type(item) not in (int, float) for item in raw):
                raise ValueError(f"{name} {key} must be a JSON numeric list")
            array = np.asarray(raw, dtype=np.float64)
            if array.shape != (dimension,) or not np.isfinite(array).all():
                raise ValueError(f"Invalid {name} {key} shape or nonfinite statistics")
            arrays[key] = array
        if np.any(arrays["std"] < 0):
            raise ValueError(f"{name} standard deviations must be nonnegative")
        expected_scale = np.where(arrays["std"] < MINIMUM_STD, 1.0, arrays["std"])
        if not np.array_equal(arrays["scale"], expected_scale):
            raise ValueError(f"{name} scale does not follow the fixed small-std rule")
    # Reject non-JSON scalars/containers elsewhere in the schema as well.
    try:
        return json.loads(json.dumps(metadata, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("A/V preprocessing metadata must be finite JSON data") from error


def metadata_fingerprint(metadata: dict) -> str:
    canonical = json.dumps(validate_preprocessing(metadata), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fit_preprocessing(source_path: str | Path) -> dict:
    """Fit original train once, without writing files or examining other splits."""
    path = Path(source_path).resolve()
    archive = read_pickle(path)
    train = archive["train"]
    summary = summarize_raw_split(train, "train")
    attention_lengths = np.asarray(train["text_bert"])[:, 1].sum(axis=1).astype(np.int64)
    positions = np.arange(50)[None, :]
    content = (positions >= 1) & (positions < attention_lengths[:, None] - 1)
    metadata = {
        "version": PREPROCESSING_VERSION,
        "rule": copy.deepcopy(PREPROCESSING_RULE),
        "source": {
            "path": str(path), "sha256": _sha256(path), "split": "train",
            "sample_count": summary["sample_count"], "content_rows": int(content.sum()),
        },
        "modalities": {},
    }
    for name, dimension in DIMENSIONS.items():
        source = np.asarray(train[name])
        finite_source = np.nan_to_num(source, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        with np.errstate(over="ignore"):
            inputs = finite_source.astype(np.float32)
        if not np.isfinite(inputs).all():
            raise ValueError(f"train/{name}: finite values overflowed float32")
        observed = content & np.any(inputs != 0, axis=-1)
        rows = inputs[observed].astype(np.float64)
        if not len(rows):
            raise ValueError(f"train/{name}: no observed content rows for standardization")
        mean = rows.mean(axis=0, dtype=np.float64)
        std = rows.std(axis=0, dtype=np.float64, ddof=0)
        metadata["modalities"][name] = {
            "dimension": dimension,
            "observed_rows": int(observed.sum()),
            "native_zero_rows": int((content & ~observed).sum()),
            "mean": mean.tolist(), "std": std.tolist(),
            "scale": np.where(std < MINIMUM_STD, 1.0, std).tolist(),
        }
    return validate_preprocessing(metadata)


class AVNormalizer(nn.Module):
    """Frozen persistent channel statistics shared by complete/missing views."""

    def __init__(self, metadata: dict):
        super().__init__()
        self._metadata = validate_preprocessing(metadata)
        for name in DIMENSIONS:
            for key in ("mean", "std", "scale"):
                self.register_buffer(
                    f"{name}_{key}",
                    torch.tensor(self._metadata["modalities"][name][key], dtype=torch.float64),
                    persistent=True,
                )

    def metadata(self) -> dict:
        result = copy.deepcopy(self._metadata)
        for name in DIMENSIONS:
            for key in ("mean", "std", "scale"):
                result["modalities"][name][key] = getattr(self, f"{name}_{key}").detach().cpu().tolist()
        return validate_preprocessing(result)

    def forward(self, value: torch.Tensor, lengths: torch.Tensor, modality: str) -> torch.Tensor:
        if modality not in DIMENSIONS:
            raise ValueError(f"Unknown A/V modality: {modality}")
        if value.ndim != 3 or value.shape[-1] != DIMENSIONS[modality] or not value.is_floating_point():
            raise ValueError(f"{modality} must be floating [batch, time, {DIMENSIONS[modality]}]")
        if lengths.shape != (value.shape[0],) or lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("A/V lengths must contain one integer per sample")
        if torch.any(lengths < 1) or torch.any(lengths > value.shape[1]):
            raise ValueError("A/V lengths must retain valid content positions")
        # Read observability in the original scale, before mean subtraction.
        content = torch.arange(value.shape[1], device=value.device)[None, :] < lengths.to(value.device)[:, None]
        observed = content & torch.any(value != 0, dim=-1)
        mean = getattr(self, f"{modality}_mean").to(device=value.device, dtype=value.dtype)
        scale = getattr(self, f"{modality}_scale").to(device=value.device, dtype=value.dtype)
        normalized = (value - mean) / scale
        return torch.where(observed[..., None], normalized, torch.zeros_like(value))
