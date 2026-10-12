"""Attachment 2 loader for the complete-input regression baseline.

The aligned pickle contains a CLS placeholder at audio/vision position 0 and a
SEP placeholder at attention_mask.sum()-1. They are verified to be zero before
we shift audio/vision left by one slot. LSTM lengths count content tokens only.
Interior zero rows are valid positions (including naturally unavailable vision)
and are retained. No fitted normalization, resplitting, or missingness is added.
"""

from __future__ import annotations

from pathlib import Path
import pickle
import warnings
from typing import Iterable

import numpy as np


EXPECTED_COUNTS = {"train": 3395, "valid": 728, "test": 727}
LENGTH_POLICY = (
    "Validate contiguous BERT attention mask and CLS/SEP placement; audio/vision "
    "CLS, SEP and padding slots must be zero after nonfinite replacement. Shift "
    "audio/vision one slot left, right-pad with zero to 50, and use mask.sum()-2 "
    "as each LSTM length. Retain every content slot, including all-zero rows."
)
FINITE_POLICY = (
    "Replace NaN, +Inf and -Inf in audio/vision with zero before float32 conversion; "
    "report counts. Reject nonfinite labels or text and float32 overflow. No "
    "normalization or statistics fitted to any split."
)


def read_pickle(path: str | Path) -> dict:
    """Read the local, trusted competition pickle; never unpickle unknown files."""
    with Path(path).open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, dict) or set(data) != set(EXPECTED_COUNTS):
        raise ValueError("Expected the original train/valid/test split dictionary")
    return data


def summarize_raw_split(fields: dict, split: str) -> dict:
    """Validate layout and return aggregate structural metadata, never predictions."""
    text = np.asarray(fields["text_bert"])
    size = len(fields["id"])
    if size != EXPECTED_COUNTS[split]:
        raise ValueError(f"{split}: expected {EXPECTED_COUNTS[split]} samples, got {size}")
    if text.shape != (size, 3, 50):
        raise ValueError(f"{split}: unexpected text_bert shape {text.shape}")
    if not np.isfinite(text).all() or not np.equal(text, np.floor(text)).all():
        raise ValueError(f"{split}: text_bert must contain finite integers")
    mask = text[:, 1, :]
    if not np.isin(mask, [0, 1]).all():
        raise ValueError(f"{split}: attention mask must be binary")
    lengths = mask.sum(axis=1).astype(np.int64)
    positions = np.arange(50)[None, :]
    rows = np.arange(size)
    if not np.equal(mask, positions < lengths[:, None]).all():
        raise ValueError(f"{split}: attention mask is not a contiguous prefix")
    if np.any(lengths < 3):
        raise ValueError(f"{split}: each sample needs CLS, content and SEP")
    if not np.all(text[:, 0, 0] == 101) or not np.all(text[rows, 0, lengths - 1] == 102):
        raise ValueError(f"{split}: expected bert-base-uncased CLS=101 and SEP=102")
    if not np.all(text[:, 0, :][positions >= lengths[:, None]] == 0):
        raise ValueError(f"{split}: expected PAD token 0 outside the attention mask")
    if not np.all((text[:, 0, :] >= 0) & (text[:, 0, :] < 30522)):
        raise ValueError(f"{split}: token ID outside bert-base-uncased vocabulary")
    if not np.all(text[:, 2, :] == 0):
        raise ValueError(f"{split}: expected single-sentence token type IDs")
    labels = np.asarray(fields["regression_labels"])
    if labels.shape not in ((size,), (size, 1)) or not np.isfinite(labels).all():
        raise ValueError(f"{split}: regression labels must be finite, one per sample")

    content = (positions >= 1) & (positions < lengths[:, None] - 1)
    features = {}
    for name, dimension in (("audio", 74), ("vision", 35)):
        array = np.asarray(fields[name])
        if array.shape != (size, 50, dimension):
            raise ValueError(f"{split}: unexpected {name} shape {array.shape}")
        finite = np.isfinite(array)
        # A nonfinite entry is replaced with zero by the fixed finite policy.
        nonzero_rows = np.any(finite & (array != 0), axis=-1)
        cls_count = int(nonzero_rows[:, 0].sum())
        sep_count = int(nonzero_rows[rows, lengths - 1].sum())
        padding_count = int((nonzero_rows & (positions >= lengths[:, None])).sum())
        if cls_count or sep_count or padding_count:
            raise ValueError(f"{split}: {name} has data in CLS/SEP/padding slots; refusing to truncate")
        features[name] = {
            "shape": list(array.shape),
            "source_dtype": str(array.dtype),
            "output_dtype": "float32",
            "nonfinite_values_replaced": int((~finite).sum()),
            "nan_count": int(np.isnan(array).sum()),
            "positive_inf_count": int(np.isposinf(array).sum()),
            "negative_inf_count": int(np.isneginf(array).sum()),
            "nonzero_cls_rows": cls_count,
            "nonzero_sep_rows": sep_count,
            "nonzero_padding_rows": padding_count,
            "zero_content_rows_retained": int((~nonzero_rows & content).sum()),
            "all_zero_content_samples_retained": int((~np.any(nonzero_rows & content, axis=1)).sum()),
        }
    return {
        "sample_count": size,
        "text_bert_shape": list(text.shape),
        "text_output_dtype": "int64",
        "attention_mask_binary": True,
        "attention_mask_contiguous_prefix": True,
        "cls_id": 101,
        "sep_id": 102,
        "pad_id": 0,
        "token_id_range": [int(text[:, 0].min()), int(text[:, 0].max())],
        "attention_length_range": [int(lengths.min()), int(lengths.max())],
        "content_length_range": [int(lengths.min() - 2), int(lengths.max() - 2)],
        "token_type_ids": [0],
        "labels_shape": list(labels.shape),
        "labels_finite": True,
        "features": features,
    }


class BaselineDataset:
    """Map-style torch Dataset protocol; torch is imported only when requested."""

    def __init__(self, fields: dict, split: str):
        self.split = split
        self.summary = summarize_raw_split(fields, split)
        self.text = np.array(fields["text_bert"], dtype=np.int64, copy=True)
        self.audio_lengths = self.text[:, 1].sum(axis=1, dtype=np.int64) - 2
        self.vision_lengths = self.audio_lengths.copy()
        self.ids = [str(item) for item in fields["id"]]
        self.labels = np.asarray(fields["regression_labels"], dtype=np.float32).reshape(-1).copy()
        if not np.isfinite(self.labels).all():
            raise ValueError(f"{split}: labels overflowed float32")
        for name in ("audio", "vision"):
            source = np.asarray(fields[name])
            count = self.summary["features"][name]["nonfinite_values_replaced"]
            if count:
                warnings.warn(f"{split}/{name}: replacing {count} nonfinite values with zero", stacklevel=2)
                source = np.nan_to_num(source, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
            output = np.zeros(source.shape, dtype=np.float32)
            with np.errstate(over="ignore"):
                output[:, :-1] = source[:, 1:]
            if not np.isfinite(output).all():
                raise ValueError(f"{split}/{name}: finite values overflowed float32")
            setattr(self, name, output)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict:
        import torch

        return {
            "text": torch.from_numpy(self.text[index]),
            "audio": torch.from_numpy(self.audio[index]),
            "vision": torch.from_numpy(self.vision[index]),
            "audio_lengths": torch.tensor(self.audio_lengths[index], dtype=torch.long),
            "vision_lengths": torch.tensor(self.vision_lengths[index], dtype=torch.long),
            "labels": torch.tensor(self.labels[index], dtype=torch.float32),
            "id": self.ids[index],
            "index": int(index),
        }


def load_splits(path: str | Path, splits: Iterable[str] | str = ("train", "valid")) -> dict[str, BaselineDataset]:
    """Expose requested original splits only; no test Dataset is built for training.

    Pickle is a single serialized container, so reading it deserializes the full
    archive. Only selected splits are validated/adapted and retained in memory.
    """
    requested = (splits,) if isinstance(splits, str) else tuple(splits)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Request one or more unique split names")
    unknown = set(requested) - set(EXPECTED_COUNTS)
    if unknown:
        raise ValueError(f"Unknown splits: {sorted(unknown)}")
    source = read_pickle(path)
    return {split: BaselineDataset(source[split], split) for split in requested}
