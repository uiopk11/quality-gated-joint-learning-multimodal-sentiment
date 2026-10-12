"""Verified, complete-input batches using the unchanged Q2 reading conventions.

Only the selected split is adapted. Pickle necessarily deserializes the entire
trusted archive. No statistics are fitted, and raw A/V values are not normalized
here. Source data is read-only and is never exported by this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .runtime import configure_runtime

HANDOFF_ROOT = Path(configure_runtime()).resolve()

import numpy as np
import torch

from baseline.data import BaselineDataset, read_pickle
from dataset_construction.reader import apply_missing


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_data_sha256():
    """Read the unmodified formal provenance and cross-check preprocessing."""
    provenance = json.loads((HANDOFF_ROOT / "checkpoint/provenance.json").read_text(encoding="utf-8"))
    preprocessing = json.loads((HANDOFF_ROOT / "checkpoint/preprocessing.json").read_text(encoding="utf-8"))
    expected = provenance["data_binding"]["data_sha256"]
    if expected != preprocessing["source"]["sha256"]:
        raise ValueError("Copied provenance and preprocessing disagree on data SHA-256")
    return expected


def _default_indices(lengths, batch_size):
    """Prefer the first two different lengths, then earliest remaining samples."""
    if not 1 <= batch_size <= len(lengths):
        raise ValueError("batch_size is outside the selected split")
    chosen = [0]
    if batch_size >= 2:
        other = np.flatnonzero(lengths != lengths[0])
        chosen.append(int(other[0]) if len(other) else 1)
    for index in range(len(lengths)):
        if len(chosen) >= batch_size:
            break
        if index not in chosen:
            chosen.append(index)
    return chosen


def load_complete_batch(data_path, *, split="valid", indices=None, batch_size=2,
                        device="cpu", require_inside_handoff=False):
    """Return a real complete batch, verifying the exact formal data SHA first.

    ``indices`` are zero-based indices in the original, unchanged split. With
    indices omitted, the first sample and first sample of different body length
    are used. ``require_inside_handoff=True`` is mandatory for our smoke test.
    Receivers may read their own trusted local data with the default False.

    Model fields: text int64 [B,3,50], audio float32 [B,50,74], vision float32
    [B,50,35], audio_lengths/vision_lengths int64 [B]. A/V body j is already
    shifted to row j; text body j remains at row j+1. Source slot coordinates
    refer to the original aligned array, not seconds or raw character offsets.
    Metadata fields are ignored by the formal prediction path.
    """
    path = Path(data_path).expanduser().resolve()
    if require_inside_handoff and not path.is_relative_to(HANDOFF_ROOT):
        raise ValueError("Smoke input must be a copied dataset inside q3_handoff")
    actual_sha = sha256_file(path)
    if actual_sha != expected_data_sha256():
        raise ValueError("Dataset SHA-256 does not match the formal M0 data binding")
    if split not in ("train", "valid", "test"):
        raise ValueError("split must be train, valid or test")
    archive = read_pickle(path)
    fields = archive[split]
    dataset = BaselineDataset(fields, split)
    chosen = (_default_indices(dataset.audio_lengths, batch_size) if indices is None
              else list(indices))
    if not chosen or any(type(i) is not int or i < 0 or i >= len(dataset) for i in chosen):
        raise ValueError("indices must contain valid zero-based integer split indices")
    if len(set(chosen)) != len(chosen):
        raise ValueError("Duplicate sample indices are not allowed")
    items = []
    for index in chosen:
        item = dataset[index]
        # This call uses an all-False mask solely to verify the formal clean
        # reader protocol. It does not instantiate any missing-modality model.
        av = []
        for name in ("audio", "vision"):
            source = np.asarray(fields[name][index])
            av.append(np.nan_to_num(source, copy=True, nan=0.0, posinf=0.0, neginf=0.0))
        clean = apply_missing(fields["text_bert"][index], *av,
                              np.zeros((3, 50), dtype=np.bool_))
        for key in ("text", "audio", "vision", "audio_lengths", "vision_lengths"):
            expected = torch.from_numpy(np.asarray(clean[key]))
            if not torch.equal(item[key], expected):
                raise RuntimeError("BaselineDataset differs from all-False apply_missing: " + key)
        items.append(item)
    target = torch.device(device)
    tensor_keys = ("text", "audio", "vision", "audio_lengths", "vision_lengths", "labels")
    batch = {key: torch.stack([item[key] for item in items]).to(target) for key in tensor_keys}
    batch["id"] = [item["id"] for item in items]
    batch["index"] = torch.tensor(chosen, dtype=torch.long, device=target)
    batch["split"] = split
    batch["raw_text"] = [str(fields["raw_text"][i]) for i in chosen] if "raw_text" in fields else [None] * len(chosen)
    positions = torch.arange(50, device=target).expand(len(chosen), -1)
    content = positions < batch["audio_lengths"][:, None]
    batch["content_positions"] = torch.where(content, positions, -1)
    batch["source_positions"] = torch.where(content, positions + 1, -1)
    batch["coordinate_metadata"] = {
        "body_coordinate": "zero-based j in [0, content_length)",
        "source_slot": "j+1 in original aligned text/audio/vision arrays",
        "text_input_slot": "j+1",
        "audio_vision_input_slot": "j",
        "invalid_position_sentinel": -1,
        "time_unit": None,
        "timestamps_available": False,
        "raw_text_character_offsets_available": False,
        "note": "50 storage slots are not 50 seconds; no timestamp or character alignment is inferred.",
    }
    batch["data_metadata"] = {
        "path": str(path), "sha256": actual_sha, "split": split,
        "selected_indices": chosen, "sample_count_in_split": len(dataset),
        "baseline_matches_all_false_apply_missing": True,
        "raw_text_available": "raw_text" in fields,
        "split_validation": dataset.summary,
        "external_normalization": False,
    }
    return batch
