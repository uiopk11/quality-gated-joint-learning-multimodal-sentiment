"""Apply saved masks before BERT; expose prediction input separately from truth.

All public item arrays are fresh copies. No training/model code is imported.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from baseline.data import read_pickle
from .common import (DEFAULT_OUTPUT, array_fingerprint, code_provenance, json_fingerprint,
                     read_json, safe_artifact_path, sha256_file, source_metadata,
                     tokenizer_info, verify_inventory)


def shift_left(array: np.ndarray) -> np.ndarray:
    """Drop the original CLS slot; zero-fill the final slot, preserving 50 steps."""
    out = np.zeros_like(array)
    out[:-1] = array[1:]
    return out


def apply_missing(text_bert, audio, vision, missing, *, unk_id=100) -> dict:
    """Return numpy model inputs from one raw sample and a [3,50] boolean mask."""
    text = np.array(text_bert, dtype=np.int64, copy=True)
    mask = np.asarray(missing)
    if text.shape != (3, 50) or mask.shape != (3, 50) or mask.dtype != np.bool_:
        raise ValueError("Expected text_bert[3,50] and boolean missing[3,50]")
    length = int(text[1].sum()) - 2
    content = (np.arange(50) >= 1) & (np.arange(50) <= length)
    if length < 1 or length > 48 or np.any(mask & ~content[None]):
        raise ValueError("Missing mask outside valid content")
    text[0, mask[0]] = unk_id
    result = {"text": text, "audio_lengths": np.int64(length), "vision_lengths": np.int64(length)}
    for m, (name, value, dim) in enumerate((("audio", audio, 74), ("vision", vision, 35)), 1):
        original = np.asarray(value)
        if original.shape != (50, dim) or not np.isfinite(original).all():
            raise ValueError(f"Invalid {name} input")
        array = np.array(original, dtype=np.float32, copy=True)
        if not np.isfinite(array).all():
            raise ValueError(f"{name} overflow during float32 conversion")
        array[mask[m]] = 0
        result[name] = shift_left(array)
    return result


def reconstruction_support(missing, observed, content, *, audio_steps=50, vision_steps=50):
    """Return per-position supports, not a loss and not prediction input.

    Text is in post-CLS 49-slot BERT coordinates; A/V are shifted 50-slot
    coordinates, optionally cropped to the later LSTM batch maximum length.
    The future trainer must return zero for empty support and normalize each
    modality by selected positions times its feature dimension.
    """
    missing, observed, content = map(np.asarray, (missing, observed, content))
    if missing.shape != (3, 50) or observed.shape != (3, 50) or content.shape != (50,):
        raise ValueError("Support mask shape mismatch")
    if any(value.dtype != np.bool_ for value in (missing, observed, content)):
        raise ValueError("Support masks must be boolean")
    length = int(content.sum())
    if not (length <= audio_steps <= 50 and length <= vision_steps <= 50):
        raise ValueError("LSTM mask crop cannot truncate valid content")
    if np.any(missing & ~content[None]) or np.any(observed & ~content[None]):
        raise ValueError("Support outside content")
    support = missing & observed & content[None]
    return {"text": support[0, 1:].copy(),
            "audio": shift_left(support[1])[:audio_steps],
            "vision": shift_left(support[2])[:vision_steps]}


class MissingDataset:
    """Map-style Dataset compatible with the baseline prediction batch keys.

    Paired non-complete views require a panel to enforce its shared eligible
    cohort. Only train/main_mixed may explicitly include the original reference.
    Simulator masks/support are available via supervision(), not __getitem__.
    """
    def __init__(self, artifact_dir=DEFAULT_OUTPUT, source_path=None, split="train",
                 family="main_mixed", condition_id="mixed_v1_s0", panel=None,
                 include_reference=False):
        self.root = Path(artifact_dir).resolve()
        manifest = read_json(self.root / "manifest.json")
        acceptance_path = self.root / "acceptance.json"
        if manifest.get("status") != "accepted" or not acceptance_path.is_file():
            raise ValueError("This data version has not passed full acceptance")
        acceptance = read_json(acceptance_path)
        if acceptance.get("passed") is not True or acceptance.get("manifest_fingerprint") != json_fingerprint(manifest):
            raise ValueError("Acceptance record does not match this manifest")
        if code_provenance() != manifest["code_sha256"] or tokenizer_info() != manifest["tokenizer"]:
            raise ValueError("Construction/reader code or tokenizer changed; verify a new version")
        verify_inventory(self.root, manifest)
        if include_reference and (split != "train" or family != "main_mixed"):
            raise ValueError("Original reference is only exposed for train/main_mixed")
        entries = [entry for entry in manifest["views"] if
                   (entry["split"], entry["evaluation_family"], entry["condition_id"]) == (split, family, condition_id)]
        if len(entries) != 1:
            raise ValueError("Unknown or ambiguous split/family/condition")
        entry = entries[0]
        path = Path(source_path or manifest["source"]["path"])
        if sha256_file(path) != manifest["source"]["sha256"]:
            raise ValueError("Reader source fingerprint mismatch")
        archive = read_pickle(path)
        # Retain only the requested split and required fields; no raw_text or
        # cached continuous text enters the item/prediction interface.
        self._fields = {key: archive[split][key] for key in
                        ("text_bert", "audio", "vision", "id", "regression_labels")}
        del archive
        self.unk_id = manifest["tokenizer"]["unk_id"]
        self._source, _ = source_metadata(self._fields, split, self.unk_id)
        if array_fingerprint(np.asarray(self._fields["regression_labels"])) != manifest["source"]["label_fingerprints"][split]:
            raise ValueError("Labels differ from frozen source")
        with np.load(self.root / "source" / f"{split}.npz", allow_pickle=False) as frozen:
            for key, value in self._source.items():
                if not np.array_equal(value, frozen[key]):
                    raise ValueError(f"Source identity/observability mismatch: {key}")
        with np.load(safe_artifact_path(self.root, entry["mask_file"]), allow_pickle=False) as saved:
            self._missing, self._applicable = saved["missing"].copy(), saved["applicable"].copy()
        if panel is not None:
            panels = read_json(self.root / "panels" / f"{split}.json")
            if family != "paired" or panel not in panels or condition_id not in panels[panel]["condition_ids"]:
                raise ValueError("Condition is not part of the requested paired panel")
            self.indices = np.asarray(panels[panel]["indices"], dtype=np.int64)
        else:
            if family == "paired" and condition_id != "complete":
                raise ValueError("Select a panel explicitly to use the common paired cohort")
            self.indices = np.flatnonzero(self._applicable)
        if not self._applicable[self.indices].all():
            raise ValueError("Panel selected an inapplicable sample")
        self.split, self.family, self.condition_id = split, family, condition_id
        self.include_reference = include_reference

    def __len__(self):
        return len(self.indices)

    def numpy_item(self, position: int) -> dict:
        index = int(self.indices[position])
        fields = self._fields
        values = (fields["text_bert"][index], fields["audio"][index], fields["vision"][index])
        result = apply_missing(*values, self._missing[index], unk_id=self.unk_id)
        result.update(labels=np.float32(np.asarray(fields["regression_labels"][index]).item()),
                      id=str(fields["id"][index]), index=index)
        if self.include_reference:
            result["reference"] = apply_missing(*values, np.zeros((3, 50), dtype=bool), unk_id=self.unk_id)
        return result

    def __getitem__(self, position: int) -> dict:
        import torch

        def tensorize(value):
            if isinstance(value, dict):
                return {key: tensorize(item) for key, item in value.items()}
            if isinstance(value, (np.ndarray, np.generic)):
                return torch.from_numpy(np.asarray(value))
            return value
        return tensorize(self.numpy_item(position))

    def supervision(self, position: int, *, audio_steps=50, vision_steps=50) -> dict:
        if self.split != "train" or self.family != "main_mixed":
            raise ValueError("Reconstruction supervision is only exposed for train/main_mixed")
        index = int(self.indices[position])
        support = reconstruction_support(self._missing[index], self._source["observed"][index],
                                         self._source["content"][index],
                                         audio_steps=audio_steps, vision_steps=vision_steps)
        return {"reconstruction": support, "missing_raw": self._missing[index].copy(),
                "original_observed_raw": self._source["observed"][index].copy(),
                "content_raw": self._source["content"][index].copy(), "index": index}
