"""Verified, cached access to the immutable missing-data artifact.

The frozen construction reader remains authoritative for validating each raw
split and applying masks.  A store loads a raw split at most once, so evaluating
nineteen conditions does not repeatedly deserialize the large source pickle.
No mask is resampled, and evaluation items never expose reconstruction truth.
"""

from __future__ import annotations

import copy
from pathlib import Path
from src.runtime import resolve_path, source_path
from src.compat.provenance import validate_frozen_code, validate_tokenizer
from typing import Any

import numpy as np

from src.data.construction.common import (
    DEFAULT_OUTPUT,
    array_fingerprint,
    code_provenance,
    json_fingerprint,
    read_json,
    safe_artifact_path,
    sha256_file,
    tokenizer_info,
    verify_inventory,
)
from src.data.construction.reader import MissingDataset
from src.data.retirement import retired_files


class _ViewDataset(MissingDataset):
    """Borrow a verified split without re-running the frozen constructor."""

    def __init__(self, anchor: MissingDataset, entry: dict, missing: np.ndarray,
                 applicable: np.ndarray, indices: np.ndarray, *, training: bool,
                 panel: str | None, cohort: str):
        self.root = anchor.root
        self._fields = anchor._fields
        self._source = anchor._source
        self.unk_id = anchor.unk_id
        self._missing, self._applicable = missing, applicable
        self.indices = np.array(indices, dtype=np.int64, copy=True)
        self.indices.flags.writeable = False
        self.split = entry["split"]
        self.family = entry["evaluation_family"]
        self.condition_id = entry["condition_id"]
        self.training = bool(training)
        self.include_reference = bool(training)
        self.panel, self.cohort = panel, cohort
        self.view_identity = {
            "split": self.split, "evaluation_family": self.family,
            "condition_id": self.condition_id, "panel": panel,
            "cohort": cohort, "sample_count": len(self.indices),
            "mask_array_sha256": entry["mask_array_sha256"],
        }

    def numpy_item(self, position: int) -> dict:
        result = super().numpy_item(position)
        if self.training:
            # Reuse the accepted support definition exactly: artificial mask
            # intersected with content and the original observed-row proxy.
            # Simulator raw masks are intentionally not included in a batch.
            result["reconstruction"] = super().supervision(position)["reconstruction"]
        return result

    def supervision(self, position: int, *, audio_steps=50, vision_steps=50) -> dict:
        if not self.training:
            raise ValueError("Evaluation datasets do not expose reconstruction supervision")
        return super().supervision(position, audio_steps=audio_steps, vision_steps=vision_steps)


class MixDataStore:
    """Access one accepted data version, retaining a single raw copy per split.

    ``dataset`` enforces panel intersections for non-complete paired views.
    ``prediction_dataset`` is the explicitly named evaluation-cache exception:
    it exposes a condition's own applicable cohort for one canonical inference
    pass. Scoring must then select the panel's common original indices.

    ``binding`` preserves the original accepted data identity and historical
    eligible indices for checkpoint provenance, never labels. Retired views in
    that identity are unavailable through the dataset interfaces.
    Labels/IDs for a requested split are available separately via ``cohort``.
    DataLoaders should use ``drop_last=False``; this class retains every index.
    """

    def __init__(self, artifact_dir=DEFAULT_OUTPUT, source_path=None):
        self.root = resolve_path(artifact_dir)
        manifest_path = self.root / "manifest.json"
        acceptance_path = self.root / "acceptance.json"
        manifest = read_json(manifest_path)
        if manifest.get("status") != "accepted" or not acceptance_path.is_file():
            raise ValueError("This data version has not passed full acceptance")
        acceptance = read_json(acceptance_path)
        canonical_fingerprint = json_fingerprint(manifest)
        if acceptance.get("passed") is not True or acceptance.get("manifest_fingerprint") != canonical_fingerprint:
            raise ValueError("Acceptance record does not match this manifest")
        validate_frozen_code(manifest["code_sha256"])
        validate_tokenizer(manifest["tokenizer"], tokenizer_info())
        verify_inventory(self.root, manifest)
        self.source_path = resolve_path(source_path or manifest["source"]["path"]).resolve()
        if sha256_file(self.source_path) != manifest["source"]["sha256"]:
            raise ValueError("Reader source fingerprint mismatch")
        self._manifest = copy.deepcopy(manifest)
        self._manifest_file_sha256 = sha256_file(manifest_path)
        self._acceptance_file_sha256 = sha256_file(acceptance_path)
        self._anchors: dict[str, MissingDataset] = {}
        self._entries: dict[tuple[str, str, str], dict] = {}
        self._masks: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
        self._panels: dict[str, dict] = {}
        mask_views = []
        condition_indices: dict[str, dict] = {}
        retired = retired_files(self.root, manifest)
        self._retired_views: set[tuple[str, str, str]] = set()
        seen_keys = set()

        for entry in manifest["views"]:
            key = (entry["split"], entry["evaluation_family"], entry["condition_id"])
            if key in seen_keys:
                raise ValueError("Duplicate split/family/condition in manifest")
            seen_keys.add(key)
            if entry["mask_file"] in retired:
                self._retired_views.add(key)
                # The retirement validator binds this to the original manifest
                # and proves that both historical cohorts were all-applicable.
                # Preserve checkpoint identity without loading a deleted mask.
                applicable_indices = list(range(entry["applicable_count"]))
                applicable_count = entry["applicable_count"]
            else:
                self._entries[key] = copy.deepcopy(entry)
                missing, applicable = self._load_verified_masks(entry)
                self._masks[key] = missing, applicable
                applicable_indices = np.flatnonzero(applicable).astype(int).tolist()
                applicable_count = int(applicable.sum())
            split, family, condition = key
            condition_indices.setdefault(split, {})[f"{family}/{condition}"] = applicable_indices
            mask_views.append({
                "split": split, "evaluation_family": family, "condition_id": condition,
                "mask_array_sha256": entry["mask_array_sha256"],
                "mask_file_sha256": manifest["files"][entry["mask_file"]]["sha256"],
                "applicable_count": applicable_count,
            })
        for split in ("train", "valid", "test"):
            relative = f"panels/{split}.json"
            if relative in manifest["files"]:
                self._verify_file_now(relative)
                self._panels[split] = read_json(safe_artifact_path(self.root, relative))
            else:
                self._panels[split] = {}
        self._validate_panels()
        self._binding = {
            "dataset_version": manifest["dataset_version"],
            "mask_manifest_sha256": self._manifest_file_sha256,
            "manifest_fingerprint": canonical_fingerprint,
            "data_sha256": manifest["source"]["sha256"],
            "recipe_sha256": manifest["recipe_sha256"],
            "exploratory_recipe_sha256": manifest.get("exploratory_recipe_sha256"),
            "tokenizer_vocab_sha256": manifest["tokenizer"]["vocab_sha256"],
            "label_fingerprints": copy.deepcopy(manifest["source"]["label_fingerprints"]),
            "mask_views": mask_views, "condition_indices": condition_indices,
        }

    def _verify_file_now(self, relative: str) -> None:
        record = self._manifest["files"].get(relative)
        if record is None:
            raise ValueError(f"Artifact file is not covered by the accepted inventory: {relative}")
        path = safe_artifact_path(self.root, relative)
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Artifact integrity failure: {relative}")

    def _load_verified_masks(self, entry: dict) -> tuple[np.ndarray, np.ndarray]:
        self._verify_file_now(entry["mask_file"])
        with np.load(safe_artifact_path(self.root, entry["mask_file"]), allow_pickle=False) as archive:
            missing, applicable = archive["missing"].copy(), archive["applicable"].copy()
        expected_shape = tuple(entry["shape"])
        if missing.dtype != np.bool_ or missing.shape != expected_shape or len(expected_shape) != 3 or expected_shape[1:] != (3, 50):
            raise ValueError("Saved missing-mask shape or dtype disagrees with the manifest")
        if applicable.dtype != np.bool_ or applicable.shape != (len(missing),):
            raise ValueError("Saved applicability shape or dtype is invalid")
        if int(applicable.sum()) != entry["applicable_count"] or array_fingerprint(missing) != entry["mask_array_sha256"]:
            raise ValueError("Saved mask or applicability fingerprint disagrees with the manifest")
        missing.flags.writeable = False
        applicable.flags.writeable = False
        return missing, applicable

    def _validate_panels(self) -> None:
        for split, panels in self._panels.items():
            for name, panel in panels.items():
                conditions = panel["condition_ids"]
                if not conditions or len(conditions) != len(set(conditions)):
                    raise ValueError(f"Invalid panel conditions: {split}/{name}")
                keys = [(split, "paired", condition) for condition in conditions]
                if any(key not in self._masks for key in keys):
                    raise ValueError(f"Unknown panel condition: {split}/{name}")
                common = np.logical_and.reduce([self._masks[key][1] for key in keys])
                if panel["indices"] != np.flatnonzero(common).astype(int).tolist():
                    raise ValueError(f"Panel is not the common applicable cohort: {split}/{name}")

    def _anchor(self, split: str) -> MissingDataset:
        if split not in ("train", "valid", "test"):
            raise ValueError("Unknown split")
        if split not in self._anchors:
            if (sha256_file(self.root / "manifest.json") != self._manifest_file_sha256
                    or sha256_file(self.root / "acceptance.json") != self._acceptance_file_sha256):
                raise ValueError("Artifact manifest or acceptance changed after store verification")
            # The reader revalidates acceptance, provenance, the retained
            # inventory, the source file, labels, IDs and observability.
            anchor = MissingDataset(artifact_dir=self.root, source_path=self.source_path,
                                    split=split, family="main_mixed", condition_id="mixed_v1_s0")
            for value in (*anchor._fields.values(), *anchor._source.values()):
                if isinstance(value, np.ndarray):
                    value.flags.writeable = False
            expected = len(anchor._source["ids"])
            for key, (missing, _) in self._masks.items():
                if key[0] != split:
                    continue
                if len(missing) != expected or np.any(missing & ~anchor._source["content"][:, None, :]):
                    raise ValueError("Cached mask does not match the verified split content")
            for name, panel in self._panels[split].items():
                expected_ids = [str(anchor._source["ids"][index]) for index in panel["indices"]]
                if panel["ids"] != expected_ids:
                    raise ValueError(f"Panel identity differs from the source: {split}/{name}")
            self._anchors[split] = anchor
        return self._anchors[split]

    def _entry(self, split: str, family: str, condition_id: str) -> tuple[tuple, dict]:
        key = (split, family, condition_id)
        if key in self._retired_views:
            raise ValueError("The exploratory_a3like research view has been retired and is unavailable; "
                             "select a retained main_mixed or paired view.")
        if key not in self._entries:
            raise ValueError("Unknown split/family/condition")
        return key, self._entries[key]

    def _dataset(self, split: str, family: str, condition_id: str, *, panel: str | None,
                 training: bool, prediction_cache: bool) -> _ViewDataset:
        key, entry = self._entry(split, family, condition_id)
        if training and (split != "train" or family != "main_mixed" or condition_id != "mixed_v1_s0"):
            raise ValueError("Training reference/support is only allowed for train/main_mixed")
        missing, applicable = self._masks[key]
        if panel is not None:
            panels = self._panels[split]
            if family != "paired" or panel not in panels or condition_id not in panels[panel]["condition_ids"]:
                raise ValueError("Condition is not part of the requested paired panel")
            indices = np.asarray(panels[panel]["indices"], dtype=np.int64)
            cohort = "panel_intersection"
        else:
            if family == "paired" and condition_id != "complete" and not prediction_cache:
                raise ValueError("Select a panel explicitly to use the common paired cohort")
            indices = np.flatnonzero(applicable)
            cohort = "condition_applicable" if prediction_cache else "view_applicable"
        if not applicable[indices].all():
            raise ValueError("Dataset cohort contains an inapplicable sample")
        return _ViewDataset(self._anchor(split), entry, missing, applicable, indices,
                            training=training, panel=panel, cohort=cohort)

    def dataset(self, split: str, family="main_mixed", condition_id="mixed_v1_s0",
                panel=None, training=False) -> _ViewDataset:
        return self._dataset(split, family, condition_id, panel=panel,
                             training=bool(training), prediction_cache=False)

    def prediction_dataset(self, split: str, condition_id: str, family="paired") -> _ViewDataset:
        """Canonical inference cohort; panel-specific scoring must filter indices."""
        if split not in ("valid", "test"):
            raise ValueError("Canonical evaluation prediction is only available for valid/test")
        return self._dataset(split, family, condition_id, panel=None,
                             training=False, prediction_cache=True)

    def panels(self, split: str) -> dict:
        if split not in self._panels:
            raise ValueError("Unknown split")
        return copy.deepcopy(self._panels[split])

    def cohort(self, split: str) -> list[dict[str, Any]]:
        """Original identity/labels in source order, loaded only for this split."""
        anchor = self._anchor(split)
        fields = anchor._fields
        return [{"index": index, "id": str(fields["id"][index]),
                 "y_true": float(np.float32(np.asarray(fields["regression_labels"][index]).item()))}
                for index in range(len(anchor._source["ids"]))]

    def binding(self) -> dict:
        """Return original label-free checkpoint provenance, not available views."""
        return copy.deepcopy(self._binding)
