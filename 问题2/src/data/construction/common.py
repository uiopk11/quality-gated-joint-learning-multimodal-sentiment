"""Source validation, canonical fingerprints and artifact IO.

Code assisted by OpenAI Codex. This module never changes the source pickle.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from src.runtime import resolve_path, source_path

import numpy as np

from src.data.aligned import EXPECTED_COUNTS, read_pickle, summarize_raw_split
from src.data.retirement import retired_files

from src.runtime import ROOT as PROJECT
DEFAULT_SOURCE = resolve_path("data/aligned_50.pkl")
DEFAULT_OUTPUT = resolve_path("dataset_construction/artifacts/mix_v1")
EXPECTED_SOURCE_SHA256 = "66e867aa74bc70a844e806e5571e371c9abb4a35f9e2887ce9b4d97ff2cb8fcd"
MODALITIES = ("T", "A", "V")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def json_fingerprint(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def array_fingerprint(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(canonical_json({"dtype": array.dtype.str, "shape": list(array.shape)}).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def code_provenance() -> dict:
    from src.compat.provenance import current_provenance
    return current_provenance("construction", PROJECT)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def tokenizer_info() -> dict:
    vocab_path = resolve_path("pretrained/bert-base-uncased/vocab.txt")
    vocab = vocab_path.read_text(encoding="utf-8").splitlines()
    ids = {name: vocab.index(token) for name, token in
           (("unk_id", "[UNK]"), ("cls_id", "[CLS]"), ("sep_id", "[SEP]"), ("pad_id", "[PAD]"))}
    if (ids["cls_id"], ids["sep_id"], ids["pad_id"]) != (101, 102, 0):
        raise ValueError("Tokenizer special IDs disagree with the existing baseline")
    return {**ids, "vocab_path": str(vocab_path), "vocab_sha256": sha256_file(vocab_path)}


def source_metadata(fields: dict, split: str, unk_id: int) -> tuple[dict, dict]:
    summary = summarize_raw_split(fields, split)
    text = np.asarray(fields["text_bert"])
    lengths = text[:, 1].sum(axis=1).astype(np.int64) - 2
    positions = np.arange(50)[None, :]
    content = (positions >= 1) & (positions <= lengths[:, None])
    observed = np.zeros((len(lengths), 3, 50), dtype=bool)
    observed[:, 0] = content & (text[:, 0] != unk_id)
    for m, name in enumerate(("audio", "vision"), 1):
        features = np.asarray(fields[name])
        # Frozen source is finite. Refuse a changed/nonfinite source rather than
        # invent reliable reconstruction targets after numerical replacement.
        if not np.isfinite(features).all():
            raise ValueError(f"{split}/{name}: construction requires finite source features")
        observed[:, m] = content & np.any(features != 0, axis=-1)
    ids = np.asarray([str(item) for item in fields["id"]])
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"{split}: duplicate IDs")
    return {"ids": ids, "lengths": lengths, "content": content, "observed": observed}, summary


def load_source(path: Path, expected_sha256: str = EXPECTED_SOURCE_SHA256):
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"Source SHA256 mismatch: {actual}; expected {expected_sha256}")
    raw = read_pickle(path)
    tokenizer = tokenizer_info()
    sources, summaries = {}, {}
    for split in EXPECTED_COUNTS:
        sources[split], summaries[split] = source_metadata(raw[split], split, tokenizer["unk_id"])
    return raw, sources, summaries, tokenizer


def safe_artifact_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes its directory")
    return path


def verify_inventory(root: Path, manifest: dict) -> None:
    retired = retired_files(root, manifest)
    for relative, record in manifest["files"].items():
        if relative in retired:
            continue
        path = safe_artifact_path(root, relative)
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Artifact integrity failure: {relative}")


def load_saved_split(root: Path, manifest: dict, split: str) -> dict:
    if retired_files(root, manifest):
        raise ValueError("The exploratory research views have been retired; historical construction "
                         "acceptance cannot be regenerated. Use run.py verify to validate retained data.")
    views = []
    for entry in manifest["views"]:
        if entry["split"] != split:
            continue
        with np.load(safe_artifact_path(root, entry["mask_file"]), allow_pickle=False) as archive:
            masks, applicable = archive["missing"].copy(), archive["applicable"].copy()
        metadata = [json.loads(line) for line in safe_artifact_path(root, entry["metadata_file"]).read_text(encoding="utf-8").splitlines()]
        views.append({"evaluation_family": entry["evaluation_family"], "condition_id": entry["condition_id"],
                      "masks": masks, "applicable": applicable, "metadata": metadata})
    return {"views": views, "panels": read_json(root / "panels" / f"{split}.json")}
