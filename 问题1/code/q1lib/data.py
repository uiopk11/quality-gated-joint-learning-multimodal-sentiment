from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    video_id: str
    clip_id: str
    text: str
    label: float
    annotation: str
    video_path: str


@dataclass(frozen=True)
class WordInterval:
    word_index: int
    word: str
    start: float
    end: float


def normalize_clip_id(value: Any) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def make_sample_id(video_id: str, clip_id: str) -> str:
    safe_video = re.sub(r"[^A-Za-z0-9_-]", "_", video_id.strip())
    safe_clip = re.sub(r"[^A-Za-z0-9_-]", "_", clip_id.strip())
    return f"{safe_video}__{safe_clip}"


def normalize_transcript(text: str) -> tuple[str, list[str]]:
    text = str(text).replace("’", "'").replace("‘", "'")
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?", text)
    normalized = " ".join(words)
    return normalized, words


def load_manifest(label_file: Path, data_root: Path) -> list[SampleRecord]:
    frame = pd.read_excel(
        label_file,
        dtype={"video_id": str, "clip_id": str, "text": str, "annotation": str},
    )
    required = {"video_id", "clip_id", "text", "label", "annotation"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"label file is missing columns: {sorted(missing)}")
    records: list[SampleRecord] = []
    seen: set[str] = set()
    for row in frame.to_dict(orient="records"):
        video_id = str(row["video_id"]).strip()
        clip_id = normalize_clip_id(row["clip_id"])
        sample_id = make_sample_id(video_id, clip_id)
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id in labels: {sample_id}")
        seen.add(sample_id)
        video_path = data_root / video_id / f"{clip_id}.mp4"
        if pd.isna(row["text"]) or not str(row["text"]).strip():
            raise ValueError(f"empty text for sample {sample_id}")
        records.append(
            SampleRecord(
                sample_id=sample_id,
                video_id=video_id,
                clip_id=clip_id,
                text=str(row["text"]),
                label=float(row["label"]),
                annotation=str(row["annotation"]),
                video_path=str(video_path.resolve()),
            )
        )
    return records


def validate_manifest(
    records: Iterable[SampleRecord],
    data_root: Path,
    expected_count: int | None = 100,
) -> pd.DataFrame:
    rows = []
    for record in records:
        path = Path(record.video_path)
        normalized_text, words = normalize_transcript(record.text)
        rows.append(
            {
                **asdict(record),
                "video_exists": path.is_file(),
                "text_exists": bool(str(record.text).strip()),
                "normalized_text": normalized_text,
                "normalized_word_count": len(words),
            }
        )
    frame = pd.DataFrame(rows)
    if expected_count is not None and len(frame) != expected_count:
        raise ValueError(f"expected {expected_count} label rows, found {len(frame)}")
    missing = frame.loc[~frame["video_exists"], ["sample_id", "video_path"]]
    if not missing.empty:
        raise FileNotFoundError("videos missing for labels:\n" + missing.to_string(index=False))
    expected_paths = {str(Path(item.video_path).resolve()).casefold() for item in records}
    actual_paths = {str(path.resolve()).casefold() for path in Path(data_root).rglob("*.mp4")}
    extra_paths = sorted(actual_paths - expected_paths)
    if extra_paths:
        raise ValueError("videos without label rows:\n" + "\n".join(extra_paths))
    return frame


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def finite_or_raise(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        bad = int(np.size(array) - np.isfinite(array).sum())
        raise ValueError(f"{name} contains {bad} NaN/inf values")


def pad_or_truncate(
    array: np.ndarray,
    length: int,
    padding_value: float = 0.0,
    preserve_last: bool = False,
) -> tuple[np.ndarray, bool]:
    if array.shape[0] > length:
        if preserve_last and length >= 2:
            output = np.concatenate([array[: length - 1], array[-1:]], axis=0)
        else:
            output = array[:length]
        return output, True
    shape = (length,) + array.shape[1:]
    output = np.full(shape, padding_value, dtype=array.dtype)
    output[: array.shape[0]] = array
    return output, False
