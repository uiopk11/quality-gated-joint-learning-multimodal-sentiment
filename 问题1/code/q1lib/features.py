from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import opensmile
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from .data import WordInterval, finite_or_raise


@dataclass
class FrameFeatures:
    values: np.ndarray
    times: np.ndarray
    names: list[str]
    valid: np.ndarray
    starts: np.ndarray | None = None
    ends: np.ndarray | None = None


@dataclass
class TokenFeatures:
    values: np.ndarray
    tokens: list[str]
    word_ids: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    attention_mask: np.ndarray
    model_truncated: bool


def _seconds(values: pd.Index) -> np.ndarray:
    if isinstance(values, pd.TimedeltaIndex):
        return values.total_seconds().to_numpy(dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


class BertFeatureExtractor:
    def __init__(self, model_path: Path, hidden_layer: int, device: str, model_max_tokens: int):
        self.model_path = model_path
        self.hidden_layer = int(hidden_layer)
        self.model_max_tokens = int(model_max_tokens)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(self.device).eval()

    def extract(
        self,
        words: Sequence[str],
        intervals: Sequence[WordInterval | None] | None = None,
    ) -> TokenFeatures:
        words = list(words)
        if intervals is not None and len(intervals) != len(words):
            raise ValueError("BERT words and intervals must have the same length")
        encoded = self.tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=True,
            truncation=True,
            max_length=self.model_max_tokens,
            return_tensors="pt",
            return_attention_mask=True,
        )
        word_ids_list = encoded.word_ids(batch_index=0)
        model_truncated = bool(word_ids_list and max((x for x in word_ids_list if x is not None), default=-1) < len(words) - 1)
        model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = self.model(**model_inputs, output_hidden_states=True)
        hidden_states = output.hidden_states
        if hidden_states is None:
            raise RuntimeError("BERT did not return hidden states")
        values = hidden_states[self.hidden_layer][0].detach().cpu().numpy().astype(np.float32)
        token_ids = encoded["input_ids"][0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        word_ids = np.asarray([-1 if item is None else item for item in word_ids_list], dtype=np.int32)
        starts = np.full(len(tokens), -1.0, dtype=np.float64)
        ends = np.full(len(tokens), -1.0, dtype=np.float64)
        for index, word_id in enumerate(word_ids):
            if word_id >= 0 and intervals is not None and intervals[word_id] is not None:
                starts[index] = intervals[word_id].start
                ends[index] = intervals[word_id].end
        attention = encoded["attention_mask"][0].cpu().numpy().astype(np.uint8)
        finite_or_raise("BERT token features", values)
        return TokenFeatures(values, tokens, word_ids, starts, ends, attention, model_truncated)


class OpenSmileFeatureExtractor:
    def __init__(
        self,
        feature_set: str,
        feature_level: str,
        delta_fields: Sequence[str],
        target_dimension: int,
    ):
        try:
            set_value = getattr(opensmile.FeatureSet, feature_set)
            level_value = getattr(opensmile.FeatureLevel, feature_level)
        except AttributeError as exc:
            raise ValueError(f"invalid OpenSMILE configuration: {feature_set}/{feature_level}") from exc
        self.feature_set_name = feature_set
        self.feature_level_name = feature_level
        self.delta_fields = list(delta_fields)
        self.target_dimension = int(target_dimension)
        self.smile = opensmile.Smile(feature_set=set_value, feature_level=level_value)
        missing = sorted(set(self.delta_fields) - set(self.smile.feature_names))
        if missing:
            raise ValueError(f"OpenSMILE delta fields are unavailable: {missing}")
        if len(self.smile.feature_names) + len(self.delta_fields) != self.target_dimension:
            raise ValueError(
                f"OpenSMILE dimension mismatch: {len(self.smile.feature_names)} base + "
                f"{len(self.delta_fields)} deltas != {self.target_dimension}"
            )

    def extract(self, wav_path: Path) -> FrameFeatures:
        frame = self.smile.process_file(str(wav_path)).reset_index()
        if frame.empty:
            raise ValueError(f"OpenSMILE returned no frames: {wav_path}")
        if not {"start", "end"}.issubset(frame.columns):
            raise ValueError("OpenSMILE output does not contain start/end timestamps")
        starts = pd.to_timedelta(frame.pop("start")).dt.total_seconds().to_numpy(dtype=np.float64)
        ends = pd.to_timedelta(frame.pop("end")).dt.total_seconds().to_numpy(dtype=np.float64)
        frame = frame.drop(columns=[column for column in ("file",) if column in frame.columns])
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        values = numeric.to_numpy(dtype=np.float32)
        valid = np.isfinite(values).all(axis=1)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        times = (starts + ends) / 2.0
        delta_values: list[np.ndarray] = []
        delta_names: list[str] = []
        for field in self.delta_fields:
            column = values[:, numeric.columns.get_loc(field)].astype(np.float64)
            if len(column) <= 1:
                derivative = np.zeros_like(column)
            else:
                safe_times = np.maximum.accumulate(times)
                if np.any(np.diff(safe_times) <= 0):
                    derivative = np.gradient(column)
                else:
                    derivative = np.gradient(column, safe_times)
            delta_values.append(derivative.astype(np.float32))
            delta_names.append(f"{field}__delta_per_second")
        if delta_values:
            values = np.column_stack([values, *delta_values]).astype(np.float32)
        names = list(numeric.columns) + delta_names
        if values.shape[1] != self.target_dimension:
            raise ValueError(
                f"OpenSMILE produced {values.shape[1]} fields, expected {self.target_dimension}"
            )
        finite_or_raise("OpenSMILE frame features", values)
        return FrameFeatures(values, times, names, valid.astype(np.uint8), starts, ends)


def read_openface_features(csv_path: Path, config: dict[str, Any]) -> FrameFeatures:
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"OpenFace returned no frames: {csv_path}")
    frame.columns = [str(column).strip() for column in frame.columns]
    if "timestamp" not in frame.columns:
        raise ValueError(f"OpenFace CSV has no timestamp column: {csv_path}")
    names = [str(item) for item in config["feature_fields"]]
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"OpenFace CSV is missing configured fields: {missing}")
    if len(names) != int(config["target_dimension"]):
        raise ValueError("OpenFace configured dimension does not match selected fields")
    numeric = frame[names].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float32)
    times = pd.to_numeric(frame["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
    confidence = (
        pd.to_numeric(frame["confidence"], errors="coerce").to_numpy(dtype=np.float64)
        if "confidence" in frame.columns
        else np.ones(len(frame), dtype=np.float64)
    )
    success = (
        pd.to_numeric(frame["success"], errors="coerce").fillna(0).to_numpy(dtype=np.float64) > 0
        if "success" in frame.columns
        else np.ones(len(frame), dtype=bool)
    )
    valid = (
        success
        & (confidence >= float(config["confidence_threshold"]))
        & np.isfinite(times)
        & np.isfinite(values).all(axis=1)
    )
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values[~valid] = 0.0
    finite_or_raise("OpenFace frame features", values)
    return FrameFeatures(values, times, names, valid.astype(np.uint8))


def _pool(values: np.ndarray, statistics: Sequence[str]) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for statistic in statistics:
        if statistic == "mean":
            outputs.append(values.mean(axis=0))
        elif statistic == "std":
            outputs.append(values.std(axis=0, ddof=0))
        elif statistic == "max":
            outputs.append(values.max(axis=0))
        elif statistic == "min":
            outputs.append(values.min(axis=0))
        elif statistic == "median":
            outputs.append(np.median(values, axis=0))
        else:
            raise ValueError(f"unsupported pooling statistic: {statistic}")
    return np.concatenate(outputs).astype(np.float32)


def pooled_feature_names(frame_names: Sequence[str], statistics: Sequence[str]) -> list[str]:
    return [f"{name}__{statistic}" for statistic in statistics for name in frame_names]


def aggregate_frames_by_words(
    frames: FrameFeatures,
    intervals: Sequence[WordInterval],
    statistics: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    dimension = len(frames.names) * len(statistics)
    features = np.zeros((len(intervals), dimension), dtype=np.float32)
    mask = np.zeros(len(intervals), dtype=np.uint8)
    mappings: list[np.ndarray] = []
    for word_index, interval in enumerate(intervals):
        indices = np.flatnonzero(
            (frames.times >= interval.start)
            & (frames.times <= interval.end)
            & frames.valid.astype(bool)
        )
        mappings.append(indices.astype(np.int32))
        if len(indices):
            features[word_index] = _pool(frames.values[indices], statistics)
            mask[word_index] = 1
    finite_or_raise("word pooled features", features)
    return features, mask, mappings


def aggregate_frames_by_optional_words(
    frames: FrameFeatures,
    intervals: Sequence[WordInterval | None],
    statistics: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    dimension = len(frames.names) * len(statistics)
    features = np.zeros((len(intervals), dimension), dtype=np.float32)
    mask = np.zeros(len(intervals), dtype=np.uint8)
    mappings: list[np.ndarray] = []
    for word_index, interval in enumerate(intervals):
        if interval is None:
            mappings.append(np.asarray([], dtype=np.int32))
            continue
        indices = np.flatnonzero(
            (frames.times >= interval.start)
            & (frames.times <= interval.end)
            & frames.valid.astype(bool)
        ).astype(np.int32)
        mappings.append(indices)
        if len(indices):
            features[word_index] = _pool(frames.values[indices], statistics)
            mask[word_index] = 1
    finite_or_raise("optional word pooled features", features)
    return features, mask, mappings


def expand_words_to_tokens(
    word_features: np.ndarray,
    word_mask: np.ndarray,
    word_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    token_features = np.zeros((len(word_ids), word_features.shape[1]), dtype=np.float32)
    token_mask = np.zeros(len(word_ids), dtype=np.uint8)
    valid = word_ids >= 0
    if np.any(valid):
        token_features[valid] = word_features[word_ids[valid]]
        token_mask[valid] = word_mask[word_ids[valid]]
    return token_features, token_mask


def resample_frames_to_time_bins(
    frames: FrameFeatures,
    duration: float,
    length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool a frame sequence onto a fixed real-time grid without using text."""
    if duration <= 0 or length <= 0:
        raise ValueError("duration and length must be positive")
    output = np.zeros((length, frames.values.shape[1]), dtype=np.float32)
    mask = np.zeros(length, dtype=np.uint8)
    valid = frames.valid.astype(bool) & np.isfinite(frames.times)
    if not np.any(valid):
        return output, mask
    clipped = np.clip(frames.times[valid], 0.0, np.nextafter(duration, 0.0))
    bins = np.minimum((clipped / duration * length).astype(np.int64), length - 1)
    valid_values = frames.values[valid]
    for bin_index in np.unique(bins):
        selected = valid_values[bins == bin_index]
        if len(selected):
            output[bin_index] = selected.mean(axis=0)
            mask[bin_index] = 1
    finite_or_raise("time-bin features", output)
    return output, mask
