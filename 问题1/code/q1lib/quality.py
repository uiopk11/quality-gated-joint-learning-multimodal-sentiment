from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf
import torch

from .data import WordInterval, normalize_transcript
from .number_normalization import ComparisonUnit, build_comparison_units


@dataclass(frozen=True)
class ASRWord:
    index: int
    word: str
    start: float
    end: float


@dataclass(frozen=True)
class QualityResult:
    asr_text: str
    asr_words: list[ASRWord]
    detected_language: str
    peak_dbfs: float
    rms_dbfs: float
    signal_activity_ratio: float
    speech_present: bool
    exact_match_count: int
    reliable_match_count: int
    official_word_recall: float
    asr_word_precision: float
    word_f1: float
    character_similarity: float
    consistency_score: float
    alignment_type: str
    official_to_asr: list[int]
    official_match_mask: list[int]
    official_asr_starts: list[float | None]
    official_asr_ends: list[float | None]
    official_comparison_unit_count: int
    asr_comparison_unit_count: int
    numeric_equivalence_count: int

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["asr_words"] = [asdict(item) for item in self.asr_words]
        return payload


def _dbfs(value: float) -> float:
    return float(20.0 * np.log10(max(value, 1e-12)))


def analyze_signal(
    wav_path: Path,
    silence_peak_dbfs: float,
    energy_margin_db: float,
) -> dict[str, float]:
    signal, sample_rate = sf.read(str(wav_path), always_2d=False, dtype="float32")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if len(signal) == 0:
        return {"peak_dbfs": -240.0, "rms_dbfs": -240.0, "signal_activity_ratio": 0.0}
    peak = float(np.max(np.abs(signal), initial=0.0))
    rms = float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))
    frame_length = max(1, int(round(0.030 * sample_rate)))
    hop_length = max(1, int(round(0.010 * sample_rate)))
    if len(signal) < frame_length:
        frame_rms = np.asarray([rms], dtype=np.float64)
    else:
        starts = np.arange(0, len(signal) - frame_length + 1, hop_length)
        frame_rms = np.asarray(
            [np.sqrt(np.mean(np.square(signal[s : s + frame_length], dtype=np.float64))) for s in starts]
        )
    frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-12))
    noise_floor = float(np.percentile(frame_db, 20)) if len(frame_db) else -240.0
    threshold = max(float(silence_peak_dbfs), noise_floor + float(energy_margin_db))
    activity = float(np.mean(frame_db >= threshold)) if len(frame_db) else 0.0
    return {"peak_dbfs": _dbfs(peak), "rms_dbfs": _dbfs(rms), "signal_activity_ratio": activity}


def _normalize_word(value: str) -> str:
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?", value.casefold())
    return words[0] if words else ""


def match_word_sequences(
    official_words: Sequence[str],
    asr_words: Sequence[ASRWord],
    minimum_contiguous_matches: int = 1,
) -> tuple[list[int], list[int], list[float | None], list[float | None], dict[str, float]]:
    official = [_normalize_word(item) for item in official_words]
    recognized = [_normalize_word(item.word) for item in asr_words]
    official_units = build_comparison_units(official)
    recognized_units = build_comparison_units(recognized)
    matcher = difflib.SequenceMatcher(
        a=[item.value for item in official_units],
        b=[item.value for item in recognized_units],
        autojunk=False,
    )
    blocks = matcher.get_matching_blocks()
    matched = sum(block.size for block in blocks)
    mapping = [-1] * len(official)
    starts: list[float | None] = [None] * len(official)
    ends: list[float | None] = [None] * len(official)
    numeric_equivalence_count = 0
    for block in blocks:
        if block.size < minimum_contiguous_matches:
            continue
        for offset in range(block.size):
            official_unit: ComparisonUnit = official_units[block.a + offset]
            recognized_unit: ComparisonUnit = recognized_units[block.b + offset]
            recognized_indices = list(recognized_unit.source_indices)
            if not recognized_indices:
                continue
            first_asr = recognized_indices[0]
            last_asr = recognized_indices[-1]
            if first_asr >= len(asr_words) or last_asr >= len(asr_words):
                continue
            start = float(asr_words[first_asr].start)
            end = float(asr_words[last_asr].end)
            official_indices = list(official_unit.source_indices)
            if end <= start or not official_indices:
                continue
            step = (end - start) / len(official_indices)
            for position, official_index in enumerate(official_indices):
                relative = min(position, len(recognized_indices) - 1)
                mapping[official_index] = recognized_indices[relative]
                starts[official_index] = start + position * step
                ends[official_index] = start + (position + 1) * step
            if (
                official_unit.is_number
                and recognized_unit.is_number
                and official_unit.surface != recognized_unit.surface
            ):
                numeric_equivalence_count += 1
    recall = matched / max(1, len(official_units))
    precision = matched / max(1, len(recognized_units))
    f1 = 2.0 * recall * precision / max(1e-12, recall + precision)
    official_text = " ".join(item.value for item in official_units)
    recognized_text = " ".join(item.value for item in recognized_units)
    character_similarity = difflib.SequenceMatcher(
        a=official_text, b=recognized_text, autojunk=False
    ).ratio()
    return mapping, [int(index >= 0) for index in mapping], starts, ends, {
        "exact_match_count": float(matched),
        "reliable_match_count": float(sum(index >= 0 for index in mapping)),
        "official_word_recall": float(recall),
        "asr_word_precision": float(precision),
        "word_f1": float(f1),
        "character_similarity": float(character_similarity),
        "official_comparison_unit_count": float(len(official_units)),
        "asr_comparison_unit_count": float(len(recognized_units)),
        "numeric_equivalence_count": float(numeric_equivalence_count),
    }


def intervals_from_asr_matches(
    official_words: Sequence[str],
    asr_words: Sequence[ASRWord],
    mapping: Sequence[int],
) -> tuple[list[WordInterval], list[int]]:
    intervals: list[WordInterval] = []
    official_indices: list[int] = []
    for official_index, asr_index in enumerate(mapping):
        if asr_index < 0:
            continue
        item = asr_words[asr_index]
        if item.end <= item.start:
            continue
        intervals.append(WordInterval(len(intervals), str(official_words[official_index]), item.start, item.end))
        official_indices.append(official_index)
    return intervals, official_indices


class WhisperAuditor:
    def __init__(self, model_path: Path, device: str, chunk_length_seconds: int):
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"ASR model is missing: {model_path}. Run 环境/download_whisper_model.py first."
            )
        if device == "auto":
            device_index = 0 if torch.cuda.is_available() else -1
        elif device.startswith("cuda"):
            device_index = int(device.split(":", 1)[1]) if ":" in device else 0
        else:
            device_index = -1
        dtype = torch.float16 if device_index >= 0 else torch.float32
        from transformers import pipeline as transformers_pipeline

        self.pipe = transformers_pipeline(
            "automatic-speech-recognition",
            model=str(model_path),
            device=device_index,
            dtype=dtype,
            chunk_length_s=int(chunk_length_seconds),
            ignore_warning=True,
        )

    def transcribe(self, wav_path: Path) -> tuple[str, list[ASRWord], str]:
        signal, sample_rate = sf.read(str(wav_path), always_2d=False, dtype="float32")
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        kwargs: dict[str, Any] = {"return_timestamps": "word"}
        try:
            result = self.pipe(
                {"array": signal, "sampling_rate": sample_rate},
                return_language=True,
                **kwargs,
            )
        except (TypeError, ValueError):
            result = self.pipe({"array": signal, "sampling_rate": sample_rate}, **kwargs)
        text = str(result.get("text", "")).strip()
        words: list[ASRWord] = []
        for chunk in result.get("chunks", []):
            timestamp = chunk.get("timestamp") or (None, None)
            start, end = timestamp
            normalized, pieces = normalize_transcript(str(chunk.get("text", "")))
            if not pieces or start is None or end is None or float(end) <= float(start):
                continue
            duration = (float(end) - float(start)) / len(pieces)
            for piece_index, piece in enumerate(pieces):
                piece_start = float(start) + piece_index * duration
                piece_end = float(start) + (piece_index + 1) * duration
                words.append(ASRWord(len(words), piece, piece_start, piece_end))
        language_value = result.get("language")
        language = str(language_value).strip() if language_value else "not_reported"
        return text, words, language


def evaluate_quality(
    official_text: str,
    wav_path: Path,
    auditor: WhisperAuditor,
    config: dict[str, Any],
    low_threshold: float,
    high_threshold: float,
    minimum_contiguous_matches: int,
) -> QualityResult:
    signal = analyze_signal(
        wav_path,
        float(config["silence_peak_dbfs"]),
        float(config["energy_margin_db"]),
    )
    asr_text, asr_words, language = auditor.transcribe(wav_path)
    _, official_words = normalize_transcript(official_text)
    mapping, match_mask, starts, ends, metrics = match_word_sequences(
        official_words, asr_words, minimum_contiguous_matches
    )
    speech_present = bool(
        signal["signal_activity_ratio"] >= float(config["minimum_signal_activity_ratio"])
        and len(asr_words) >= int(config["minimum_asr_words_for_speech"])
    )
    score = 0.70 * metrics["word_f1"] + 0.30 * metrics["character_similarity"]
    if not speech_present:
        score = 0.0
        alignment_type = "no_word_alignment"
    elif score >= high_threshold:
        alignment_type = "full_mfa"
    elif score >= low_threshold and any(match_mask):
        alignment_type = "partial_asr_timestamp"
    else:
        alignment_type = "no_word_alignment"
    if language.casefold() in {"", "unknown", "none"}:
        language = "not_reported"
    return QualityResult(
        asr_text=asr_text,
        asr_words=asr_words,
        detected_language=language,
        peak_dbfs=signal["peak_dbfs"],
        rms_dbfs=signal["rms_dbfs"],
        signal_activity_ratio=signal["signal_activity_ratio"],
        speech_present=speech_present,
        exact_match_count=int(metrics["exact_match_count"]),
        reliable_match_count=int(metrics["reliable_match_count"]),
        official_word_recall=metrics["official_word_recall"],
        asr_word_precision=metrics["asr_word_precision"],
        word_f1=metrics["word_f1"],
        character_similarity=metrics["character_similarity"],
        consistency_score=float(score),
        alignment_type=alignment_type,
        official_to_asr=mapping,
        official_match_mask=match_mask,
        official_asr_starts=starts,
        official_asr_ends=ends,
        official_comparison_unit_count=int(metrics["official_comparison_unit_count"]),
        asr_comparison_unit_count=int(metrics["asr_comparison_unit_count"]),
        numeric_equivalence_count=int(metrics["numeric_equivalence_count"]),
    )
