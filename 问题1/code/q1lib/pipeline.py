from __future__ import annotations

import difflib
import importlib.metadata
import json
import logging
import platform
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from .config import Q1Config
from .data import SampleRecord, WordInterval, load_manifest, normalize_transcript, pad_or_truncate, sha256_file, validate_manifest, write_json
from .features import BertFeatureExtractor, OpenSmileFeatureExtractor, aggregate_frames_by_optional_words, expand_words_to_tokens, pooled_feature_names, read_openface_features, resample_frames_to_time_bins
from .quality import ASRWord, WhisperAuditor, evaluate_quality
from .tools import extract_audio, locate_textgrid, probe_media, read_word_intervals, run_mfa_align, run_openface


LOGGER = logging.getLogger("q1_pipeline")


def derive_mfa_window(
    quality: dict[str, Any],
    audio_duration: float,
    padding_seconds: float,
    require_edge_anchors: bool = True,
) -> dict[str, Any]:
    """Locate the official transcript inside the presented audio timeline."""
    masks = [int(item) for item in quality.get("official_match_mask", [])]
    starts = quality.get("official_asr_starts", [])
    ends = quality.get("official_asr_ends", [])
    result: dict[str, Any] = {
        "eligible": False,
        "reason": "missing_official_asr_timestamps",
        "official_anchor_start": None,
        "official_anchor_end": None,
        "window_start": None,
        "window_end": None,
        "prefix_extra_seconds": None,
        "suffix_extra_seconds": None,
        "is_windowed": False,
    }
    if not masks or len(starts) != len(masks) or len(ends) != len(masks):
        return result
    if require_edge_anchors and (masks[0] != 1 or masks[-1] != 1):
        result["reason"] = "official_boundary_word_not_reliably_matched"
        return result
    anchor_start = starts[0]
    anchor_end = ends[-1]
    if anchor_start is None or anchor_end is None:
        return result
    anchor_start = float(anchor_start)
    anchor_end = float(anchor_end)
    if not 0.0 <= anchor_start < anchor_end <= float(audio_duration) + 0.1:
        result["reason"] = "invalid_official_asr_window"
        return result
    padding = max(0.0, float(padding_seconds))
    window_start = max(0.0, anchor_start - padding)
    window_end = min(float(audio_duration), anchor_end + padding)
    if window_end - window_start <= 0.10:
        result["reason"] = "official_asr_window_too_short"
        return result
    prefix = max(0.0, anchor_start)
    suffix = max(0.0, float(audio_duration) - anchor_end)
    result.update(
        {
            "eligible": True,
            "reason": "ok",
            "official_anchor_start": anchor_start,
            "official_anchor_end": anchor_end,
            "window_start": window_start,
            "window_end": window_end,
            "prefix_extra_seconds": prefix,
            "suffix_extra_seconds": suffix,
            "is_windowed": bool(window_start > 1e-6 or window_end < float(audio_duration) - 1e-6),
        }
    )
    return result


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_runtime_info(config: Q1Config | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version, "platform": platform.platform(), "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"), "torch": torch.__version__,
        "transformers": _package_version("transformers"), "opensmile": _package_version("opensmile"),
        "praatio": _package_version("praatio"), "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "openface": "2.2.0", "mfa": "3.3.10",
    }
    if config is not None:
        try:
            completed = subprocess.run([str(config.tools["ffmpeg"]), "-version"], capture_output=True,
                                       text=True, encoding="utf-8", errors="replace", check=False)
            info["ffmpeg"] = completed.stdout.splitlines()[0] if completed.stdout else "unknown"
        except OSError as exc:
            info["ffmpeg"] = f"unavailable: {exc}"
        info.update({
            "bert_model_path": str(config.text["bert_model"]),
            "bert_hidden_layer": int(config.text["hidden_layer"]),
            "asr_model_path": str(config.quality["asr_model"]),
            "opensmile_feature_set": str(config.audio["opensmile_feature_set"]),
            "opensmile_feature_level": str(config.audio["opensmile_feature_level"]),
        })
    return info


def _json_indices(indices: np.ndarray) -> str:
    return json.dumps(indices.astype(int).tolist(), ensure_ascii=False)


def _path_or_empty(path: Path | None) -> str:
    return str(path.resolve()) if path is not None and path.exists() else ""


def _sequence_mapping(left: Sequence[str], right: Sequence[str]) -> tuple[list[int], float]:
    matcher = difflib.SequenceMatcher(
        a=[str(item).casefold() for item in left], b=[str(item).casefold() for item in right],
        autojunk=False,
    )
    mapping = [-1] * len(left)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            mapping[block.a + offset] = block.b + offset
    return mapping, matcher.ratio()


def _speech_bins(asr_words: Sequence[ASRWord], duration: float, length: int) -> np.ndarray:
    mask = np.zeros(length, dtype=np.uint8)
    if duration <= 0:
        return mask
    for item in asr_words:
        start = max(0, min(length - 1, int(np.floor(item.start / duration * length))))
        end = max(start + 1, min(length, int(np.ceil(item.end / duration * length))))
        mask[start:end] = 1
    return mask


def _time_bin_bounds(duration: float, length: int) -> tuple[np.ndarray, np.ndarray]:
    if duration <= 0 or length <= 0:
        raise ValueError("duration and length must be positive")
    edges = np.linspace(0.0, float(duration), int(length) + 1, dtype=np.float64)
    return edges[:-1], edges[1:]


def _normalize_language_field(value: Any) -> str:
    text = str(value or "").strip()
    if text in {"", "unknown", "english_likely", "unknown_or_non_english"}:
        return "not_reported"
    return text


def _describe_openface_field(field: str) -> tuple[str, str, str]:
    if field.startswith("gaze_0_"):
        return "left_eye_gaze", "Left-eye gaze direction vector component", "unit vector component"
    if field.startswith("gaze_1_"):
        return "right_eye_gaze", "Right-eye gaze direction vector component", "unit vector component"
    if field.startswith("gaze_angle_"):
        return "gaze_angle", "Gaze direction angle", "radians"
    if field.startswith("pose_T"):
        return "head_translation", "Estimated head translation in camera coordinates", "millimetres"
    if field.startswith("pose_R"):
        return "head_rotation", "Estimated head rotation", "radians"
    if field.endswith("_r"):
        return "action_unit_intensity", f"Facial action-unit intensity ({field[:-2]})", "OpenFace intensity scale 0-5"
    if field.endswith("_c"):
        return "action_unit_presence", f"Facial action-unit presence ({field[:-2]})", "binary 0/1 before pooling"
    return "openface_field", f"Configured OpenFace field {field}", "OpenFace native scale"


class Q1Pipeline:
    def __init__(self, config: Q1Config):
        self.config = config
        self.records = load_manifest(config.label_file, config.data_root)
        self.bert: BertFeatureExtractor | None = None
        self.smile: OpenSmileFeatureExtractor | None = None
        self.auditor: WhisperAuditor | None = None
        self.active_mfa_output_dir = config.mfa_output_dir

    def _is_full_run(self, records: Sequence[SampleRecord]) -> bool:
        return {item.sample_id for item in records} == {item.sample_id for item in self.records}

    @property
    def manifest_csv(self) -> Path:
        return self.config.output_root / "manifest" / "manifest.csv"

    def select_records(self, sample_ids: set[str] | None = None, limit: int | None = None) -> list[SampleRecord]:
        records = self.records
        if sample_ids:
            records = [item for item in records if item.sample_id in sample_ids]
            missing = sample_ids - {item.sample_id for item in records}
            if missing:
                raise KeyError(f"unknown sample ids: {sorted(missing)}")
        return records[:limit] if limit is not None else records

    def build_manifest(self) -> pd.DataFrame:
        frame = validate_manifest(self.records, self.config.data_root, expected_count=100)
        frame.to_csv(self.manifest_csv, index=False, encoding="utf-8-sig")
        return frame

    def prepare_mfa_corpus(self, records: Iterable[SampleRecord]) -> pd.DataFrame:
        records = list(records)
        rows: list[dict[str, Any]] = []
        for record in records:
            wav_path = self.config.mfa_corpus_dir / f"{record.sample_id}.wav"
            lab_path = self.config.mfa_corpus_dir / f"{record.sample_id}.lab"
            normalized, words = normalize_transcript(record.text)
            try:
                if not normalized:
                    raise ValueError("normalized official transcript is empty")
                extract_audio(Path(record.video_path), wav_path, Path(self.config.tools["ffmpeg"]),
                              int(self.config.audio["sample_rate"]), int(self.config.audio["channels"]),
                              bool(self.config.runtime["overwrite"]),
                              self.config.output_root / "logs" / f"{record.sample_id}.log")
                lab_path.write_text(normalized + "\n", encoding="utf-8")
                audio_info = sf.info(str(wav_path))
                rows.append({"sample_id": record.sample_id, "status": "ok", "normalized_text": normalized,
                             "word_count": len(words), "wav_path": str(wav_path.resolve()),
                             "lab_path": str(lab_path.resolve()), "audio_duration_seconds": float(audio_info.duration),
                             "audio_sample_rate": int(audio_info.samplerate), "error": ""})
            except Exception as exc:
                rows.append({"sample_id": record.sample_id, "status": "failed", "normalized_text": normalized,
                             "word_count": len(words), "wav_path": str(wav_path.resolve()),
                             "lab_path": str(lab_path.resolve()), "audio_duration_seconds": None,
                             "audio_sample_rate": None, "error": f"{type(exc).__name__}: {exc}"})
                if not self.config.runtime["continue_on_error"]:
                    raise
        frame = pd.DataFrame(rows)
        suffix = "" if self._is_full_run(records) else "_partial"
        frame.to_csv(self.config.output_root / "manifest" / f"mfa_corpus_manifest{suffix}.csv",
                     index=False, encoding="utf-8-sig")
        return frame

    def _load_auditor(self) -> WhisperAuditor:
        if self.auditor is None:
            self.auditor = WhisperAuditor(Path(self.config.quality["asr_model"]),
                                          str(self.config.quality["asr_device"]),
                                          int(self.config.quality["asr_chunk_length_seconds"]))
        return self.auditor

    def run_quality_audit(self, records: Iterable[SampleRecord]) -> pd.DataFrame:
        records = list(records)
        auditor = self._load_auditor()
        rows: list[dict[str, Any]] = []
        low = float(self.config.alignment["low_threshold"])
        high = float(self.config.alignment["high_threshold"])
        for record in records:
            LOGGER.info("quality audit %s", record.sample_id)
            sample_dir = self.config.output_root / "samples" / record.sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            try:
                result = evaluate_quality(record.text, self.config.mfa_corpus_dir / f"{record.sample_id}.wav",
                                          auditor, self.config.quality, low, high,
                                          int(self.config.alignment["minimum_contiguous_matches"]))
                payload = result.to_json()
                payload.update({"sample_id": record.sample_id, "official_text": record.text,
                                "low_threshold": low, "high_threshold": high,
                                "threshold_status": self.config.alignment["threshold_status"]})
                write_json(sample_dir / "quality.json", payload)
                pd.DataFrame([asdict(item) for item in result.asr_words],
                             columns=["index", "word", "start", "end"]).to_csv(
                    sample_dir / "asr_words.csv", index=False, encoding="utf-8-sig")
                rows.append({"sample_id": record.sample_id, "status": "ok",
                             "speech_present": result.speech_present,
                             "detected_language": result.detected_language,
                             "peak_dbfs": result.peak_dbfs, "rms_dbfs": result.rms_dbfs,
                             "signal_activity_ratio": result.signal_activity_ratio,
                             "asr_word_count": len(result.asr_words),
                             "reliable_match_count": result.reliable_match_count,
                             "official_comparison_unit_count": result.official_comparison_unit_count,
                             "asr_comparison_unit_count": result.asr_comparison_unit_count,
                             "numeric_equivalence_count": result.numeric_equivalence_count,
                             "official_word_recall": result.official_word_recall,
                             "asr_word_precision": result.asr_word_precision, "word_f1": result.word_f1,
                             "character_similarity": result.character_similarity,
                             "consistency_score": result.consistency_score,
                             "alignment_type": result.alignment_type, "error": ""})
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                LOGGER.exception("quality audit failed: %s", record.sample_id)
                write_json(sample_dir / "quality_failure.json",
                           {"sample_id": record.sample_id, "error": error, "traceback": traceback.format_exc()})
                rows.append({"sample_id": record.sample_id, "status": "failed", "error": error})
                if not self.config.runtime["continue_on_error"]:
                    raise
        frame = pd.DataFrame(rows)
        full_run = self._is_full_run(records)
        destination = "quality_audit.csv" if full_run else "quality_audit_partial.csv"
        frame.to_csv(self.config.output_root / "summary" / destination,
                     index=False, encoding="utf-8-sig")
        return frame

    def _load_quality(self, sample_id: str) -> dict[str, Any]:
        path = self.config.output_root / "samples" / sample_id / "quality.json"
        if not path.is_file():
            raise FileNotFoundError(f"quality audit is missing for {sample_id}; run --stage quality first")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["detected_language"] = _normalize_language_field(payload.get("detected_language"))
        return payload

    def align_mfa(self, records: Iterable[SampleRecord]) -> None:
        records = list(records)
        full_run = self._is_full_run(records)
        gated = (self.config.mfa_gated_corpus_dir if full_run
                 else self.config.mfa_partial_gated_corpus_dir)
        output_dir = (self.config.mfa_output_dir if full_run
                      else self.config.mfa_partial_output_dir)
        self.active_mfa_output_dir = output_dir
        for pattern in ("*.wav", "*.lab"):
            for path in gated.glob(pattern):
                path.unlink()
        for path in output_dir.rglob("*.TextGrid"):
            path.unlink()
        window_rows: list[dict[str, Any]] = []
        selected: list[SampleRecord] = []
        padding = float(self.config.alignment.get("mfa_window_padding_seconds", 0.03))
        require_edges = bool(self.config.alignment.get("mfa_require_edge_anchors", True))
        for record in records:
            quality = self._load_quality(record.sample_id)
            if quality.get("alignment_type") != "full_mfa":
                continue
            source_wav = self.config.mfa_corpus_dir / f"{record.sample_id}.wav"
            audio_duration = float(sf.info(str(source_wav)).duration)
            window = derive_mfa_window(quality, audio_duration, padding, require_edges)
            window_rows.append(
                {
                    "sample_id": record.sample_id,
                    "requested_alignment_type": quality.get("alignment_type"),
                    **window,
                    "audio_duration_seconds": audio_duration,
                }
            )
            if not window["eligible"]:
                LOGGER.warning("%s: full MFA downgraded: %s", record.sample_id, window["reason"])
                continue
            signal, sample_rate = sf.read(str(source_wav), always_2d=False, dtype="float32")
            start_index = max(0, int(np.floor(float(window["window_start"]) * sample_rate)))
            end_index = min(len(signal), int(np.ceil(float(window["window_end"]) * sample_rate)))
            if end_index <= start_index:
                LOGGER.warning("%s: empty MFA crop after sample conversion", record.sample_id)
                continue
            sf.write(
                str(gated / f"{record.sample_id}.wav"),
                signal[start_index:end_index],
                sample_rate,
                subtype="PCM_16",
            )
            shutil.copy2(self.config.mfa_corpus_dir / f"{record.sample_id}.lab",
                         gated / f"{record.sample_id}.lab")
            selected.append(record)
        suffix = "" if full_run else "_partial"
        window_columns = [
            "sample_id", "requested_alignment_type", "eligible", "reason",
            "official_anchor_start", "official_anchor_end", "window_start", "window_end",
            "prefix_extra_seconds", "suffix_extra_seconds", "is_windowed",
            "audio_duration_seconds",
        ]
        pd.DataFrame(window_rows, columns=window_columns).to_csv(
            self.config.output_root / "summary" / f"mfa_alignment_windows{suffix}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        if not selected:
            LOGGER.info("no samples passed the full-MFA gate")
            return
        run_mfa_align(corpus_dir=gated, output_dir=output_dir,
                      mfa_env=Path(self.config.tools["mfa_env"]), mfa_root=self.config.mfa_root,
                      dictionary=str(self.config.tools["mfa_dictionary"]),
                      acoustic_model=str(self.config.tools["mfa_acoustic_model"]), overwrite=True,
                      log_path=self.config.output_root / "logs" / "mfa_gated_align.log")

    def _load_extractors(self) -> None:
        if self.bert is None:
            self.bert = BertFeatureExtractor(Path(self.config.text["bert_model"]),
                                             int(self.config.text["hidden_layer"]),
                                             str(self.config.text["device"]),
                                             int(self.config.text["model_max_tokens"]))
        if self.smile is None:
            self.smile = OpenSmileFeatureExtractor(str(self.config.audio["opensmile_feature_set"]),
                                                   str(self.config.audio["opensmile_feature_level"]),
                                                   list(self.config.audio["delta_fields"]),
                                                   int(self.config.audio["target_dimension"]))

    @staticmethod
    def _standardize(values: np.ndarray, length: int, padding_value: float,
                     preserve_last: bool) -> np.ndarray:
        return pad_or_truncate(values, length, padding_value, preserve_last)[0]

    def _resolve_official_intervals(
        self, record: SampleRecord, official_words: Sequence[str], quality: dict[str, Any]
    ) -> tuple[list[WordInterval | None], str, Path | None, float | None, dict[str, Any] | None]:
        intervals: list[WordInterval | None] = [None] * len(official_words)
        requested = str(quality["alignment_type"])
        actual = requested
        textgrid_path: Path | None = None
        mfa_match_ratio: float | None = None
        mfa_window: dict[str, Any] | None = None
        if requested == "full_mfa":
            try:
                wav_path = self.config.mfa_corpus_dir / f"{record.sample_id}.wav"
                mfa_window = derive_mfa_window(
                    quality,
                    float(sf.info(str(wav_path)).duration),
                    float(self.config.alignment.get("mfa_window_padding_seconds", 0.03)),
                    bool(self.config.alignment.get("mfa_require_edge_anchors", True)),
                )
                if not mfa_window["eligible"]:
                    raise ValueError(str(mfa_window["reason"]))
                search_roots = [self.active_mfa_output_dir]
                for candidate in (self.config.mfa_output_dir, self.config.mfa_partial_output_dir):
                    if candidate not in search_roots:
                        search_roots.append(candidate)
                located: list[Path] = []
                for root in search_roots:
                    try:
                        located.append(locate_textgrid(root, record.sample_id))
                        break
                    except FileNotFoundError:
                        continue
                if not located:
                    raise FileNotFoundError(f"no TextGrid found for {record.sample_id}")
                textgrid_path = located[0]
                mfa_intervals = read_word_intervals(textgrid_path)
                mapping, mfa_match_ratio = _sequence_mapping(official_words, [item.word for item in mfa_intervals])
                if mfa_match_ratio < float(self.config.alignment["mfa_minimum_word_match_ratio"]):
                    raise ValueError(f"MFA word match ratio is only {mfa_match_ratio:.3f}")
                offset = float(mfa_window["window_start"])
                for official_index, mfa_index in enumerate(mapping):
                    if mfa_index >= 0:
                        source = mfa_intervals[mfa_index]
                        intervals[official_index] = WordInterval(official_index, official_words[official_index],
                                                                 source.start + offset, source.end + offset)
                actual = "full_mfa_windowed" if mfa_window["is_windowed"] else "full_mfa"
            except Exception as exc:
                LOGGER.warning("%s: MFA rejected; using matched ASR timestamps: %s", record.sample_id, exc)
                actual = "partial_asr_timestamp_fallback"
        if requested == "partial_asr_timestamp" or actual.endswith("fallback"):
            asr_words = [ASRWord(**item) for item in quality.get("asr_words", [])]
            stored_starts = quality.get("official_asr_starts", [])
            stored_ends = quality.get("official_asr_ends", [])
            if len(stored_starts) == len(official_words) and len(stored_ends) == len(official_words):
                for official_index, (start, end) in enumerate(zip(stored_starts, stored_ends)):
                    if start is not None and end is not None and float(end) > float(start):
                        intervals[official_index] = WordInterval(
                            official_index, official_words[official_index], float(start), float(end)
                        )
            else:
                for official_index, asr_index in enumerate(quality.get("official_to_asr", [])):
                    if 0 <= asr_index < len(asr_words):
                        source = asr_words[asr_index]
                        intervals[official_index] = WordInterval(
                            official_index, official_words[official_index], source.start, source.end
                        )
        return intervals, actual, textgrid_path, mfa_match_ratio, mfa_window

    def process_sample(self, record: SampleRecord) -> dict[str, Any]:
        self._load_extractors()
        assert self.bert is not None and self.smile is not None
        started = time.perf_counter()
        sample_dir = self.config.output_root / "samples" / record.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.config.output_root / "logs" / f"{record.sample_id}.log"
        video_path = Path(record.video_path)
        wav_path = self.config.mfa_corpus_dir / f"{record.sample_id}.wav"
        lab_path = self.config.mfa_corpus_dir / f"{record.sample_id}.lab"
        quality = self._load_quality(record.sample_id)
        media = probe_media(video_path, Path(self.config.tools["ffprobe"]), log_path)
        if not media["has_video"] or not media["has_audio"]:
            raise ValueError(f"sample lacks required stream: {media}")
        duration = float(media["duration_seconds"] or sf.info(str(wav_path)).duration)
        normalized_text, official_words = normalize_transcript(record.text)
        official_intervals, alignment_type, textgrid_path, mfa_match_ratio, mfa_window = self._resolve_official_intervals(
            record, official_words, quality)
        token = self.bert.extract(official_words, official_intervals)
        audio_frames = self.smile.extract(wav_path)
        openface_csv = run_openface(Path(self.config.tools["openface"]), video_path,
                                    sample_dir / "openface_raw", bool(self.config.runtime["overwrite"]),
                                    bool(self.config.runtime["openface_quiet"]), log_path)
        vision_frames = read_openface_features(openface_csv, self.config.vision)

        audio_stats = list(self.config.audio["pooling"])
        vision_stats = list(self.config.vision["pooling"])
        audio_words, audio_word_mask, audio_mapping = aggregate_frames_by_optional_words(
            audio_frames, official_intervals, audio_stats)
        vision_words, vision_word_mask, vision_mapping = aggregate_frames_by_optional_words(
            vision_frames, official_intervals, vision_stats)
        official_word_match = np.asarray([int(item is not None) for item in official_intervals], dtype=np.uint8)
        audio_words *= official_word_match[:, None]
        vision_words *= official_word_match[:, None]
        audio_word_mask &= official_word_match
        vision_word_mask &= official_word_match
        audio_tokens, audio_token_mask = expand_words_to_tokens(audio_words, audio_word_mask, token.word_ids)
        vision_tokens, vision_token_mask = expand_words_to_tokens(vision_words, vision_word_mask, token.word_ids)
        _, match_token_mask = expand_words_to_tokens(official_word_match[:, None].astype(np.float32),
                                                     official_word_match, token.word_ids)
        speech_token_mask = (match_token_mask.copy() if quality.get("speech_present", False)
                             else np.zeros_like(match_token_mask))
        cross_mask = (token.attention_mask.astype(bool) & match_token_mask.astype(bool)
                      & audio_token_mask.astype(bool) & vision_token_mask.astype(bool)
                      & (token.word_ids >= 0)).astype(np.uint8)

        unaligned_length = int(self.config.sequence["unaligned_length"])
        audio_unaligned, audio_unaligned_mask = resample_frames_to_time_bins(audio_frames, duration, unaligned_length)
        vision_unaligned, vision_unaligned_mask = resample_frames_to_time_bins(vision_frames, duration, unaligned_length)
        unaligned_bin_starts, unaligned_bin_ends = _time_bin_bounds(duration, unaligned_length)
        asr_words = [ASRWord(**item) for item in quality.get("asr_words", [])]
        speech_unaligned_mask = _speech_bins(asr_words, duration, unaligned_length)

        standard_length = int(self.config.sequence["standard_length"])
        padding_value = float(self.config.sequence["padding_value"])
        preserve_last = self.config.sequence["truncation"] == "keep_special_tokens"
        standard = {
            "text": self._standardize(token.values, standard_length, padding_value, preserve_last),
            "audio": self._standardize(audio_tokens, standard_length, padding_value, preserve_last),
            "vision": self._standardize(vision_tokens, standard_length, padding_value, preserve_last),
            "text_attention_mask": self._standardize(token.attention_mask, standard_length, 0, preserve_last),
            "speech_present_mask": self._standardize(speech_token_mask, standard_length, 0, preserve_last),
            "official_asr_match_mask": self._standardize(match_token_mask, standard_length, 0, preserve_last),
            "audio_valid_mask": self._standardize(audio_token_mask, standard_length, 0, preserve_last),
            "vision_valid_mask": self._standardize(vision_token_mask, standard_length, 0, preserve_last),
            "cross_modal_mask": self._standardize(cross_mask, standard_length, 0, preserve_last),
            "word_ids": self._standardize(token.word_ids, standard_length, -1, preserve_last),
            "token_starts": self._standardize(token.starts, standard_length, -1.0, preserve_last),
            "token_ends": self._standardize(token.ends, standard_length, -1.0, preserve_last),
            "tokens": self._standardize(np.asarray(token.tokens, dtype="<U64"), standard_length,
                                         "[PAD]", preserve_last),
        }
        truncated = len(token.tokens) > standard_length
        word_starts = np.asarray([item.start if item else -1.0 for item in official_intervals], dtype=np.float64)
        word_ends = np.asarray([item.end if item else -1.0 for item in official_intervals], dtype=np.float64)
        audio_names = pooled_feature_names(audio_frames.names, audio_stats)
        vision_names = pooled_feature_names(vision_frames.names, vision_stats)
        if len(audio_names) != 74 or len(vision_names) != 35:
            raise ValueError(f"fixed dimensions violated: audio={len(audio_names)}, vision={len(vision_names)}")

        feature_path = sample_dir / "features.npz"
        arrays: dict[str, np.ndarray] = {
            "text_full": token.values, "audio_full": audio_tokens, "vision_full": vision_tokens,
            "text_attention_mask_full": token.attention_mask, "speech_present_mask_full": speech_token_mask,
            "official_asr_match_mask_full": match_token_mask, "audio_valid_mask_full": audio_token_mask,
            "vision_valid_mask_full": vision_token_mask, "cross_modal_mask_full": cross_mask,
            "tokens_full": np.asarray(token.tokens, dtype="<U64"), "word_ids_full": token.word_ids,
            "token_starts_full": token.starts, "token_ends_full": token.ends,
            "word_starts": word_starts, "word_ends": word_ends,
            "words": np.asarray(official_words, dtype="<U128"),
            "official_word_match_mask": official_word_match,
            "audio_word_features": audio_words, "vision_word_features": vision_words,
            "audio_word_valid_mask": audio_word_mask, "vision_word_valid_mask": vision_word_mask,
            "text_standard": standard["text"], "audio_standard": standard["audio"],
            "vision_standard": standard["vision"],
            "aligned_text": standard["text"], "aligned_audio": standard["audio"],
            "aligned_vision": standard["vision"],
            "text_attention_mask": standard["text_attention_mask"],
            "speech_present_mask": standard["speech_present_mask"],
            "official_asr_match_mask": standard["official_asr_match_mask"],
            "audio_valid_mask": standard["audio_valid_mask"],
            "vision_valid_mask": standard["vision_valid_mask"],
            "cross_modal_mask": standard["cross_modal_mask"],
            "tokens": standard["tokens"], "word_ids": standard["word_ids"],
            "token_starts": standard["token_starts"], "token_ends": standard["token_ends"],
            "unaligned_text": standard["text"], "unaligned_audio": audio_unaligned,
            "unaligned_vision": vision_unaligned, "unaligned_audio_valid_mask": audio_unaligned_mask,
            "unaligned_vision_valid_mask": vision_unaligned_mask,
            "unaligned_speech_present_mask": speech_unaligned_mask,
            "unaligned_bin_starts": unaligned_bin_starts,
            "unaligned_bin_ends": unaligned_bin_ends,
            "audio_feature_names": np.asarray(audio_names, dtype="<U160"),
            "vision_feature_names": np.asarray(vision_names, dtype="<U160"),
        }
        if self.config.runtime["save_frame_level_features"]:
            arrays.update({
                "audio_frames": audio_frames.values, "audio_frame_times": audio_frames.times,
                "audio_frame_starts": audio_frames.starts, "audio_frame_ends": audio_frames.ends,
                "audio_frame_valid_mask": audio_frames.valid,
                "audio_frame_feature_names": np.asarray(audio_frames.names, dtype="<U160"),
                "vision_frames": vision_frames.values, "vision_frame_times": vision_frames.times,
                "vision_frame_valid_mask": vision_frames.valid,
                "vision_frame_feature_names": np.asarray(vision_frames.names, dtype="<U160"),
            })
        np.savez_compressed(feature_path, **arrays)

        word_rows = []
        for index, word in enumerate(official_words):
            interval = official_intervals[index]
            word_rows.append({
                "sample_id": record.sample_id, "word_index": index, "word": word,
                "start": interval.start if interval else "", "end": interval.end if interval else "",
                "duration": interval.end - interval.start if interval else "",
                "alignment_source": "MFA" if alignment_type.startswith("full_mfa") and interval else (
                    "ASR" if interval else "none"),
                "official_asr_match": int(official_word_match[index]),
                "audio_frame_count": len(audio_mapping[index]),
                "audio_frame_indices": _json_indices(audio_mapping[index]),
                "audio_valid": int(audio_word_mask[index]),
                "vision_frame_count": len(vision_mapping[index]),
                "vision_frame_indices": _json_indices(vision_mapping[index]),
                "vision_valid": int(vision_word_mask[index]),
            })
        word_alignment_path = sample_dir / "word_alignment.csv"
        pd.DataFrame(word_rows).to_csv(word_alignment_path, index=False, encoding="utf-8-sig")

        token_rows = []
        for index, token_text in enumerate(token.tokens):
            word_id = int(token.word_ids[index])
            token_rows.append({
                "sample_id": record.sample_id, "token_index": index, "token": token_text,
                "word_index": word_id, "word": official_words[word_id] if word_id >= 0 else "",
                "start": token.starts[index] if token.starts[index] >= 0 else "",
                "end": token.ends[index] if token.ends[index] >= 0 else "",
                "is_special_token": int(word_id < 0),
                "text_attention_mask": int(token.attention_mask[index]),
                "speech_present_mask": int(speech_token_mask[index]),
                "official_asr_match_mask": int(match_token_mask[index]),
                "audio_valid_mask": int(audio_token_mask[index]),
                "vision_valid_mask": int(vision_token_mask[index]),
                "cross_modal_mask": int(cross_mask[index]),
            })
        token_alignment_path = sample_dir / "token_alignment.csv"
        pd.DataFrame(token_rows).to_csv(token_alignment_path, index=False, encoding="utf-8-sig")

        integrity = bool(standard["text"].shape == (50, 768)
                         and standard["audio"].shape == (50, 74)
                         and standard["vision"].shape == (50, 35)
                         and audio_unaligned.shape == (500, 74)
                         and vision_unaligned.shape == (500, 35))
        metadata = {
            "sample_id": record.sample_id, "video_id": record.video_id, "clip_id": record.clip_id,
            "label": record.label, "annotation": record.annotation, "official_text": record.text,
            "normalized_official_text": normalized_text, "official_words": official_words,
            "asr_text": quality.get("asr_text", ""), "asr_words": quality.get("asr_words", []),
            "source": {"video_path": str(video_path.resolve()), "video_sha256": sha256_file(video_path),
                       "label_file": str(self.config.label_file)}, "media": media,
            "dimensions": {"aligned_text": list(standard["text"].shape),
                           "aligned_audio": list(standard["audio"].shape),
                           "aligned_vision": list(standard["vision"].shape),
                           "unaligned_text": list(standard["text"].shape),
                           "unaligned_audio": list(audio_unaligned.shape),
                           "unaligned_vision": list(vision_unaligned.shape)},
            "feature_fields": {"audio": audio_names, "vision": vision_names},
            "lengths": {"official_words": len(official_words), "bert_tokens": len(token.tokens),
                        "standard_length": standard_length, "unaligned_length": unaligned_length,
                        "audio_frames": len(audio_frames.values),
                        "audio_valid_frames": int(audio_frames.valid.sum()),
                        "vision_frames": len(vision_frames.values),
                        "vision_valid_frames": int(vision_frames.valid.sum())},
            "quality": {**{key: quality.get(key) for key in (
                "speech_present", "detected_language", "peak_dbfs", "rms_dbfs",
                "signal_activity_ratio", "official_word_recall", "asr_word_precision", "word_f1",
                "character_similarity", "consistency_score", "official_comparison_unit_count",
                "asr_comparison_unit_count", "numeric_equivalence_count")},
                "requested_alignment_type": quality.get("alignment_type"),
                "actual_alignment_type": alignment_type,
                "mfa_alignment_window": mfa_window,
                "mfa_transcript_word_match_ratio": mfa_match_ratio,
                "openface_valid_frame_ratio": float(vision_frames.valid.mean()),
                "audio_valid_word_ratio": float(audio_word_mask.mean()) if len(audio_word_mask) else 0.0,
                "vision_valid_word_ratio": float(vision_word_mask.mean()) if len(vision_word_mask) else 0.0,
                "cross_modal_token_ratio": float(cross_mask.sum() / max(1, (token.word_ids >= 0).sum()))},
            "rules": {"official_text_policy": "immutable BERT input; ASR never replaces it",
                      "alignment_gate": "full MFA / matched ASR timestamps / no word alignment",
                      "mfa_window_rule": (
                          "ASR locates reliable official boundary words; MFA runs only inside that "
                          "window and its timestamps are shifted back to the presented media timeline"
                      ),
                      "bert_time_axis": (
                          "natural-word intervals are mapped to BERT WordPiece tokens by word_ids; "
                          "all subwords inherit the parent word interval"
                      ),
                      "presentation_timeline_policy": (
                          "FFmpeg/OpenFace presented MP4 timeline is authoritative; hidden edit-list "
                          "preroll remains outside aligned_50"
                      ),
                      "numeric_matching": (
                          "English number words and equivalent Arabic numerals are canonicalized only "
                          "for official-ASR comparison; source transcripts remain unchanged"
                      ),
                      "audio_dimension": "65 ComParE_2016 LLD + 9 configured first derivatives",
                      "vision_dimension": "35 configured OpenFace fields",
                      "word_pooling": "mean of valid frame features inside reliable word interval",
                      "unaligned_view": "mean pooling on 500 equal-duration bins of real AV timeline",
                      "unaligned_bin_rule": "bin k is [k*D/500, (k+1)*D/500), with the final end equal to D",
                      "subword_rule": "WordPiece inherits reliable parent interval; otherwise masked",
                      "truncation": self.config.sequence["truncation"], "padding_value": padding_value},
            "status": {"model_truncated_at_bert_limit": token.model_truncated,
                       "standard_view_truncated": truncated, "success": True,
                       "integrity_passed": integrity},
            "files": {"wav": str(wav_path.resolve()), "lab": str(lab_path.resolve()),
                      "textgrid": _path_or_empty(textgrid_path),
                      "openface_csv": str(openface_csv.resolve()), "features": str(feature_path.resolve()),
                      "word_alignment": str(word_alignment_path.resolve()),
                      "token_alignment": str(token_alignment_path.resolve()),
                      "quality": str((sample_dir / "quality.json").resolve()),
                      "asr_words": str((sample_dir / "asr_words.csv").resolve()),
                      "log": str(log_path.resolve())},
            "runtime": collect_runtime_info(self.config), "config": self.config.raw,
            "elapsed_seconds": time.perf_counter() - started,
        }
        metadata_path = sample_dir / "metadata.json"
        write_json(metadata_path, metadata)
        return {
            "sample_id": record.sample_id, "video_id": record.video_id, "clip_id": record.clip_id,
            "annotation": record.annotation, "label": record.label, "status": "ok",
            "video_exists": True, "text_exists": bool(record.text.strip()),
            "video_duration_seconds": duration,
            "audio_duration_seconds": float(sf.info(str(wav_path)).duration),
            "visual_valid_duration_seconds": float(vision_frames.valid.astype(bool).sum() / media["fps"]
                                                   if media.get("fps") else 0.0),
            "audio_frame_count": len(audio_frames.values),
            "audio_valid_frame_count": int(audio_frames.valid.sum()),
            "vision_frame_count": len(vision_frames.values),
            "vision_valid_frame_count": int(vision_frames.valid.sum()),
            "natural_word_count": len(official_words), "bert_token_count": len(token.tokens),
            "text_feature_dim": 768, "audio_frame_dim": 74, "audio_feature_dim": 74,
            "vision_frame_dim": 35, "vision_feature_dim": 35,
            "original_sequence_length": len(token.tokens), "standard_sequence_length": 50,
            "unaligned_sequence_length": 500, "alignment_granularity": alignment_type,
            "standard_view_truncated": truncated,
            "padding_count": max(0, standard_length - len(token.tokens)),
            "speech_present": quality.get("speech_present"),
            "detected_language": quality.get("detected_language"),
            "text_audio_consistency_score": quality.get("consistency_score"),
            "alignment_type": alignment_type, "mfa_status": "used" if textgrid_path else "not_used",
            "mfa_transcript_word_match_ratio": mfa_match_ratio,
            "mfa_window_start": mfa_window.get("window_start") if mfa_window else None,
            "mfa_window_end": mfa_window.get("window_end") if mfa_window else None,
            "mfa_prefix_extra_seconds": mfa_window.get("prefix_extra_seconds") if mfa_window else None,
            "mfa_suffix_extra_seconds": mfa_window.get("suffix_extra_seconds") if mfa_window else None,
            "openface_valid_frame_ratio": float(vision_frames.valid.mean()),
            "audio_valid_word_ratio": float(audio_word_mask.mean()) if len(audio_word_mask) else 0.0,
            "vision_valid_word_ratio": float(vision_word_mask.mean()) if len(vision_word_mask) else 0.0,
            "feature_file": str(feature_path.resolve()), "metadata_file": str(metadata_path.resolve()),
            "alignment_file": str(token_alignment_path.resolve()), "integrity_passed": integrity,
            "warning": "" if alignment_type == "full_mfa" else f"quality-gated: {alignment_type}",
            "error": "", "elapsed_seconds": metadata["elapsed_seconds"],
        }

    def process_features(self, records: Iterable[SampleRecord]) -> pd.DataFrame:
        records = list(records)
        rows: list[dict[str, Any]] = []
        for record in records:
            LOGGER.info("processing %s", record.sample_id)
            try:
                rows.append(self.process_sample(record))
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                LOGGER.exception("sample failed: %s", record.sample_id)
                write_json(self.config.output_root / "samples" / record.sample_id / "failure.json",
                           {"sample_id": record.sample_id, "error": error_text,
                            "traceback": traceback.format_exc()})
                rows.append({"sample_id": record.sample_id, "video_id": record.video_id,
                             "clip_id": record.clip_id, "annotation": record.annotation,
                             "label": record.label, "status": "failed",
                             "video_exists": Path(record.video_path).is_file(),
                             "text_exists": bool(record.text.strip()),
                             "mfa_status": "failed_or_unavailable", "integrity_passed": False,
                             "error": error_text})
                if not self.config.runtime["continue_on_error"]:
                    raise
        frame = pd.DataFrame(rows)
        summary_root = self.config.output_root / "summary"
        full_run = self._is_full_run(records)
        suffix = "" if full_run else "_partial"
        frame.to_csv(summary_root / f"all_samples{suffix}.csv", index=False, encoding="utf-8-sig")
        frame.loc[frame["status"] != "ok"].to_csv(summary_root / f"failures{suffix}.csv", index=False,
                                                   encoding="utf-8-sig")
        alignment_series = (
            frame["alignment_type"]
            if "alignment_type" in frame.columns
            else pd.Series("unknown", index=frame.index)
        )
        warning_mask = frame["status"].eq("ok") & (
            ~frame["integrity_passed"].fillna(False)
            | ~alignment_series.astype(str).str.startswith("full_mfa")
        )
        frame.loc[warning_mask].to_csv(summary_root / f"warnings{suffix}.csv", index=False,
                                        encoding="utf-8-sig")
        self._write_modality_summary(frame, suffix)
        return frame

    def _write_modality_summary(self, frame: pd.DataFrame, suffix: str = "") -> None:
        rows: list[dict[str, Any]] = []
        for item in frame.to_dict(orient="records"):
            common = {"sample_id": item.get("sample_id"), "video_id": item.get("video_id"),
                      "clip_id": item.get("clip_id"), "status": item.get("status"),
                      "alignment_type": item.get("alignment_type")}
            specifications = (
                ("text", item.get("bert_token_count"), "bert_tokens", None, None, 768, 50),
                ("audio", item.get("audio_frame_count"), "opensmile_frames",
                 item.get("audio_duration_seconds"), item.get("audio_duration_seconds"), 74, 500),
                ("vision", item.get("vision_frame_count"), "openface_frames",
                 item.get("video_duration_seconds"), item.get("visual_valid_duration_seconds"), 35, 500),
            )
            for modality, effective_length, length_unit, source_duration, valid_duration, dimension, raw_length in specifications:
                rows.append({**common, "modality": modality,
                             "original_effective_length": effective_length,
                             "effective_length_unit": length_unit,
                             "source_duration_seconds": source_duration,
                             "valid_duration_seconds": valid_duration,
                             "feature_dimension": dimension,
                             "aligned_sequence_length": 50,
                             "unaligned_sequence_length": raw_length,
                             "alignment_granularity": item.get("alignment_granularity"),
                             "feature_file": item.get("feature_file")})
        pd.DataFrame(rows).to_csv(
                                  self.config.output_root / "summary" / f"all_samples_by_modality{suffix}.csv",
                                  index=False, encoding="utf-8-sig")
        self._write_feature_dictionary(frame, suffix)

    def _write_feature_dictionary(self, frame: pd.DataFrame, suffix: str = "") -> None:
        successful = frame.loc[frame["status"].eq("ok")]
        if successful.empty:
            return
        feature_path = Path(str(successful.iloc[0]["feature_file"]))
        with np.load(feature_path, allow_pickle=False) as data:
            schemas = {
                "audio": [str(item) for item in data["audio_feature_names"].tolist()],
                "vision": [str(item) for item in data["vision_feature_names"].tolist()],
            }
        rows: list[dict[str, Any]] = []
        for modality, names in schemas.items():
            for index, name in enumerate(names):
                base = name.rsplit("__mean", 1)[0]
                if modality == "audio":
                    derivative = "__delta_per_second" in base
                    category = "first_derivative" if derivative else "ComParE_2016_LLD"
                    description = ("Per-second first derivative of " + base.replace("__delta_per_second", "")
                                   if derivative else "OpenSMILE ComParE_2016 low-level descriptor " + base)
                    unit = "native OpenSMILE scale per second" if derivative else "native OpenSMILE scale"
                    source = "OpenSMILE 2.6.0 / ComParE_2016 LowLevelDescriptors"
                else:
                    category, description, unit = _describe_openface_field(base)
                    source = "OpenFace 2.2.0"
                rows.append({"modality": modality, "feature_index": index,
                             "feature_name": name, "base_field": base,
                             "category": category, "description": description,
                             "unit_or_scale": unit, "source": source,
                             "word_pooling": "mean over valid frames in the reliable word interval",
                             "time_bin_pooling": "mean over valid frames in the equal-duration bin"})
        pd.DataFrame(rows).to_csv(
            self.config.output_root / "summary" / f"feature_dictionary{suffix}.csv",
            index=False, encoding="utf-8-sig")

    def save_run_provenance(self) -> None:
        summary_root = self.config.output_root / "summary"
        shutil.copy2(self.config.config_path, summary_root / "config_used.yaml")
        write_json(summary_root / "runtime.json", collect_runtime_info(self.config))

    def run(self, stage: str, sample_ids: set[str] | None = None,
            limit: int | None = None, skip_mfa: bool = False) -> None:
        self.build_manifest()
        records = self.select_records(sample_ids=sample_ids, limit=limit)
        self.save_run_provenance()
        if stage in {"prepare", "quality", "align", "all"}:
            corpus_frame = self.prepare_mfa_corpus(records)
            if (corpus_frame["status"] != "ok").any() and not self.config.runtime["continue_on_error"]:
                raise RuntimeError("audio preparation failed")
        if stage in {"quality", "all"}:
            quality_frame = self.run_quality_audit(records)
            if (quality_frame["status"] != "ok").any() and not self.config.runtime["continue_on_error"]:
                raise RuntimeError("quality audit failed")
        if stage in {"align", "all"} and not skip_mfa:
            self.align_mfa(records)
        if stage in {"features", "all"}:
            self.process_features(records)


def configure_logging(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(output_root / "pipeline.log", encoding="utf-8")],
                        force=True)
