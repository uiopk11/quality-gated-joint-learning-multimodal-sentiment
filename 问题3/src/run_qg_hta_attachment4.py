from __future__ import annotations

import argparse
import csv
import difflib
import importlib
import json
import math
import os
import pickle
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from transformers import AutoTokenizer


Q3_ROOT = Path(__file__).resolve().parents[1]
Q1_ROOT = Q3_ROOT.parent / "问题1" / "code"
RESOURCE_ROOT = Path(os.environ.get("Q1_RESOURCE_ROOT", str(Q3_ROOT / "external_q1"))).expanduser().resolve()
OFFICIAL_DATA_ROOT = Path(os.environ.get("E_TOPIC_DATA_ROOT", str(Q3_ROOT / "official_data"))).expanduser().resolve()
if str(Q1_ROOT) not in sys.path:
    sys.path.insert(0, str(Q1_ROOT))

from q1lib.data import normalize_transcript  # noqa: E402
from q1lib.pipeline import derive_mfa_window  # noqa: E402
from q1lib.quality import WhisperAuditor, evaluate_quality  # noqa: E402
from q1lib.tools import (  # noqa: E402
    extract_audio,
    locate_textgrid,
    probe_media,
    read_word_intervals,
    run_mfa_align,
)


ATTACHMENT4_ROOT = (
    OFFICIAL_DATA_ROOT
    / "附件4-可解释专项视频样本与特征文件"
    / "附件4-可解释专项视频样本与特征文件"
)
ALIGNED_ROOT = ATTACHMENT4_ROOT / "对齐版本"
Q1_CONFIG = {
    "ffmpeg": RESOURCE_ROOT / ".envs" / "q1" / "Library" / "bin" / "ffmpeg.exe",
    "ffprobe": RESOURCE_ROOT / ".envs" / "q1" / "Library" / "bin" / "ffprobe.exe",
    "mfa_env": RESOURCE_ROOT / ".envs" / "mfa",
    "mfa_root": RESOURCE_ROOT / "q1_work" / "mfa_data",
    "mfa_dictionary": "english_us_arpa",
    "mfa_acoustic_model": "english_us_arpa",
    "asr_model": RESOURCE_ROOT / "模型" / "whisper-small",
    "bert_model": RESOURCE_ROOT / "模型" / "bert-base-uncased",
}
Q3_WORK_ROOT = Q3_ROOT / "results" / "qg_hta" / "_work"
RESULT_ROOT = Q3_ROOT / "results" / "qg_hta"
WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")


def _load_pickle(path: Path) -> dict[str, Any]:
    sys.modules.setdefault("numpy._core", np.core)
    for module_name in ("numeric", "multiarray", "fromnumeric", "shape_base", "_multiarray_umath"):
        try:
            module = importlib.import_module(f"numpy.core.{module_name}")
        except ModuleNotFoundError:
            continue
        sys.modules.setdefault(f"numpy._core.{module_name}", module)
    with path.open("rb") as stream:
        value = pickle.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected dictionary")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _validate_tools() -> None:
    required = {
        "Q1 source": Q1_ROOT / "q1lib" / "quality.py",
        "ffmpeg": Q1_CONFIG["ffmpeg"],
        "ffprobe": Q1_CONFIG["ffprobe"],
        "MFA environment": Q1_CONFIG["mfa_env"],
        "MFA root": Q1_CONFIG["mfa_root"],
        "Whisper model": Q1_CONFIG["asr_model"],
        "BERT tokenizer": Q1_CONFIG["bert_model"] / "tokenizer.json",
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing QG-HTA dependency:\n" + "\n".join(missing))


def _official_word_spans(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    matches = list(WORD_PATTERN.finditer(text.replace("’", "'").replace("‘", "'")))
    return [item.group(0) for item in matches], [(item.start(), item.end()) for item in matches]


def _sequence_mapping(left: list[str], right: list[str]) -> tuple[list[int], float]:
    matcher = difflib.SequenceMatcher(
        a=[str(item).casefold() for item in left],
        b=[str(item).casefold() for item in right],
        autojunk=False,
    )
    mapping = [-1] * len(left)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            mapping[block.a + offset] = block.b + offset
    return mapping, float(matcher.ratio())


def _verify_and_tokenize(record: dict[str, Any], tokenizer) -> dict[str, Any]:
    raw_text = str(record["raw_text"])
    text = np.asarray(record["text_bert"], dtype=np.int64)
    if text.shape != (3, 50):
        raise ValueError("text_bert must have shape [3,50]")
    encoded = tokenizer(
        raw_text,
        add_special_tokens=True,
        max_length=50,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
    )
    for row, key in enumerate(("input_ids", "attention_mask", "token_type_ids")):
        if not np.array_equal(np.asarray(encoded[key], dtype=np.int64), text[row]):
            raise ValueError(f"tokenizer mismatch for {key}")
    total = int(text[1].sum())
    length = total - 2
    tokens = tokenizer.convert_ids_to_tokens(text[0, 1 : length + 1].tolist())
    offsets = [[int(start), int(end)] for start, end in encoded["offset_mapping"][1 : length + 1]]
    return {"body_length": length, "wordpieces": tokens, "offsets": offsets}


def _word_intervals_from_quality(words: list[str], quality: dict[str, Any]) -> list[dict[str, Any] | None]:
    if quality.get("alignment_type") == "no_word_alignment":
        return [None] * len(words)
    starts = quality.get("official_asr_starts", [])
    ends = quality.get("official_asr_ends", [])
    mask = quality.get("official_match_mask", [])
    output: list[dict[str, Any] | None] = []
    for index, word in enumerate(words):
        if index >= len(mask) or not int(mask[index]) or starts[index] is None or ends[index] is None:
            output.append(None)
            continue
        start = float(starts[index])
        end = float(ends[index])
        output.append(
            {"word": word, "start": start, "end": end, "source": "partial_asr_timestamp"}
            if end > start >= 0 else None
        )
    return output


def _map_tokens_to_words(
    raw_text: str,
    token_info: dict[str, Any],
    word_spans: list[tuple[int, int]],
    word_intervals: list[dict[str, Any] | None],
    media: dict[str, Any],
) -> list[dict[str, Any]]:
    fps = float(media["fps"]) if media.get("fps") else None
    presented_frames = int(media["video_frame_count"]) if media.get("video_frame_count") is not None else None
    rows = []
    for body_position, (token, offset) in enumerate(zip(token_info["wordpieces"], token_info["offsets"])):
        start_char, end_char = offset
        overlaps = [
            index for index, (word_start, word_end) in enumerate(word_spans)
            if max(start_char, word_start) < min(end_char, word_end)
        ]
        word_index = overlaps[0] if len(overlaps) == 1 else None
        interval = word_intervals[word_index] if word_index is not None else None
        if interval is not None:
            time_start = float(interval["start"])
            time_end = float(interval["end"])
            frame_start = max(0, int(math.floor(time_start * fps))) if fps else None
            frame_end = int(math.ceil(time_end * fps)) if fps else None
            if frame_end is not None and presented_frames is not None:
                frame_start = min(frame_start, max(0, presented_frames - 1))
                frame_end = min(max(frame_start + 1, frame_end), presented_frames)
            key_frame = (
                min(max(0, int(round((time_start + time_end) * 0.5 * fps))), presented_frames - 1)
                if fps and presented_frames and presented_frames > 0 else None
            )
            status = "verified_word_interval"
            method = str(interval["source"])
        else:
            time_start = time_end = None
            frame_start = frame_end = key_frame = None
            status = "no_reliable_word_interval"
            method = None
        rows.append({
            "body_position": body_position,
            "source_slot": body_position + 1,
            "wordpiece": token,
            "text_char_start": start_char,
            "text_char_end": end_char,
            "original_text_fragment": raw_text[start_char:end_char],
            "official_word_index": word_index,
            "official_word": None if word_index is None else raw_text[word_spans[word_index][0] : word_spans[word_index][1]],
            "time_start_seconds": time_start,
            "time_end_seconds": time_end,
            "video_frame_start": frame_start,
            "video_frame_end_exclusive": frame_end,
            "video_key_frame": key_frame,
            "timestamp_status": status,
            "alignment_method": method,
        })
    return rows


def _mfa_intervals(
    sample_id: str,
    words: list[str],
    output_dir: Path,
    window: dict[str, Any],
    quality: dict[str, Any],
) -> tuple[list[dict[str, Any] | None], dict[str, Any]]:
    fallback = _word_intervals_from_quality(words, quality)
    try:
        textgrid = locate_textgrid(output_dir, sample_id)
        raw_intervals = read_word_intervals(textgrid)
        mapping, ratio = _sequence_mapping(words, [item.word for item in raw_intervals])
        if ratio < 0.90:
            return fallback, {"mfa_status": "fallback_low_word_match", "mfa_word_match_ratio": ratio}
        shift = float(window["window_start"])
        result: list[dict[str, Any] | None] = []
        for word, interval_index in zip(words, mapping):
            if interval_index < 0:
                result.append(None)
                continue
            interval = raw_intervals[interval_index]
            result.append({
                "word": word,
                "start": float(interval.start) + shift,
                "end": float(interval.end) + shift,
                "source": "full_mfa_windowed" if window["is_windowed"] else "full_mfa",
            })
        if result[0] is None or result[-1] is None:
            return fallback, {"mfa_status": "fallback_missing_edge_word", "mfa_word_match_ratio": ratio}
        return result, {"mfa_status": "used", "mfa_word_match_ratio": ratio}
    except Exception as exc:
        return fallback, {"mfa_status": f"fallback_error:{type(exc).__name__}", "mfa_word_match_ratio": None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply read-only Q1 QG-HTA logic to attachment 4")
    parser.add_argument("--samples", nargs="*", type=int, default=list(range(1, 21)))
    parser.add_argument("--asr-device", default="auto")
    args = parser.parse_args()
    samples = sorted({int(item) for item in args.samples})
    if not samples or any(item < 1 or item > 20 for item in samples):
        raise ValueError("samples must be non-empty and within 1..20")
    _validate_tools()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(Q1_CONFIG["bert_model"], local_files_only=True, use_fast=True)
    auditor = WhisperAuditor(Q1_CONFIG["asr_model"], args.asr_device, 30)
    tag = "all" if samples == list(range(1, 21)) else "samples_" + "_".join(f"{item:02d}" for item in samples)
    work_root = Q3_WORK_ROOT / tag
    audio_root = work_root / "audio"
    mfa_corpus = work_root / "mfa_corpus"
    mfa_output = work_root / "mfa_output"
    log_root = work_root / "logs"
    for path in (audio_root, mfa_corpus, mfa_output, log_root, RESULT_ROOT / "samples"):
        path.mkdir(parents=True, exist_ok=True)

    contexts: dict[str, dict[str, Any]] = {}
    selected_for_mfa = []
    for number in samples:
        sample_id = f"{number:02d}"
        record = _load_pickle(ALIGNED_ROOT / f"{sample_id}.pkl")
        raw_text = str(record["raw_text"])
        video_path = ALIGNED_ROOT / "videos" / f"{sample_id}.mp4"
        wav_path = audio_root / f"{sample_id}.wav"
        extract_audio(video_path, wav_path, Q1_CONFIG["ffmpeg"], 16000, 1, False, log_root / f"{sample_id}.log")
        media = probe_media(video_path, Q1_CONFIG["ffprobe"], log_root / f"{sample_id}.log")
        token_info = _verify_and_tokenize(record, tokenizer)
        normalized_text, words = normalize_transcript(raw_text)
        direct_words, word_spans = _official_word_spans(raw_text)
        if [item.casefold() for item in direct_words] != [item.casefold() for item in words]:
            raise ValueError(f"{sample_id}: official word parsing mismatch")
        quality_result = evaluate_quality(
            raw_text,
            wav_path,
            auditor,
            {
                "silence_peak_dbfs": -50.0,
                "energy_margin_db": 12.0,
                "minimum_signal_activity_ratio": 0.02,
                "minimum_asr_words_for_speech": 1,
            },
            0.35,
            0.75,
            2,
        )
        quality = quality_result.to_json()
        audio_duration = float(sf.info(str(wav_path)).duration)
        window = derive_mfa_window(quality, audio_duration, 0.03, True)
        context = {
            "sample_id": sample_id,
            "raw_text": raw_text,
            "video_path": str(video_path.resolve()),
            "wav_path": str(wav_path.resolve()),
            "media": media,
            "token_info": token_info,
            "words": words,
            "word_spans": word_spans,
            "quality": quality,
            "window": window,
        }
        contexts[sample_id] = context
        _write_json(RESULT_ROOT / "samples" / sample_id / "quality.json", {
            "sample_id": sample_id,
            "official_text": raw_text,
            "low_threshold": 0.35,
            "high_threshold": 0.75,
            "minimum_contiguous_matches": 2,
            **quality,
        })
        if quality["alignment_type"] == "full_mfa" and window["eligible"]:
            signal, sample_rate = sf.read(str(wav_path), always_2d=False, dtype="float32")
            start_index = max(0, int(np.floor(float(window["window_start"]) * sample_rate)))
            end_index = min(len(signal), int(np.ceil(float(window["window_end"]) * sample_rate)))
            if end_index > start_index:
                sf.write(str(mfa_corpus / f"{sample_id}.wav"), signal[start_index:end_index], sample_rate, subtype="PCM_16")
                (mfa_corpus / f"{sample_id}.lab").write_text(normalized_text + "\n", encoding="utf-8")
                selected_for_mfa.append(sample_id)
        print(f"quality attachment4/{sample_id}: {quality['alignment_type']}", flush=True)

    mfa_run_error = None
    if selected_for_mfa:
        try:
            run_mfa_align(
                corpus_dir=mfa_corpus,
                output_dir=mfa_output,
                mfa_env=Q1_CONFIG["mfa_env"],
                mfa_root=Q1_CONFIG["mfa_root"],
                dictionary=Q1_CONFIG["mfa_dictionary"],
                acoustic_model=Q1_CONFIG["mfa_acoustic_model"],
                overwrite=True,
                log_path=log_root / "mfa_align.log",
            )
        except Exception as exc:
            mfa_run_error = f"{type(exc).__name__}: {exc}"

    all_token_rows = []
    summaries = []
    for sample_id, context in contexts.items():
        quality = context["quality"]
        if sample_id in selected_for_mfa and mfa_run_error is None:
            intervals, mfa_info = _mfa_intervals(
                sample_id, context["words"], mfa_output, context["window"], quality)
        else:
            intervals = _word_intervals_from_quality(context["words"], quality)
            if quality["alignment_type"] == "full_mfa":
                reason = mfa_run_error or context["window"]["reason"]
                mfa_info = {"mfa_status": f"fallback:{reason}", "mfa_word_match_ratio": None}
            else:
                mfa_info = {"mfa_status": "not_requested", "mfa_word_match_ratio": None}
        token_rows = _map_tokens_to_words(
            context["raw_text"], context["token_info"], context["word_spans"], intervals, context["media"])
        for row in token_rows:
            row["sample_id"] = sample_id
            row["video_path"] = context["video_path"]
            row["fps"] = context["media"].get("fps")
            row["presented_frame_count"] = context["media"].get("video_frame_count")
        all_token_rows.extend(token_rows)
        verified = sum(row["timestamp_status"] == "verified_word_interval" for row in token_rows)
        summary = {
            "sample_id": sample_id,
            "body_length": context["token_info"]["body_length"],
            "verified_token_count": verified,
            "verified_token_ratio": verified / max(1, len(token_rows)),
            "requested_alignment_type": quality["alignment_type"],
            "final_alignment_method": next((item["source"] for item in intervals if item is not None), "no_word_alignment"),
            "consistency_score": quality["consistency_score"],
            "speech_present": quality["speech_present"],
            "mfa_window_eligible": context["window"]["eligible"],
            "mfa_window_start": context["window"]["window_start"],
            "mfa_window_end": context["window"]["window_end"],
            **mfa_info,
            "duration_seconds": context["media"].get("duration_seconds"),
            "fps": context["media"].get("fps"),
            "presented_frame_count": context["media"].get("video_frame_count"),
        }
        summaries.append(summary)
        _write_json(RESULT_ROOT / "samples" / sample_id / "timeline.json", {
            "sample": summary,
            "media": context["media"],
            "quality": quality,
            "mfa_window": context["window"],
            "tokens": token_rows,
        })
        print(f"timeline attachment4/{sample_id}: {verified}/{len(token_rows)} tokens", flush=True)

    suffix = "" if tag == "all" else f"_{tag}"
    timeline_path = RESULT_ROOT / f"token_timeline{suffix}.csv"
    summary_path = RESULT_ROOT / f"alignment_summary{suffix}.json"
    with timeline_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_token_rows[0]))
        writer.writeheader()
        writer.writerows(all_token_rows)
    report = {
        "status": "completed" if mfa_run_error is None else "completed_with_mfa_fallback",
        "sample_count": len(summaries),
        "token_count": len(all_token_rows),
        "verified_token_count": sum(item["verified_token_count"] for item in summaries),
        "verified_token_ratio": sum(item["verified_token_count"] for item in summaries) / max(1, len(all_token_rows)),
        "alignment_type_counts": {
            name: sum(item["final_alignment_method"] == name for item in summaries)
            for name in sorted({item["final_alignment_method"] for item in summaries})
        },
        "mfa_run_error": mfa_run_error,
        "timeline_csv": str(timeline_path.resolve()),
        "samples": summaries,
        "q1_source_policy": "read-only import; no Q1 file or Q1 output modified",
    }
    _write_json(summary_path, report)
    print(json.dumps({key: report[key] for key in ("status", "sample_count", "token_count", "verified_token_ratio", "alignment_type_counts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
