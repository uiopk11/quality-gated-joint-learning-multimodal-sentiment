from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from q1lib import load_config
from q1lib.data import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Question 1 generated features and traceability")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--summary", type=Path, default=None, help="optional summary CSV to audit")
    return parser.parse_args()


def check_sample(
    sample_id: str,
    sample_dir: Path,
    standard_length: int,
    unaligned_length: int,
) -> list[str]:
    errors: list[str] = []
    required = ["features.npz", "metadata.json", "word_alignment.csv", "token_alignment.csv"]
    for filename in required:
        if not (sample_dir / filename).is_file():
            errors.append(f"missing {filename}")
    if errors:
        return errors

    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    data = np.load(sample_dir / "features.npz", allow_pickle=False)
    required_arrays = {
        "text_full",
        "audio_full",
        "vision_full",
        "text_standard",
        "audio_standard",
        "vision_standard",
        "text_attention_mask",
        "audio_valid_mask",
        "vision_valid_mask",
        "cross_modal_mask",
        "speech_present_mask",
        "official_asr_match_mask",
        "unaligned_text",
        "unaligned_audio",
        "unaligned_vision",
        "unaligned_audio_valid_mask",
        "unaligned_vision_valid_mask",
        "unaligned_speech_present_mask",
        "unaligned_bin_starts",
        "unaligned_bin_ends",
        "audio_feature_names",
        "vision_feature_names",
        "word_starts",
        "word_ends",
    }
    missing_arrays = required_arrays - set(data.files)
    if missing_arrays:
        errors.append(f"missing arrays: {sorted(missing_arrays)}")
        return errors

    for name in (
        "text_full", "audio_full", "vision_full", "text_standard", "audio_standard",
        "vision_standard", "unaligned_text", "unaligned_audio", "unaligned_vision",
    ):
        if not np.isfinite(data[name]).all():
            errors.append(f"{name} contains NaN/inf")
    token_lengths = {data[name].shape[0] for name in ("text_full", "audio_full", "vision_full")}
    if len(token_lengths) != 1:
        errors.append(f"full token lengths disagree: {sorted(token_lengths)}")
    for name in (
        "text_standard",
        "audio_standard",
        "vision_standard",
        "text_attention_mask",
        "audio_valid_mask",
        "vision_valid_mask",
        "cross_modal_mask",
        "speech_present_mask",
        "official_asr_match_mask",
    ):
        if data[name].shape[0] != standard_length:
            errors.append(f"{name} length={data[name].shape[0]}, expected={standard_length}")
    expected_shapes = {
        "text_standard": (standard_length, 768),
        "audio_standard": (standard_length, 74),
        "vision_standard": (standard_length, 35),
        "unaligned_text": (standard_length, 768),
        "unaligned_audio": (unaligned_length, 74),
        "unaligned_vision": (unaligned_length, 35),
        "unaligned_audio_valid_mask": (unaligned_length,),
        "unaligned_vision_valid_mask": (unaligned_length,),
        "unaligned_speech_present_mask": (unaligned_length,),
        "unaligned_bin_starts": (unaligned_length,),
        "unaligned_bin_ends": (unaligned_length,),
    }
    for name, expected in expected_shapes.items():
        if data[name].shape != expected:
            errors.append(f"{name} shape={data[name].shape}, expected={expected}")
    if len(data["audio_feature_names"]) != 74 or len(set(data["audio_feature_names"].tolist())) != 74:
        errors.append("audio feature schema is not 74 unique fields")
    if len(data["vision_feature_names"]) != 35 or len(set(data["vision_feature_names"].tolist())) != 35:
        errors.append("vision feature schema is not 35 unique fields")

    bin_starts = data["unaligned_bin_starts"].astype(float)
    bin_ends = data["unaligned_bin_ends"].astype(float)
    if np.any(bin_ends <= bin_starts) or np.any(np.diff(bin_starts) < 0):
        errors.append("unaligned time bins are not positive and monotonic")
    if len(bin_starts) > 1 and not np.allclose(bin_starts[1:], bin_ends[:-1], atol=1e-9):
        errors.append("unaligned time bins are not contiguous")

    text_mask = data["text_attention_mask"].astype(bool)
    audio_mask = data["audio_valid_mask"].astype(bool)
    vision_mask = data["vision_valid_mask"].astype(bool)
    cross_mask = data["cross_modal_mask"].astype(bool)
    if np.any(cross_mask & ~(text_mask & audio_mask & vision_mask)):
        errors.append("cross_modal_mask is not a subset of all modality masks")

    starts = data["word_starts"].astype(float)
    ends = data["word_ends"].astype(float)
    duration = metadata.get("media", {}).get("duration_seconds")
    if duration is not None and (
        not np.isclose(bin_starts[0], 0.0, atol=1e-9)
        or not np.isclose(bin_ends[-1], float(duration), atol=1e-6)
    ):
        errors.append("unaligned time bins do not span the complete media duration")
    valid_intervals = (starts >= 0) | (ends >= 0)
    if len(starts) != len(ends) or np.any((starts >= 0) != (ends >= 0)):
        errors.append("word interval start/end missingness disagrees")
    valid_starts = starts[valid_intervals]
    valid_ends = ends[valid_intervals]
    if np.any(valid_ends <= valid_starts):
        errors.append("valid word interval has non-positive duration")
    if len(valid_starts) > 1 and (
        np.any(np.diff(valid_starts) < 0) or np.any(valid_starts[1:] < valid_ends[:-1] - 1e-4)
    ):
        errors.append("valid word intervals are non-monotonic or overlapping")
    if duration is not None and (
        np.any(valid_starts < -1e-6) or np.any(valid_ends > float(duration) + 0.1)
    ):
        errors.append("valid word interval exceeds media duration")

    word_ids = data["word_ids_full"].astype(int)
    token_starts = data["token_starts_full"].astype(float)
    token_ends = data["token_ends_full"].astype(float)
    for token_index, word_index in enumerate(word_ids):
        if word_index < 0:
            continue
        if word_index >= len(starts):
            errors.append("BERT token references an out-of-range official word")
            break
        expected_start = starts[word_index]
        expected_end = ends[word_index]
        if not (
            np.isclose(token_starts[token_index], expected_start, atol=1e-6)
            and np.isclose(token_ends[token_index], expected_end, atol=1e-6)
        ):
            errors.append("BERT WordPiece timestamp does not inherit its official-word interval")
            break

    actual_alignment = str(metadata.get("quality", {}).get("actual_alignment_type", ""))
    mfa_window = metadata.get("quality", {}).get("mfa_alignment_window")
    if actual_alignment.startswith("full_mfa"):
        if not isinstance(mfa_window, dict) or not mfa_window.get("eligible"):
            errors.append("full MFA sample lacks an eligible ASR-localized window")
        else:
            window_start = float(mfa_window["window_start"])
            window_end = float(mfa_window["window_end"])
            if len(valid_starts) and (
                np.any(valid_starts < window_start - 0.1)
                or np.any(valid_ends > window_end + 0.1)
            ):
                errors.append("MFA word interval exceeds its ASR-localized window")
            if actual_alignment == "full_mfa_windowed" and not mfa_window.get("is_windowed"):
                errors.append("windowed MFA label disagrees with window metadata")

    for path_name in ("video_path",):
        value = metadata.get("source", {}).get(path_name)
        if not value or not Path(value).is_file():
            errors.append(f"source path missing: {path_name}")
    for path_name in (
        "wav", "openface_csv", "features", "word_alignment", "token_alignment", "quality", "asr_words"
    ):
        value = metadata.get("files", {}).get(path_name)
        if not value or not Path(value).is_file():
            errors.append(f"output path missing: {path_name}")
    return errors


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    manifest_path = config.output_root / "manifest" / "manifest.csv"
    if args.summary is not None:
        summary_path = args.summary.resolve()
    elif args.allow_partial and (config.output_root / "summary" / "all_samples_partial.csv").is_file():
        summary_path = config.output_root / "summary" / "all_samples_partial.csv"
    else:
        summary_path = config.output_root / "summary" / "all_samples.csv"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("manifest.csv or all_samples.csv is missing")
    manifest = pd.read_csv(manifest_path)
    summary = pd.read_csv(summary_path)
    if not args.allow_partial and len(manifest) != 100:
        raise AssertionError(f"manifest must contain 100 samples, found {len(manifest)}")
    if not args.allow_partial and len(summary) != 100:
        raise AssertionError(f"summary must contain 100 samples, found {len(summary)}")

    summary_root = config.output_root / "summary"
    suffix = "_partial" if args.allow_partial else ""
    modality_path = summary_root / f"all_samples_by_modality{suffix}.csv"
    dictionary_path = summary_root / f"feature_dictionary{suffix}.csv"
    windows_path = summary_root / f"mfa_alignment_windows{suffix}.csv"
    if not modality_path.is_file() or not dictionary_path.is_file() or not windows_path.is_file():
        raise FileNotFoundError("modality summary, feature dictionary or MFA window table is missing")
    modality = pd.read_csv(modality_path)
    required_modality_columns = {
        "sample_id", "modality", "original_effective_length", "effective_length_unit",
        "source_duration_seconds", "valid_duration_seconds", "feature_dimension",
        "aligned_sequence_length", "unaligned_sequence_length", "alignment_granularity",
        "feature_file",
    }
    missing_columns = required_modality_columns - set(modality.columns)
    if missing_columns:
        raise AssertionError(f"modality summary is missing columns: {sorted(missing_columns)}")
    expected_modality_rows = len(summary) * 3
    if len(modality) != expected_modality_rows:
        raise AssertionError(
            f"modality summary must contain {expected_modality_rows} rows, found {len(modality)}"
        )
    counts = modality.groupby("sample_id")["modality"].agg(lambda values: set(values))
    if not all(value == {"text", "audio", "vision"} for value in counts):
        raise AssertionError("each sample must have exactly text, audio and vision modality rows")
    feature_dictionary = pd.read_csv(dictionary_path)
    dictionary_counts = feature_dictionary.groupby("modality").size().to_dict()
    if dictionary_counts != {"audio": 74, "vision": 35}:
        raise AssertionError(f"feature dictionary dimensions are wrong: {dictionary_counts}")
    windows = pd.read_csv(windows_path)
    required_window_columns = {
        "sample_id", "eligible", "reason", "official_anchor_start", "official_anchor_end",
        "window_start", "window_end", "prefix_extra_seconds", "suffix_extra_seconds",
        "is_windowed", "audio_duration_seconds",
    }
    if required_window_columns - set(windows.columns):
        raise AssertionError("MFA alignment-window table is missing required fields")

    rows: list[dict[str, Any]] = []
    summary_ids = set(summary["sample_id"].astype(str))
    expected_ids = set(manifest["sample_id"].astype(str))
    if not args.allow_partial and summary_ids != expected_ids:
        missing = sorted(expected_ids - summary_ids)
        extra = sorted(summary_ids - expected_ids)
        raise AssertionError(f"summary/manifest IDs differ; missing={missing}, extra={extra}")

    for item in summary.to_dict(orient="records"):
        sample_id = str(item["sample_id"])
        if item.get("status") != "ok":
            errors = [str(item.get("error", "sample processing failed"))]
        else:
            errors = check_sample(
                sample_id,
                config.output_root / "samples" / sample_id,
                int(config.sequence["standard_length"]),
                int(config.sequence["unaligned_length"]),
            )
        rows.append(
            {
                "sample_id": sample_id,
                "audit_passed": not errors,
                "error_count": len(errors),
                "errors": " | ".join(errors),
            }
        )

    report = pd.DataFrame(rows)
    destination = config.output_root / "summary" / f"audit_report{suffix}.csv"
    report.to_csv(destination, index=False, encoding="utf-8-sig")
    payload = {
        "manifest_count": len(manifest),
        "summary_count": len(summary),
        "passed_count": int(report["audit_passed"].sum()),
        "failed_count": int((~report["audit_passed"]).sum()),
        "all_passed": bool(report["audit_passed"].all()),
        "observed_video_duration_seconds": {
            "minimum": float(pd.to_numeric(summary["video_duration_seconds"]).min()),
            "maximum": float(pd.to_numeric(summary["video_duration_seconds"]).max()),
        },
        "alignment_type_counts": {
            str(key): int(value) for key, value in summary["alignment_type"].value_counts().items()
        },
        "modality_summary_count": int(len(modality)),
        "feature_dictionary_count": int(len(feature_dictionary)),
        "mfa_window_candidate_count": int(len(windows)),
        "mfa_window_eligible_count": int(
            windows["eligible"].astype(str).str.casefold().eq("true").sum()
        ),
        "report": str(destination.resolve()),
    }
    write_json(config.output_root / "summary" / f"audit_summary{suffix}.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if payload["all_passed"] else 1)


if __name__ == "__main__":
    main()
