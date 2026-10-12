"""将问题1的100个单样本特征合并为一个可追溯的统一特征文件。

直接在VS Code中运行本文件即可。默认输出：
    输出/summary/question1_features_100.pkl.gz

统一文件保留标准化双视图、全部核心掩码、时间边界、样本标签、
300行样本—模态全量表、109行特征字典，以及逐样本质量与对齐记录。
原有100个features.npz不会被修改或删除。
"""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "输出"
SUMMARY_ROOT = OUTPUT_ROOT / "summary"
SAMPLES_ROOT = OUTPUT_ROOT / "samples"
DEFAULT_OUTPUT = SUMMARY_ROOT / "question1_features_100.pkl.gz"

EXPECTED_SHAPES = {
    "aligned_text": (50, 768),
    "aligned_audio": (50, 74),
    "aligned_vision": (50, 35),
    "unaligned_text": (50, 768),
    "unaligned_audio": (500, 74),
    "unaligned_vision": (500, 35),
    "text_attention_mask": (50,),
    "speech_present_mask": (50,),
    "official_asr_match_mask": (50,),
    "audio_valid_mask": (50,),
    "vision_valid_mask": (50,),
    "cross_modal_mask": (50,),
    "unaligned_audio_valid_mask": (500,),
    "unaligned_vision_valid_mask": (500,),
    "unaligned_speech_present_mask": (500,),
    "unaligned_bin_starts": (500,),
    "unaligned_bin_ends": (500,),
    "token_starts": (50,),
    "token_ends": (50,),
    "word_ids": (50,),
    "tokens": (50,),
}


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """将DataFrame转换为不含NaN的普通Python记录。"""
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.parent.parent.resolve()).as_posix()


def _check_array(sample_id: str, name: str, value: np.ndarray) -> None:
    expected = EXPECTED_SHAPES[name]
    if value.shape != expected:
        raise ValueError(f"{sample_id}: {name} shape={value.shape}, expected={expected}")
    if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
        raise ValueError(f"{sample_id}: {name} contains NaN or infinity")


def build_combined_file(output_path: Path = DEFAULT_OUTPUT) -> Path:
    all_samples = pd.read_csv(SUMMARY_ROOT / "all_samples.csv", dtype={"clip_id": str})
    by_modality = pd.read_csv(
        SUMMARY_ROOT / "all_samples_by_modality.csv", dtype={"clip_id": str}
    )
    feature_dictionary = pd.read_csv(SUMMARY_ROOT / "feature_dictionary.csv")
    audit_summary = _read_json(SUMMARY_ROOT / "audit_summary.json")

    if len(all_samples) != 100 or all_samples["sample_id"].nunique() != 100:
        raise ValueError("all_samples.csv must contain exactly 100 unique samples")
    if len(by_modality) != 300:
        raise ValueError("all_samples_by_modality.csv must contain 300 sample-modality rows")
    modality_counts = by_modality["modality"].value_counts().to_dict()
    if modality_counts != {"text": 100, "audio": 100, "vision": 100}:
        raise ValueError(f"unexpected modality counts: {modality_counts}")
    if len(feature_dictionary) != 109:
        raise ValueError("feature_dictionary.csv must contain 109 audio/vision fields")
    if not bool(audit_summary.get("all_passed")):
        raise ValueError("audit_summary.json does not report all_passed=true")

    sample_ids = all_samples["sample_id"].astype(str).tolist()
    stacked: dict[str, list[np.ndarray]] = {name: [] for name in EXPECTED_SHAPES}
    sample_details: list[dict[str, Any]] = []
    audio_names: np.ndarray | None = None
    vision_names: np.ndarray | None = None

    for sample_id in sample_ids:
        sample_dir = SAMPLES_ROOT / sample_id
        feature_path = sample_dir / "features.npz"
        metadata_path = sample_dir / "metadata.json"
        quality_path = sample_dir / "quality.json"
        word_path = sample_dir / "word_alignment.csv"
        token_path = sample_dir / "token_alignment.csv"
        required = (feature_path, metadata_path, quality_path, word_path, token_path)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{sample_id}: missing required files: {missing}")

        with np.load(feature_path, allow_pickle=False) as data:
            source_names = {
                "aligned_text": "aligned_text",
                "aligned_audio": "aligned_audio",
                "aligned_vision": "aligned_vision",
                "unaligned_text": "unaligned_text",
                "unaligned_audio": "unaligned_audio",
                "unaligned_vision": "unaligned_vision",
                "text_attention_mask": "text_attention_mask",
                "speech_present_mask": "speech_present_mask",
                "official_asr_match_mask": "official_asr_match_mask",
                "audio_valid_mask": "audio_valid_mask",
                "vision_valid_mask": "vision_valid_mask",
                "cross_modal_mask": "cross_modal_mask",
                "unaligned_audio_valid_mask": "unaligned_audio_valid_mask",
                "unaligned_vision_valid_mask": "unaligned_vision_valid_mask",
                "unaligned_speech_present_mask": "unaligned_speech_present_mask",
                "unaligned_bin_starts": "unaligned_bin_starts",
                "unaligned_bin_ends": "unaligned_bin_ends",
                "token_starts": "token_starts",
                "token_ends": "token_ends",
                "word_ids": "word_ids",
                "tokens": "tokens",
            }
            for output_name, source_name in source_names.items():
                value = np.asarray(data[source_name])
                _check_array(sample_id, output_name, value)
                stacked[output_name].append(value.copy())

            current_audio_names = np.asarray(data["audio_feature_names"]).astype(str)
            current_vision_names = np.asarray(data["vision_feature_names"]).astype(str)
            if audio_names is None:
                audio_names = current_audio_names
                vision_names = current_vision_names
            elif not np.array_equal(audio_names, current_audio_names):
                raise ValueError(f"{sample_id}: audio feature field order differs")
            elif not np.array_equal(vision_names, current_vision_names):
                raise ValueError(f"{sample_id}: vision feature field order differs")

        metadata = _read_json(metadata_path)
        quality = _read_json(quality_path)
        sample_details.append(
            {
                "sample_id": sample_id,
                "source_video_sha256": metadata["source"]["video_sha256"],
                "source_video_path": metadata["source"]["video_path"],
                "metadata": metadata,
                "quality": quality,
                "word_alignment": _records(pd.read_csv(word_path)),
                "token_alignment": _records(pd.read_csv(token_path)),
                "original_files": {
                    "features": _relative(feature_path),
                    "metadata": _relative(metadata_path),
                    "quality": _relative(quality_path),
                    "word_alignment": _relative(word_path),
                    "token_alignment": _relative(token_path),
                },
            }
        )

    arrays = {name: np.stack(values, axis=0) for name, values in stacked.items()}
    if not np.array_equal(arrays["aligned_text"], arrays["unaligned_text"]):
        raise ValueError("aligned_text and unaligned_text differ; refusing silent deduplication")
    text = arrays.pop("aligned_text")
    arrays.pop("unaligned_text")

    payload: dict[str, Any] = {
        "schema": {
            "name": "QG-HTA-question1-features",
            "version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sample_count": 100,
            "sample_axis_order": sample_ids,
            "aligned_shapes": {
                "text": [100, 50, 768],
                "audio": [100, 50, 74],
                "vision": [100, 50, 35],
            },
            "unaligned_shapes": {
                "text": [100, 50, 768],
                "audio": [100, 500, 74],
                "vision": [100, 500, 35],
            },
            "text_policy": "official label-100.xlsx text; ASR never replaces it",
            "padding_value": 0.0,
        },
        "id": np.asarray(sample_ids),
        "video_id": all_samples["video_id"].astype(str).to_numpy(),
        "clip_id": all_samples["clip_id"].astype(str).to_numpy(),
        "annotations": all_samples["annotation"].astype(str).to_numpy(),
        "regression_labels": all_samples["label"].to_numpy(dtype=np.float32),
        "raw_text": np.asarray(
            [detail["metadata"]["official_text"] for detail in sample_details]
        ),
        "aligned_50": {
            "text": text,
            "audio": arrays["aligned_audio"],
            "vision": arrays["aligned_vision"],
            "text_attention_mask": arrays["text_attention_mask"],
            "speech_present_mask": arrays["speech_present_mask"],
            "official_asr_match_mask": arrays["official_asr_match_mask"],
            "audio_valid_mask": arrays["audio_valid_mask"],
            "vision_valid_mask": arrays["vision_valid_mask"],
            "cross_modal_mask": arrays["cross_modal_mask"],
            "tokens": arrays["tokens"],
            "word_ids": arrays["word_ids"],
            "token_starts": arrays["token_starts"],
            "token_ends": arrays["token_ends"],
        },
        "unaligned_500": {
            # 文本轴与aligned_50完全相同，复用同一数组对象以减少pickle体积。
            "text": text,
            "audio": arrays["unaligned_audio"],
            "vision": arrays["unaligned_vision"],
            "audio_valid_mask": arrays["unaligned_audio_valid_mask"],
            "vision_valid_mask": arrays["unaligned_vision_valid_mask"],
            "speech_present_mask": arrays["unaligned_speech_present_mask"],
            "bin_starts": arrays["unaligned_bin_starts"],
            "bin_ends": arrays["unaligned_bin_ends"],
        },
        "feature_names": {
            "text": [f"bert_hidden_{index}" for index in range(768)],
            "audio": audio_names.tolist() if audio_names is not None else [],
            "vision": vision_names.tolist() if vision_names is not None else [],
        },
        "sample_summary": _records(all_samples),
        "sample_modality_summary": _records(by_modality),
        "feature_dictionary": _records(feature_dictionary),
        "audit_summary": audit_summary,
        "sample_details": sample_details,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wb", compresslevel=6) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # 写后重新读取关键结构，避免只生成了一个不可用的大文件。
    with gzip.open(output_path, "rb") as handle:
        check = pickle.load(handle)
    if check["schema"]["sample_count"] != 100:
        raise RuntimeError("combined file post-write validation failed: sample_count")
    if check["aligned_50"]["text"].shape != (100, 50, 768):
        raise RuntimeError("combined file post-write validation failed: aligned text")
    if check["unaligned_500"]["audio"].shape != (100, 500, 74):
        raise RuntimeError("combined file post-write validation failed: unaligned audio")
    if len(check["sample_modality_summary"]) != 300:
        raise RuntimeError("combined file post-write validation failed: modality table")

    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(f"Combined feature file: {output_path.resolve()}")
    print(f"Samples: {check['schema']['sample_count']}")
    print(f"Sample-modality rows: {len(check['sample_modality_summary'])}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.2f} MiB")
    print(f"SHA256: {digest}")
    return output_path


def main() -> None:
    build_combined_file()


if __name__ == "__main__":
    main()
