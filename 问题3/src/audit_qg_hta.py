# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：DeepSeek（深度求索）
# 版本/型号：DeepSeek-V4.1-Flash
# 开发机构/公司：杭州深度求索人工智能基础技术研究有限公司
# 版本颁布日期：2026年9月10日
# 版本信息来源：https://deepseek.com/news/deepseek-v4-1-flash/

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


Q3_ROOT = Path(__file__).resolve().parents[1]
Q1_ROOT = Q3_ROOT.parent / "问题1" / "code"
RESULT_ROOT = Q3_ROOT / "results" / "qg_hta"
TIMELINE_PATH = RESULT_ROOT / "token_timeline.csv"
SUMMARY_PATH = RESULT_ROOT / "alignment_summary.json"
EVIDENCE_PATH = Q3_ROOT / "results" / "tables" / "attachment4_top_local_evidence_timed.csv"
OUTPUT_PATH = Q3_ROOT / "results" / "audit" / "qg_hta_audit.json"
EXPECTED_Q1_SOURCE_DIGEST = "729d74dfd8ab3e3342e211ca313c58c4ffb5d66d397eae58c1dbc68ae5e2fbf6"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _q1_source_digest() -> tuple[int, str]:
    tracked = sorted(
        path for path in Q1_ROOT.rglob("*")
        if path.is_file() and (path.suffix.lower() in {".py", ".yaml", ".yml"} or path.name == "README.md")
    )
    digest = hashlib.sha256()
    for path in tracked:
        digest.update(str(path.relative_to(Q1_ROOT)).replace("\\", "/").encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return len(tracked), digest.hexdigest()


def main() -> None:
    timeline = _read_csv(TIMELINE_PATH)
    evidence = _read_csv(EVIDENCE_PATH)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if summary["sample_count"] != 20 or len(timeline) != 564 or len(evidence) != 354:
        raise ValueError("unexpected sample/token/evidence count")
    keys = [(row["sample_id"], int(row["body_position"])) for row in timeline]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate timeline key")
    if set(row["sample_id"] for row in timeline) != {f"{index:02d}" for index in range(1, 21)}:
        raise ValueError("timeline sample coverage mismatch")

    verified = []
    for row in timeline:
        status = row["timestamp_status"]
        if status == "verified_word_interval":
            start = float(row["time_start_seconds"])
            end = float(row["time_end_seconds"])
            duration = float(next(item["duration_seconds"] for item in summary["samples"] if item["sample_id"] == row["sample_id"]))
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration + 0.1):
                raise ValueError(f"invalid time interval: {row}")
            frame_start = int(row["video_frame_start"])
            frame_end = int(row["video_frame_end_exclusive"])
            frame_count = int(row["presented_frame_count"])
            key_frame = int(row["video_key_frame"])
            if not (0 <= frame_start < frame_end <= frame_count and frame_start <= key_frame < frame_end):
                raise ValueError(f"invalid video frame interval: {row}")
            verified.append(row)
        elif any(row[field] for field in (
            "time_start_seconds", "time_end_seconds", "video_frame_start",
            "video_frame_end_exclusive", "video_key_frame",
        )):
            raise ValueError("unverified token contains inferred time/frame values")

    sample15 = [row for row in timeline if row["sample_id"] == "15"]
    if not sample15 or any(row["timestamp_status"] == "verified_word_interval" for row in sample15):
        raise ValueError("no_word_alignment sample 15 must not expose word timestamps")

    mapped_evidence = [row for row in evidence if row["timestamp_status"] in {
        "qg_hta_verified", "qg_hta_partial_window_coverage"}]
    unmapped_evidence = [row for row in evidence if row["timestamp_status"] == "qg_hta_no_reliable_interval_in_window"]
    if len(mapped_evidence) + len(unmapped_evidence) != len(evidence):
        raise ValueError("unexpected mapped/unmapped evidence count")
    for row in mapped_evidence:
        if not row["time_start_seconds"] or not row["time_end_seconds"]:
            raise ValueError("mapped evidence lacks time interval")
    for row in unmapped_evidence:
        if row["time_start_seconds"] or row["time_end_seconds"]:
            raise ValueError("unmapped evidence contains fabricated time interval")

    keyframe_paths = [Path(row["video_keyframe_image"]) for row in evidence if row["video_keyframe_image"]]
    if len(keyframe_paths) != len(set(keyframe_paths)) or not all(path.is_file() for path in keyframe_paths):
        raise ValueError("keyframe file coverage mismatch")
    if any(row["sample_id"] == "13" and row["modality"] == "vision" for row in evidence):
        raise ValueError("all-zero visual sample 13 must not expose visual local evidence")
    if any(row.get("time_mapping_is_official", "").lower() != "false" for row in evidence):
        raise ValueError("QG-HTA mapping must be marked non-official")
    q1_file_count, q1_digest = _q1_source_digest()
    if q1_digest != EXPECTED_Q1_SOURCE_DIGEST:
        raise ValueError("Q1 source/config fingerprint changed")

    report = {
        "status": "passed",
        "sample_count": summary["sample_count"],
        "token_count": len(timeline),
        "verified_token_count": len(verified),
        "verified_token_ratio": len(verified) / len(timeline),
        "alignment_type_counts": summary["alignment_type_counts"],
        "evidence_rows": len(evidence),
        "mapped_evidence_rows": len(mapped_evidence),
        "fully_covered_evidence_rows": sum(row["timestamp_status"] == "qg_hta_verified" for row in evidence),
        "partially_covered_evidence_rows": sum(row["timestamp_status"] == "qg_hta_partial_window_coverage" for row in evidence),
        "unmapped_evidence_rows": len(unmapped_evidence),
        "keyframe_image_count": len(keyframe_paths),
        "evidence_status_counts": dict(Counter(row["timestamp_status"] for row in evidence)),
        "q1_tracked_source_config_files": q1_file_count,
        "q1_source_config_digest": q1_digest,
        "q1_unchanged": True,
        "claim_boundary": "Only quality-gated reconstructed QG-HTA word intervals are reported; they are not official feature-row timestamps and unavailable intervals remain blank.",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
