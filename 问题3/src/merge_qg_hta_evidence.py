from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from pathlib import Path


Q3_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_ROOT = Path(os.environ.get("Q1_RESOURCE_ROOT", str(Q3_ROOT / "external_q1"))).expanduser().resolve()
TIMELINE_PATH = Q3_ROOT / "results" / "qg_hta" / "token_timeline.csv"
EVIDENCE_PATH = Q3_ROOT / "results" / "tables" / "attachment4_top_local_evidence.csv"
OUTPUT_PATH = Q3_ROOT / "results" / "tables" / "attachment4_top_local_evidence_timed.csv"
REPORT_PATH = Q3_ROOT / "results" / "qg_hta" / "evidence_time_mapping_audit.json"
KEYFRAME_ROOT = Q3_ROOT / "results" / "qg_hta" / "keyframes"
FFMPEG = RESOURCE_ROOT / ".envs" / "q1" / "Library" / "bin" / "ffmpeg.exe"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _number(value: str) -> float | None:
    return None if value in ("", None) else float(value)


def _integer(value: str) -> int | None:
    return None if value in ("", None) else int(float(value))


def _extract_keyframe(video_path: str, timestamp: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(FFMPEG), "-y", "-v", "error", "-ss", f"{timestamp:.6f}",
        "-i", video_path, "-frames:v", "1", "-q:v", "2", str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError(f"keyframe extraction failed: {completed.stderr[-1000:]}")


def main() -> None:
    timeline = _read_csv(TIMELINE_PATH)
    evidence = _read_csv(EVIDENCE_PATH)
    by_sample_position = {
        (row["sample_id"], int(row["body_position"])): row for row in timeline
    }
    if len(by_sample_position) != len(timeline):
        raise ValueError("duplicate sample/body_position in QG-HTA timeline")
    output = []
    mapped = 0
    full_coverage = 0
    keyframes = 0
    failures = []
    for row in evidence:
        sample_id = row["sample_id"]
        start = int(row["body_start"])
        end = int(row["body_end_exclusive"])
        tokens = [by_sample_position[(sample_id, position)] for position in range(start, end)]
        timed = [item for item in tokens if item["timestamp_status"] == "verified_word_interval"]
        enriched = dict(row)
        enriched["qg_hta_token_coverage"] = len(timed) / max(1, len(tokens))
        enriched["alignment_methods"] = ";".join(sorted({item["alignment_method"] for item in timed if item["alignment_method"]}))
        enriched["video_frame_start"] = ""
        enriched["video_frame_end_exclusive"] = ""
        enriched["video_key_frame"] = ""
        enriched["video_keyframe_image"] = ""
        enriched["time_mapping_source"] = "reconstructed_qg_hta"
        enriched["time_mapping_is_official"] = False
        enriched["feature_row_timestamp_recovered"] = False
        enriched["input_observation_status"] = "available"
        if timed:
            time_start = min(float(item["time_start_seconds"]) for item in timed)
            time_end = max(float(item["time_end_seconds"]) for item in timed)
            frame_starts = [_integer(item["video_frame_start"]) for item in timed]
            frame_ends = [_integer(item["video_frame_end_exclusive"]) for item in timed]
            key_frame_values = [_integer(item["video_key_frame"]) for item in timed]
            enriched["time_start_seconds"] = time_start
            enriched["time_end_seconds"] = time_end
            enriched["video_frame_start"] = min(item for item in frame_starts if item is not None)
            enriched["video_frame_end_exclusive"] = max(item for item in frame_ends if item is not None)
            enriched["video_key_frame"] = next(item for item in key_frame_values if item is not None)
            enriched["timestamp_status"] = "qg_hta_verified" if len(timed) == len(tokens) else "qg_hta_partial_window_coverage"
            mapped += 1
            full_coverage += int(len(timed) == len(tokens))
            if row["modality"] == "vision":
                midpoint = (time_start + time_end) * 0.5
                image_path = KEYFRAME_ROOT / f"{sample_id}_{row['target']}_rank{row['rank']}.jpg"
                try:
                    _extract_keyframe(row["video_path"], midpoint, image_path)
                    enriched["video_keyframe_image"] = str(image_path.resolve())
                    keyframes += 1
                except Exception as exc:
                    failures.append({"sample_id": sample_id, "target": row["target"], "rank": row["rank"], "error": str(exc)})
        else:
            enriched["time_start_seconds"] = ""
            enriched["time_end_seconds"] = ""
            enriched["timestamp_status"] = "qg_hta_no_reliable_interval_in_window"
        output.append(enriched)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    report = {
        "status": "passed" if not failures else "completed_with_keyframe_failures",
        "evidence_rows": len(output),
        "mapped_time_rows": mapped,
        "fully_covered_rows": full_coverage,
        "unmapped_rows": len(output) - mapped,
        "visual_evidence_rows": sum(row["modality"] == "vision" for row in evidence),
        "keyframe_images": keyframes,
        "keyframe_failures": failures,
        "output_csv": str(OUTPUT_PATH.resolve()),
        "claim_boundary": "Only verified QG-HTA word intervals are mapped; 50 slots were not divided uniformly.",
        "time_mapping_provenance": "Reconstructed from raw text/audio/video by QG-HTA; not supplied by Attachment 4 and not the original feature extractor's row timestamps.",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
