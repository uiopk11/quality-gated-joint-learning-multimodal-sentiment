from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from q3_data import ALIGNED_ROOT, Q3_ROOT, UNALIGNED_ROOT, load_attachment4_sample


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    samples = []
    for number in range(1, 21):
        batch, metadata = load_attachment4_sample(number)
        unaligned_path = UNALIGNED_ROOT / f"{number:02d}.pkl"
        video_path = ALIGNED_ROOT / "videos" / f"{number:02d}.mp4"
        samples.append({
            **metadata,
            "aligned_sha256": sha256(Path(metadata["aligned_pickle"])),
            "unaligned_pickle": str(unaligned_path),
            "unaligned_sha256": sha256(unaligned_path),
            "video_sha256": sha256(video_path),
            "audio_nonzero_body_rows": int(np.any(batch["audio"][0, : metadata["body_length"]].numpy() != 0, axis=1).sum()),
            "vision_nonzero_body_rows": int(np.any(batch["vision"][0, : metadata["body_length"]].numpy() != 0, axis=1).sum()),
        })
    report = {
        "status": "passed_with_documented_input_anomalies",
        "sample_count": len(samples),
        "body_length_min": min(item["body_length"] for item in samples),
        "body_length_max": max(item["body_length"] for item in samples),
        "timestamps_available_in_aligned_feature_files": False,
        "all_zero_audio_samples": [item["sample_id"] for item in samples if item["audio_nonzero_body_rows"] == 0],
        "all_zero_vision_samples": [item["sample_id"] for item in samples if item["vision_nonzero_body_rows"] == 0],
        "truncated_text_samples": [item["sample_id"] for item in samples if item["truncated_wordpiece_count"] > 0],
        "attachment2_test_overlap_samples": [item["sample_id"] for item in samples if item["overlaps_attachment2_test"]],
        "claim_boundary": "All-zero aligned modalities are unavailable observations, not usable local evidence; reconstructed QG-HTA times are not official feature-row timestamps.",
        "samples": samples,
    }
    output = Q3_ROOT / "results" / "audit"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "attachment4_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(path), "sample_count": len(samples),
                      "all_zero_vision_samples": report["all_zero_vision_samples"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
