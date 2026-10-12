from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from q1lib import load_config
from q1lib.data import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize text-audio consistency scores for manual threshold review"
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    source = config.output_root / "summary" / "quality_audit.csv"
    if not source.is_file():
        raise FileNotFoundError("quality_audit.csv is missing; run the full quality stage first")
    frame = pd.read_csv(source)
    usable = frame.loc[frame["status"].eq("ok")].copy()
    if usable.empty:
        raise ValueError("quality_audit.csv contains no successful rows")
    scores = pd.to_numeric(usable["consistency_score"], errors="coerce").dropna()
    quantiles = {f"q{int(q * 100):02d}": float(scores.quantile(q)) for q in np.linspace(0, 1, 11)}
    low = float(config.alignment["low_threshold"])
    high = float(config.alignment["high_threshold"])
    usable["configured_group"] = np.select(
        [usable["consistency_score"] >= high, usable["consistency_score"] >= low],
        ["full_mfa", "partial_asr_timestamp"],
        default="no_word_alignment",
    )
    detail_rows: list[dict[str, str]] = []
    for sample_id in usable["sample_id"].astype(str):
        detail_path = config.output_root / "samples" / sample_id / "quality.json"
        detail = json.loads(detail_path.read_text(encoding="utf-8")) if detail_path.is_file() else {}
        metadata_path = config.output_root / "samples" / sample_id / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        detail_rows.append(
            {
                "sample_id": sample_id,
                "official_text": str(detail.get("official_text", "")),
                "asr_text": str(detail.get("asr_text", "")),
                "video_path": str(metadata.get("source", {}).get("video_path", "")),
            }
        )
    usable = usable.merge(pd.DataFrame(detail_rows), how="left", on="sample_id")
    usable["distance_to_nearest_threshold"] = np.minimum(
        (usable["consistency_score"] - low).abs(),
        (usable["consistency_score"] - high).abs(),
    )
    usable["review_priority"] = np.where(
        usable["configured_group"].eq("partial_asr_timestamp"),
        "highest",
        np.where(usable["distance_to_nearest_threshold"] <= 0.10, "high", "normal"),
    )
    review_path = config.output_root / "summary" / "quality_manual_review.csv"
    previous_reviews = None
    if review_path.is_file():
        previous = pd.read_csv(review_path, dtype=str, keep_default_na=False)
        required = {"sample_id", "manual_review", "manual_note"}
        if required.issubset(previous.columns):
            previous_reviews = previous[list(required)].drop_duplicates("sample_id", keep="last")
    if previous_reviews is None:
        usable["manual_review"] = ""
        usable["manual_note"] = ""
    else:
        usable = usable.merge(previous_reviews, how="left", on="sample_id")
        usable[["manual_review", "manual_note"]] = usable[
            ["manual_review", "manual_note"]
        ].fillna("")
    priority_order = pd.Categorical(
        usable["review_priority"], categories=["highest", "high", "normal"], ordered=True
    )
    usable = usable.assign(_priority_order=priority_order).sort_values(
        ["_priority_order", "distance_to_nearest_threshold", "consistency_score"]
    ).drop(columns="_priority_order")
    usable.to_csv(review_path, index=False, encoding="utf-8-sig")
    payload = {
        "successful_samples": int(len(usable)),
        "score_quantiles": quantiles,
        "configured_thresholds": {"low": low, "high": high},
        "configured_group_counts": {
            str(key): int(value) for key, value in usable["configured_group"].value_counts().items()
        },
        "threshold_status": config.alignment["threshold_status"],
        "manual_review_file": str(review_path.resolve()),
        "instruction": (
            "Fill manual_review with full/partial/none after listening and checking official text; "
            "then revise config thresholds only if the reviewed boundary supports it."
        ),
    }
    destination = config.output_root / "summary" / "quality_score_distribution.json"
    write_json(destination, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
