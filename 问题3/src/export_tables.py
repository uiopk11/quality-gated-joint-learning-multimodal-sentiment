# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：DeepSeek（深度求索）
# 版本/型号：DeepSeek-V4.1-Flash
# 开发机构/公司：杭州深度求索人工智能基础技术研究有限公司
# 版本颁布日期：2026年9月10日
# 版本信息来源：https://deepseek.com/news/deepseek-v4-1-flash/

from __future__ import annotations

import csv
import json
from pathlib import Path


Q3_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = Q3_ROOT / "results" / "attachment4" / "samples"
TABLE_ROOT = Q3_ROOT / "results" / "tables"
MODALITIES = ("text", "audio", "vision")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(SAMPLE_ROOT.glob("*.json"))]
    if len(records) != 20:
        raise ValueError("Expected 20 explanation records")
    top_rows = []
    position_rows = []
    for record in records:
        sample = record["sample"]
        for modality in MODALITIES:
            local = record["local_explanation"][modality]
            for target, key in (("classification", "top_classification_windows"),
                                ("intensity", "top_intensity_windows")):
                for rank, window in enumerate(local[key], start=1):
                    top_rows.append({
                        "sample_id": sample["sample_id"],
                        "modality": modality,
                        "target": target,
                        "rank": rank,
                        "body_start": window["body_start"],
                        "body_end_exclusive": window["body_end"],
                        "source_slot_start": window["source_slot_start"],
                        "source_slot_end_exclusive": window["source_slot_end_exclusive"],
                        "wordpieces": " ".join(window["wordpieces"]),
                        "original_text_fragment": window["original_text_fragment"],
                        "classification_margin_delta": window["classification_margin_delta"],
                        "intensity_delta": window["intensity_delta"],
                        "absolute_classification_delta": window["absolute_classification_delta"],
                        "absolute_intensity_delta": window["absolute_intensity_delta"],
                        "video_path": sample["video"],
                        "time_start_seconds": "",
                        "time_end_seconds": "",
                        "timestamp_status": "pending reliable forced alignment; not inferred from 50 slots",
                    })
            for position in local["positions"]:
                position_rows.append({
                    "sample_id": sample["sample_id"],
                    "modality": modality,
                    **position,
                })
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    top_path = TABLE_ROOT / "attachment4_top_local_evidence.csv"
    position_path = TABLE_ROOT / "attachment4_position_importance.csv"
    write_csv(top_path, top_rows)
    write_csv(position_path, position_rows)
    report = {
        "status": "completed",
        "top_evidence_rows": len(top_rows),
        "position_rows": len(position_rows),
        "top_evidence_csv": str(top_path),
        "position_importance_csv": str(position_path),
        "timestamp_status": "not available; no equal-duration inference was used",
    }
    report_path = TABLE_ROOT / "export_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
