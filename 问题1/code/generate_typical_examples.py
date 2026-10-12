# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：DeepSeek（深度求索）
# 版本/型号：DeepSeek-V4.1-Flash
# 开发机构/公司：杭州深度求索人工智能基础技术研究有限公司
# 版本颁布日期：2026年9月10日
# 版本信息来源：https://deepseek.com/news/deepseek-v4-1-flash/

"""Regenerate formal typical-sample figures and a human-verification worksheet."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from q1lib import load_config


EXAMPLES = (
    ("-3g5yACwYnA__13", "normal_full_mfa"),
    ("-9y-fZ3swSY__0", "boundary_partial_alignment"),
    ("-mJ2ud6oKI8__1", "silent_no_word_alignment"),
)
HUMAN_COLUMNS = (
    "human_audio_timestamp_check",
    "human_video_correspondence_check",
    "human_mask_check",
    "human_note",
)

# Typical-sample figures use Times/Georgia serif typography in
# visualize_alignment.py.  Scale > 1 keeps section titles and captions readable
# after the wide figure is reduced to page width.
FIGURE_FONT_SCALE = "1.28"


def main() -> None:
    script_root = Path(__file__).resolve().parent
    config = load_config(script_root / "config.yaml")
    review_path = config.output_root / "summary" / "typical_samples_validation.csv"
    previous: dict[str, dict[str, str]] = {}
    if review_path.is_file():
        old = pd.read_csv(review_path, dtype=str, keep_default_na=False)
        previous = {str(row["sample_id"]): row for row in old.to_dict(orient="records")}

    rows = []
    for sample_id, role in EXAMPLES:
        figure_path = config.output_root / "figures" / f"{sample_id}_alignment.png"
        subprocess.run(
            [sys.executable, str(script_root / "visualize_alignment.py"),
             f"--sample-id={sample_id}", "--font-scale", FIGURE_FONT_SCALE,
             "--output", str(figure_path)],
            check=True,
        )
        sample_dir = config.output_root / "samples" / sample_id
        metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
        old_row = previous.get(sample_id, {})
        row = {
            "sample_id": sample_id,
            "role": role,
            "alignment_type": metadata["quality"]["actual_alignment_type"],
            "consistency_score": metadata["quality"]["consistency_score"],
            "official_text": metadata["official_text"],
            "asr_text": metadata["asr_text"],
            "figure_path": str(figure_path.resolve()),
            "automatic_integrity_passed": metadata["status"]["integrity_passed"],
        }
        for column in HUMAN_COLUMNS:
            row[column] = old_row.get(column, "")
        rows.append(row)
    pd.DataFrame(rows).to_csv(review_path, index=False, encoding="utf-8-sig")
    print(f"Created {len(rows)} figures and review worksheet: {review_path}")


if __name__ == "__main__":
    main()
