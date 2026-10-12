# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：DeepSeek（深度求索）
# 版本/型号：DeepSeek-V4.1-Flash
# 开发机构/公司：杭州深度求索人工智能基础技术研究有限公司
# 版本颁布日期：2026年9月10日
# 版本信息来源：https://deepseek.com/news/deepseek-v4-1-flash/

"""Export the final attachment-4 CSV with package-relative keyframe paths."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "tables" / "attachment4_formal_prediction_explanation_summary.csv"
TARGET = ROOT / "附件4_测试集预测与解释结果.csv"
SOURCE_FRAMES = ROOT / "results" / "qg_hta" / "keyframes"
TARGET_FRAMES = ROOT / "evidence" / "keyframes"


def main() -> None:
    with SOURCE.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if fields is None:
            raise ValueError("Missing CSV header")
        rows = list(reader)
    if len(rows) != 20 or len({row["sample_id"] for row in rows}) != 20:
        raise ValueError("Expected 20 distinct attachment-4 samples")
    TARGET_FRAMES.mkdir(parents=True, exist_ok=True)
    for row in rows:
        for field in fields:
            if not field.endswith("_keyframe_image") or not row[field]:
                continue
            filename = Path(row[field].replace("\\", "/")).name
            source = SOURCE_FRAMES / filename
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = TARGET_FRAMES / filename
            if not destination.exists():
                shutil.copyfile(source, destination)
            row[field] = f"evidence/keyframes/{filename}"
    with TARGET.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(TARGET, "rows:", len(rows))


if __name__ == "__main__":
    main()
