# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：DeepSeek（深度求索）
# 版本/型号：DeepSeek-V4.1-Flash
# 开发机构/公司：杭州深度求索人工智能基础技术研究有限公司
# 版本颁布日期：2026年9月10日
# 版本信息来源：https://deepseek.com/news/deepseek-v4-1-flash/

"""Minimal, safe reader for one Question 1 feature package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from q1lib import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read one generated multimodal feature package")
    parser.add_argument("--sample-id", default="-3g5yACwYnA__13")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    sample_dir = config.output_root / "samples" / args.sample_id
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    with np.load(sample_dir / "features.npz", allow_pickle=False) as data:
        result = {
            "sample_id": args.sample_id,
            "alignment_type": metadata["quality"]["actual_alignment_type"],
            "aligned_text": list(data["aligned_text"].shape),
            "aligned_audio": list(data["aligned_audio"].shape),
            "aligned_vision": list(data["aligned_vision"].shape),
            "unaligned_audio": list(data["unaligned_audio"].shape),
            "unaligned_vision": list(data["unaligned_vision"].shape),
            "available_arrays": sorted(data.files),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
