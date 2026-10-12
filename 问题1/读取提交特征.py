# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：DeepSeek（深度求索）
# 版本/型号：DeepSeek-V4.1-Flash
# 开发机构/公司：杭州深度求索人工智能基础技术研究有限公司
# 版本颁布日期：2026年9月10日
# 版本信息来源：https://deepseek.com/news/deepseek-v4-1-flash/

"""Inspect the submitted 100-sample feature archive without rerunning extraction."""

from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "archive",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("问题1_100条多模态时序特征.pkl.gz"),
    )
    args = parser.parse_args()
    # Only load this archive from a trusted source; pickle is not a safe format
    # for files supplied by unknown parties.
    with gzip.open(args.archive, "rb") as stream:
        data = pickle.load(stream)
    print("samples:", len(data["id"]))
    for view in ("aligned_50", "unaligned_500"):
        print(view)
        for name, value in data[view].items():
            if hasattr(value, "shape"):
                print("  ", name, value.shape)


if __name__ == "__main__":
    main()
