# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：DeepSeek（深度求索）
# 版本/型号：DeepSeek-V4.1-Flash
# 开发机构/公司：杭州深度求索人工智能基础技术研究有限公司
# 版本颁布日期：2026年9月10日
# 版本信息来源：https://deepseek.com/news/deepseek-v4-1-flash/

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT_ROOT / "模型" / "whisper-small"


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="openai/whisper-small",
        local_dir=DESTINATION,
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.model",
            "*.safetensors",
            "*.py",
        ],
    )
    print(DESTINATION.resolve())


if __name__ == "__main__":
    main()
