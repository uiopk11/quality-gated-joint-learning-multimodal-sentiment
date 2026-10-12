# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：DeepSeek（深度求索）
# 版本/型号：DeepSeek-V4.1-Flash
# 开发机构/公司：杭州深度求索人工智能基础技术研究有限公司
# 版本颁布日期：2026年9月10日
# 版本信息来源：https://deepseek.com/news/deepseek-v4-1-flash/

from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "模型" / "bert-base-uncased"
MODEL_ID = "google-bert/bert-base-uncased"

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID)
tokenizer.save_pretrained(MODEL_DIR)
model.save_pretrained(MODEL_DIR, safe_serialization=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
batch = tokenizer("Environment smoke test.", return_tensors="pt")
batch = {key: value.to(device) for key, value in batch.items()}
with torch.inference_mode():
    hidden = model(**batch).last_hidden_state

print(f"model={MODEL_ID}")
print(f"saved_to={MODEL_DIR}")
print(f"device={device}")
print(f"shape={tuple(hidden.shape)}")
