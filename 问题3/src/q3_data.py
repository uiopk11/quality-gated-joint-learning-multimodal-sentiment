from __future__ import annotations

import pickle
import importlib
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
Q3_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_DATA_ROOT = Path(
    os.environ.get("E_TOPIC_DATA_ROOT", str(Q3_ROOT / "official_data"))
).expanduser().resolve()
_CACHE_ROOT = Q3_ROOT / "results" / "runtime_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_CACHE_ROOT / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_CACHE_ROOT / "transformers"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
from transformers import BertTokenizerFast


ATTACHMENT4_ROOT = (
    OFFICIAL_DATA_ROOT
    / "附件4-可解释专项视频样本与特征文件"
    / "附件4-可解释专项视频样本与特征文件"
)
ALIGNED_ROOT = ATTACHMENT4_ROOT / "对齐版本"
UNALIGNED_ROOT = ATTACHMENT4_ROOT / "未对齐版本"
HANDOFF_ROOT = Q3_ROOT / "q3_handoff"
VOCAB_PATH = HANDOFF_ROOT / "original_source" / "pretrained" / "bert-base-uncased" / "vocab.txt"
ATTACHMENT2_TEST_OVERLAP = {
    "03": ("GAVpYuhMZAw$_$6", 330),
    "07": ("ChhZna-aBK4$_$8", 310),
    "08": ("2m58ShI1QSI$_$2", 195),
    "12": ("gE7kUqMqQ9g$_$2", 543),
    "15": ("SKTyBOhDX6U$_$9", 416),
}


def _load_pickle(path: Path) -> dict:
    # Attachment 4 was serialized by NumPy 2.x (``numpy._core`` module paths),
    # while the formal M1 environment is pinned to NumPy 1.26.4.  Alias only
    # the historical import locations used by pickle; arrays and dtypes are
    # still created by the installed NumPy and the source file stays untouched.
    sys.modules.setdefault("numpy._core", np.core)
    for module_name in ("numeric", "multiarray", "fromnumeric", "shape_base", "_multiarray_umath"):
        try:
            module = importlib.import_module(f"numpy.core.{module_name}")
        except ModuleNotFoundError:
            continue
        sys.modules.setdefault(f"numpy._core.{module_name}", module)
    with path.open("rb") as stream:
        value = pickle.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a dictionary")
    return value


def _validate_text(text: np.ndarray, sample_id: str) -> int:
    if text.shape != (3, 50) or not np.issubdtype(text.dtype, np.integer):
        raise ValueError(f"{sample_id}: text_bert must be integer [3,50]")
    ids, attention, token_types = text
    if not np.isin(attention, [0, 1]).all():
        raise ValueError(f"{sample_id}: attention mask is not binary")
    total = int(attention.sum())
    if total < 3 or total > 50 or not np.array_equal(attention, np.arange(50) < total):
        raise ValueError(f"{sample_id}: invalid contiguous attention mask")
    if ids[0] != 101 or ids[total - 1] != 102 or np.any(ids[total:] != 0):
        raise ValueError(f"{sample_id}: invalid CLS/SEP/PAD placement")
    if np.any(token_types != 0) or np.any(ids < 0) or np.any(ids >= 30522):
        raise ValueError(f"{sample_id}: invalid token ids or token types")
    return total - 2


def _shift_av(source: np.ndarray, length: int, dim: int, name: str, sample_id: str) -> np.ndarray:
    value = np.asarray(source)
    if value.shape != (50, dim):
        raise ValueError(f"{sample_id}: {name} must be [50,{dim}]")
    value = np.nan_to_num(value, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(value[0] != 0) or np.any(value[length + 1 :] != 0) or np.any(value[length + 1] != 0):
        raise ValueError(f"{sample_id}: {name} has values in CLS/SEP/padding slots")
    output = np.zeros((50, dim), dtype=np.float32)
    output[:-1] = value[1:].astype(np.float32, copy=False)
    if not np.isfinite(output).all():
        raise ValueError(f"{sample_id}: {name} overflows float32")
    return output


def vocabulary() -> list[str]:
    vocab = VOCAB_PATH.read_text(encoding="utf-8").splitlines()
    if len(vocab) != 30522:
        raise ValueError("Unexpected bert-base-uncased vocabulary size")
    return vocab


@lru_cache(maxsize=1)
def fast_tokenizer() -> BertTokenizerFast:
    return BertTokenizerFast.from_pretrained(
        str(VOCAB_PATH.parent), local_files_only=True, do_lower_case=True)


def verified_text_offsets(raw_text: str, text: np.ndarray, length: int) -> list[list[int]]:
    encoded = fast_tokenizer()(
        raw_text,
        add_special_tokens=True,
        max_length=50,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
    )
    for row, key in enumerate(("input_ids", "attention_mask", "token_type_ids")):
        if not np.array_equal(np.asarray(encoded[key], dtype=np.int64), text[row]):
            raise ValueError(f"Tokenizer reconstruction disagrees with attachment4 text_bert/{key}")
    offsets = [[int(start), int(end)] for start, end in encoded["offset_mapping"][1 : length + 1]]
    if any(start < 0 or end < start or end > len(raw_text) for start, end in offsets):
        raise ValueError("Tokenizer returned invalid character offsets")
    return offsets


def load_attachment4_sample(sample_number: int | str, device: str = "cpu") -> tuple[dict, dict]:
    number = f"{int(sample_number):02d}"
    path = ALIGNED_ROOT / f"{number}.pkl"
    record = _load_pickle(path)
    required = {"raw_text", "audio", "vision", "id", "text", "text_bert"}
    if set(record) != required:
        raise ValueError(f"{path}: unexpected fields {sorted(record)}")
    sample_id = str(record["id"])
    if sample_id != number:
        raise ValueError(f"{path}: id={sample_id!r}")
    text = np.asarray(record["text_bert"], dtype=np.int64)
    length = _validate_text(text, sample_id)
    audio = _shift_av(record["audio"], length, 74, "audio", sample_id)
    vision = _shift_av(record["vision"], length, 35, "vision", sample_id)
    target = torch.device(device)
    batch = {
        "text": torch.from_numpy(text).unsqueeze(0).to(target),
        "audio": torch.from_numpy(audio).unsqueeze(0).to(target),
        "vision": torch.from_numpy(vision).unsqueeze(0).to(target),
        "audio_lengths": torch.tensor([length], dtype=torch.long, device=target),
        "vision_lengths": torch.tensor([length], dtype=torch.long, device=target),
        "id": [sample_id],
        "index": torch.tensor([int(number) - 1], dtype=torch.long, device=target),
        "split": "attachment4_aligned",
        "raw_text": [str(record["raw_text"])],
    }
    vocab = vocabulary()
    token_ids = text[0, 1 : length + 1].tolist()
    raw_text = str(record["raw_text"])
    text_offsets = verified_text_offsets(raw_text, text, length)
    full_wordpiece_count = len(fast_tokenizer()(raw_text, add_special_tokens=False)["input_ids"])
    audio_nonzero_body_rows = int(np.any(audio[:length] != 0, axis=1).sum())
    vision_nonzero_body_rows = int(np.any(vision[:length] != 0, axis=1).sum())
    overlap = ATTACHMENT2_TEST_OVERLAP.get(number)
    metadata = {
        "sample_id": sample_id,
        "raw_text": raw_text,
        "body_length": length,
        "token_ids": token_ids,
        "wordpieces": [vocab[index] for index in token_ids],
        "text_offsets": text_offsets,
        "original_token_fragments": [raw_text[start:end] for start, end in text_offsets],
        "body_positions": list(range(length)),
        "source_slots": list(range(1, length + 1)),
        "aligned_pickle": str(path),
        "video": str(ALIGNED_ROOT / "videos" / f"{number}.mp4"),
        "timestamps_available": False,
        "raw_text_offsets_verified": True,
        "full_wordpiece_count": full_wordpiece_count,
        "retained_wordpiece_count": length,
        "truncated_wordpiece_count": max(0, full_wordpiece_count - length),
        "model_visible_text_end_char": text_offsets[-1][1] if text_offsets else 0,
        "audio_nonzero_body_rows": audio_nonzero_body_rows,
        "vision_nonzero_body_rows": vision_nonzero_body_rows,
        "audio_observation_status": "available" if audio_nonzero_body_rows else "unavailable_all_zero",
        "vision_observation_status": "available" if vision_nonzero_body_rows else "unavailable_all_zero",
        "overlaps_attachment2_test": overlap is not None,
        "attachment2_test_id": overlap[0] if overlap else None,
        "attachment2_test_index": overlap[1] if overlap else None,
    }
    return batch, metadata


def iter_attachment4(device: str = "cpu", sample_numbers: Iterable[int] = range(1, 21)):
    for number in sample_numbers:
        yield load_attachment4_sample(number, device=device)
