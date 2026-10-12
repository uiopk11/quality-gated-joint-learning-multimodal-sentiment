"""Controlled raw-input perturbations, with body-token coordinates throughout.

All spans are zero-based half-open ``[start, end)`` BODY coordinates. In the
source pickle and ``batch['text']`` body j is stored at slot j+1. The formal
dataset reader has already shifted A/V left: body j is ``batch['audio'][:, j]``
and ``batch['vision'][:, j]``. There are 50 storage slots, not 50 seconds.

Bounds may be integers or [B] integer sequences/tensors. A span is intersected
with each sample's actual body length; start beyond that length is a no-op.
Negative or reversed bounds are rejected. Padding, CLS/SEP, attention masks,
token-type IDs, sequence lengths and metadata are preserved. Every returned
batch is a recursive copy. A/V values remain in their original physical units;
the frozen model performs its own standardization exactly once.

Perturbation sensitivity is not an identified causal contribution.
"""
from __future__ import annotations

from .runtime import configure_runtime

_PACKAGE_ROOT = configure_runtime()

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

_INTEGRAL_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


def _clone(value: Any) -> Any:
    """Copy tensors without detaching, keeping dtype/device and optional grads."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        values = [_clone(item) for item in value]
        return type(value)(*values) if hasattr(value, '_fields') else tuple(values)
    return copy.deepcopy(value)


def special_token_ids(bert_dir: str | Path | None = None) -> dict[str, int]:
    """Resolve and cross-check special IDs using the copied local tokenizer.

    ``bert_dir`` is an optional alternative directory inside this handoff. It
    may not escape the handoff, so a masking call cannot silently use a tokenizer
    from the original project or the network. No transformer model is loaded.
    """
    root = _PACKAGE_ROOT.resolve()
    folder = (Path(bert_dir) if bert_dir is not None else
              root / 'original_source' / 'pretrained' / 'bert-base-uncased').resolve()
    if not folder.is_relative_to(root):
        raise ValueError('Tokenizer directory must be inside q3_handoff')
    vocab = (folder / 'vocab.txt').read_text(encoding='utf-8').splitlines()
    tokens = {'unk_id': '[UNK]', 'cls_id': '[CLS]', 'sep_id': '[SEP]', 'pad_id': '[PAD]'}
    if any(vocab.count(token) != 1 for token in tokens.values()):
        raise ValueError('Copied vocabulary must contain each special token exactly once')
    ids = {key: vocab.index(token) for key, token in tokens.items()}
    if ids != {'unk_id': 100, 'cls_id': 101, 'sep_id': 102, 'pad_id': 0}:
        raise ValueError('Special IDs disagree with the formal M0 vocabulary')
    for filename in ('tokenizer_config.json', 'special_tokens_map.json'):
        path = folder / filename
        if not path.is_file():
            continue
        config = json.loads(path.read_text(encoding='utf-8-sig'))
        for key, token in tokens.items():
            configured = config.get(key.replace('_id', '_token'), token)
            if isinstance(configured, dict):
                configured = configured.get('content')
            if configured != token:
                raise ValueError(f'{filename} conflicts with vocabulary for {key}')
    tokenizer_path = folder / 'tokenizer.json'
    if tokenizer_path.is_file():
        tokenizer = json.loads(tokenizer_path.read_text(encoding='utf-8-sig'))
        tokenizer_vocab = tokenizer.get('model', {}).get('vocab', {})
        if any(tokenizer_vocab.get(token) != ids[key] for key, token in tokens.items()):
            raise ValueError('tokenizer.json special IDs conflict with vocab.txt')
    return ids


def _validate_batch(batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    text = batch.get('text')
    if not isinstance(text, torch.Tensor) or text.ndim != 3 or text.shape[1:] != (3, 50):
        raise ValueError("batch['text'] must be an integer tensor [B,3,50]")
    if text.dtype not in _INTEGRAL_DTYPES or text.shape[0] < 1:
        raise ValueError('Text must have integer dtype and nonempty batch dimension')
    attention = text[:, 1]
    if not bool(((attention == 0) | (attention == 1)).all()):
        raise ValueError('Attention mask must contain only 0 and 1')
    lengths = attention.to(dtype=torch.long).sum(-1) - 2
    if not bool(((lengths >= 1) & (lengths <= 48)).all()):
        raise ValueError('Formal M0 requires 1..48 valid body positions')
    positions = torch.arange(50, device=text.device)[None]
    if not torch.equal(attention.bool(), positions < lengths[:, None] + 2):
        raise ValueError('Attention mask must contain contiguous CLS, body and SEP slots')
    ids = special_token_ids()
    last_token = text[:, 0].gather(1, (lengths + 1)[:, None]).reshape(-1)
    if not bool((text[:, 0, 0] == ids['cls_id']).all()) or not bool((last_token == ids['sep_id']).all()):
        raise ValueError('Expected preserved CLS at slot 0 and SEP after the body')
    for name, dim in (('audio', 74), ('vision', 35)):
        value = batch.get(name)
        if not isinstance(value, torch.Tensor) or value.shape != (text.shape[0], 50, dim):
            raise ValueError(f"batch['{name}'] must be [B,50,{dim}] after reader alignment")
        if not value.is_floating_point():
            raise ValueError(f'{name} must be floating point, in raw input units')
        saved_lengths = batch.get(name + '_lengths')
        if (not isinstance(saved_lengths, torch.Tensor) or saved_lengths.dtype not in _INTEGRAL_DTYPES
                or saved_lengths.shape != (text.shape[0],)):
            raise ValueError(f'{name}_lengths must be an integer tensor [B]')
        if not torch.equal(saved_lengths.to(device=text.device, dtype=torch.long), lengths):
            raise ValueError(f'{name}_lengths must match aligned text body lengths')
    return text, lengths


def _bounds(value: Any, batch_size: int, device: torch.device, name: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, device=device)
    except (ValueError, TypeError, RuntimeError) as exc:
        raise ValueError(f'{name} must be an integer or an integer sequence [B]') from exc
    if tensor.dtype not in _INTEGRAL_DTYPES:
        raise ValueError(f'{name} must contain integers (not floats or booleans)')
    if tensor.ndim == 0:
        tensor = tensor.expand(batch_size)
    if tensor.shape != (batch_size,):
        raise ValueError(f'{name} must be a scalar or [B]')
    tensor = tensor.to(dtype=torch.long)
    if not bool((tensor >= 0).all()):
        raise ValueError(f'{name} cannot be negative')
    return tensor


def _mask_span(batch: Mapping[str, Any], start: Any, end: Any, modality: str) -> dict[str, Any]:
    text, lengths = _validate_batch(batch)
    starts = _bounds(start, text.shape[0], text.device, 'start')
    ends = _bounds(end, text.shape[0], text.device, 'end')
    if not bool((ends >= starts).all()):
        raise ValueError('end must be greater than or equal to start for every sample')
    result = _clone(batch)
    if modality == 'text':
        # Convert source text slots to BODY positions; CLS has body position -1.
        body_positions = torch.arange(50, device=text.device)[None] - 1
    else:
        # A/V already drop the source CLS slot in the formal dataset reader.
        body_positions = torch.arange(50, device=text.device)[None]
    selected = ((body_positions >= starts[:, None]) & (body_positions < ends[:, None])
                & (body_positions >= 0) & (body_positions < lengths[:, None]))
    if modality == 'text':
        result['text'][:, 0] = result['text'][:, 0].masked_fill(selected, special_token_ids()['unk_id'])
    else:
        value = result[modality]
        result[modality] = value.masked_fill(selected.to(value.device).unsqueeze(-1), 0)
    return result


def mask_text_span(batch: Mapping[str, Any], start: Any, end: Any) -> dict[str, Any]:
    """Replace body tokens in [start,end) with the copied vocabulary's UNK ID."""
    return _mask_span(batch, start, end, 'text')


def mask_audio_span(batch: Mapping[str, Any], start: Any, end: Any) -> dict[str, Any]:
    """Zero raw 74-feature audio rows at body [start,end), preserving all lengths."""
    return _mask_span(batch, start, end, 'audio')


def mask_vision_span(batch: Mapping[str, Any], start: Any, end: Any) -> dict[str, Any]:
    """Zero raw 35-feature vision rows at body [start,end), preserving all lengths."""
    return _mask_span(batch, start, end, 'vision')


def mask_modality(batch: Mapping[str, Any], modality: str) -> dict[str, Any]:
    """Mask every valid BODY position of exactly 'text', 'audio' or 'vision'."""
    if modality not in ('text', 'audio', 'vision'):
        raise ValueError("modality must be 'text', 'audio' or 'vision'")
    return _mask_span(batch, 0, 48, modality)
