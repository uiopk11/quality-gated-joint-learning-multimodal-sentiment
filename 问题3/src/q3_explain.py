from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import torch


Q3_ROOT = Path(__file__).resolve().parents[1]
HANDOFF_ROOT = Q3_ROOT / "q3_handoff"
if str(HANDOFF_ROOT) not in sys.path:
    sys.path.insert(0, str(HANDOFF_ROOT))

from q3_adapter.masking import (  # noqa: E402
    mask_audio_span,
    mask_modality,
    mask_text_span,
    mask_vision_span,
)
from q3_adapter.targets import fixed_target_class, score_prediction  # noqa: E402


MODALITIES = ("text", "audio", "vision")
MASK_SPAN = {
    "text": mask_text_span,
    "audio": mask_audio_span,
    "vision": mask_vision_span,
}


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().reshape(-1)[0])


def _scores(prediction: dict, target: torch.Tensor) -> tuple[float, float]:
    scored = score_prediction(prediction, target)
    return _scalar(scored["class_margin"]), _scalar(scored["intensity"])


def _coalition_batch(batch: dict, retained: frozenset[str]) -> dict:
    result = batch
    for modality in MODALITIES:
        if modality not in retained:
            result = mask_modality(result, modality)
    return result


def explain_modalities(adapter, batch: dict) -> dict:
    """Exact 3-player Shapley sensitivity over all eight modality coalitions."""
    original = adapter.predict(batch)
    target = fixed_target_class(original)
    values: dict[frozenset[str], tuple[float, float]] = {}
    predictions: dict[frozenset[str], dict] = {}
    for size in range(4):
        for members in itertools.combinations(MODALITIES, size):
            coalition = frozenset(members)
            prediction = original if size == 3 else adapter.predict(_coalition_batch(batch, coalition))
            predictions[coalition] = prediction
            values[coalition] = _scores(prediction, target)

    shapley = {modality: [0.0, 0.0] for modality in MODALITIES}
    factorial = math.factorial
    n = len(MODALITIES)
    for modality in MODALITIES:
        others = [item for item in MODALITIES if item != modality]
        for size in range(n):
            for members in itertools.combinations(others, size):
                coalition = frozenset(members)
                weight = factorial(size) * factorial(n - size - 1) / factorial(n)
                without = values[coalition]
                with_modality = values[coalition | {modality}]
                for output in range(2):
                    shapley[modality][output] += weight * (with_modality[output] - without[output])

    full = frozenset(MODALITIES)
    empty = frozenset()
    loo = {}
    for modality in MODALITIES:
        removed = full - {modality}
        loo[modality] = {
            "classification_margin_delta": values[full][0] - values[removed][0],
            "intensity_delta": values[full][1] - values[removed][1],
        }

    def summarize(output_index: int) -> dict:
        signed = {name: shapley[name][output_index] for name in MODALITIES}
        denominator = sum(abs(value) for value in signed.values())
        shares = {name: (abs(value) / denominator if denominator > 1e-12 else 0.0)
                  for name, value in signed.items()}
        influential = max(MODALITIES, key=lambda name: (shares[name], -MODALITIES.index(name)))
        supporting = max(MODALITIES, key=lambda name: signed[name])
        opposing = min(MODALITIES, key=lambda name: signed[name])
        return {
            "signed_shapley": signed,
            "absolute_share": shares,
            "most_influential_modality": influential,
            "primary_supporting_modality": supporting if signed[supporting] > 0 else None,
            "primary_opposing_modality": opposing if signed[opposing] < 0 else None,
        }

    coalition_records = []
    for coalition, value in sorted(values.items(), key=lambda item: (len(item[0]), sorted(item[0]))):
        prediction = predictions[coalition]
        coalition_records.append({
            "retained_modalities": [name for name in MODALITIES if name in coalition],
            "classification_margin": value[0],
            "intensity": value[1],
            "pred_class": int(prediction["pred_class"].detach().cpu()[0]),
            "probabilities": prediction["probabilities"].detach().cpu()[0].tolist(),
        })
    class_efficiency_error = sum(shapley[m][0] for m in MODALITIES) - (values[full][0] - values[empty][0])
    reg_efficiency_error = sum(shapley[m][1] for m in MODALITIES) - (values[full][1] - values[empty][1])
    return {
        "target_class": int(target.detach().cpu()[0]),
        "original_prediction": {
            "logits": original["logits"].detach().cpu()[0].tolist(),
            "probabilities": original["probabilities"].detach().cpu()[0].tolist(),
            "pred_class": int(original["pred_class"].detach().cpu()[0]),
            "intensity": _scalar(original["intensity"]),
            "classification_margin": values[full][0],
        },
        "classification": summarize(0),
        "regression": summarize(1),
        "leave_one_modality_out": loo,
        "coalitions": coalition_records,
        "efficiency_error": {
            "classification": class_efficiency_error,
            "regression": reg_efficiency_error,
        },
        "interpretation_scope": "masking sensitivity relative to UNK/zero-row baselines, not causal effect",
    }


def _stack_model_batches(batches: list[dict]) -> dict:
    if not batches:
        raise ValueError("Cannot stack an empty batch list")
    tensor_keys = ("text", "audio", "vision", "audio_lengths", "vision_lengths")
    return {key: torch.cat([batch[key] for batch in batches], dim=0) for key in tensor_keys}


def _batched_predict(adapter, batches: list[dict], batch_size: int) -> dict:
    fields = {"logits": [], "probabilities": [], "pred_class": [], "intensity": []}
    for offset in range(0, len(batches), batch_size):
        prediction = adapter.predict(_stack_model_batches(batches[offset : offset + batch_size]))
        for key in fields:
            fields[key].append(prediction[key].detach().cpu())
    return {key: torch.cat(parts, dim=0) for key, parts in fields.items()}


def mask_position_set(batch: dict, modality: str, positions: list[int]) -> dict:
    """Mask an arbitrary body-position set by composing verified one-slot masks."""
    result = batch
    for position in sorted(set(int(value) for value in positions)):
        result = MASK_SPAN[modality](result, position, position + 1)
    return result


def validate_local_fidelity(adapter, batch: dict, local_explanations: dict,
                            fractions: tuple[float, ...] = (0.1, 0.2, 0.3),
                            random_repeats: int = 20, seed: int = 20260925,
                            batch_size: int = 16) -> dict:
    """Compare top-ranked deletion with equal-size random deletion.

    Importance rankings and validation perturbations use the same explicit
    masking baseline, so this is a model-faithfulness check, not a causal or
    human-rationale benchmark. Absolute output change is used because a highly
    influential position may support or oppose the original output.
    """
    if random_repeats < 1:
        raise ValueError("random_repeats must be positive")
    original = adapter.predict(batch)
    target = fixed_target_class(original)
    base = score_prediction(original, target)
    base_cls = _scalar(base["class_margin"])
    base_reg = _scalar(base["intensity"])
    requests = []
    variants = []
    rng = np.random.default_rng(seed)
    unavailable_modalities = []
    for modality in MODALITIES:
        positions = local_explanations[modality]["positions"]
        if local_explanations[modality].get("observation_status") == "unavailable_all_zero":
            unavailable_modalities.append(modality)
            continue
        length = len(positions)
        rankings = {
            "classification": sorted(range(length), key=lambda index: positions[index]["classification_absolute"], reverse=True),
            "regression": sorted(range(length), key=lambda index: positions[index]["intensity_absolute"], reverse=True),
        }
        for target_name, ranking in rankings.items():
            for fraction in fractions:
                if not 0 < fraction <= 1:
                    raise ValueError("fractions must be in (0,1]")
                count = max(1, min(length, int(math.ceil(length * fraction))))
                top_positions = ranking[:count]
                requests.append((modality, target_name, fraction, "top", top_positions))
                variants.append(mask_position_set(batch, modality, top_positions))
                for repeat in range(random_repeats):
                    random_positions = sorted(rng.choice(length, size=count, replace=False).tolist())
                    requests.append((modality, target_name, fraction, f"random_{repeat}", random_positions))
                    variants.append(mask_position_set(batch, modality, random_positions))
    predictions = _batched_predict(adapter, variants, batch_size)
    repeated_target = target.detach().cpu().repeat(len(variants))
    scores = score_prediction(predictions, repeated_target)
    cls_changes = np.abs(base_cls - scores["class_margin"].detach().cpu().numpy())
    reg_changes = np.abs(base_reg - scores["intensity"].detach().cpu().numpy())
    grouped = {}
    for index, request in enumerate(requests):
        modality, target_name, fraction, kind, positions = request
        key = (modality, target_name, fraction)
        grouped.setdefault(key, {"top": None, "random": [], "top_positions": None})
        change = float(cls_changes[index] if target_name == "classification" else reg_changes[index])
        if kind == "top":
            grouped[key]["top"] = change
            grouped[key]["top_positions"] = positions
        else:
            grouped[key]["random"].append(change)
    output = {modality: {"classification": [], "regression": []} for modality in MODALITIES}
    for (modality, target_name, fraction), values in grouped.items():
        random_values = np.asarray(values["random"], dtype=np.float64)
        top_change = float(values["top"])
        mean = float(random_values.mean())
        std = float(random_values.std(ddof=0))
        percentile = float((random_values <= top_change).mean())
        output[modality][target_name].append({
            "deleted_fraction": fraction,
            "deleted_count": len(values["top_positions"]),
            "top_positions": values["top_positions"],
            "top_absolute_output_change": top_change,
            "random_mean_absolute_output_change": mean,
            "random_std_absolute_output_change": std,
            "top_to_random_mean_ratio": top_change / mean if mean > 1e-12 else None,
            "top_percentile_among_random": percentile,
            "top_at_least_random_mean": top_change >= mean,
        })
    comparisons = [item for modality in output.values() for target in modality.values() for item in target]
    return {
        "protocol": {
            "fractions": list(fractions),
            "random_repeats": random_repeats,
            "seed": seed,
            "metric": "absolute change of fixed-class margin or raw intensity",
            "claim_boundary": "masking faithfulness only; not causal or human-rationale agreement",
            "unavailable_modalities_excluded": unavailable_modalities,
        },
        "results": output,
        "aggregate": {
            "comparison_count": len(comparisons),
            "top_at_least_random_mean_rate": sum(item["top_at_least_random_mean"] for item in comparisons) / len(comparisons),
            "mean_top_percentile": sum(item["top_percentile_among_random"] for item in comparisons) / len(comparisons),
        },
    }


def explain_local(adapter, batch: dict, metadata: dict, modality: str,
                  window_sizes: tuple[int, ...] = (1, 3, 5), batch_size: int = 16) -> dict:
    if modality not in MODALITIES:
        raise ValueError(f"Unknown modality {modality}")
    length = int(metadata["body_length"])
    if modality in ("audio", "vision"):
        status = metadata[f"{modality}_observation_status"]
        if status == "unavailable_all_zero":
            positions = [{
                "body_position": index,
                "source_slot": index + 1,
                "wordpiece": metadata["wordpieces"][index],
                "text_char_start": metadata["text_offsets"][index][0],
                "text_char_end": metadata["text_offsets"][index][1],
                "original_text_fragment": metadata["original_token_fragments"][index],
                "classification_signed": 0.0,
                "classification_absolute": 0.0,
                "classification_share": 0.0,
                "intensity_signed": 0.0,
                "intensity_absolute": 0.0,
                "intensity_share": 0.0,
            } for index in range(length)]
            return {
                "modality": modality,
                "observation_status": status,
                "window_sizes": [],
                "aggregation": "not computed because the complete modality observation is all zero",
                "positions": positions,
                "windows": [],
                "top_classification_windows": [],
                "top_intensity_windows": [],
                "classification_zero_total": True,
                "intensity_zero_total": True,
                "per_width_classification_position_absolute": {},
                "window_scale_rank_stability": [],
            }
    widths = sorted({width for width in window_sizes if 1 <= width <= length})
    original = adapter.predict(batch)
    target = fixed_target_class(original)
    original_scores = score_prediction(original, target)
    windows = []
    perturbed = []
    for width in widths:
        for start in range(0, length - width + 1):
            end = start + width
            windows.append((width, start, end))
            perturbed.append(MASK_SPAN[modality](batch, start, end))
    predictions = _batched_predict(adapter, perturbed, batch_size)
    repeated_target = target.detach().cpu().repeat(len(windows))
    scored = score_prediction(predictions, repeated_target)
    original_margin = _scalar(original_scores["class_margin"])
    original_intensity = _scalar(original_scores["intensity"])
    delta_cls = original_margin - scored["class_margin"].detach().cpu().numpy()
    delta_reg = original_intensity - scored["intensity"].detach().cpu().numpy()

    position_signed_cls = np.zeros(length, dtype=np.float64)
    position_signed_reg = np.zeros(length, dtype=np.float64)
    position_abs_cls = np.zeros(length, dtype=np.float64)
    position_abs_reg = np.zeros(length, dtype=np.float64)
    counts = np.zeros(length, dtype=np.int64)
    records = []
    per_width_position_abs = {}

    def original_fragment(start: int, end: int) -> str:
        offsets = metadata["text_offsets"][start:end]
        if not offsets:
            return ""
        return metadata["raw_text"][offsets[0][0] : offsets[-1][1]]
    for width in widths:
        per_width_position_abs[width] = np.zeros(length, dtype=np.float64)
        per_width_counts = np.zeros(length, dtype=np.int64)
        for index, (candidate_width, start, end) in enumerate(windows):
            if candidate_width != width:
                continue
            per_width_position_abs[width][start:end] += abs(float(delta_cls[index])) / width
            per_width_counts[start:end] += 1
        per_width_position_abs[width] = np.divide(
            per_width_position_abs[width], per_width_counts,
            out=np.zeros(length), where=per_width_counts > 0)

    def rankdata(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=np.float64)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[order[end]] == values[order[start]]:
                end += 1
            ranks[order[start:end]] = (start + end - 1) / 2.0
            start = end
        return ranks

    stability = []
    for left_index, left in enumerate(widths):
        for right in widths[left_index + 1 :]:
            a = rankdata(per_width_position_abs[left])
            b = rankdata(per_width_position_abs[right])
            if float(a.std()) <= 1e-12 or float(b.std()) <= 1e-12:
                correlation = None
            else:
                correlation = float(np.corrcoef(a, b)[0, 1])
            stability.append({"left_window": left, "right_window": right,
                              "spearman_classification": correlation})

    for index, (width, start, end) in enumerate(windows):
        cls_value = float(delta_cls[index])
        reg_value = float(delta_reg[index])
        position_signed_cls[start:end] += cls_value / width
        position_signed_reg[start:end] += reg_value / width
        position_abs_cls[start:end] += abs(cls_value) / width
        position_abs_reg[start:end] += abs(reg_value) / width
        counts[start:end] += 1
        records.append({
            "window_size": width,
            "body_start": start,
            "body_end": end,
            "source_slot_start": start + 1,
            "source_slot_end_exclusive": end + 1,
            "wordpieces": metadata["wordpieces"][start:end],
            "original_text_fragment": original_fragment(start, end),
            "classification_margin_delta": cls_value,
            "intensity_delta": reg_value,
            "absolute_classification_delta": abs(cls_value),
            "absolute_intensity_delta": abs(reg_value),
        })
    arrays = (position_signed_cls, position_signed_reg, position_abs_cls, position_abs_reg)
    for array in arrays:
        np.divide(array, counts, out=array, where=counts > 0)
    cls_total = float(position_abs_cls.sum())
    reg_total = float(position_abs_reg.sum())
    cls_share = position_abs_cls / cls_total if cls_total > 1e-12 else np.zeros(length)
    reg_share = position_abs_reg / reg_total if reg_total > 1e-12 else np.zeros(length)

    def top_windows(key: str, limit: int = 3) -> list[dict]:
        chosen = []
        occupied: set[int] = set()
        for record in sorted(records, key=lambda item: item[key], reverse=True):
            positions = set(range(record["body_start"], record["body_end"]))
            if positions.isdisjoint(occupied):
                chosen.append(record)
                occupied.update(positions)
            if len(chosen) >= limit:
                break
        return chosen

    positions = []
    for index in range(length):
        positions.append({
            "body_position": index,
            "source_slot": index + 1,
            "wordpiece": metadata["wordpieces"][index],
            "text_char_start": metadata["text_offsets"][index][0],
            "text_char_end": metadata["text_offsets"][index][1],
            "original_text_fragment": metadata["original_token_fragments"][index],
            "classification_signed": float(position_signed_cls[index]),
            "classification_absolute": float(position_abs_cls[index]),
            "classification_share": float(cls_share[index]),
            "intensity_signed": float(position_signed_reg[index]),
            "intensity_absolute": float(position_abs_reg[index]),
            "intensity_share": float(reg_share[index]),
        })
    return {
        "modality": modality,
        "observation_status": "available",
        "window_sizes": widths,
        "aggregation": "mean of signed or absolute window delta divided by window width over covering windows",
        "positions": positions,
        "windows": records,
        "top_classification_windows": top_windows("absolute_classification_delta"),
        "top_intensity_windows": top_windows("absolute_intensity_delta"),
        "classification_zero_total": cls_total <= 1e-12,
        "intensity_zero_total": reg_total <= 1e-12,
        "per_width_classification_position_absolute": {
            str(width): values.tolist() for width, values in per_width_position_abs.items()
        },
        "window_scale_rank_stability": stability,
    }
