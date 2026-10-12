"""Deterministic, label-free missing-view construction for Attachment 2.

Only NumPy is needed. Masks always use original 50-slot coordinates, with True
meaning artificial removal. Metadata intervals use zero-based content coordinates
and half-open [start, end) intervals. This module never reads a data file or label.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


MODALITIES = ("T", "A", "V")
OTHER_MODALITIES = ("T", "A", "V", "TA", "TV", "AV")


def default_recipe() -> dict[str, Any]:
    """Return the frozen, JSON-serializable first main/paired recipe."""
    return {
        "schema_version": 1,
        "recipe_id": "mixed_v1",
        "condition_id": "mixed_v1_s0",
        "root_seed": 20260924,
        "rng": "numpy.Generator(PCG64)",
        "seed_derivation": "SHA256(canonical UTF-8 JSON identity), first 16 bytes big-endian",
        "seed_identity_fields": ["root_seed", "recipe_id", "family", "split", "index", "id", "condition", "stage"],
        "quota_seed_scope": "quota/* stages use index=-1,id=''; all other stages use original sample index and ID",
        "max_attempts_per_k": 32,
        "class_order": ["NONE", "TAV", "OTHER"],
        "groups": [0.20, 0.60, 0.20],
        "other_modalities": list(OTHER_MODALITIES),
        "rates": [0.10, 0.20, 0.40, 0.60],
        "single_fraction": 1.0 / 3.0,
        "shared_only_fraction": 0.75,
        "extra_max_length": 3,
        "extra_k_divisor": 4,
        "rounding": "r=0 -> 0; L>=2 -> min(L-1,max(1,floor(r*L+0.5))); L<=1 -> 0",
        "quota_algorithm": "hierarchical largest remainder, ties use listed category order; seeded permutation",
        "segment_algorithm": "uniform positive composition of k; uniform weak composition of remaining gaps",
        "preservation": "retain >=1 original observed proxy per selected modality when any existed",
        "fallback": "at most max_attempts_per_k per k; decrease common k by one, finally zero",
        "coordinates": "masks original-50; intervals content-0-based half-open; original_position=content_position+1",
        "mask_semantics": "true means artificial removal",
        "pairing": {
            "base_rate": 0.40,
            "rate_panel": [0.0, 0.20, 0.40, 0.60, 0.80],
            "shape_min_length": 20,
            "shapes": [[8], [4, 4], [2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1]],
            "position_order": ["head", "center", "tail"],
            "random_attempts": 32,
            "failure": "keep fixed-budget candidate and mark inapplicable; compare common applicable intersection",
        },
    }


def default_exploratory_recipe() -> dict[str, Any]:
    """Return an independent, explicitly exploratory evaluation recipe."""
    return {
        "schema_version": 1,
        "recipe_id": "a3like_v1",
        "condition_id": "a3like_v1_s0",
        "root_seed": 20260924,
        "rng": "numpy.Generator(PCG64)",
        "seed_derivation": "SHA256(canonical UTF-8 JSON identity), first 16 bytes big-endian",
        "quota_seed_scope": "quota/* stages use index=-1,id=''; all other stages use original sample index and ID",
        "max_attempts_per_k": 32,
        "class_order": ["NONE", "TAV"],
        "groups": [0.10, 0.90],
        "complete_fraction": 0.10,
        "rates": [0.10, 0.20, 0.30, 0.40],
        "sampling": "uniform positions without replacement; identical T/A/V mask",
        "fallback": "at most max_attempts_per_k per k; decrease common k by one, finally zero",
        "preservation": "same original observed proxy protection as main",
        "eligible_splits": ["valid", "test"],
        "source_note": "Exploratory design informed by prior Attachment 3 shape observations; no Attachment 3 file is read by generator.",
    }


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _rng(recipe: dict, family: str, split: str, index: int, sample_id: str,
         condition: str, stage: str) -> np.random.Generator:
    identity = {
        "root_seed": int(recipe["root_seed"]), "recipe_id": recipe["recipe_id"],
        "family": family, "split": split, "index": int(index), "id": sample_id,
        "condition": condition, "stage": stage,
    }
    seed = int.from_bytes(hashlib.sha256(_canonical(identity)).digest()[:16], "big")
    return np.random.Generator(np.random.PCG64(seed))


def largest_remainder(total: int, weights: list[float]) -> list[int]:
    """Integer quotas; equal remainders retain the provided category order."""
    weights_array = np.asarray(weights, dtype=np.float64)
    if total < 0 or not len(weights_array) or not np.isfinite(weights_array).all() or np.any(weights_array < 0) or weights_array.sum() <= 0:
        raise ValueError("Invalid quota weights or total")
    raw = total * weights_array / weights_array.sum()
    quotas = np.floor(raw).astype(np.int64)
    order = sorted(range(len(weights)), key=lambda i: (-float(raw[i] - quotas[i]), i))
    for i in order[:int(total - quotas.sum())]:
        quotas[i] += 1
    return quotas.tolist()


def missing_count(length: int, rate: float) -> int:
    if rate == 0 or length <= 1:
        return 0
    return min(length - 1, max(1, int(np.floor(rate * length + 0.5))))


def _assign(indices: np.ndarray, labels: list, weights: list[float], rng: np.random.Generator) -> dict[int, Any]:
    shuffled = rng.permutation(indices)
    counts = largest_remainder(len(indices), weights)
    result, start = {}, 0
    for label, count in zip(labels, counts):
        for index in shuffled[start:start + count]:
            result[int(index)] = label
        start += count
    return result


def _weak_composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    if total < 0 or parts < 1:
        raise ValueError("Infeasible weak composition")
    if parts == 1:
        return [total]
    # Each placement of bars corresponds to one weak composition, uniformly.
    bars = np.sort(rng.choice(total + parts - 1, size=parts - 1, replace=False))
    return (np.diff(np.concatenate(([-1], bars, [total + parts - 1]))) - 1).astype(int).tolist()


def _positive_composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    if not 1 <= parts <= total:
        raise ValueError("Infeasible positive composition")
    if parts == 1:
        return [total]
    cuts = np.sort(rng.choice(np.arange(1, total), size=parts - 1, replace=False))
    return np.diff(np.concatenate(([0], cuts, [total]))).astype(int).tolist()


def _place_segments(length: int, segment_lengths: list[int], rng: np.random.Generator) -> np.ndarray:
    count = len(segment_lengths)
    k = sum(segment_lengths)
    if count < 1 or length < k + count - 1 or any(size < 1 for size in segment_lengths):
        raise ValueError("Infeasible separated segments")
    gaps = _weak_composition(length - k - (count - 1), count + 1, rng)
    mask = np.zeros(length, dtype=bool)
    cursor = gaps[0]
    for j, size in enumerate(segment_lengths):
        mask[cursor:cursor + size] = True
        cursor += size + gaps[j + 1] + (1 if j < count - 1 else 0)
    return mask


def _common_candidate(length: int, k: int, shape: str, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    maximum = min(k, length - k + 1)
    segments = 1 if shape == "single" or maximum < 2 else int(rng.integers(2, maximum + 1))
    sizes = _positive_composition(k, segments, rng)
    return _place_segments(length, sizes, rng), {"requested_shape": shape, "drawn_segments": segments,
                                               "multisegment_fallback": bool(shape == "multi" and maximum < 2)}


def _selected(modality_set: str) -> list[int]:
    return [j for j, name in enumerate(MODALITIES) if name in modality_set]


def _protection_failures(mask: np.ndarray, observed: np.ndarray, selected: list[int]) -> list[str]:
    return [MODALITIES[m] for m in selected if observed[m].any() and not (observed[m] & ~mask[m]).any()]


def _intervals(content_mask: np.ndarray) -> list[list[int]]:
    padded = np.concatenate(([False], content_mask, [False])).astype(np.int8)
    changes = np.diff(padded)
    return [[int(start), int(end)] for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))]


def _metadata(source: dict, split: str, family: str, condition: str, index: int,
              mask: np.ndarray, applicable: bool, plan: dict, reasons: list[str],
              attempts: dict, stages: list[str], exclusion_reasons: list[str] | None = None) -> dict:
    length = int(source["lengths"][index])
    observed = source["observed"][index]
    content = source["content"][index]
    modalities = {}
    for m, name in enumerate(MODALITIES):
        intervals = _intervals(mask[m, 1:length + 1])
        sizes = [end - start for start, end in intervals]
        k = int(mask[m].sum())
        modalities[name] = {
            "intervals": intervals, "k": k, "rate": k / length if length else None,
            "segment_lengths": sizes, "segment_count": len(intervals),
            "gaps": [intervals[j][0] - intervals[j - 1][1] for j in range(1, len(intervals))],
            "max_segment_length": max(sizes, default=0),
            "mean_segment_length": float(np.mean(sizes)) if sizes else 0.0,
            "max_segment_fraction": max(sizes, default=0) / length if length else None,
            "relative_intervals": [[start / length, end / length] for start, end in intervals] if length else [],
            "touches_start": bool(intervals and intervals[0][0] == 0),
            "touches_end": bool(intervals and intervals[-1][1] == length),
            "native_unobserved": int((content & ~observed[m]).sum()),
            "overlap_native_unobserved": int((mask[m] & content & ~observed[m]).sum()),
            "observed_removed": int((mask[m] & observed[m]).sum()),
            "observed_remaining": int((~mask[m] & observed[m]).sum()),
            "original_all_unobserved": bool(not observed[m].any()),
        }
    selected = [m for m in range(3) if mask[m].any()]
    actual_set = "".join(MODALITIES[m] for m in selected) or "NONE"
    if not selected:
        synchrony = "none"
    elif len(selected) == 1:
        synchrony = "single_modality"
    elif all(np.array_equal(mask[selected[0]], mask[m]) for m in selected[1:]):
        synchrony = "shared"
    elif plan.get("synchrony") == "independent":
        synchrony = "independent"
    else:
        synchrony = "shared_plus_extra"
    common_k = int(np.logical_and.reduce(mask[selected], axis=0).sum()) if selected else 0
    unions = int(mask.any(axis=0).sum())
    overlaps = {a + b: int((mask[MODALITIES.index(a)] & mask[MODALITIES.index(b)]).sum())
                for a, b in (("T", "A"), ("T", "V"), ("A", "V"))}
    return {
        "index": int(index), "id": str(source["ids"][index]), "split": split,
        "evaluation_family": family, "condition_id": condition, "length": length,
        "applicable": bool(applicable), "plan": plan, "modalities": modalities,
        "actual": {"modality_set": actual_set, "synchrony": synchrony, "any_missing": bool(selected),
                   "slot_rate": int(mask.sum()) / (3 * length) if length else None,
                   "common_k": common_k, "union_k": unions, "pairwise_overlap": overlaps,
                   "tav_overlap": int(mask.all(axis=0).sum()),
                   "extra_k": int(mask.sum()) - common_k * len(selected)},
        "attempts": attempts, "reasons": list(dict.fromkeys(reasons)),
        "seed_keys": stages, "exclusion_reasons": exclusion_reasons or [],
    }


def _view(condition: str, family: str, count: int) -> dict:
    return {"evaluation_family": family, "condition_id": condition,
            "masks": np.zeros((count, 3, 50), dtype=bool),
            "applicable": np.ones(count, dtype=bool), "metadata": []}


def _validate_source(source: dict) -> int:
    if set(source) != {"ids", "lengths", "content", "observed"}:
        raise ValueError("Generator source must contain exactly ids, lengths, content and observed; labels are forbidden")
    n = len(source["ids"])
    lengths = np.asarray(source["lengths"])
    content = np.asarray(source["content"])
    observed = np.asarray(source["observed"])
    if lengths.shape != (n,) or lengths.dtype.kind not in "iu" or np.any((lengths < 0) | (lengths > 48)):
        raise ValueError("Content lengths must be integer values from 0 to 48")
    expected = (np.arange(50)[None] >= 1) & (np.arange(50)[None] < lengths[:, None] + 1)
    if content.shape != (n, 50) or content.dtype != bool or not np.array_equal(content, expected):
        raise ValueError("content must be the original-coordinate contiguous content support")
    if observed.shape != (n, 3, 50) or observed.dtype != bool or np.any(observed & ~content[:, None]):
        raise ValueError("observed must be a boolean subset of content for each modality")
    if len(set(str(x) for x in source["ids"])) != n:
        raise ValueError("Sample IDs must be unique within a split")
    return n


def _validate_recipe(recipe: dict, exploratory: bool = False) -> None:
    if recipe["rng"] != "numpy.Generator(PCG64)" or recipe["max_attempts_per_k"] < 1:
        raise ValueError("Unsupported RNG or invalid retry limit")
    if any(not 0 < rate < 1 for rate in recipe["rates"]):
        raise ValueError("Missing rates must lie strictly between zero and one")
    if exploratory:
        if recipe["class_order"] != ["NONE", "TAV"]:
            raise ValueError("Unexpected exploratory category order")
    elif (recipe["class_order"] != ["NONE", "TAV", "OTHER"] or recipe["other_modalities"] != list(OTHER_MODALITIES)
          or not 0 <= recipe["single_fraction"] <= 1 or not 0 <= recipe["shared_only_fraction"] <= 1
          or recipe["extra_max_length"] < 0 or recipe["extra_k_divisor"] < 1):
        raise ValueError("Unsupported main category order or invalid main parameters")


def _mixed_assignments(n: int, split: str, recipe: dict, exploratory: bool) -> tuple[dict, dict, dict, dict, list[str]]:
    family = "exploratory_a3like" if exploratory else "main_mixed"
    condition = recipe["condition_id"]
    all_indices = np.arange(n)
    stages = ["quota/groups"]
    group = _assign(all_indices, recipe["class_order"], recipe["groups"], _rng(recipe, family, split, -1, "", condition, stages[-1]))
    if not exploratory:
        indices = np.array([i for i in range(n) if group[i] == "OTHER"])
        stages.append("quota/other_modalities")
        group.update(_assign(indices, recipe["other_modalities"], [1.0] * 6,
                             _rng(recipe, family, split, -1, "", condition, stages[-1])))
    rates, shapes, relations = {}, {}, {}
    categories = ["TAV"] if exploratory else ["TAV", *recipe["other_modalities"]]
    for category in categories:
        indices = np.array([i for i in range(n) if group[i] == category])
        stage = f"quota/{category}/rates"
        stages.append(stage)
        rates.update(_assign(indices, recipe["rates"], [1.0] * len(recipe["rates"]),
                             _rng(recipe, family, split, -1, "", condition, stage)))
        if exploratory:
            shapes.update({int(i): "positions" for i in indices})
            relations.update({int(i): "shared" for i in indices})
            continue
        stage = f"quota/{category}/shape"
        stages.append(stage)
        shapes.update(_assign(indices, ["single", "multi"], [recipe["single_fraction"], 1 - recipe["single_fraction"]],
                              _rng(recipe, family, split, -1, "", condition, stage)))
        if len(category) > 1:
            stage = f"quota/{category}/synchrony"
            stages.append(stage)
            relations.update(_assign(indices, ["shared", "shared_plus_extra"],
                                     [recipe["shared_only_fraction"], 1 - recipe["shared_only_fraction"]],
                                     _rng(recipe, family, split, -1, "", condition, stage)))
        else:
            relations.update({int(i): "single_modality" for i in indices})
    return group, rates, shapes, relations, stages


def _generate_mixed(split: str, source: dict, recipe: dict, exploratory: bool = False) -> dict:
    n = len(source["ids"])
    family = "exploratory_a3like" if exploratory else "main_mixed"
    condition = recipe["condition_id"]
    view = _view(condition, family, n)
    groups, rates, shapes, relations, quota_stages = _mixed_assignments(n, split, recipe, exploratory)
    for i in range(n):
        length = int(source["lengths"][i])
        selected = _selected(groups[i])
        rate = float(rates.get(i, 0.0))
        initial_k = missing_count(length, rate)
        plan = {"modality_set": groups[i], "rate": rate, "shape": shapes.get(i, "none"),
                "synchrony": relations.get(i, "none"), "initial_k": initial_k}
        reasons, stages, by_k = [], list(quota_stages), {}
        mask = view["masks"][i]
        observed = source["observed"][i]
        if rate and length <= 1:
            reasons.append("short_sequence_no_local_missing")
        accepted_k = 0
        for k in range(initial_k, 0, -1):
            accepted = False
            for attempt in range(int(recipe["max_attempts_per_k"])):
                stage = f"candidate/k{k}/attempt{attempt}"
                stages.append(stage)
                rng = _rng(recipe, family, split, i, str(source["ids"][i]), condition, stage)
                if exploratory:
                    common = np.zeros(length, dtype=bool)
                    common[rng.choice(length, size=k, replace=False)] = True
                    shape_info = {"requested_shape": "positions"}
                else:
                    common, shape_info = _common_candidate(length, k, shapes[i], rng)
                mask[:] = False
                mask[selected, 1:length + 1] = common
                by_k[str(k)] = attempt + 1
                failures = _protection_failures(mask, observed, selected)
                if not failures:
                    accepted = True
                    accepted_k = k
                    plan["accepted_shape_draw"] = shape_info
                    if shape_info.get("multisegment_fallback"):
                        reasons.append("multisegment_infeasible_fell_back_to_single")
                    break
                reasons.extend(f"candidate_would_remove_all_observed_{name}" for name in failures)
            if accepted:
                break
            reasons.append(f"retry_limit_reached_at_k_{k}")
        if not accepted_k:
            mask[:] = False
        if accepted_k < initial_k:
            reasons.append("common_budget_reduced_for_observation_protection")
            if not accepted_k:
                reasons.append("no_additional_missing_after_fallback")
        plan["accepted_base_k"] = accepted_k
        if not exploratory and relations.get(i) == "shared_plus_extra":
            stage = "extra/modality_and_length"
            stages.append(stage)
            rng = _rng(recipe, family, split, i, str(source["ids"][i]), condition, stage)
            extra_modality = int(rng.choice(selected))
            maximum = min(int(recipe["extra_max_length"]), accepted_k // int(recipe["extra_k_divisor"]), length - 1 - accepted_k)
            requested = int(rng.integers(1, maximum + 1)) if maximum >= 1 else 0
            plan.update(extra_modality=MODALITIES[extra_modality], extra_requested_length=requested,
                        extra_max_length=maximum, extra_applied_length=0)
            if not requested:
                reasons.append("extra_budget_unavailable")
            else:
                for size in range(requested, 0, -1):
                    valid_starts = []
                    for start in range(length - size + 1):
                        positions = slice(start + 1, start + size + 1)
                        if mask[extra_modality, positions].any():
                            continue
                        candidate = mask.copy()
                        candidate[extra_modality, positions] = True
                        if not _protection_failures(candidate, observed, selected):
                            valid_starts.append(start)
                    if valid_starts:
                        stage = f"extra/start/length{size}"
                        stages.append(stage)
                        rng = _rng(recipe, family, split, i, str(source["ids"][i]), condition, stage)
                        start = int(rng.choice(valid_starts))
                        mask[extra_modality, start + 1:start + size + 1] = True
                        plan["extra_applied_length"] = size
                        plan["extra_interval"] = [start, start + size]
                        if size < requested:
                            reasons.append("extra_length_reduced_for_feasibility")
                        break
                if not plan["extra_applied_length"]:
                    reasons.append("extra_interval_unavailable")
        view["metadata"].append(_metadata(source, split, family, condition, i, mask, True, plan, reasons,
                                          {"total": sum(by_k.values()), "by_k": by_k}, stages))
    return view


def _paired_definitions(recipe: dict) -> list[dict]:
    base = float(recipe["pairing"]["base_rate"])
    if base != 0.4 or recipe["pairing"]["rate_panel"] != [0.0, 0.2, 0.4, 0.6, 0.8]:
        raise ValueError("Changing paired rates requires new canonical condition IDs/version")
    definitions = [{"condition": "complete", "modality_set": "NONE", "rate": 0.0, "shape": "none", "synchrony": "none"}]
    for modality in (*OTHER_MODALITIES, "TAV"):
        definitions.append({"condition": f"center_r40_{modality}", "modality_set": modality, "rate": base,
                            "shape": "single", "position": "center", "synchrony": "shared" if len(modality) > 1 else "single_modality"})
    for rate in (0.2, 0.6, 0.8):
        definitions.append({"condition": f"center_r{int(rate * 100)}_TAV", "modality_set": "TAV", "rate": rate,
                            "shape": "single", "position": "center", "synchrony": "shared"})
    for position in ("head", "tail"):
        definitions.append({"condition": f"{position}_r40_TAV", "modality_set": "TAV", "rate": base,
                            "shape": "single", "position": position, "synchrony": "shared"})
    expected_shapes = [[8], [4, 4], [2, 2, 2, 2], [1] * 8]
    if recipe["pairing"]["shapes"] != expected_shapes:
        raise ValueError("Changing paired shapes requires new canonical condition IDs/version")
    for segments in expected_shapes:
        definitions.append({"condition": f"shape_{len(segments)}x{segments[0]}", "modality_set": "TAV", "rate": None,
                            "shape": "fixed_segments", "segment_lengths": segments, "synchrony": "shared"})
    for relationship in ("shared", "independent"):
        definitions.append({"condition": f"sync_{relationship}_r40", "modality_set": "TAV", "rate": base,
                            "shape": "single", "position": "random", "synchrony": relationship})
    return definitions


def _generate_paired(split: str, source: dict, recipe: dict) -> list[dict]:
    n = len(source["ids"])
    views = []
    for definition in _paired_definitions(recipe):
        condition = definition["condition"]
        view = _view(condition, "paired", n)
        selected = _selected(definition["modality_set"])
        for i in range(n):
            length = int(source["lengths"][i])
            mask = view["masks"][i]
            shape = definition["shape"]
            fixed = shape == "fixed_segments"
            k = sum(definition["segment_lengths"]) if fixed else missing_count(length, definition["rate"])
            plan = {key: value for key, value in definition.items() if key != "condition"}
            plan.update(initial_k=k, accepted_base_k=k)
            reasons, exclusions, stages = [], [], []
            attempts = 0
            if not selected:
                pass
            elif fixed and length < int(recipe["pairing"]["shape_min_length"]):
                exclusions.append("length_below_shape_panel_minimum")
                plan["candidate_materialized"] = False
            elif k == 0:
                exclusions.append("short_sequence_no_local_missing")
                plan["candidate_materialized"] = False
            else:
                random_layout = fixed or definition.get("position") == "random"
                limit = int(recipe["pairing"]["random_attempts"]) if random_layout else 1
                for attempt in range(limit):
                    attempts += 1
                    mask[:] = False
                    if random_layout:
                        stage = f"candidate/k{k}/attempt{attempt}"
                        stages.append(stage)
                        rng = _rng(recipe, "paired", split, i, str(source["ids"][i]), condition, stage)
                    if fixed:
                        common = _place_segments(length, definition["segment_lengths"], rng)
                        mask[selected, 1:length + 1] = common
                    elif definition["position"] != "random":
                        start = {"head": 0, "center": (length - k) // 2, "tail": length - k}[definition["position"]]
                        mask[selected, start + 1:start + k + 1] = True
                    elif definition["synchrony"] == "shared":
                        start = int(rng.integers(0, length - k + 1))
                        mask[selected, start + 1:start + k + 1] = True
                    else:
                        for modality in selected:
                            start = int(rng.integers(0, length - k + 1))
                            mask[modality, start + 1:start + k + 1] = True
                    failures = _protection_failures(mask, source["observed"][i], selected)
                    if not failures:
                        break
                    reasons.extend(f"candidate_would_remove_all_observed_{name}" for name in failures)
                plan["candidate_materialized"] = True
                if failures:
                    exclusions.extend(f"observation_protection_failed_{name}" for name in failures)
                    if random_layout:
                        exclusions.append("fixed_budget_retry_limit_reached")
            applicable = not exclusions
            view["applicable"][i] = applicable
            view["metadata"].append(_metadata(source, split, "paired", condition, i, mask, applicable, plan, reasons,
                                              {"total": attempts, "by_k": {str(k): attempts} if attempts else {}}, stages, exclusions))
        views.append(view)
    return views


def _panels(source: dict, views: list[dict]) -> dict:
    by_condition = {view["condition_id"]: view for view in views if view["evaluation_family"] == "paired"}
    conditions = {
        "complete": ["complete"],
        "modality": [f"center_r40_{modality}" for modality in (*OTHER_MODALITIES, "TAV")],
        "rate": ["complete", "center_r20_TAV", "center_r40_TAV", "center_r60_TAV", "center_r80_TAV"],
        "position": ["head_r40_TAV", "center_r40_TAV", "tail_r40_TAV"],
        "shape": ["shape_1x8", "shape_2x4", "shape_4x2", "shape_8x1"],
        "synchrony": ["sync_shared_r40", "sync_independent_r40"],
    }
    result = {}
    for panel, condition_ids in conditions.items():
        applicable = np.logical_and.reduce([by_condition[key]["applicable"] for key in condition_ids])
        indices = np.flatnonzero(applicable).astype(int).tolist()
        excluded = []
        for index in np.flatnonzero(~applicable):
            reasons = [f"{key}:{reason}" for key in condition_ids
                       for reason in by_condition[key]["metadata"][int(index)]["exclusion_reasons"]]
            excluded.append({"index": int(index), "id": str(source["ids"][index]), "reasons": reasons})
        result[panel] = {"condition_ids": condition_ids, "indices": indices,
                         "ids": [str(source["ids"][i]) for i in indices], "excluded": excluded}
        if panel == "position":
            coincidences = []
            for i in range(len(source["ids"])):
                equal_pairs = [[a, b] for j, a in enumerate(condition_ids) for b in condition_ids[j + 1:]
                               if np.array_equal(by_condition[a]["masks"][i], by_condition[b]["masks"][i])]
                if equal_pairs:
                    coincidences.append({"index": i, "id": str(source["ids"][i]), "condition_pairs": equal_pairs,
                                         "applicable": bool(applicable[i])})
            result[panel]["position_coincidences"] = coincidences
    return result


def generate_split(split: str, source: dict, recipe: dict, exploratory_recipe: dict | None = None) -> dict:
    """Build fixed main and optional evaluation views without ever reading labels.

    source contains exactly ids[N], lengths[N], content[N,50], observed[N,3,50].
    Every view retains all N source rows; paired exclusions are explicit flags.
    """
    if split not in ("train", "valid", "test"):
        raise ValueError("Unknown split")
    _validate_source(source)
    _validate_recipe(recipe)
    views = [_generate_mixed(split, source, recipe)]
    panels = {}
    if split != "train":
        views.extend(_generate_paired(split, source, recipe))
        panels = _panels(source, views)
        if exploratory_recipe is not None:
            _validate_recipe(exploratory_recipe, exploratory=True)
            views.append(_generate_mixed(split, source, exploratory_recipe, exploratory=True))
    return {"views": views, "panels": panels}
