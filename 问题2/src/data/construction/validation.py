"""Independent, model-free checks and summaries for fixed missing-data views.

The source arrays and saved masks are the authority here.  In particular, this
module never infers artificial missingness from zeros in a corrupted feature.
All failures use ValueError so validation remains active under ``python -O``.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any

import numpy as np


MODALITIES = ("T", "A", "V")
PAIRED_IDS = (
    "complete", "center_r40_T", "center_r40_A", "center_r40_V",
    "center_r40_TA", "center_r40_TV", "center_r40_AV", "center_r40_TAV",
    "center_r20_TAV", "center_r60_TAV", "center_r80_TAV",
    "head_r40_TAV", "tail_r40_TAV", "shape_1x8", "shape_2x4",
    "shape_4x2", "shape_8x1", "sync_shared_r40", "sync_independent_r40",
)


def _check(ok: Any, message: str) -> None:
    if not bool(ok):
        raise ValueError(message)


def _bool_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    _check(array.dtype == np.dtype(bool), f"{name}: expected bool dtype")
    _check(array.shape == shape, f"{name}: expected shape {shape}, got {array.shape}")
    return array


def _source_arrays(source: dict) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    ids = list(source["ids"])
    n = len(ids)
    _check(len(set(str(item) for item in ids)) == n, "source: duplicate sample IDs")
    lengths = np.asarray(source["lengths"])
    _check(lengths.shape == (n,), "source: lengths shape mismatch")
    _check(np.issubdtype(lengths.dtype, np.integer), "source: lengths must be integers")
    _check(np.all((lengths >= 0) & (lengths <= 48)), "source: lengths outside 0..48")
    content = _bool_array(source["content"], (n, 50), "source.content")
    expected = (np.arange(50)[None, :] >= 1) & (np.arange(50)[None, :] <= lengths[:, None])
    _check(np.array_equal(content, expected), "source: content is not original-coordinate positions 1..L")
    observed = _bool_array(source["observed"], (n, 3, 50), "source.observed")
    _check(not np.any(observed & ~content[:, None, :]), "source: observed proxy outside content")
    return ids, lengths, content, observed


def _intervals(row: np.ndarray, length: int) -> list[list[int]]:
    """Compute intervals in content coordinates, independently of the generator."""
    positions = np.flatnonzero(row[1 : length + 1])
    result: list[list[int]] = []
    for position in positions.tolist():
        if result and result[-1][1] == position:
            result[-1][1] = position + 1
        else:
            result.append([position, position + 1])
    return result


def _modality_set(value: Any) -> str:
    if value is None or value in ("", "none", "NONE", "complete"):
        return ""
    if isinstance(value, (list, tuple)):
        value = "".join(value)
    _check(isinstance(value, str), f"invalid modality set {value!r}")
    _check(len(value) == len(set(value)) and set(value) <= set(MODALITIES), f"invalid modality set {value!r}")
    return "".join(m for m in MODALITIES if m in value)


def _target_k(length: int, rate: float) -> int:
    if rate == 0 or length <= 1:
        return 0
    return min(length - 1, max(1, math.floor(rate * length + 0.5)))


def _equal_number(actual: Any, expected: float | None, message: str) -> None:
    if expected is None:
        _check(actual is None, message + ": expected null for undefined rate")
    else:
        _check(isinstance(actual, (int, float, np.integer, np.floating)) and not isinstance(actual, bool), message + ": not numeric")
        _check(math.isfinite(float(actual)) and math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12), message)


def _metadata_check(split: str, ids: list, lengths: np.ndarray, observed: np.ndarray,
                    view: dict, index: int) -> None:
    row = view["metadata"][index]
    family, condition = view["evaluation_family"], view["condition_id"]
    name = f"{split}/{family}/{condition}/{index}"
    length = int(lengths[index])
    masks = view["masks"][index]
    applicable = bool(view["applicable"][index])
    _check(row["index"] == index and str(row["id"]) == str(ids[index]), name + ": metadata identity/order mismatch")
    _check(row["split"] == split and row["length"] == length, name + ": metadata split/length mismatch")
    _check(row["evaluation_family"] == family and row["condition_id"] == condition, name + ": metadata view mismatch")
    _check(row["applicable"] == applicable, name + ": applicability mismatch")
    actual_set = "".join(m for j, m in enumerate(MODALITIES) if masks[j].any())
    _check(_modality_set(row["actual"]["modality_set"]) == actual_set, name + ": actual modality set mismatch")
    _check(row["actual"]["any_missing"] == bool(masks.any()), name + ": any_missing mismatch")
    _equal_number(row["actual"]["slot_rate"], float(masks.sum()) / (3 * length) if length else None, name + ": slot rate mismatch")
    plan_set = _modality_set(row["plan"]["modality_set"])
    _check(set(actual_set) <= set(plan_set), name + ": unselected modality was corrupted")
    actual_selected = [j for j, modality in enumerate(MODALITIES) if modality in actual_set]
    common_count = int(np.logical_and.reduce(masks[actual_selected], axis=0).sum()) if actual_selected else 0
    if not actual_selected:
        synchrony = "none"
    elif len(actual_selected) == 1:
        synchrony = "single_modality"
    elif all(np.array_equal(masks[actual_selected[0]], masks[j]) for j in actual_selected[1:]):
        synchrony = "shared"
    elif row["plan"]["synchrony"] == "independent":
        synchrony = "independent"
    else:
        synchrony = "shared_plus_extra"
    actual_expected = {"common_k": common_count, "union_k": int(masks.any(axis=0).sum()),
                       "tav_overlap": int(masks.all(axis=0).sum()), "synchrony": synchrony,
                       "extra_k": int(masks.sum()) - common_count * len(actual_selected)}
    for key, expected in actual_expected.items():
        _check(row["actual"][key] == expected, name + f": actual {key} mismatch")
    for left, right in ((0, 1), (0, 2), (1, 2)):
        pair = MODALITIES[left] + MODALITIES[right]
        _check(row["actual"]["pairwise_overlap"][pair] == int((masks[left] & masks[right]).sum()), name + ": pairwise overlap mismatch")
    if "initial_k" in row["plan"] and "rate" in row["plan"] and not condition.startswith("shape_"):
        _check(row["plan"]["initial_k"] == _target_k(length, float(row["plan"]["rate"])), name + ": initial rounded budget mismatch")
    for modality_index, modality in enumerate(MODALITIES):
        data = row["modalities"][modality]
        mask = masks[modality_index]
        intervals = _intervals(mask, length)
        segment_lengths = [end - start for start, end in intervals]
        gaps = [intervals[j + 1][0] - intervals[j][1] for j in range(len(intervals) - 1)]
        count = int(mask.sum())
        native_count = length - int(observed[index, modality_index].sum())
        overlap = int(np.count_nonzero(mask & ~observed[index, modality_index]))
        removed = int(np.count_nonzero(mask & observed[index, modality_index]))
        remaining = int(np.count_nonzero(~mask & observed[index, modality_index]))
        expected_values = {
            "intervals": intervals, "k": count, "segment_lengths": segment_lengths,
            "segment_count": len(intervals), "gaps": gaps,
            "touches_start": bool(intervals and intervals[0][0] == 0),
            "touches_end": bool(intervals and intervals[-1][1] == length),
            "native_unobserved": native_count, "overlap_native_unobserved": overlap,
            "observed_removed": removed, "observed_remaining": remaining,
            "original_all_unobserved": int(observed[index, modality_index].sum()) == 0,
        }
        for key, expected in expected_values.items():
            _check(data[key] == expected, f"{name}/{modality}: metadata {key} mismatch")
        _equal_number(data["rate"], count / length if length else None, f"{name}/{modality}: actual rate mismatch")
        _check(data["max_segment_length"] == max(segment_lengths, default=0), f"{name}/{modality}: longest segment mismatch")
        _equal_number(data["mean_segment_length"], count / len(intervals) if intervals else 0.0, f"{name}/{modality}: mean segment length mismatch")
        _equal_number(data["max_segment_fraction"], max(segment_lengths, default=0) / length if length else None, f"{name}/{modality}: relative longest segment mismatch")
        expected_relative = [[start / length, end / length] for start, end in intervals] if length else []
        _check(data["relative_intervals"] == expected_relative, f"{name}/{modality}: relative coordinates mismatch")
    _check(isinstance(row["reasons"], list), name + ": reasons must be a list")
    _check(isinstance(row["exclusion_reasons"], list), name + ": exclusion reasons must be a list")
    _check(isinstance(row["attempts"]["total"], int) and row["attempts"]["total"] >= 0, name + ": invalid attempts")
    _check(row["attempts"]["total"] == sum(row["attempts"]["by_k"].values()), name + ": attempt counts disagree")
    if not applicable:
        _check(bool(row["exclusion_reasons"]), name + ": excluded row needs a reason")
    else:
        _check(not row["exclusion_reasons"], name + ": applicable row has exclusion reasons")


def _protected(mask: np.ndarray, observed: np.ndarray) -> bool:
    original = observed.sum(axis=1)
    remaining = (observed & ~mask).sum(axis=1)
    return bool(np.all((original == 0) | (remaining > 0)))


def _paired_check(condition: str, masks: np.ndarray, applicable: np.ndarray,
                  lengths: np.ndarray, observed: np.ndarray, name: str) -> None:
    if condition == "complete":
        _check(applicable.all() and not masks.any(), name + ": complete must be uncorrupted and applicable to all")
        return
    for index, value in enumerate(lengths):
        length = int(value)
        mask = masks[index]
        label = f"{name}/{index}"
        if condition.startswith(("center_", "head_", "tail_")):
            position, rate_part, selected = condition.split("_")
            count = _target_k(length, int(rate_part[1:]) / 100)
            start = {"head": 0, "center": (length - count) // 2, "tail": length - count}[position]
            expected = np.zeros((3, 50), dtype=bool)
            for modality_index, modality in enumerate(MODALITIES):
                if modality in selected:
                    expected[modality_index, start + 1 : start + count + 1] = True
            expected_applicable = count > 0 and _protected(expected, observed[index])
            _check(bool(applicable[index]) == expected_applicable, label + ": incorrect deterministic-condition applicability")
            _check(np.array_equal(mask, expected), label + ": wrong fixed position/budget/modalities")
        elif condition.startswith("shape_"):
            parts = condition[6:].split("x")
            segments, segment_length = int(parts[0]), int(parts[1])
            if length < 20:
                _check(not applicable[index], label + ": shape panel requires L >= 20")
            if length >= 20:
                _check(np.array_equal(mask[0], mask[1]) and np.array_equal(mask[0], mask[2]), label + ": shape masks must be synchronous")
                intervals = _intervals(mask[0], length)
                _check(len(intervals) == segments and all(end - start == segment_length for start, end in intervals), label + ": wrong shape or adjacent supposedly separate gaps")
                _check(int(mask[0].sum()) == 8, label + ": shape panel budget must be eight")
                _check(bool(applicable[index]) == _protected(mask, observed[index]), label + ": shape applicability disagrees with retained candidate")
            else:
                _check(not mask.any(), label + ": too-short shape row must not materialize a candidate")
        elif condition in ("sync_shared_r40", "sync_independent_r40"):
            count = _target_k(length, 0.4)
            if not count:
                _check(not applicable[index], label + ": nonzero-rate condition needs a nonzero budget")
            if count:
                for row in mask:
                    _check(int(row.sum()) == count and len(_intervals(row, length)) == 1, label + ": synchrony panel requires one interval and unchanged budget")
                if condition == "sync_shared_r40":
                    _check(np.array_equal(mask[0], mask[1]) and np.array_equal(mask[0], mask[2]), label + ": shared condition is not shared")
                _check(bool(applicable[index]) == _protected(mask, observed[index]), label + ": synchrony applicability disagrees with retained candidate")
        else:
            raise ValueError(name + ": unknown paired condition")


def _panels_check(source_ids: list, result: dict, paired: dict) -> dict:
    panels = result.get("panels", {})
    required = [
        {f"center_r40_{m}" for m in ("T", "A", "V", "TA", "TV", "AV", "TAV")},
        {"complete", "center_r20_TAV", "center_r40_TAV", "center_r60_TAV", "center_r80_TAV"},
        {"head_r40_TAV", "center_r40_TAV", "tail_r40_TAV"},
        {"shape_1x8", "shape_2x4", "shape_4x2", "shape_8x1"},
        {"sync_shared_r40", "sync_independent_r40"},
    ]
    seen_sets = []
    summary = {}
    for panel_name, panel in panels.items():
        conditions = panel["condition_ids"]
        _check(bool(conditions) and len(conditions) == len(set(conditions)), f"panel {panel_name}: empty/duplicate conditions")
        _check(set(conditions) <= set(paired), f"panel {panel_name}: unknown condition")
        seen_sets.append(set(conditions))
        intersection = np.logical_and.reduce([paired[c]["applicable"] for c in conditions])
        indices = np.flatnonzero(intersection).tolist()
        excluded_indices = np.flatnonzero(~intersection).tolist()
        _check(panel["indices"] == indices, f"panel {panel_name}: not the shared applicable intersection")
        _check([str(item) for item in panel["ids"]] == [str(source_ids[i]) for i in indices], f"panel {panel_name}: ID/index mismatch")
        excluded = panel["excluded"]
        _check([entry["index"] for entry in excluded] == excluded_indices, f"panel {panel_name}: missing/extra exclusions")
        for entry in excluded:
            _check(str(entry["id"]) == str(source_ids[entry["index"]]) and bool(entry["reasons"]), f"panel {panel_name}: untraceable exclusion")
            expected_reasons = [f"{condition}:{reason}" for condition in conditions
                                for reason in paired[condition]["metadata"][entry["index"]]["exclusion_reasons"]]
            _check(entry["reasons"] == expected_reasons, f"panel {panel_name}: exclusion reasons disagree with condition records")
        if set(conditions) == {"head_r40_TAV", "center_r40_TAV", "tail_r40_TAV"}:
            coincidences = []
            for index, sample_id in enumerate(source_ids):
                pairs = [[left, right] for j, left in enumerate(conditions) for right in conditions[j + 1:]
                         if np.array_equal(paired[left]["masks"][index], paired[right]["masks"][index])]
                if pairs:
                    coincidences.append({"index": index, "id": str(sample_id), "condition_pairs": pairs,
                                         "applicable": bool(intersection[index])})
            _check(panel.get("position_coincidences") == coincidences, f"panel {panel_name}: position coincidences mismatch")
        summary[panel_name] = {"applicable": len(indices), "excluded": len(excluded_indices)}
    for required_set in required:
        _check(required_set in seen_sets, "paired panels: required comparison panel missing")
    return summary


def _expected_quotas(total: int, labels: list, weights: list[float]) -> dict:
    denominator = sum(weights)
    shares = [total * weight / denominator for weight in weights]
    counts = [math.floor(share) for share in shares]
    remainder_order = sorted(range(len(labels)), key=lambda j: (-(shares[j] - counts[j]), j))
    for j in remainder_order[: total - sum(counts)]:
        counts[j] += 1
    return dict(zip(labels, counts))


def _quota_check(view: dict, recipe: dict, exploratory: bool) -> None:
    rows = view["metadata"]
    count = len(rows)
    group_labels = ["NONE", "TAV"] if exploratory else ["NONE", "TAV", "OTHER"]
    expected = _expected_quotas(count, group_labels, recipe["groups"])
    actual = Counter()
    for row in rows:
        selected = _modality_set(row["plan"]["modality_set"])
        actual["NONE" if not selected else "TAV" if selected == "TAV" else "OTHER"] += 1
    for key, amount in expected.items():
        _check(actual[key] == amount, f"{view['condition_id']}: planned {key} quota mismatch")
    if exploratory:
        category_counts = {"TAV": expected["TAV"]}
    else:
        category_counts = {"TAV": expected["TAV"], **_expected_quotas(expected["OTHER"], recipe["other_modalities"], [1.0] * 6)}
    for category, amount in category_counts.items():
        category_rows = [row for row in rows if _modality_set(row["plan"]["modality_set"]) == category]
        _check(len(category_rows) == amount, f"{view['condition_id']}: {category} subtype quota mismatch")
        expected_rates = _expected_quotas(amount, recipe["rates"], [1.0] * len(recipe["rates"]))
        rates = Counter(row["plan"]["rate"] for row in category_rows)
        _check(all(rates[rate] == quota for rate, quota in expected_rates.items()), f"{view['condition_id']}: {category} rate quota mismatch")
        if not exploratory:
            shapes = Counter(row["plan"]["shape"] for row in category_rows)
            expected_shapes = _expected_quotas(amount, ["single", "multi"], [recipe["single_fraction"], 1 - recipe["single_fraction"]])
            _check(all(shapes[shape] == quota for shape, quota in expected_shapes.items()), f"{view['condition_id']}: {category} shape quota mismatch")
            if len(category) > 1:
                relations = Counter(row["plan"]["synchrony"] for row in category_rows)
                expected_relations = _expected_quotas(amount, ["shared", "shared_plus_extra"], [recipe["shared_only_fraction"], 1 - recipe["shared_only_fraction"]])
                _check(all(relations[key] == quota for key, quota in expected_relations.items()), f"{view['condition_id']}: {category} synchrony quota mismatch")
    for row in rows:
        initial = int(row["plan"]["initial_k"])
        accepted = int(row["plan"]["accepted_base_k"])
        attempts = row["attempts"]["by_k"]
        _check(0 <= accepted <= initial, "invalid fallback budget")
        _check(all(1 <= value <= recipe["max_attempts_per_k"] for value in attempts.values()), "retry limit exceeded")
        expected_k_values = set(range(initial, max(accepted, 1) - 1, -1))
        _check({int(key) for key in attempts} == expected_k_values, "retry stages do not match one-by-one budget fallback")
        _check(all(attempts[str(k)] == recipe["max_attempts_per_k"] for k in expected_k_values if k != accepted), "budget was lowered before exhausting the retry limit")
        _check(accepted == row["actual"]["common_k"], "accepted base budget differs from actual shared mask")
        if accepted < initial:
            _check("common_budget_reduced_for_observation_protection" in row["reasons"], "unrecorded budget reduction")


def validate_split(split: str, source: dict, result: dict, recipe: dict,
                   exploratory_recipe: dict | None = None) -> dict:
    """Validate all generated views; return a compact JSON-serializable audit.

    This function is deliberately independent of generation helpers.  It checks
    source coordinate definitions, recomputes every saved interval/statistic,
    checks paired contrasts, and raises ValueError on the first discrepancy.
    """
    try:
        return _validate_split(split, source, result, recipe, exploratory_recipe)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f"invalid validation input schema: {error}") from error


def _validate_split(split: str, source: dict, result: dict, recipe: dict,
                    exploratory_recipe: dict | None = None) -> dict:
    _check(split in ("train", "valid", "test"), "unknown split")
    ids, lengths, content, observed = _source_arrays(source)
    n = len(ids)
    views = result["views"]
    keys: set[tuple[str, str]] = set()
    paired = {}
    main_count = exploratory_count = 0
    audits = []
    for view in views:
        family, condition = view["evaluation_family"], view["condition_id"]
        key = (family, condition)
        _check(key not in keys, f"duplicate view {key}")
        keys.add(key)
        name = f"{split}/{family}/{condition}"
        masks = _bool_array(view["masks"], (n, 3, 50), name + ".masks")
        applicable = _bool_array(view["applicable"], (n,), name + ".applicable")
        _check(len(view["metadata"]) == n, name + ": metadata must cover every source sample")
        _check(not np.any(masks & ~content[:, None, :]), name + ": special slot or padding was masked")
        counts = masks.sum(axis=2)
        _check(np.all(counts <= np.maximum(lengths - 1, 0)[:, None]), name + ": artificially removed all content")
        for index in range(n):
            if applicable[index]:
                _check(_protected(masks[index], observed[index]), f"{name}/{index}: removed every original observed position")
            _metadata_check(split, ids, lengths, observed, view, index)
        if family == "main_mixed":
            main_count += 1
            _check(condition == "mixed_v1_s0" and applicable.all(), name + ": main view must retain all samples")
            _quota_check(view, recipe, exploratory=False)
            for index, row in enumerate(view["metadata"]):
                _mixed_row_check(view, index, row, int(lengths[index]), name)
        elif family == "exploratory_a3like":
            exploratory_count += 1
            _check(split != "train", name + ": exploratory train is forbidden")
            _check(condition == "a3like_v1_s0" and applicable.all(), name + ": supplemental view must retain all samples")
            _quota_check(view, exploratory_recipe, exploratory=True)
            _check(np.array_equal(masks[:, 0], masks[:, 1]) and np.array_equal(masks[:, 0], masks[:, 2]), name + ": supplemental masks are not TAV shared")
            for row in view["metadata"]:
                _check(_modality_set(row["plan"]["modality_set"]) in ("", "TAV"), name + ": wrong supplemental planned modality set")
                _check(float(row["plan"]["rate"]) in (0.0, 0.1, 0.2, 0.3, 0.4), name + ": wrong supplemental rate")
        elif family == "paired":
            _check(split != "train" and condition in PAIRED_IDS, name + ": invalid paired view")
            paired[condition] = view
            for row in view["metadata"]:
                plan = row["plan"]
                selected = _modality_set(plan["modality_set"])
                if condition == "complete":
                    _check(not selected and plan["rate"] == 0, name + ": invalid complete condition plan")
                elif condition.startswith(("center_", "head_", "tail_")):
                    position, rate_part, expected_set = condition.split("_")
                    _check(selected == expected_set and plan["rate"] == int(rate_part[1:]) / 100 and plan["position"] == position,
                           name + ": fixed condition plan does not match its identifier")
                elif condition.startswith("shape_"):
                    segments, segment_length = (int(part) for part in condition[6:].split("x"))
                    _check(selected == "TAV" and plan["rate"] is None and plan["initial_k"] == 8
                           and plan["segment_lengths"] == [segment_length] * segments, name + ": fixed shape plan mismatch")
                else:
                    _check(selected == "TAV" and plan["rate"] == 0.4 and plan["synchrony"] == condition.split("_")[1],
                           name + ": synchrony condition plan mismatch")
            _paired_check(condition, masks, applicable, lengths, observed, name)
        else:
            raise ValueError(name + ": unknown evaluation family")
        audits.append({"evaluation_family": family, "condition_id": condition,
                       "rows": n, "applicable": int(applicable.sum()),
                       "artificial_positions": counts.sum(axis=0).astype(int).tolist()})
    _check(main_count == 1, "exactly one main mixed view is required")
    expected_exploratory = int(exploratory_recipe is not None and split != "train")
    _check(exploratory_count == expected_exploratory, "exploratory view presence does not match its configuration")
    if split == "train":
        _check(not result.get("panels"), "train must not contain paired panels")
        panels = {}
    else:
        _check(set(paired) == set(PAIRED_IDS), "valid/test must contain all 19 paired conditions")
        panels = _panels_check(ids, result, paired)
    return {"status": "passed", "split": split, "sample_count": n,
            "view_count": len(views), "views": audits, "panels": panels,
            "checks": ["content_coordinates", "special_slots_and_padding", "local_missingness",
                       "observed_information_protection", "metadata_recomputed_from_masks",
                       "paired_budget_shape_intersection", "supplemental_boundaries"]}


def _mixed_row_check(view: dict, index: int, row: dict, length: int, name: str) -> None:
    selected = _modality_set(row["plan"]["modality_set"])
    rate = float(row["plan"]["rate"])
    label = f"{name}/{index}"
    _check(rate in (0.0, 0.1, 0.2, 0.4, 0.6), label + ": unsupported main rate")
    _check((not selected) == (rate == 0), label + ": complete/affected rate mismatch")
    rows = view["masks"][index]
    if not selected:
        _check(not rows.any(), label + ": complete plan has artificial missingness")
        return
    initial = _target_k(length, rate)
    chosen = [MODALITIES.index(modality) for modality in selected]
    common = np.logical_and.reduce(rows[chosen])
    common_count = int(common.sum())
    _check(common_count <= initial, label + ": base mask exceeds the rounded budget")
    common_intervals = _intervals(common, length)
    shape = row["plan"]["shape"]
    if common_count:
        if shape == "single":
            _check(len(common_intervals) == 1, label + ": single-segment base is not continuous")
        elif shape == "multi":
            maximum = min(common_count, length - common_count + 1)
            if maximum >= 2:
                _check(2 <= len(common_intervals) <= maximum, label + ": feasible multi-segment base did not remain multi-segment")
            else:
                _check(len(common_intervals) == 1 and "multisegment_infeasible_fell_back_to_single" in row["reasons"], label + ": unrecorded multi-to-single fallback")
        else:
            raise ValueError(label + ": unexpected main base shape")
        shape_draw = row["plan"]["accepted_shape_draw"]
        _check(shape_draw["requested_shape"] == shape and shape_draw["drawn_segments"] == len(common_intervals), label + ": accepted base shape metadata mismatch")
    residuals = rows[chosen] & ~common
    extra_indices = np.flatnonzero(residuals.any(axis=1))
    _check(len(extra_indices) <= 1, label + ": more than one modality has an extra gap")
    if len(extra_indices):
        extra_index = chosen[int(extra_indices[0])]
        extra = rows[extra_index] & ~common
        extra_count = int(extra.sum())
        limit = min(3, common_count // 4, length - 1 - common_count)
        _check(extra_count <= limit and len(_intervals(extra, length)) == 1, label + ": invalid added interval")
        _check(row["plan"].get("extra_modality") == MODALITIES[extra_index], label + ": added modality is not the planned modality")
    elif len(chosen) == 1:
        _check(int(rows[chosen[0]].sum()) <= initial, label + ": single-modality budget increased")
    _check(row["actual"]["common_k"] == common_count, label + ": actual common count mismatch")
    relationship = row["plan"]["synchrony"]
    if len(chosen) == 1:
        _check(relationship == "single_modality", label + ": single modality has wrong synchrony plan")
    elif relationship == "shared":
        _check(not residuals.any(), label + ": shared-only plan has an extra gap")
    elif relationship == "shared_plus_extra":
        planned_extra = row["plan"]["extra_modality"]
        _check(planned_extra in selected, label + ": extra modality is not selected")
        maximum = min(3, common_count // 4, length - 1 - common_count)
        requested = row["plan"]["extra_requested_length"]
        applied = row["plan"]["extra_applied_length"]
        _check(row["plan"]["extra_max_length"] == maximum and 0 <= requested <= max(0, maximum), label + ": invalid planned extra budget")
        _check(0 <= applied <= requested and int(residuals.sum()) == applied, label + ": applied extra budget mismatch")
        if applied:
            extra_mask = rows[MODALITIES.index(planned_extra)] & ~common
            _check(_intervals(extra_mask, length) == [row["plan"]["extra_interval"]], label + ": saved extra interval mismatch")
    else:
        raise ValueError(label + ": invalid main synchrony plan")


def _histogram(values: list[int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(values).items())}


def _segment_buckets(values: list[int]) -> dict:
    counts = {str(i): int(sum(value == i for value in values)) for i in (1, 2, 3, 4)}
    counts[">=5"] = int(sum(value >= 5 for value in values))
    total = len(values)
    return {"count": total, "counts": counts,
            "fractions": {key: value / total if total else None for key, value in counts.items()}}


def _rate_summary(counts: np.ndarray, lengths: np.ndarray) -> dict:
    positive = lengths > 0
    denominator = int(lengths.sum())
    return {"positions": int(counts.sum()), "content_positions": denominator,
            "micro_rate": float(counts.sum()) / denominator if denominator else None,
            "macro_rate": float(np.mean(counts[positive] / lengths[positive])) if positive.any() else None,
            "undefined_rate_rows": int((~positive).sum())}


def _view_statistics(source: dict, view: dict) -> dict:
    ids, lengths, content, observed = _source_arrays(source)
    masks = np.asarray(view["masks"])
    applicable = np.asarray(view["applicable"])
    indices = np.flatnonzero(applicable)
    use_lengths = lengths[indices]
    metadata = view["metadata"]
    modality_stats = {}
    for mi, modality in enumerate(MODALITIES):
        segment_lengths: list[int] = []
        segments_per_sample: list[int] = []
        gaps: list[int] = []
        longest: list[int] = []
        all_singletons = touching_start = touching_end = 0
        centroids: list[float] = []
        by_length_rate: dict[str, dict] = {}
        for i in indices.tolist():
            length = int(lengths[i])
            intervals = _intervals(masks[i, mi], length)
            local_lengths = [end - start for start, end in intervals]
            segment_lengths.extend(local_lengths)
            segments_per_sample.append(len(intervals))
            gaps.extend(intervals[j + 1][0] - intervals[j][1] for j in range(len(intervals) - 1))
            longest.append(max(local_lengths, default=0))
            all_singletons += bool(local_lengths and max(local_lengths) == 1)
            touching_start += bool(intervals and intervals[0][0] == 0)
            touching_end += bool(intervals and intervals[-1][1] == length)
            positions = np.flatnonzero(masks[i, mi, 1 : length + 1])
            if len(positions):
                centroids.append(float(np.mean((positions + 0.5) / length)))
            length_group = "0-1" if length <= 1 else "2-7" if length <= 7 else "8-19" if length <= 19 else "20-48"
            key = f"L={length_group},target={metadata[i]['plan']['rate']}"
            group = by_length_rate.setdefault(key, {"samples": 0, "affected_samples": 0, "segments": [], "all_singleton_samples": 0})
            group["samples"] += 1
            group["affected_samples"] += int(bool(local_lengths))
            group["segments"].extend(local_lengths)
            group["all_singleton_samples"] += int(bool(local_lengths and max(local_lengths) == 1))
        for group in by_length_rate.values():
            group["segment_length_buckets"] = _segment_buckets(group.pop("segments"))
        chosen_masks = masks[indices, mi]
        chosen_observed = observed[indices, mi]
        counts = chosen_masks.sum(axis=1)
        native = content[indices] & ~chosen_observed
        original_all_unobserved = (chosen_observed.sum(axis=1) == 0)
        all_indices = indices[original_all_unobserved]
        modality_stats[modality] = {
            **_rate_summary(counts, use_lengths),
            "affected_samples": int((counts > 0).sum()),
            "count_histogram": _histogram(counts.astype(int).tolist()),
            "segment_length_buckets": _segment_buckets(segment_lengths),
            "segments_per_sample_histogram": _histogram(segments_per_sample),
            "longest_segment_histogram": _histogram(longest),
            "gap_histogram": _histogram(gaps),
            "all_singleton_affected_samples": int(all_singletons),
            "touching_start_samples": int(touching_start), "touching_end_samples": int(touching_end),
            "mean_missing_centroid_relative": float(np.mean(centroids)) if centroids else None,
            "native_unobserved_positions": int(native.sum()),
            "artificial_native_overlap": int((chosen_masks & native).sum()),
            "observed_removed_positions": int((chosen_masks & chosen_observed).sum()),
            "observed_remaining_positions": int((~chosen_masks & chosen_observed).sum()),
            "original_all_unobserved_samples": int(original_all_unobserved.sum()),
            "original_all_unobserved_ids": [str(ids[i]) for i in all_indices],
            "length_and_target_rate": by_length_rate,
        }
    counts_all = masks[indices].sum(axis=(1, 2))
    positive = use_lengths > 0
    reasons = Counter(reason for i in indices.tolist() for reason in metadata[i]["reasons"])
    actual_sets = Counter("".join(m for mi, m in enumerate(MODALITIES) if masks[i, mi].any()) or "none" for i in indices.tolist())
    planned_sets = Counter(_modality_set(metadata[i]["plan"]["modality_set"]) or "none" for i in indices.tolist())
    pairwise = {}
    for left, right in ((0, 1), (0, 2), (1, 2)):
        pairwise[MODALITIES[left] + MODALITIES[right]] = int((masks[indices, left] & masks[indices, right]).sum())
    return {
        "evaluation_family": view["evaluation_family"], "condition_id": view["condition_id"],
        "stored_rows": len(ids), "applicable_samples": int(applicable.sum()),
        "excluded_samples": int((~applicable).sum()),
        "excluded": [{"index": i, "id": str(ids[i]), "reasons": metadata[i]["exclusion_reasons"]} for i in np.flatnonzero(~applicable).tolist()],
        "planned_modality_counts": dict(sorted(planned_sets.items())),
        "actual_modality_counts": dict(sorted(actual_sets.items())),
        "no_artificial_missing_samples": int(sum(not masks[i].any() for i in indices.tolist())),
        "length_histogram": _histogram(use_lengths.astype(int).tolist()),
        "target_rate_counts": dict(sorted(Counter(str(metadata[i]["plan"]["rate"]) for i in indices.tolist()).items())),
        "planned_synchrony_counts": dict(sorted(Counter(str(metadata[i]["plan"]["synchrony"]) for i in indices.tolist()).items())),
        "actual_synchrony_counts": dict(sorted(Counter(str(metadata[i]["actual"]["synchrony"]) for i in indices.tolist()).items())),
        "slot_micro_rate": float(counts_all.sum()) / (3 * int(use_lengths.sum())) if use_lengths.sum() else None,
        "slot_macro_rate": float(np.mean(counts_all[positive] / (3 * use_lengths[positive]))) if positive.any() else None,
        "pairwise_overlap_positions": pairwise,
        "three_way_overlap_positions": int(np.logical_and.reduce(masks[indices], axis=1).sum()),
        "attempts_total": int(sum(metadata[i]["attempts"]["total"] for i in indices.tolist())),
        "rows_with_retries": int(sum(metadata[i]["attempts"]["total"] > 1 for i in indices.tolist())),
        "reason_counts": dict(sorted(reasons.items())), "modalities": modality_stats,
    }


def statistics_split(split: str, source: dict, result: dict) -> dict:
    """Summarize final masks, keeping each view and its denominator separate."""
    ids, lengths, content, observed = _source_arrays(source)
    source_stats = {
        "sample_count": len(ids), "content_positions": int(lengths.sum()),
        "length_histogram": _histogram(lengths.astype(int).tolist()),
        "native_unobserved_positions": {m: int((content & ~observed[:, j]).sum()) for j, m in enumerate(MODALITIES)},
        "original_all_unobserved_samples": {m: int((observed[:, j].sum(axis=1) == 0).sum()) for j, m in enumerate(MODALITIES)},
    }
    panels = {}
    for name, panel in result.get("panels", {}).items():
        panels[name] = {"condition_ids": panel["condition_ids"], "applicable_samples": len(panel["indices"]),
                        "applicable_indices": panel["indices"], "applicable_ids": panel["ids"],
                        "excluded_samples": len(panel["excluded"]), "excluded": panel["excluded"]}
        if "position_coincidences" in panel:
            panels[name]["position_coincidences"] = panel["position_coincidences"]
    return {"split": split, "source": source_stats,
            "views": [_view_statistics(source, view) for view in result["views"]], "panels": panels,
            "notes": ["All rates use final artificial masks; native zeros/UNK are separate proxies.",
                      "Per-view summaries use that view's applicable rows; paired comparisons use panel intersections.",
                      "Each segment is a maximal consecutive run in content coordinates; adjacent intervals are merged."]}
