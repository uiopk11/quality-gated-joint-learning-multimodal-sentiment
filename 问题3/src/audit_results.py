# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：DeepSeek（深度求索）
# 版本/型号：DeepSeek-V4.1-Flash
# 开发机构/公司：杭州深度求索人工智能基础技术研究有限公司
# 版本颁布日期：2026年9月10日
# 版本信息来源：https://deepseek.com/news/deepseek-v4-1-flash/

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path


Q3_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = Q3_ROOT / "results" / "attachment4" / "samples"
AUDIT_ROOT = Q3_ROOT / "results" / "audit"
MODALITIES = ("text", "audio", "vision")


def main() -> None:
    paths = sorted(SAMPLE_ROOT.glob("*.json"))
    if [path.stem for path in paths] != [f"{index:02d}" for index in range(1, 21)]:
        raise ValueError("Expected exactly attachment4 sample JSON files 01..20")
    class_counts = Counter()
    influential_cls = Counter()
    influential_reg = Counter()
    supporting_cls = Counter()
    supporting_reg = Counter()
    intensities = []
    margins = []
    probability_errors = []
    efficiency_cls = []
    efficiency_reg = []
    stability = []
    sample_summaries = []
    unavailable_observations = []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        sample = record["sample"]
        for modality in ("audio", "vision"):
            if sample.get(f"{modality}_observation_status") == "unavailable_all_zero":
                unavailable_observations.append({"sample_id": sample["sample_id"], "modality": modality})
        if sample["sample_id"] != path.stem:
            raise ValueError(f"{path}: sample id mismatch")
        explanation = record["modality_explanation"]
        prediction = explanation["original_prediction"]
        probabilities = prediction["probabilities"]
        numeric_values = prediction["logits"] + probabilities + [prediction["intensity"], prediction["classification_margin"]]
        if not all(math.isfinite(float(value)) for value in numeric_values):
            raise ValueError(f"{path}: nonfinite prediction")
        probability_errors.append(abs(sum(probabilities) - 1.0))
        efficiency_cls.append(abs(float(explanation["efficiency_error"]["classification"])))
        efficiency_reg.append(abs(float(explanation["efficiency_error"]["regression"])))
        for target in ("classification", "regression"):
            shares = explanation[target]["absolute_share"]
            if abs(sum(shares.values()) - 1.0) > 1e-6 and any(abs(value) > 1e-12 for value in explanation[target]["signed_shapley"].values()):
                raise ValueError(f"{path}: {target} shares do not sum to one")
        for modality in MODALITIES:
            local = record["local_explanation"][modality]
            if len(local["positions"]) != int(sample["body_length"]):
                raise ValueError(f"{path}: local position count mismatch for {modality}")
            for pair in local["window_scale_rank_stability"]:
                value = pair["spearman_classification"]
                if value is not None:
                    stability.append(float(value))
        pred_class = int(prediction["pred_class"])
        class_counts[pred_class] += 1
        intensities.append(float(prediction["intensity"]))
        margins.append(float(prediction["classification_margin"]))
        influential_cls[explanation["classification"]["most_influential_modality"]] += 1
        influential_reg[explanation["regression"]["most_influential_modality"]] += 1
        supporting_cls[str(explanation["classification"]["primary_supporting_modality"])] += 1
        supporting_reg[str(explanation["regression"]["primary_supporting_modality"])] += 1
        sample_summaries.append({
            "sample_id": sample["sample_id"],
            "pred_class": pred_class,
            "intensity": prediction["intensity"],
            "margin": prediction["classification_margin"],
            "most_influential_classification": explanation["classification"]["most_influential_modality"],
            "most_influential_intensity": explanation["regression"]["most_influential_modality"],
        })
    report = {
        "status": "passed",
        "sample_count": len(paths),
        "class_order": {"0": "negative", "1": "neutral", "2": "positive"},
        "predicted_class_counts": {str(key): class_counts[key] for key in range(3)},
        "intensity_range": [min(intensities), max(intensities)],
        "classification_margin_range": [min(margins), max(margins)],
        "most_influential_modality_counts": {
            "classification": dict(influential_cls), "regression": dict(influential_reg)},
        "primary_supporting_modality_counts": {
            "classification": dict(supporting_cls), "regression": dict(supporting_reg)},
        "max_probability_sum_error": max(probability_errors),
        "max_shapley_efficiency_error": {
            "classification": max(efficiency_cls), "regression": max(efficiency_reg)},
        "window_scale_spearman": {
            "count": len(stability),
            "mean": sum(stability) / len(stability) if stability else None,
            "min": min(stability) if stability else None,
            "max": max(stability) if stability else None,
        },
        "samples": sample_summaries,
        "unavailable_observations": unavailable_observations,
        "attachment2_test_overlap_samples": [
            item["sample_id"] for item in (json.loads(path.read_text(encoding="utf-8"))["sample"] for path in paths)
            if item.get("overlaps_attachment2_test")
        ],
        "truncated_text_samples": [
            item["sample_id"] for item in (json.loads(path.read_text(encoding="utf-8"))["sample"] for path in paths)
            if int(item.get("truncated_wordpiece_count", 0)) > 0
        ],
        "claim_boundary": "All attribution values are masking sensitivities, not causal effects.",
    }
    if report["max_probability_sum_error"] > 1e-5:
        raise ValueError("Probability sum error exceeds tolerance")
    if max(report["max_shapley_efficiency_error"].values()) > 1e-6:
        raise ValueError("Shapley efficiency error exceeds tolerance")
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    output = AUDIT_ROOT / "q3_results_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "output": str(output), **{k: report[k] for k in ("predicted_class_counts", "intensity_range", "most_influential_modality_counts")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
