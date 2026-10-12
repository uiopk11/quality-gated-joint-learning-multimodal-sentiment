from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


Q3_ROOT = Path(__file__).resolve().parents[1]
HANDOFF_ROOT = Q3_ROOT / "q3_handoff"

if str(HANDOFF_ROOT) not in sys.path:
    sys.path.insert(0, str(HANDOFF_ROOT))

from q3_adapter.model_adapter import CLASS_ORDER, load_frozen_m0  # noqa: E402
from q3_data import load_attachment4_sample  # noqa: E402
from q3_explain import MODALITIES, explain_local, explain_modalities  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen EMT explanations for attachment 4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", nargs="*", type=int, default=list(range(1, 21)))
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--perturbation-batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    output_root = Q3_ROOT / "results" / "attachment4"
    detail_root = output_root / "samples"
    detail_root.mkdir(parents=True, exist_ok=True)
    adapter = load_frozen_m0(device=args.device)
    summaries = []
    for number in args.samples:
        batch, metadata = load_attachment4_sample(number, device=args.device)
        modality = explain_modalities(adapter, batch)
        local = {
            name: explain_local(
                adapter, batch, metadata, name,
                window_sizes=tuple(args.window_sizes),
                batch_size=args.perturbation_batch_size,
            )
            for name in MODALITIES
        }
        result = {
            "protocol": {
                "model": "M1 complete-input joint predictor (legacy run id mt_m0_avstd_s1111_v1)",
                "predictor": "frozen complete-input M1 EMT multitask model",
                "classification_target": "fixed original argmax class logit margin",
                "regression_target": "raw intensity output",
                "modality_method": "exact three-player Shapley masking sensitivity over 8 coalitions",
                "local_method": "multi-scale contiguous occlusion",
                "window_sizes": args.window_sizes,
                "causal_claim": False,
            },
            "sample": metadata,
            "modality_explanation": modality,
            "local_explanation": local,
        }
        sample_path = detail_root / f"{int(number):02d}.json"
        sample_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        prediction = modality["original_prediction"]
        summaries.append({
            "sample_id": metadata["sample_id"],
            "raw_text": metadata["raw_text"],
            "pred_class": prediction["pred_class"],
            "pred_label": CLASS_ORDER[prediction["pred_class"]],
            "p_negative": prediction["probabilities"][0],
            "p_neutral": prediction["probabilities"][1],
            "p_positive": prediction["probabilities"][2],
            "intensity": prediction["intensity"],
            "class_margin": prediction["classification_margin"],
            "full_wordpiece_count": metadata["full_wordpiece_count"],
            "retained_wordpiece_count": metadata["retained_wordpiece_count"],
            "truncated_wordpiece_count": metadata["truncated_wordpiece_count"],
            "model_visible_text_end_char": metadata["model_visible_text_end_char"],
            "audio_observation_status": metadata["audio_observation_status"],
            "vision_observation_status": metadata["vision_observation_status"],
            "overlaps_attachment2_test": metadata["overlaps_attachment2_test"],
            "attachment2_test_id": metadata["attachment2_test_id"],
            "attachment2_test_index": metadata["attachment2_test_index"],
            "most_influential_modality_classification": modality["classification"]["most_influential_modality"],
            "supporting_modality_classification": modality["classification"]["primary_supporting_modality"],
            "opposing_modality_classification": modality["classification"]["primary_opposing_modality"],
            "most_influential_modality_intensity": modality["regression"]["most_influential_modality"],
            "supporting_modality_intensity": modality["regression"]["primary_supporting_modality"],
            "opposing_modality_intensity": modality["regression"]["primary_opposing_modality"],
            **{
                f"{name}_classification_shapley": modality["classification"]["signed_shapley"][name]
                for name in MODALITIES
            },
            **{
                f"{name}_classification_share": modality["classification"]["absolute_share"][name]
                for name in MODALITIES
            },
            **{
                f"{name}_intensity_shapley": modality["regression"]["signed_shapley"][name]
                for name in MODALITIES
            },
            **{
                f"{name}_intensity_share": modality["regression"]["absolute_share"][name]
                for name in MODALITIES
            },
            "detail_json": str(sample_path),
        })
        print(f"completed attachment4/{int(number):02d}", flush=True)
    csv_path = output_root / "attachment4_predictions_and_modality_effects.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    summary_path = output_root / "run_summary.json"
    summary_path.write_text(json.dumps({
        "status": "completed",
        "sample_count": len(summaries),
        "device": args.device,
        "window_sizes": args.window_sizes,
        "prediction_csv": str(csv_path),
        "sample_detail_directory": str(detail_root),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "summary": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
