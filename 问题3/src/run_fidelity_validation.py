from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


Q3_ROOT = Path(__file__).resolve().parents[1]
HANDOFF_ROOT = Q3_ROOT / "q3_handoff"

if str(HANDOFF_ROOT) not in sys.path:
    sys.path.insert(0, str(HANDOFF_ROOT))

from q3_adapter.model_adapter import load_frozen_m0  # noqa: E402
from q3_data import load_attachment4_sample  # noqa: E402
from q3_explain import validate_local_fidelity  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", nargs="*", type=int, default=list(range(1, 21)))
    parser.add_argument("--random-repeats", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    adapter = load_frozen_m0(device=args.device)
    output_root = Q3_ROOT / "results" / "validation" / "samples"
    output_root.mkdir(parents=True, exist_ok=True)
    aggregates = []
    for number in args.samples:
        explanation_path = Q3_ROOT / "results" / "attachment4" / "samples" / f"{number:02d}.json"
        explanation = json.loads(explanation_path.read_text(encoding="utf-8"))
        batch, metadata = load_attachment4_sample(number, device=args.device)
        validation = validate_local_fidelity(
            adapter, batch, explanation["local_explanation"],
            random_repeats=args.random_repeats,
            seed=20260925 + int(number), batch_size=args.batch_size,
        )
        output = {"sample": metadata, "fidelity_validation": validation}
        (output_root / f"{number:02d}.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        aggregates.append(validation["aggregate"])
        print(f"validated attachment4/{number:02d}", flush=True)
    summary = {
        "status": "completed",
        "sample_count": len(aggregates),
        "comparison_count": sum(item["comparison_count"] for item in aggregates),
        "top_at_least_random_mean_rate": sum(
            item["top_at_least_random_mean_rate"] * item["comparison_count"] for item in aggregates
        ) / sum(item["comparison_count"] for item in aggregates),
        "mean_top_percentile": sum(
            item["mean_top_percentile"] * item["comparison_count"] for item in aggregates
        ) / sum(item["comparison_count"] for item in aggregates),
        "claim_boundary": "Faithfulness to frozen M1 under the declared masking baseline, not causal validity.",
    }
    summary_path = Q3_ROOT / "results" / "validation" / "fidelity_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
