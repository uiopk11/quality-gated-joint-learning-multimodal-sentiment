"""Terminal presentation of the single saved evaluation, without extra reports."""
import json
from pathlib import Path


def _number(value, percentage=False):
    if value is None:
        return "N/A"
    return f"{value:.2%}" if percentage else f"{value:.4f}"


def present_results(root, run, metrics):
    """Print this run's valid/test scores and the authoritative absolute path."""
    run = Path(run).resolve()
    destination = run / "evaluation.json"
    evaluation = json.loads(destination.read_text(encoding="utf-8-sig"))
    if metrics["split"] not in ("valid", "test"):
        raise ValueError("Only valid or test results can be presented")
    sources = evaluation["metadata"]["sources"]
    for key in ("checkpoint_sha256", "data_sha256", "selected_epoch"):
        if sources[key] != metrics[key]:
            raise ValueError(f"Cannot present results with different {key}")
    if evaluation["decision_rule"]["neutral_threshold"] != metrics["neutral_threshold"]:
        raise ValueError("Cannot present results with different neutral_threshold")
    lines = [f"\nRun: {run.name} | best epoch: {metrics['selected_epoch']} | neutral threshold: {metrics['neutral_threshold']:g}",
             f"{'Split':<8}{'N':>6}{'Accuracy':>12}{'Macro-F1':>12}{'MAE':>12}{'Pearson':>12}"]
    for split in ("valid", "test"):
        item = evaluation["summary"][split]
        if item["status"] != "complete":
            lines.append(f"{split:<8} not_run")
            continue
        values = [_number(item["Accuracy"], True), _number(item["Macro_F1"], True),
                  _number(item["MAE"]), _number(item["Pearson"])]
        lines.append(f"{split:<8}{item['n']:>6}" + "".join(f"{value:>12}" for value in values))
    print("\n".join(lines))
    print(f"\nEvaluation: {destination}\n", flush=True)
    return destination
