"""Route M0/M2 to the preserved standardized MAE regression implementation.

This module owns no model, loss, threshold search, training loop, or metric
formula.  The historical runners remain responsible for those operations and
for authenticating their saved checkpoints and frozen validation thresholds.
The four-model Accuracy protocol remains readable through its original runner.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import math
from pathlib import Path

from src.runtime import ROOT


METHODS = ("M0", "M2")
_CLI_OVERRIDES = {
    "device", "seed", "epochs", "batch_size", "eval_batch_size",
    "gradient_accumulation_steps",
}


def _method(method):
    if method not in METHODS:
        raise ValueError("The restored MAE baseline adapter supports only M0/M2")
    return method


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _defaults(method):
    _method(method)
    if method == "M0":
        config = _read(ROOT / "configs/baseline.json")
    else:
        from src.training import missing_standardized
        config = missing_standardized.default_config("M2")
    # These were the effective settings of the previous standardized 16/4
    # baselines.  Do not load an experiment's outputs as a configuration source.
    config.update(epochs=20, batch_size=16, gradient_accumulation_steps=4,
                  eval_batch_size=8, early_stop_patience=8 if method == "M0" else 5,
                  early_stop_min_delta=1e-6)
    return config


def _training_overrides(method, overrides):
    if not isinstance(overrides, Mapping):
        raise ValueError("Baseline configuration overrides must be a mapping")
    overrides = deepcopy(dict(overrides))
    if "training" in overrides:
        if set(overrides) != {"training", "evaluation_protocol"}:
            raise ValueError("A saved baseline config must contain training and evaluation_protocol only")
        if overrides["evaluation_protocol"] != _read(ROOT / "configs/evaluation.json"):
            raise ValueError("Restored baselines require the legacy valid Macro_F1 threshold protocol")
        overrides = overrides["training"]
        if not isinstance(overrides, Mapping):
            raise ValueError("Saved training configuration must be a mapping")
        overrides = deepcopy(dict(overrides))
    if "method" in overrides and overrides["method"] != method:
        raise ValueError("config method conflicts with the requested restored baseline")
    # The original complete-input config has no method field.  Its checkpoint
    # writer records method=M0 explicitly; preserve the original config shape.
    if method == "M0":
        overrides.pop("method", None)
    return overrides


def _validate(method, config):
    _method(method)
    config = _training_overrides(method, config)
    expected = set(_defaults(method))
    missing, unknown = expected - set(config), set(config) - expected
    if missing or unknown:
        raise ValueError(
            "Restored MAE baseline config has invalid keys; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}. "
            "Do not reuse a four_models_accuracy_v1 config for new MAE training."
        )
    try:
        json.dumps(config, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Baseline config must contain finite JSON values") from error
    if method == "M2":
        from src.training import missing_standardized
        return missing_standardized.validate_config(config)
    # The original M0 runner keeps these checks inside its CLI main().  Repeat
    # only argument validation here so its numerical train() can be called
    # directly without rewriting argv or changing the original runner.
    for key in ("epochs", "batch_size", "eval_batch_size", "gradient_accumulation_steps",
                "early_stop_patience", "cpu_threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["seed"]) is not int or not 0 <= config["seed"] <= 2**32 - 1:
        raise ValueError("seed must be an integer between 0 and 4294967295")
    delta = config["early_stop_min_delta"]
    if type(delta) not in (int, float) or not math.isfinite(delta) or delta <= 0:
        raise ValueError("early_stop_min_delta must be positive and finite")
    if config["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if config["missing_rate"] != [0.0, 0.0, 0.0] or config["regression_only"] is not True:
        raise ValueError("Restored M0 supports complete-input regression only")
    return config


def make_config(method, overrides=None, cli_overrides=None):
    """Create legacy settings without consulting or modifying any saved run."""
    config = _defaults(method)
    if overrides is not None:
        config.update(_training_overrides(method, overrides))
    if cli_overrides is not None:
        if not isinstance(cli_overrides, Mapping):
            raise ValueError("CLI overrides must be a mapping")
        unknown = set(cli_overrides) - _CLI_OVERRIDES
        if unknown:
            raise ValueError(f"Unsupported baseline CLI overrides: {sorted(unknown)}")
        config.update({key: value for key, value in cli_overrides.items() if value is not None})
    return _validate(method, config)


def _new_output(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output}")
    return output


def train(method, config, output):
    """Run the unchanged historical MAE loop and its post-selection calibration."""
    config = _validate(method, config)
    output = _new_output(output)
    if method == "M0":
        from src.training import baseline_standardized
        return baseline_standardized.train(config, output, smoke=False)
    from src.training import missing_standardized
    return missing_standardized.train(config, output)


def smoke(method, config, output):
    """Use the historical bounded development probe, never a formal experiment."""
    config = _validate(method, config)
    output = _new_output(output)
    if method == "M0":
        from src.training import baseline_standardized
        config.update(epochs=1, eval_batch_size=2, gradient_accumulation_steps=2)
        return baseline_standardized.train(config, output, smoke=True)
    from src.training import missing_standardized
    return missing_standardized.smoke(config, output)


def final_evaluate(output, split, device_override=None, paired=False, evaluation_output=None):
    """Keep the saved legacy rule and the original final-test execution path.

    Legacy M2 test includes mixed test and both valid/test paired panels.  Legacy
    M0 test evaluates clean test.  Their original writers update the run in place;
    the command-boundary reporting wrapper preserves previous report snapshots.
    """
    if split != "test":
        raise ValueError("Restored MAE baselines support evaluate --split test; use report for saved valid results")
    if evaluation_output is not None:
        raise ValueError("Restored MAE evaluation uses the legacy run directory; --evaluation-output is unsupported")
    if device_override not in (None, "cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    output = Path(output).resolve()
    checkpoint_path = output / "best.pt"
    # Inspect only enough to select the correct original reader.  That reader
    # remains responsible for full provenance, preprocessing and rule validation.
    import torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    method = checkpoint.get("method")
    if (checkpoint.get("code_version") != "av_standardized_v1"
            or checkpoint.get("format_version") != 1 or checkpoint.get("purpose") != "train"
            or method not in METHODS
            or checkpoint.get("selection_split") != "valid"
            or checkpoint.get("selection_metric") != "MAE"):
        raise ValueError("Restored evaluation requires an original av_standardized_v1 M0/M2 valid-MAE checkpoint")
    if method == "M2" and (checkpoint.get("selection_family"), checkpoint.get("selection_condition")) != (
        "main_mixed", "mixed_v1_s0"
    ):
        raise ValueError("Restored M2 requires the original mixed-valid MAE checkpoint")
    del checkpoint
    if method == "M0":
        if paired:
            raise ValueError("Restored M0 uses clean final test; --paired is only supported for M2")
        from src.training import baseline_standardized
        return baseline_standardized.final_test(checkpoint_path, device_override)
    from src.training import missing_standardized
    print("Restored M2 legacy test includes mixed test and the frozen valid/test paired panels.", flush=True)
    return missing_standardized.test(output, device_override)
