"""Complete, validated configuration for the first multitask protocol."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Mapping

from baseline.model import OFFICIAL_MOSEI_DEFAULTS


CODE_VERSION = "multitask_avstd_v1"
PROTOCOL_VERSION = "multitask_accuracy_v1"
CLASS_ORDER = ["negative", "neutral", "positive"]
DECISION_RULE = "logits_argmax"
TIE_TOLERANCE = 1e-12

# These values supplement the original architecture; the file contains all
# effective values, including the previously implicit Adam defaults.
_RUNTIME_DEFAULTS = {
    "data_path": "data/aligned_50.pkl",
    "artifact_path": "dataset_construction/artifacts/mix_v1",
    "seed": 1111, "epochs": 20, "early_stop_patience": 8,
    "early_stop_min_delta": TIE_TOLERANCE,
    "batch_size": 16, "eval_batch_size": 16,
    "gradient_accumulation_steps": 4, "num_workers": 0,
    "device": "cuda", "cpu_threads": 4,
    "learning_rate_bert": 2e-5, "learning_rate_audio": 1e-4,
    "learning_rate_video": 1e-4, "learning_rate_other": 1e-4,
    "weight_decay_bert": 0.001, "weight_decay_audio": 0.001,
    "weight_decay_video": 0.001, "weight_decay_other": 0.001,
    "optimizer": "Adam", "adam_betas": [0.9, 0.999],
    "adam_eps": 1e-8, "adam_amsgrad": False,
    "classification_weight": 1.0, "regression_weight": 0.5,
    "prediction_reference_weight": 1.0, "prediction_missing_weight": 1.0,
    "consistency_weight": 1.0, "reconstruction_weight": 1.0,
    "reconstruction_loss": "SmoothL1Loss",
    "classification_loss": "CrossEntropyLoss", "regression_loss": "L1Loss",
    "class_weights": None, "label_smoothing": 0.0,
    "code_version": CODE_VERSION, "protocol_version": PROTOCOL_VERSION,
    "decision_rule": DECISION_RULE, "class_order": CLASS_ORDER,
    "neutral_threshold": None,
    "initialization": "same-seed local base BERT plus fresh EMT; no complete-run warm start",
    "singleton_batchnorm_policy": "running statistics only for physical batch=1",
}
_SCHEMA = {**OFFICIAL_MOSEI_DEFAULTS, **_RUNTIME_DEFAULTS, "method": "M0"}


def default_config(method, root=None):
    """Load the new complete preset, then set the selected model variant.

    No old configuration is rewritten. The new JSON records the architecture
    originally inherited from baseline.json and OFFICIAL_MOSEI_DEFAULTS.
    """
    if method not in ("M0", "M1", "M2"):
        raise ValueError("method must be M0, M1 or M2")
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/multitask_v1.json").read_text(encoding="utf-8-sig"))
    config.update(method=method, regression_only=(method != "M2"))
    return validate_config(config)


def validate_config(config):
    """Reject missing/unknown options and silent changes to fixed semantics.

    Runtime batch/epoch/device settings may change for bounded development;
    explicit loss weights can be used for separately declared later ablations.
    The architecture, optimizer and classification rule remain versioned.
    """
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    missing, extra = set(_SCHEMA) - set(config), set(config) - set(_SCHEMA)
    if missing or extra:
        raise ValueError(f"Invalid config keys; missing={sorted(missing)}, unknown={sorted(extra)}")
    if config["method"] not in ("M0", "M1", "M2"):
        raise ValueError("method must be M0, M1 or M2")
    if type(config["regression_only"]) is not bool or config["regression_only"] != (config["method"] != "M2"):
        raise ValueError("regression_only disagrees with method (only M2 uses DLFR)")
    for key in ("epochs", "batch_size", "eval_batch_size", "gradient_accumulation_steps",
                "early_stop_patience", "cpu_threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["seed"]) is not int or not 0 <= config["seed"] <= 2**32 - 1:
        raise ValueError("seed must be in 0..2**32-1")
    if config["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if type(config["gradient_checkpointing"]) is not bool:
        raise ValueError("gradient_checkpointing must be boolean")
    for key in ("data_path", "artifact_path", "bert_path"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a nonempty path string")
    for key in ("classification_weight", "regression_weight", "consistency_weight", "reconstruction_weight"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if config["classification_weight"] <= 0:
        raise ValueError("The multitask protocol requires classification supervision")
    if config["method"] == "M2" and any(config[key] <= 0 for key in ("consistency_weight", "reconstruction_weight")):
        raise ValueError("M2 requires both DLFR losses; use M1 for no DLFR")
    mutable = {"method", "regression_only", "epochs", "batch_size", "eval_batch_size",
               "gradient_accumulation_steps", "early_stop_patience", "cpu_threads", "seed",
               "device", "gradient_checkpointing", "data_path", "artifact_path", "bert_path",
               "classification_weight", "regression_weight", "consistency_weight", "reconstruction_weight"}
    for key in set(_SCHEMA) - mutable:
        expected = _SCHEMA[key]
        # Type checks keep booleans from silently passing numeric equality.
        if type(config[key]) is not type(expected) and not (
            type(config[key]) in (int, float) and type(expected) in (int, float)
        ):
            raise ValueError(f"{key} has the wrong type")
        if config[key] != expected:
            raise ValueError(f"{key} is fixed to {expected!r} in {CODE_VERSION}")
    try:
        json.dumps(dict(config), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("config must contain finite JSON data") from error
    return config
