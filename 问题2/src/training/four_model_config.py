"""Frozen common settings and explicit roles for the new four-model protocol."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Mapping

from src.models.baseline import OFFICIAL_MOSEI_DEFAULTS
from src.models.four_models import METHODS, is_joint, is_missing, method_name
from src.runtime import ROOT


CODE_VERSION = "four_models_avstd_v1"
PROTOCOL_VERSION = "four_models_accuracy_v1"
CLASS_ORDER = ["negative", "neutral", "positive"]
JOINT_DECISION_RULE = "logits_argmax"
REGRESSION_DECISION_RULE = "valid_symmetric_threshold"
TIE_TOLERANCE = 1e-12


def decision_rule(method):
    return JOINT_DECISION_RULE if is_joint(method) else REGRESSION_DECISION_RULE


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
    "classification_weight": 0.0, "regression_weight": 1.0,
    "prediction_reference_weight": 1.0, "prediction_missing_weight": 1.0,
    "consistency_weight": 1.0, "reconstruction_weight": 1.0,
    "reconstruction_loss": "SmoothL1Loss",
    "classification_loss": "CrossEntropyLoss", "regression_loss": "L1Loss",
    "class_weights": None, "label_smoothing": 0.0,
    "code_version": CODE_VERSION, "protocol_version": PROTOCOL_VERSION,
    "decision_rule": REGRESSION_DECISION_RULE, "class_order": CLASS_ORDER,
    "neutral_threshold": None,
    "initialization": "same-seed local base BERT plus fresh EMT; optional classifier preserves backbone RNG state",
    "singleton_batchnorm_policy": "running statistics only for physical batch=1",
}
_SCHEMA = {**OFFICIAL_MOSEI_DEFAULTS, **_RUNTIME_DEFAULTS, "method": "M0"}


def role_settings(method):
    """These fields are derived from the versioned role, never CLI overrides."""
    method = method_name(method)
    joint = is_joint(method)
    return {"method": method, "regression_only": not is_missing(method),
            "classification_weight": 1.0 if joint else 0.0,
            "regression_weight": 0.5 if joint else 1.0,
            "decision_rule": decision_rule(method)}


def default_config(method, root=None):
    method = method_name(method)
    root = Path(root).resolve() if root is not None else ROOT
    config = json.loads((root / "configs/four_models_v1.json").read_text(encoding="utf-8-sig"))
    # The preset records M0; validate before deriving another role so stale or
    # mistyped role fields in the preset cannot silently disappear.
    validate_config(config)
    config.update(role_settings(method))
    return validate_config(config)


def validate_config(config):
    """Reject old semantics and edits to the frozen paired comparison protocol."""
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    missing, extra = set(_SCHEMA) - set(config), set(config) - set(_SCHEMA)
    if missing or extra:
        raise ValueError(f"Invalid config keys; missing={sorted(missing)}, unknown={sorted(extra)}")
    if config["method"] not in METHODS:
        raise ValueError("method must be M0, M1, M2 or M3")
    expected = {**deepcopy(_SCHEMA), **role_settings(config["method"])}
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
    mutable = {"epochs", "batch_size", "eval_batch_size", "gradient_accumulation_steps",
               "early_stop_patience", "cpu_threads", "seed", "device", "gradient_checkpointing",
               "data_path", "artifact_path", "bert_path"}
    for key in set(_SCHEMA) - mutable:
        value = expected[key]
        if type(config[key]) is not type(value) and not (
            type(config[key]) in (int, float) and type(value) in (int, float)
        ):
            raise ValueError(f"{key} has the wrong type")
        if config[key] != value:
            raise ValueError(f"{key} is fixed to {value!r} for {config['method']} in {CODE_VERSION}")
    try:
        json.dumps(dict(config), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("config must contain finite JSON data") from error
    return config
