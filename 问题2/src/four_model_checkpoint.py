"""Independent four-model checkpoints: explicit roles, frozen decisions and provenance.

Legacy checkpoints retain their original meanings and are never interpreted as
new roles here. Restoration needs local weights/code, but no source dataset.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import re

import torch

from src.data.normalization import AVNormalizer, metadata_fingerprint, validate_preprocessing
from src.evaluation.four_models import fingerprint, joint_decision, validate_decision
from src.training.four_model_config import CODE_VERSION, PROTOCOL_VERSION, CLASS_ORDER, is_joint, validate_config
from src.training.four_model_selection import selection_protocol, selection_score


ROLE_MAPPING = {
    "M0": {"role": "regression_complete", "input": "complete", "prediction": "regression",
           "dlfr": "complete_input_flow", "source_version": "av_standardized_v1", "source_method": "M0"},
    "M1": {"role": "joint_complete", "input": "complete", "prediction": "joint",
           "dlfr": "complete_input_flow", "source_version": "multitask_avstd_v1", "source_method": "M0"},
    "M2": {"role": "regression_missing_full_dlfr", "input": "complete_and_missing", "prediction": "regression",
           "dlfr": "high_level_consistency_and_low_level_reconstruction", "source_version": "av_standardized_v1", "source_method": "M2"},
    "M3": {"role": "joint_missing_full_dlfr", "input": "complete_and_missing", "prediction": "joint",
           "dlfr": "high_level_consistency_and_low_level_reconstruction", "source_version": "multitask_avstd_v1", "source_method": "M2"},
}
_FIELDS = {"format_version", "code_version", "protocol_version", "purpose", "method", "role_mapping",
    "model_state_dict", "config", "training_config_sha256", "epoch", "selection_split", "selection_protocol",
    "selection_score", "valid_metrics", "decision", "decision_sha256", "decision_rule", "class_order",
    "neutral_threshold", "data_binding", "data_binding_sha256", "preprocessing", "preprocessing_sha256", "code_sha256"}
_NEW_SOURCES = ("src/four_model_checkpoint.py", "src/models/four_models.py", "src/evaluation/four_models.py",
    "src/training/four_models.py", "src/training/four_model_config.py", "src/training/four_model_losses.py",
    "src/training/four_model_selection.py", "configs/four_models_v1.json")
_fingerprint = fingerprint


def implementation_provenance(root):
    """Include inherited implementation and every version-specific dependency."""
    from src.compat.multitask import implementation_provenance as inherited

    root = Path(root).resolve()
    result = dict(inherited(root))
    for relative in _NEW_SOURCES:
        with (root / relative).open("rb") as stream:
            result[relative] = hashlib.file_digest(stream, "sha256").hexdigest()
    # The unified entry determines version routing and is part of the contract.
    for relative in ("run.py",):
        if (root / relative).is_file():
            with (root / relative).open("rb") as stream:
                result[relative] = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(sorted(result.items()))


def _validate_buffers(state, preprocessing, method):
    if not isinstance(state, dict) or not state:
        raise ValueError("Missing model_state_dict")
    expected = AVNormalizer(preprocessing).state_dict()
    for name, tensor in expected.items():
        saved = state.get("av_normalizer." + name)
        if (not isinstance(saved, torch.Tensor) or saved.dtype != tensor.dtype
                or not torch.equal(saved.detach().cpu(), tensor)):
            raise ValueError("Checkpoint preprocessing buffers differ from metadata: " + name)
    regressor, regressor_bias = state.get("post_fusion_layer_3.weight"), state.get("post_fusion_layer_3.bias")
    if (not isinstance(regressor, torch.Tensor) or regressor.ndim != 2 or regressor.shape[0] != 1
            or not isinstance(regressor_bias, torch.Tensor) or regressor_bias.shape != (1,)):
        raise ValueError("Checkpoint requires a compatible regression head")
    if is_joint(method):
        classifier, bias = state.get("cls_head.weight"), state.get("cls_head.bias")
        if (not isinstance(classifier, torch.Tensor) or classifier.shape != (3, regressor.shape[1])
                or not isinstance(bias, torch.Tensor) or bias.shape != (3,)):
            raise ValueError("Joint checkpoint requires a compatible classification head")
    elif any(name.startswith("cls_head.") for name in state):
        raise ValueError("Regression baseline must not contain a classification head")


def _validate_model_method(model, method):
    declared = [getattr(model, name) for name in ("four_model_method", "method") if hasattr(model, name)]
    if not declared or any(value != method for value in declared):
        raise ValueError("Model method differs from the four-model checkpoint/config")


def _validate_payload(checkpoint, *, allow_smoke=False):
    if not isinstance(checkpoint, dict) or set(checkpoint) != _FIELDS:
        raise ValueError("Missing or unexpected four-model checkpoint metadata fields")
    if (checkpoint["format_version"] != 1 or checkpoint["code_version"] != CODE_VERSION
            or checkpoint["protocol_version"] != PROTOCOL_VERSION):
        raise ValueError("Not a four-model checkpoint; historical checkpoints require their original version")
    purpose = checkpoint["purpose"]
    if purpose != "train" and not (allow_smoke and purpose == "smoke"):
        raise ValueError("A smoke checkpoint cannot be used for formal evaluation")
    config = validate_config(checkpoint["config"])
    if checkpoint["training_config_sha256"] != fingerprint(config):
        raise ValueError("Checkpoint config fingerprint mismatch")
    method = config["method"]
    if checkpoint["method"] != method or checkpoint["role_mapping"] != ROLE_MAPPING[method]:
        raise ValueError("Checkpoint method/source role mapping disagrees with config")
    protocol = selection_protocol(method)
    if checkpoint["selection_split"] != "valid" or checkpoint["selection_protocol"] != protocol:
        raise ValueError("Checkpoint does not match the fixed paired Accuracy selection protocol")
    decision = validate_decision(checkpoint["decision"], method)
    if checkpoint["decision_sha256"] != fingerprint(decision):
        raise ValueError("Checkpoint decision fingerprint mismatch")
    for key in ("decision_rule", "class_order", "neutral_threshold"):
        if checkpoint[key] != decision[key]:
            raise ValueError("Checkpoint decision metadata disagrees: " + key)
    epoch = checkpoint["epoch"]
    if type(epoch) is not int or not 1 <= epoch <= config["epochs"]:
        raise ValueError("Checkpoint epoch must be within the configured training budget")
    metrics = checkpoint["valid_metrics"]
    if checkpoint["selection_score"] != selection_score(method, metrics):
        raise ValueError("Saved selection score disagrees with individual valid conditions")
    for condition in protocol["conditions"]:
        result = metrics[condition]
        expected = {"split": "valid", "condition": condition, "method": method,
                    "decision_rule": decision["decision_rule"], "class_order": list(CLASS_ORDER),
                    "neutral_threshold": decision["neutral_threshold"], "prediction_clipping": False,
                    "decision": decision, "decision_sha256": fingerprint(decision)}
        if any(key not in result or result[key] != value for key, value in expected.items()):
            raise ValueError("Valid condition metadata conflicts with the frozen decision: " + condition)
        count = result.get("n")
        if type(count) is not int or count <= 0 or (purpose == "train" and count != 728):
            raise ValueError("Formal selection requires all 728 valid samples in every condition")
    fingerprint(metrics)
    binding = checkpoint["data_binding"]
    if (not isinstance(binding, dict) or not isinstance(binding.get("data_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", binding["data_sha256"])):
        raise ValueError("Checkpoint requires a valid source data binding")
    if checkpoint["data_binding_sha256"] != fingerprint(binding):
        raise ValueError("Checkpoint data binding fingerprint mismatch")
    preprocessing = validate_preprocessing(checkpoint["preprocessing"], source_sha256=binding["data_sha256"])
    if checkpoint["preprocessing_sha256"] != metadata_fingerprint(preprocessing):
        raise ValueError("Checkpoint preprocessing fingerprint mismatch")
    _validate_buffers(checkpoint["model_state_dict"], preprocessing, method)
    return checkpoint


def save_checkpoint(path, model, config, epoch, valid_metrics, data_binding, root, purpose="train", decision=None):
    validate_config(config)
    method = config["method"]
    _validate_model_method(model, method)
    if decision is None:
        decision = getattr(model, "decision", None)
    if decision is None and isinstance(valid_metrics, dict) and valid_metrics:
        decision = next(iter(valid_metrics.values())).get("decision")
    if decision is None and is_joint(method):
        decision = joint_decision(method)
    decision = validate_decision(decision, method)
    preprocessing = model.av_normalizer.metadata()
    config, valid_metrics, data_binding = copy.deepcopy((dict(config), valid_metrics, data_binding))
    payload = {"format_version": 1, "code_version": CODE_VERSION, "protocol_version": PROTOCOL_VERSION,
        "purpose": purpose, "method": method, "role_mapping": copy.deepcopy(ROLE_MAPPING[method]),
        "model_state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "config": config, "training_config_sha256": fingerprint(config), "epoch": epoch,
        "selection_split": "valid", "selection_protocol": selection_protocol(method),
        "selection_score": selection_score(method, valid_metrics), "valid_metrics": valid_metrics,
        "decision": decision, "decision_sha256": fingerprint(decision),
        "decision_rule": decision["decision_rule"], "class_order": list(CLASS_ORDER),
        "neutral_threshold": decision["neutral_threshold"],
        "data_binding": data_binding, "data_binding_sha256": fingerprint(data_binding),
        "preprocessing": preprocessing, "preprocessing_sha256": metadata_fingerprint(preprocessing),
        "code_sha256": implementation_provenance(root)}
    _validate_payload(payload, allow_smoke=(purpose == "smoke"))
    path = Path(path)
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=True)
        _validate_payload(existing, allow_smoke=(purpose == "smoke"))
        for key in ("code_sha256", "config", "purpose", "data_binding"):
            if existing[key] != payload[key]:
                raise ValueError("Refusing to replace a checkpoint from a different run/protocol")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return payload


def load_checkpoint(path, root, allow_smoke=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    _validate_payload(checkpoint, allow_smoke=allow_smoke)
    from src.compat.provenance import validate_four_model_provenance
    validate_four_model_provenance(checkpoint["code_sha256"], implementation_provenance(root), root)
    return checkpoint


def restore_model_state(model, checkpoint):
    _validate_payload(checkpoint, allow_smoke=True)
    if model.av_normalizer.metadata() != checkpoint["preprocessing"]:
        raise ValueError("Model preprocessing metadata differs from the saved checkpoint")
    _validate_model_method(model, checkpoint["method"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.decision = copy.deepcopy(checkpoint["decision"])
    return model


def load_for_prediction(checkpoint_path, root, device="cpu", allow_smoke=False):
    from src.models.four_models import build_model

    root = Path(root).resolve()
    checkpoint = load_checkpoint(checkpoint_path, root, allow_smoke=allow_smoke)
    model = build_model(checkpoint["config"], root, checkpoint["preprocessing"])
    restore_model_state(model, checkpoint)
    return model.to(device).eval()
