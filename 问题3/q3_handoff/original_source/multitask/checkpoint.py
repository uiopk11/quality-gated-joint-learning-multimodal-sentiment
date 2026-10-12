"""Strict multitask checkpoints and data-independent inference restoration."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re

import torch

from multitask.config import (CLASS_ORDER, CODE_VERSION, DECISION_RULE,
                              PROTOCOL_VERSION, validate_config)
from multitask.provenance import implementation_provenance, validate_implementation
from multitask.selection import selection_protocol, selection_score
from standardized.normalization import AVNormalizer, metadata_fingerprint, validate_preprocessing


_FIELDS = {
    "format_version", "code_version", "protocol_version", "purpose", "method",
    "model_state_dict", "config", "training_config_sha256", "epoch",
    "selection_split", "selection_protocol", "selection_score", "valid_metrics",
    "decision_rule", "class_order", "neutral_threshold", "data_binding",
    "data_binding_sha256", "preprocessing", "preprocessing_sha256", "code_sha256",
}


def _fingerprint(value):
    try:
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Checkpoint metadata must be finite JSON data") from error
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_buffers(state, preprocessing):
    if not isinstance(state, dict) or not state:
        raise ValueError("Missing model_state_dict")
    expected = AVNormalizer(preprocessing).state_dict()
    for name, tensor in expected.items():
        saved = state.get("av_normalizer." + name)
        if (not isinstance(saved, torch.Tensor) or saved.dtype != tensor.dtype
                or not torch.equal(saved.detach().cpu(), tensor)):
            raise ValueError("Checkpoint preprocessing buffers differ from metadata: " + name)
    classifier, bias = state.get("cls_head.weight"), state.get("cls_head.bias")
    regressor = state.get("post_fusion_layer_3.weight")
    if (not isinstance(classifier, torch.Tensor) or classifier.ndim != 2 or classifier.shape[0] != 3
            or not isinstance(bias, torch.Tensor) or bias.shape != (3,)
            or not isinstance(regressor, torch.Tensor) or regressor.shape != (1, classifier.shape[1])):
        raise ValueError("Checkpoint requires compatible classification and regression heads")


def _validate_model_method(model, method):
    declared = [getattr(model, name) for name in ("multitask_method", "method") if hasattr(model, name)]
    if not declared or any(value != method for value in declared):
        raise ValueError("Model method differs from the saved checkpoint or config")


def _validate_payload(checkpoint, *, allow_smoke=False):
    if not isinstance(checkpoint, dict) or set(checkpoint) != _FIELDS:
        raise ValueError("Missing or unexpected multitask checkpoint metadata fields")
    if (checkpoint["format_version"] != 1 or checkpoint["code_version"] != CODE_VERSION
            or checkpoint["protocol_version"] != PROTOCOL_VERSION):
        raise ValueError("Not a supported multitask checkpoint; legacy artifacts require their original runners")
    purpose = checkpoint["purpose"]
    if purpose != "train" and not (allow_smoke and purpose == "smoke"):
        raise ValueError("A smoke checkpoint cannot be used for formal evaluation")
    config = checkpoint["config"]
    validate_config(config)
    if checkpoint["training_config_sha256"] != _fingerprint(config):
        raise ValueError("Checkpoint config fingerprint mismatch")
    method = config["method"]
    if checkpoint["method"] != method:
        raise ValueError("Checkpoint method disagrees with config")
    if (checkpoint["selection_split"] != "valid"
            or checkpoint["selection_protocol"] != selection_protocol(method)):
        raise ValueError("Checkpoint does not match the fixed clean/mixed Accuracy selection protocol")
    if (checkpoint["decision_rule"] != DECISION_RULE or checkpoint["class_order"] != CLASS_ORDER
            or checkpoint["neutral_threshold"] is not None):
        raise ValueError("Checkpoint must use logits_argmax, fixed class order and no regression threshold")
    epoch = checkpoint["epoch"]
    if type(epoch) is not int or not 1 <= epoch <= config["epochs"]:
        raise ValueError("Checkpoint epoch must be within the configured training budget")
    metrics = checkpoint["valid_metrics"]
    score = selection_score(method, metrics)
    if checkpoint["selection_score"] != score:
        raise ValueError("Saved selection score disagrees with individual valid conditions")
    for condition in selection_protocol(method)["conditions"]:
        result = metrics[condition]
        expected = {"decision_rule": DECISION_RULE, "class_order": CLASS_ORDER,
                    "neutral_threshold": None, "prediction_clipping": False}
        if any(key not in result or type(result[key]) is not type(value) or result[key] != value
               for key, value in expected.items()):
            raise ValueError("Valid condition metadata conflicts with the fixed classification/regression protocol: " + condition)
        count = result.get("n")
        if type(count) is not int or count <= 0 or (purpose == "train" and count != 728):
            raise ValueError("Formal selection requires all 728 valid samples in every condition")
    _fingerprint(metrics)
    binding = checkpoint["data_binding"]
    if (not isinstance(binding, dict) or not isinstance(binding.get("data_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", binding["data_sha256"])):
        raise ValueError("Checkpoint requires a valid source data binding")
    if checkpoint["data_binding_sha256"] != _fingerprint(binding):
        raise ValueError("Checkpoint data binding fingerprint mismatch")
    preprocessing = validate_preprocessing(checkpoint["preprocessing"], source_sha256=binding["data_sha256"])
    if checkpoint["preprocessing_sha256"] != metadata_fingerprint(preprocessing):
        raise ValueError("Checkpoint preprocessing fingerprint mismatch")
    _validate_buffers(checkpoint["model_state_dict"], preprocessing)
    return checkpoint


def save_checkpoint(path, model, config, epoch, valid_metrics, data_binding, root, purpose="train"):
    """Save only a fully specified new artifact; never rewrite a legacy model.

    ``valid_metrics`` contains clean only for M0, clean and fixed main mixed for
    M1/M2. A smoke checkpoint is explicitly marked and refused by formal loads.
    """
    validate_config(config)
    _validate_model_method(model, config["method"])
    preprocessing = model.av_normalizer.metadata()
    config, valid_metrics, data_binding = copy.deepcopy((dict(config), valid_metrics, data_binding))
    payload = {
        "format_version": 1, "code_version": CODE_VERSION, "protocol_version": PROTOCOL_VERSION,
        "purpose": purpose, "method": config["method"],
        "model_state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "config": config, "training_config_sha256": _fingerprint(config), "epoch": epoch,
        "selection_split": "valid", "selection_protocol": selection_protocol(config["method"]),
        "selection_score": selection_score(config["method"], valid_metrics), "valid_metrics": valid_metrics,
        "decision_rule": DECISION_RULE, "class_order": list(CLASS_ORDER), "neutral_threshold": None,
        "data_binding": data_binding, "data_binding_sha256": _fingerprint(data_binding),
        "preprocessing": preprocessing, "preprocessing_sha256": metadata_fingerprint(preprocessing),
        "code_sha256": implementation_provenance(root),
    }
    _validate_payload(payload, allow_smoke=(purpose == "smoke"))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return payload


def load_checkpoint(path, root, allow_smoke=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    _validate_payload(checkpoint, allow_smoke=allow_smoke)
    validate_implementation(checkpoint["code_sha256"], implementation_provenance(root))
    return checkpoint


def restore_model_state(model, checkpoint):
    """Reject altered statistics before strict state loading can overwrite them."""
    _validate_payload(checkpoint, allow_smoke=True)
    if model.av_normalizer.metadata() != checkpoint["preprocessing"]:
        raise ValueError("Model preprocessing metadata differs from the saved checkpoint")
    _validate_model_method(model, checkpoint["method"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def load_for_prediction(checkpoint_path, root, device="cpu", allow_smoke=False):
    """Restore saved statistics and both heads without opening input datasets.

    Only local code, checkpoint and base BERT architecture/weights are accessed;
    train/valid/test data, labels, references and fitting functions are unused.
    """
    from multitask.model import build_model

    root = Path(root).resolve()
    checkpoint = load_checkpoint(checkpoint_path, root, allow_smoke=allow_smoke)
    model = build_model(checkpoint["config"], root, checkpoint["preprocessing"])
    restore_model_state(model, checkpoint)
    return model.to(device).eval()
