"""External adapter for the unchanged formal mt_m0_avstd_s1111_v1 model.

Public predictions always call the original ``multitask.model.predict_batch``.
All tensors stay on the chosen device. Class order is fixed: negative, neutral,
positive. Softmax values are uncalibrated probabilities, not confidence scores.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .runtime import configure_runtime

HANDOFF_ROOT = Path(configure_runtime()).resolve()

import torch

from multitask.checkpoint import load_for_prediction
from multitask.model import predict_batch

from .hooks import HOOK_SPECS, RepresentationHooks


CLASS_ORDER = ("negative", "neutral", "positive")


def _prediction_fields(output):
    logits = output["logits"]
    intensity = output["intensity"].reshape(-1)
    if logits.ndim != 2 or logits.shape[1] != 3 or intensity.shape != (logits.shape[0],):
        raise ValueError("Unexpected formal M0 prediction shapes")
    return {"logits": logits, "probabilities": logits.softmax(dim=-1),
            "pred_class": logits.argmax(dim=-1), "intensity": intensity}


def _shape_tree(value):
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, dict):
        return {name: _shape_tree(item) for name, item in value.items()}
    return value


class FrozenM0Adapter:
    """Read-only inference wrapper; this object is not safe for parallel forwards.

    Hooks are temporary and ordinary extraction does not replace model outputs.
    The caller may retain graphs using ``enable_grad=True``. Parameters remain
    frozen, and no optimizer, fitting operation, or prediction rewrite is used.
    """
    class_order = CLASS_ORDER

    def __init__(self, model, device, checkpoint_path):
        if getattr(model, "method", None) != "M0":
            raise ValueError("This handoff supports the formal M0 model only")
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)

    @property
    def tokenizer(self):
        return self.model.text_model.tokenizer

    def predict(self, batch, *, enable_grad=False):
        """Return logits [B,3], probabilities [B,3], class [B], intensity [B].

        ``enable_grad=True`` keeps a graph for continuous batch tensors already
        requiring gradients. To explain BERT embeddings with frozen parameters,
        use ``extract_representations(batch, enable_grad=True)`` instead.
        """
        self.model.eval()
        context = torch.enable_grad() if enable_grad else torch.no_grad()
        with context:
            return _prediction_fields(predict_batch(self.model, batch, self.device))

    def extract_representations(self, batch, *, enable_grad=False):
        """Observe exact model tensors without changing forward values.

        ``*_local`` are projected EMT inputs before positional embedding;
        ``*_local_encoder`` are encoder outputs; ``*_local_post_fusion`` are EMT
        outputs. Only ``local_masks`` positions are meaningful content. Text has
        49 returned positions because the CLS was sliced off; SEP/padding are
        present but masked. A/V sequence width is max(content length) in batch.

        With ``enable_grad=True``, three optional hooks replace BERT's complete
        embeddings output and standardized audio/vision with identical detached
        leaves. ``gradient_inputs`` are suitable for torch.autograd.grad. This
        defines local derivatives in embedding/standardized-feature space and
        does not differentiate raw token IDs or upstream normalization. Final
        post-fusion local outputs are not consumed by the prediction heads;
        text_summary/text_local_encoder are observational slices. Prefer exact
        ``text_encoded``, ``*_local`` or ``gradient_inputs`` graph tensors when
        differentiating. Parameters remain requires_grad=False.
        """
        self.model.eval()
        if enable_grad and torch.is_inference_mode_enabled():
            raise RuntimeError("Gradient extraction must run outside torch.inference_mode()")
        if enable_grad and not self.model.text_model.use_finetune:
            raise RuntimeError("Expected the formal M0 BERT path with use_finetune=True")
        context = torch.enable_grad() if enable_grad else torch.no_grad()
        with context, RepresentationHooks(self.model, continuous_grad=enable_grad) as capture:
            result = _prediction_fields(predict_batch(self.model, batch, self.device))
        result.update(capture.values)
        text_lengths = batch["text"][:, 1].sum(dim=1).to(device=self.device, dtype=torch.long) - 2
        result["lengths"] = {
            "text": text_lengths,
            "audio": batch["audio_lengths"].to(device=self.device, dtype=torch.long).reshape(-1),
            "vision": batch["vision_lengths"].to(device=self.device, dtype=torch.long).reshape(-1),
        }
        result["global_context_flat"] = result["global_context"].reshape(result["logits"].shape[0], -1)
        result["gradient_inputs"] = capture.gradient_inputs
        result["hook_metadata"] = {
            "specifications": deepcopy(HOOK_SPECS),
            "actual_tensors": _shape_tree(capture.values),
            "gradient_mode": bool(enable_grad),
            "text_word_embedding_module": "text_model.model.embeddings.word_embeddings",
            "text_complete_embedding_module": "text_model.model.embeddings",
            "local_coordinate": "body zero-based j; original source slot j+1",
            "note": "Representations and attention are not validated importance or causal contribution.",
        }
        return result


def load_frozen_m0(device="cpu", checkpoint_path=None):
    """Load through the unchanged formal loader; never open an external model.

    Original configuration, code hash, source binding and normalizer validation
    remain active. The relative local BERT path resolves inside original_source.
    Full trained BERT, EMT, two heads and A/V statistics are restored strictly.
    """
    checkpoint = (HANDOFF_ROOT / "checkpoint" / "best.pt" if checkpoint_path is None
                  else Path(checkpoint_path).expanduser().resolve())
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_relative_to(HANDOFF_ROOT):
        raise ValueError("Checkpoint must be a copied file inside this q3_handoff")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = load_for_prediction(checkpoint, HANDOFF_ROOT / "original_source", device=device,
                                allow_smoke=False)
    return FrozenM0Adapter(model, device, checkpoint)
