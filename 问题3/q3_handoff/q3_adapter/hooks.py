"""Temporary, external observation hooks for the frozen formal M0 predictor.

No original module is patched. Ordinary observation never replaces a tensor.
``continuous_grad=True`` only replaces three outputs with numerically identical
leaf tensors for local derivatives with respect to continuous representations.
These are representation derivatives, not derivatives of integer token IDs and
not causal contributions. Every handle is removed on context exit, also on error.
"""
from __future__ import annotations

from contextlib import AbstractContextManager

from .runtime import configure_runtime

configure_runtime()

import torch


HOOK_SPECS = {
    "shared_feature": {"module": "cls_head", "hook": "forward_pre_hook", "value": "inputs[0]", "shape": "[B,128]"},
    "logits": {"module": "cls_head", "hook": "forward_hook", "value": "output", "shape": "[B,3]"},
    "intensity": {"module": "post_fusion_layer_3", "hook": "forward_hook", "value": "output.reshape(-1)", "shape": "[B]"},
    "text_encoded": {"module": "text_model", "hook": "forward_hook", "value": "output", "shape": "[B,50,768]"},
    "text_summary": {"module": "text_model", "hook": "forward_hook", "value": "output[:,0]", "shape": "[B,768]"},
    "text_local_encoder": {"module": "text_model", "hook": "forward_hook", "value": "output[:,1:]", "shape": "[B,49,768]"},
    "audio_summary": {"module": "audio_model", "hook": "forward_hook", "value": "output[1]", "shape": "[B,16]"},
    "audio_local_encoder": {"module": "audio_model", "hook": "forward_hook", "value": "output[0]", "shape": "[B,max(lengths),16]"},
    "vision_summary": {"module": "video_model", "hook": "forward_hook", "value": "output[1]", "shape": "[B,32]"},
    "vision_local_encoder": {"module": "video_model", "hook": "forward_hook", "value": "output[0]", "shape": "[B,max(lengths),32]"},
    "global_context_input": {"module": "fusion", "hook": "forward_pre_hook", "value": "inputs[0]", "shape": "[B,3,128]"},
    "text_local": {"module": "fusion", "hook": "forward_pre_hook", "value": "inputs[1][0]", "shape": "[B,49,128]"},
    "audio_local": {"module": "fusion", "hook": "forward_pre_hook", "value": "inputs[1][1]", "shape": "[B,max(lengths),128]"},
    "vision_local": {"module": "fusion", "hook": "forward_pre_hook", "value": "inputs[1][2]", "shape": "[B,max(lengths),128]"},
    "local_masks": {"module": "fusion", "hook": "forward_pre_hook", "value": "inputs[2]", "shape": "three bool tensors [B,T_modality]"},
    "global_context": {"module": "fusion", "hook": "forward_hook", "value": "output[0]", "shape": "[B,3,128]"},
    "text_local_post_fusion": {"module": "fusion", "hook": "forward_hook", "value": "output[1][0]", "shape": "[B,49,128]"},
    "audio_local_post_fusion": {"module": "fusion", "hook": "forward_hook", "value": "output[1][1]", "shape": "[B,max(lengths),128]"},
    "vision_local_post_fusion": {"module": "fusion", "hook": "forward_hook", "value": "output[1][2]", "shape": "[B,max(lengths),128]"},
    "text_embedding_output": {"module": "text_model.model.embeddings", "hook": "optional forward_hook", "value": "identical detached output leaf", "shape": "[B,50,768]"},
    "audio_standardized": {"module": "av_normalizer", "hook": "optional forward_hook", "value": "identical detached output leaf when inputs[2]=='audio'", "shape": "[B,50,74]"},
    "vision_standardized": {"module": "av_normalizer", "hook": "optional forward_hook", "value": "identical detached output leaf when inputs[2]=='vision'", "shape": "[B,50,35]"},
}


def count_registered_hooks(model):
    """Count pre/forward handles without changing the model."""
    return sum(len(module._forward_hooks) + len(module._forward_pre_hooks)
               for module in model.modules())


class RepresentationHooks(AbstractContextManager):
    def __init__(self, model, *, continuous_grad=False):
        self.model = model
        self.continuous_grad = bool(continuous_grad)
        self.values = {}
        self.gradient_inputs = {}
        self.handles = []

    def _save(self, name, value):
        self.values[name] = value if self.continuous_grad else value.detach()

    def _classifier_pre(self, module, inputs):
        self._save("shared_feature", inputs[0])

    def _classifier(self, module, inputs, output):
        self._save("hook_logits", output)

    def _regressor(self, module, inputs, output):
        self._save("hook_intensity", output.reshape(-1))

    def _text(self, module, inputs, output):
        self._save("text_encoded", output)
        self._save("text_summary", output[:, 0])
        self._save("text_local_encoder", output[:, 1:])

    def _audio(self, module, inputs, output):
        self._save("audio_local_encoder", output[0])
        self._save("audio_summary", output[1])

    def _vision(self, module, inputs, output):
        self._save("vision_local_encoder", output[0])
        self._save("vision_summary", output[1])

    def _fusion_pre(self, module, inputs):
        self._save("global_context_input", inputs[0])
        for name, value in zip(("text", "audio", "vision"), inputs[1]):
            self._save(name + "_local", value)
        self.values["local_masks"] = {
            name: value.detach() for name, value in zip(("text", "audio", "vision"), inputs[2])
        }

    def _fusion(self, module, inputs, output):
        self._save("global_context", output[0])
        for name, value in zip(("text", "audio", "vision"), output[1]):
            self._save(name + "_local_post_fusion", value)

    def _embedding_leaf(self, module, inputs, output):
        leaf = output.detach().requires_grad_(True)
        self.gradient_inputs["text_embedding_output"] = leaf
        return leaf

    def _normalizer_leaf(self, module, inputs, output):
        leaf = output.detach().requires_grad_(True)
        self.gradient_inputs[str(inputs[2]) + "_standardized"] = leaf
        return leaf

    def __enter__(self):
        if self.handles:
            raise RuntimeError("A RepresentationHooks context cannot be entered twice")
        try:
            bindings = [
                (self.model.cls_head.register_forward_pre_hook, self._classifier_pre),
                (self.model.cls_head.register_forward_hook, self._classifier),
                (self.model.post_fusion_layer_3.register_forward_hook, self._regressor),
                (self.model.text_model.register_forward_hook, self._text),
                (self.model.audio_model.register_forward_hook, self._audio),
                (self.model.video_model.register_forward_hook, self._vision),
                (self.model.fusion.register_forward_pre_hook, self._fusion_pre),
                (self.model.fusion.register_forward_hook, self._fusion),
            ]
            if self.continuous_grad:
                bindings += [
                    (self.model.text_model.model.embeddings.register_forward_hook, self._embedding_leaf),
                    (self.model.av_normalizer.register_forward_hook, self._normalizer_leaf),
                ]
            for register, callback in bindings:
                self.handles.append(register(callback))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        return False
