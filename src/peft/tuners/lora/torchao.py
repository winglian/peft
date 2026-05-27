# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import warnings
from typing import Any, Optional

import torch

# from torch import nn
from peft.import_utils import is_torchao_available
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from .config import LoraConfig
from .layer import Linear


def _torchao_has_legacy_apply_tensor_subclass() -> bool:
    """Whether the installed torchao still exposes the legacy ``get_apply_tensor_subclass`` API.

    torchao moved from the ``quantize_(model, get_apply_tensor_subclass())`` callable to passing
    config objects (e.g. ``Float8WeightOnlyConfig``) and removed ``get_apply_tensor_subclass`` from
    ``torchao.quantization`` (around torchao 0.17). Capability detection is used rather than a version
    literal so this stays correct across torchao's renames. Only relevant for ``merge``/``unmerge``.
    """
    try:
        import torchao.quantization as _q
    except Exception:
        return False
    return hasattr(_q, "get_apply_tensor_subclass")


class TorchaoLoraLinear(Linear):
    """LoRA layer implementation for Linear layers using torchao data"""

    def __init__(self, *args, get_apply_tensor_subclass=None, **kwargs):
        # ``get_apply_tensor_subclass`` is only consumed by ``merge``/``unmerge`` to re-quantize the
        # base weight; it is never needed for training or inference with the adapter. It is sourced
        # from the HF quantizer's config (see ``LoraModel._create_and_replace``), so it is unavailable
        # when the base was quantized directly via ``torchao.quantize_`` (there is no HF quantizer) or
        # on torchao versions that dropped the legacy ``get_apply_tensor_subclass`` helper. Make it
        # optional so adapter injection does not fail at construction time; ``merge``/``unmerge`` raise
        # a clear error instead if it is genuinely needed but missing.
        if kwargs["config"].lora_bias:
            raise ValueError(f"{self.__class__.__name__} does not support lora_bias yet, set it to False")

        super().__init__(*args, **kwargs)
        self.get_apply_tensor_subclass = get_apply_tensor_subclass
        self._check_dtype_supported()

    def _requantize_merged_base(self, base_layer: torch.nn.Module) -> None:
        """Re-quantize ``base_layer.weight`` in place after a (un)merge.

        Requires ``get_apply_tensor_subclass``; raises a clear error when it is unavailable
        (e.g. a base quantized directly via ``torchao.quantize_``, or a torchao version that
        removed the legacy API) rather than failing cryptically.
        """
        from torchao import quantize_

        if self.get_apply_tensor_subclass is None:
            raise ValueError(
                f"{type(self).__name__} cannot merge/unmerge this adapter: the torchao re-quantization "
                "config (`get_apply_tensor_subclass`) is unavailable. This happens when the base model "
                "was quantized directly via `torchao.quantize_()` (no HF quantizer to source it from), or "
                "with a torchao version that removed the legacy `get_apply_tensor_subclass` API "
                f"(present in this install: {_torchao_has_legacy_apply_tensor_subclass()}). Training and "
                "inference with the adapter are unaffected; only merging it into the base needs this."
            )
        quantize_(base_layer, self.get_apply_tensor_subclass())

    def _check_dtype_supported(self):
        # TODO: Not required once int4_weight_only is properly supported by torchao
        from torchao.quantization import Int4Tensor

        base_layer = self.get_base_layer()
        weight = base_layer.weight
        if isinstance(weight, Int4Tensor):
            raise TypeError(f"{type(self).__name__} only supports int8 weights for now.")

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        self._check_dtype_supported()

        base_layer = self.get_base_layer()
        weight = base_layer.weight

        for active_adapter in adapter_names:
            try:
                weight = weight.dequantize()
            except NotImplementedError as exc:
                msg = (
                    f"Weights of type {type(weight).__name__} do not support dequantization (yet), which is needed to "
                    "support merging."
                )
                raise NotImplementedError(msg) from exc

            if safe_merge and not torch.isfinite(weight).all():
                raise ValueError(
                    f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                )

            weight += self.get_delta_weight(active_adapter)
            # TODO: once (if) torchao supports directly mutating the data, use that instead.
            del base_layer.weight
            base_layer.weight = weight
            self._requantize_merged_base(base_layer)
            del weight

            self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter not in self.lora_A.keys():
                continue

            base_layer = self.get_base_layer()
            weight = base_layer.weight
            try:
                weight = weight.dequantize()
            except NotImplementedError as exc:
                msg = (
                    f"Weights of type {type(weight).__name__} do not support dequantization (yet), which is needed to "
                    "support unmerging."
                )
                raise NotImplementedError(msg) from exc

            weight -= self.get_delta_weight(active_adapter)
            # We go through a dummy module because overriding the weight.data does not work, the tensor retains the old
            # data. Therefore, we need to go through quantize_, which takes a module as input, and we need to delete and
            # re-assign the weight.
            # TODO: once (if) torchao supports directly mutating the data, use that instead.
            del base_layer.weight
            base_layer.weight = weight
            self._requantize_merged_base(base_layer)
            del weight

    def __repr__(self) -> str:
        rep = super().__repr__()
        return rep.replace("lora.Linear", f"lora.{self.__class__.__name__}")


def dispatch_torchao(
    target: torch.nn.Module,
    adapter_name: str,
    config: LoraConfig,
    **kwargs: Any,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if not hasattr(target_base_layer, "weight"):
        return new_module

    if not is_torchao_available():
        return new_module

    from torchao.utils import TorchAOBaseTensor

    if isinstance(target_base_layer.weight, TorchAOBaseTensor):
        new_module = TorchaoLoraLinear(target, adapter_name, config=config, **kwargs)

    return new_module
