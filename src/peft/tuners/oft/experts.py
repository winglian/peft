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
"""Input-side OFT for fused Mixture-of-Experts weights stored as 3-D nn.Parameter.

Transformers v5 MoE layers keep all expert weights in a single fused nn.Parameter
(e.g. ``gate_up_proj`` of shape ``(num_experts, 2*inter, hidden)``) and route tokens
inside the experts ``forward``. OFT's update is multiplicative ``W' = R W``; using
``(R_e W_e) x = W_e (R_e x)`` we apply the rotation to the *activations* of each
expert instead of the weight. The base weight is read exactly as the unadapted
model reads it, so a packed/quantized expert tensor is never dequantized.

Because each token fans out to its top-k experts *inside* the experts forward, the
per-expert rotation cannot be applied by a single input pre-hook — it requires
wrapping the experts forward and consulting the routing it receives as arguments.
"""

from __future__ import annotations

import torch
from torch import nn

from .layer import OFTRotationModule


class GroupedOFTRotation(OFTRotationModule):
    """A bank of per-expert OFT rotations over one fused expert dimension.

    Stores the skew-symmetric generators as a single 3-D parameter
    ``oft_r`` of shape ``(num_experts, num_blocks, n_elements)`` and builds the
    whole ``(num_experts, num_blocks, block_size, block_size)`` rotation bank with
    one batched Cayley(-Neumann) call. Reuses ``OFTRotationModule``'s exact
    skew-symmetric / Cayley / COFT-projection math.

    The rotation acts on the ``in_features`` dimension of the activations.
    """

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        block_size: int,
        *,
        adapter_name: str = "default",
        coft: bool = False,
        eps: float = 6e-5,
        use_cayley_neumann: bool = True,
        num_cayley_neumann_terms: int = 5,
    ):
        if in_features % block_size != 0:
            raise ValueError(f"in_features ({in_features}) must be divisible by block_size ({block_size})")
        num_blocks = in_features // block_size
        n_elements = block_size * (block_size - 1) // 2
        super().__init__(
            r=num_blocks,
            n_elements=n_elements,
            block_size=block_size,
            in_features=in_features,
            coft=coft,
            eps=eps,
            use_cayley_neumann=use_cayley_neumann,
            num_cayley_neumann_terms=num_cayley_neumann_terms,
        )
        # swap the per-layer (r, n_elements) generator for a per-expert one, keyed by
        # adapter (PEFT state-dict convention). Zeros -> R = I, so init is a no-op.
        del self.weight
        self.num_experts = num_experts
        self.num_blocks = num_blocks
        self.adapter_name = adapter_name
        self.oft_R = nn.ParameterDict({adapter_name: nn.Parameter(torch.zeros(num_experts, num_blocks, n_elements))})

    @property
    def oft_r(self) -> torch.Tensor:
        return self.oft_R[self.adapter_name]

    def compute_rotation_bank(self) -> torch.Tensor:
        """``(num_experts, num_blocks, block_size, block_size)`` in one batched launch."""
        E, nb, ne = self.oft_r.shape
        flat = self.oft_r.reshape(E * nb, ne)
        if self.coft:
            with torch.no_grad():
                flat.copy_(self._project_batch(flat, eps=self.eps))
        R = self._cayley_batch(flat, self.block_size, self.use_cayley_neumann, self.num_cayley_neumann_terms)
        return R.reshape(E, nb, self.block_size, self.block_size)

    def rotate(self, x: torch.Tensor, expert_idx: int, bank: torch.Tensor | None = None) -> torch.Tensor:
        """Rotate the input-feature dim of ``x`` by expert ``expert_idx``'s rotation."""
        R = (self.compute_rotation_bank() if bank is None else bank)[expert_idx]
        xb = x.reshape(*x.shape[:-1], self.num_blocks, self.block_size)
        return torch.einsum("...rk,rkc->...rc", xb, R).reshape(x.shape)


# Modern fused MoE layers funnel all expert compute through a grouped matmul
# ``grouped_mm(input, weight, offs)`` over activations sorted by expert.
# ``RotatedGroupedWeight`` intercepts that op (and the preceding ``transpose``) to
# rotate the activations per expert and feed the packed base weight to the real
# kernel: the experts forward is untouched and a quantized base is never dequantized.


# Pluggable rotation backend (eager loop -> triton kernel -> fused).
def eager_segmented_rotate(x: torch.Tensor, bank: GroupedOFTRotation, offs: torch.Tensor) -> torch.Tensor:
    """Rotate grouped-by-expert activations per expert, keyed on ``offs`` (cumsum
    token counts). Reference backend; swap for a fused kernel via ``set_rotation_backend``."""
    R = bank.compute_rotation_bank()
    chunks, prev = [], 0
    for e in range(offs.shape[0]):
        end = int(offs[e])
        if end > prev:
            chunks.append(bank.rotate(x[prev:end], e, R))
        prev = end
    if prev < x.shape[0]:
        chunks.append(x[prev:])  # EP sentinel tail, unrotated
    return torch.cat(chunks, 0) if chunks else x


class _RotationBackend:
    fn = staticmethod(eager_segmented_rotate)


def set_rotation_backend(fn) -> None:
    """Swap the segmented per-expert rotation kernel used by the default handler."""
    _RotationBackend.fn = staticmethod(fn)


def segmented_rotate(x: torch.Tensor, bank: GroupedOFTRotation, offs: torch.Tensor) -> torch.Tensor:
    return _RotationBackend.fn(x, bank, offs)


def default_grouped_handler(func, args, kwargs, rgw: RotatedGroupedWeight) -> torch.Tensor:
    """Pre-rotate the grouped activations, then call the real grouped-GEMM with the
    PACKED base. Handles the fused gate/up split (two half-GEMMs, independent R)."""
    x = args[0]
    offs = kwargs.get("offs", args[2] if len(args) > 2 else None)
    base = rgw._packed  # [E, in, out] at grouped-mm time
    if rgw._split == 1:
        return func(segmented_rotate(x, rgw._banks[0], offs), base, offs=offs)
    half = base.shape[-1] // 2  # out dim is last -> split gate/up
    xg = segmented_rotate(x, rgw._banks[0], offs)
    xu = segmented_rotate(x, rgw._banks[1], offs)
    return torch.cat([func(xg, base[..., :half], offs=offs), func(xu, base[..., half:], offs=offs)], dim=-1)


_OP_HANDLERS: dict = {}


def register_grouped_mm_op(op, handler=None) -> None:
    """Register a grouped-GEMM entry point to intercept (e.g. scattermoe's
    ``scatter2scatter``). ``handler(func, args, kwargs, rgw)`` may fuse rotation
    into the GEMM; defaults to pre-rotate + call with the packed base."""
    if op is not None:
        _OP_HANDLERS[op] = handler or default_grouped_handler


# the fixed transformers / torch surface (not per-model) — default handler
for _f in (
    getattr(torch, "_grouped_mm", None),
    getattr(torch.nn.functional, "grouped_mm", None),
    getattr(getattr(torch.ops, "transformers", None), "grouped_mm_fallback", None),
):
    register_grouped_mm_op(_f)


class RotatedGroupedWeight(torch.Tensor):
    """Packed base weight + OFT rotation bank(s) + gate/up split metadata.

    Tensor-level identity (shares the base storage); only ``transpose`` and the
    registered grouped-GEMM ops are rewritten. The base is never dequantized."""

    @staticmethod
    def __new__(cls, packed, banks, split):
        inst = packed.as_subclass(cls)
        # NOTE: attribute is `_packed`, not `_base` — `_base` is a reserved
        # read-only property on torch.Tensor (a view's base tensor).
        inst._packed = packed
        inst._banks = banks
        inst._split = split
        return inst

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        rgw = next((a for a in args if isinstance(a, RotatedGroupedWeight)), None)
        if rgw is not None:
            if "transpose" in getattr(func, "__name__", ""):
                return RotatedGroupedWeight(rgw._packed.transpose(*args[1:]), rgw._banks, rgw._split)
            handler = _OP_HANDLERS.get(func)
            if handler is not None:
                return handler(func, args, kwargs, rgw)
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)


def _resolve_block_size(config, in_features: int) -> int:
    """Pick an OFT block size that divides ``in_features`` (from r or oft_block_size)."""
    if getattr(config, "r", 0):
        bs = in_features // config.r
    else:
        bs = config.oft_block_size or 32
    bs = min(bs, in_features)
    while bs > 1 and in_features % bs != 0:
        bs -= 1
    return bs


class OFTGroupedParametrization(nn.Module):
    """``nn.utils.parametrize`` parametrization that presents a fused expert weight
    as a ``RotatedGroupedWeight`` (input-side OFT) during forward. Holds the
    trainable per-expert generators; the base weight stays frozen as ``.original``."""

    def __init__(self, num_experts: int, in_features: int, split: int, config, adapter_name: str = "default"):
        super().__init__()
        block_size = _resolve_block_size(config, in_features)
        kw = {
            "adapter_name": adapter_name,
            "coft": config.coft,
            "eps": config.eps,
            "use_cayley_neumann": config.use_cayley_neumann,
            "num_cayley_neumann_terms": config.num_cayley_neumann_terms,
        }
        self.split = split
        if split == 2:
            self.adapter_gate = GroupedOFTRotation(num_experts, in_features, block_size, **kw)
            self.adapter_up = GroupedOFTRotation(num_experts, in_features, block_size, **kw)
            self._banks = (self.adapter_gate, self.adapter_up)
        else:
            self.adapter_down = GroupedOFTRotation(num_experts, in_features, block_size, **kw)
            self._banks = (self.adapter_down,)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return RotatedGroupedWeight(weight, self._banks, self.split)

    def right_inverse(self, weight: torch.Tensor) -> torch.Tensor:
        return weight
