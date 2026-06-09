# Copyright 2025-present the HuggingFace Inc. team.
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

"""Tests for input-side OFT over fused MoE expert ``nn.Parameter`` (``target_parameters``)."""

import pytest
import torch

from peft import OFTConfig, PeftModel, get_peft_model
from peft.tuners.oft.experts import (
    GroupedOFTRotation,
    RotatedGroupedWeight,
    eager_segmented_rotate,
    set_rotation_backend,
)

from .testing_utils import require_torch_multi_gpu


transformers = pytest.importorskip("transformers")

from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig  # noqa: E402
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts  # noqa: E402


def _make_experts(num_experts=6, hidden=32, inter=16, dtype=torch.float32):
    cfg = Qwen3MoeConfig(
        hidden_size=hidden,
        moe_intermediate_size=inter,
        num_experts=num_experts,
        num_experts_per_tok=2,
        hidden_act="silu",
    )
    experts = Qwen3MoeExperts(cfg).to(dtype)
    with torch.no_grad():
        experts.gate_up_proj.normal_(0, 0.1)
        experts.down_proj.normal_(0, 0.1)
    experts.requires_grad_(False)
    experts.config._experts_implementation = "grouped_mm"
    return cfg, experts


def _routing(tokens, cfg, dtype=torch.float32):
    top_k_index = torch.stack([torch.randperm(cfg.num_experts)[: cfg.num_experts_per_tok] for _ in range(tokens)])
    top_k_weights = torch.rand(tokens, cfg.num_experts_per_tok, dtype=dtype)
    return top_k_index, top_k_weights


def _model(dtype=torch.float32):
    cfg = Qwen3MoeConfig(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=6,
        num_experts_per_tok=2,
        vocab_size=100,
        max_position_embeddings=64,
        decoder_sparse_step=1,
        mlp_only_layers=[],
    )
    model = Qwen3MoeForCausalLM(cfg).to(dtype).eval()
    for module in model.modules():
        if hasattr(module, "config"):
            module.config._experts_implementation = "grouped_mm"
    return model


Qwen3MoeForCausalLM = transformers.Qwen3MoeForCausalLM
TARGET = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]


class TestGroupedOFTRotation:
    def test_bank_is_orthogonal(self):
        bank = GroupedOFTRotation(num_experts=4, in_features=16, block_size=8)
        bank.oft_r.data.normal_(0, 0.05)
        R = bank.compute_rotation_bank()  # (E, num_blocks, bs, bs)
        eye = torch.eye(R.shape[-1])
        err = (R @ R.transpose(-1, -2) - eye).abs().max()
        assert err < 1e-2  # Cayley-Neumann is an approximation
        assert R.shape == (4, 2, 8, 8)

    def test_identity_at_init(self):
        bank = GroupedOFTRotation(num_experts=4, in_features=16, block_size=8)  # oft_r init zeros
        x = torch.randn(5, 16)
        assert torch.allclose(bank.rotate(x, expert_idx=0), x, atol=1e-6)


class TestRotatedGroupedWeight:
    """The weight tensor-subclass that drives the standard grouped experts path."""

    def _banks(self, num_experts, in_features, block_size=8):
        bank = GroupedOFTRotation(num_experts, in_features, block_size)
        bank.oft_r.data.normal_(0, 0.05)
        return bank

    def _wrap(self, experts):
        E, hidden, inter = experts.num_experts, experts.hidden_dim, experts.intermediate_dim
        gate, up, down = self._banks(E, hidden), self._banks(E, hidden), self._banks(E, inter)
        gu = experts.gate_up_proj.detach()
        dn = experts.down_proj.detach()
        del experts.gate_up_proj, experts.down_proj
        experts.gate_up_proj = RotatedGroupedWeight(gu, (gate, up), split=2)
        experts.down_proj = RotatedGroupedWeight(dn, (down,), split=1)
        return gate, up, down

    def _weight_side_reference(self, gu, dn, gate, up, down, act_fn, x, tki, tkw, num_experts):
        # y = W (R x) computed weight-side: rotate each expert weight, then matmul.
        def rotW(W, R):
            o, nb, bs = W.shape[0], R.shape[0], R.shape[1]
            return torch.einsum("rkc,orc->ork", R, W.reshape(o, nb, bs)).reshape(o, -1)

        Rg, Ru, Rd = gate.compute_rotation_bank(), up.compute_rotation_bank(), down.compute_rotation_bank()
        out = torch.zeros_like(x)
        mask = torch.nn.functional.one_hot(tki, num_classes=num_experts).permute(2, 1, 0)
        half = gu.shape[1] // 2
        for e in range(num_experts):
            pos, tok = torch.where(mask[e])
            if tok.numel() == 0:
                continue
            xs = x[tok]
            g = xs @ rotW(gu[e][:half], Rg[e]).t()
            u = xs @ rotW(gu[e][half:], Ru[e]).t()
            d = (act_fn(g) * u) @ rotW(dn[e], Rd[e]).t()
            out.index_add_(0, tok, (d * tkw[tok, pos, None]).to(out.dtype))
        return out

    def test_input_side_matches_weight_side(self):
        cfg, experts = _make_experts()
        gu, dn = experts.gate_up_proj.detach().clone(), experts.down_proj.detach().clone()
        gate, up, down = self._wrap(experts)
        x = torch.randn(20, cfg.hidden_size)
        tki, tkw = _routing(20, cfg)

        y_input = experts(x, tki, tkw)  # standard grouped forward, unchanged
        y_weight = self._weight_side_reference(gu, dn, gate, up, down, experts.act_fn, x, tki, tkw, cfg.num_experts)
        assert torch.allclose(y_input, y_weight, atol=1e-4)

    def test_base_weight_not_materialized(self):
        _cfg, experts = _make_experts()
        packed = experts.gate_up_proj.detach()
        E, hidden = experts.num_experts, experts.hidden_dim
        del experts.gate_up_proj
        gate, up = self._banks(E, hidden), self._banks(E, hidden)
        experts.gate_up_proj = RotatedGroupedWeight(packed, (gate, up), split=2)
        # the subclass shares storage with the packed base; it is never rotated in place
        assert experts.gate_up_proj._packed.data_ptr() == packed.data_ptr()

    def test_rotation_backend_is_swappable(self):
        cfg, experts = _make_experts()
        self._wrap(experts)
        x = torch.randn(16, cfg.hidden_size)
        tki, tkw = _routing(16, cfg)
        y_default = experts(x, tki, tkw)

        calls = {"n": 0}

        def counting_backend(x, bank, offs):
            calls["n"] += 1
            return eager_segmented_rotate(x, bank, offs)

        set_rotation_backend(counting_backend)
        try:
            y_swapped = experts(x, tki, tkw)
        finally:
            set_rotation_backend(eager_segmented_rotate)
        assert calls["n"] > 0
        assert torch.allclose(y_default, y_swapped, atol=1e-5)

    def test_gate_up_use_independent_rotations(self):
        cfg, experts = _make_experts()
        gate, up, _ = self._wrap(experts)
        x = torch.randn(16, cfg.hidden_size)
        tki, tkw = _routing(16, cfg)
        y = experts(x, tki, tkw)
        # swapping the gate/up banks must change the output (a shared rotation would not)
        gate.oft_r.data, up.oft_r.data = up.oft_r.data.clone(), gate.oft_r.data.clone()
        y_swapped = experts(x, tki, tkw)
        assert not torch.allclose(y, y_swapped, atol=1e-3)


class TestOFTTargetParametersIntegration:
    def _config(self):
        return OFTConfig(oft_block_size=8, target_parameters=list(TARGET))

    def test_only_oft_params_are_trainable(self):
        model = get_peft_model(_model(), self._config())
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert trainable, "no trainable parameters"
        assert all("oft_R" in n for n in trainable)
        # 2 MoE layers x (gate, up, down) banks
        assert len(trainable) == 6

    def test_init_is_noop(self):
        model = _model()
        ids = torch.randint(0, 100, (2, 12))
        with torch.no_grad():
            base = model(input_ids=ids).logits.clone()
        model = get_peft_model(model, self._config())  # oft_r init at 0 -> identity
        with torch.no_grad():
            adapted = model(input_ids=ids).logits
        assert torch.allclose(base, adapted, atol=1e-5)

    def test_gradients_flow_to_oft_only(self):
        model = get_peft_model(_model(), self._config())
        for n, p in model.named_parameters():
            if "oft_R" in n:
                p.data.normal_(0, 0.02)
        ids = torch.randint(0, 100, (2, 12))
        model(input_ids=ids, labels=ids).loss.backward()
        oft_grads = [p.grad for n, p in model.named_parameters() if "oft_R" in n]
        assert all(g is not None and g.norm() > 0 for g in oft_grads)
        base_grads = [p.grad for n, p in model.named_parameters() if "oft_R" not in n]
        assert all(g is None for g in base_grads)

    def test_save_and_load_roundtrip(self, tmp_path):
        ids = torch.randint(0, 100, (2, 12))
        # the adapter file only stores oft_R, so both runs must share the same base weights
        torch.manual_seed(42)
        model = get_peft_model(_model(), self._config())
        for n, p in model.named_parameters():
            if "oft_R" in n:
                p.data.normal_(0, 0.02)
        with torch.no_grad():
            expected = model(input_ids=ids).logits.clone()
        model.save_pretrained(tmp_path)

        torch.manual_seed(42)
        reloaded = PeftModel.from_pretrained(_model(), tmp_path)
        with torch.no_grad():
            actual = reloaded(input_ids=ids).logits
        assert torch.allclose(expected, actual, atol=1e-5)

    def test_config_allows_target_parameters_without_target_modules(self):
        config = OFTConfig(target_parameters=list(TARGET))  # oft_block_size defaults to 32
        assert config.target_parameters == list(TARGET)
        assert config.target_modules is None


def _fsdp2_worker(rank, world_size, port, queue):
    import os

    import torch.distributed as dist
    from torch.distributed.fsdp import fully_shard

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    torch.manual_seed(0)  # identical full model on every rank before sharding
    model = get_peft_model(_model(), OFTConfig(oft_block_size=8, target_parameters=list(TARGET))).to(device)
    for n, p in model.named_parameters():
        if "oft_R" in n:
            p.data.normal_(0, 0.02)
    ids = torch.randint(0, 100, (2, 16), device=device)
    with torch.no_grad():
        ref_loss = model(input_ids=ids, labels=ids).loss.item()

    for layer in [m for m in model.modules() if type(m).__name__ == "Qwen3MoeDecoderLayer"]:
        fully_shard(layer)
    fully_shard(model)

    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    grads_ok = all(p.grad is not None for n, p in model.named_parameters() if "oft_R" in n)
    base_sharded = any(
        type(p.data).__name__ == "DTensor" for n, p in model.named_parameters() if "gate_up_proj.original" in n
    )
    if rank == 0:
        queue.put((ref_loss, out.loss.item(), grads_ok, base_sharded))
    dist.destroy_process_group()


class TestOFTTargetParametersFSDP2:
    @require_torch_multi_gpu
    def test_fully_shard_matches_unsharded(self):
        # FSDP2 per-parameter sharding: the huge fused expert weights shard across ranks while
        # the OFT adapter trains. The sharded result must match the unsharded one.
        import torch.multiprocessing as mp

        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        mp.spawn(_fsdp2_worker, args=(2, 29513, queue), nprocs=2, join=True)
        ref_loss, sharded_loss, grads_ok, base_sharded = queue.get(timeout=180)
        assert base_sharded, "base expert weight was not sharded as a DTensor"
        assert grads_ok, "oft_R did not receive gradients under FSDP2"
        assert abs(ref_loss - sharded_loss) < 1e-4
