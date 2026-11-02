"""Fused routing (MoE-Infinity #315) vs each wrapper's torch routing path.

Tiny random-weight blocks; expert selection must be identical and the dense
weights equal up to rounding. Integer-valued ("tie-heavy") gates make the
logits exact in any summation order, so ties are compared too.
"""

from types import SimpleNamespace

import pytest
import torch

from moe_store.wrappers import fused_routing

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or fused_routing._load() is None,
    reason="needs CUDA and moe_infinity.kernel.router",
)


@pytest.fixture
def torch_path(monkeypatch):
    """Context that disables the fused path inside the wrappers."""

    class _Ctx:
        def __enter__(self):
            monkeypatch.setattr(fused_routing, "_FUSED", None)
            monkeypatch.setattr(fused_routing, "_FUSED_TRIED", True)

        def __exit__(self, *exc):
            monkeypatch.undo()

    return _Ctx()


def _inputs(T, H, tie_heavy, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(T + H)
    if tie_heavy:
        v = torch.randint(-1, 2, (T, H), device="cuda", generator=g)
        keep = torch.rand(T, H, device="cuda", generator=g) < 0.05
        return (v * keep).to(dtype)
    return torch.randn(T, H, device="cuda", generator=g).to(dtype)


def _gate(E, H, tie_heavy, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(E * 7 + H)
    if tie_heavy:
        v = torch.randint(-1, 2, (E, H), device="cuda", generator=g)
        keep = torch.rand(E, H, device="cuda", generator=g) < 0.05
        w = (v * keep).to(dtype)
        w[1::5] = w[0::5][: w[1::5].shape[0]]  # exactly tied experts
        return w
    return (torch.randn(E, H, device="cuda", generator=g) * 0.02).to(dtype)


def _same(mask_a, w_a, mask_b, w_b, strict=True):
    """Tie-heavy inputs: every token identical. Random inputs: a token may
    only differ at a summation-order near-tie (cuBLAS vs tensor-core MMA),
    which must stay vanishingly rare."""
    rows = (mask_a == mask_b).all(-1)
    if strict:
        assert rows.all(), f"{(~rows).sum().item()} tokens differ"
    else:
        assert (~rows).float().mean() <= 2e-3
    torch.testing.assert_close(
        w_a[rows].float(), w_b[rows].float(), rtol=4e-2, atol=1e-4
    )


@pytest.mark.parametrize("tie_heavy", [False, True])
@pytest.mark.parametrize("T", [1, 37, 1500])
def test_qwen3_5_route(T, tie_heavy, torch_path):
    from transformers import Qwen3_5MoeTextConfig

    from moe_store.wrappers.qwen3_5_moe import SyncQwen3_5MoeSparseMoeBlock

    cfg = Qwen3_5MoeTextConfig(
        hidden_size=256, num_experts=64, num_experts_per_tok=8,
        moe_intermediate_size=32, shared_expert_intermediate_size=32,
    )  # fmt: skip
    block = SyncQwen3_5MoeSparseMoeBlock(cfg).to("cuda", torch.bfloat16)
    block.gate.weight.data = _gate(64, 256, tie_heavy)
    x = _inputs(T, 256, tie_heavy)
    mask, w, logits = block._route(x)
    with torch_path:
        ref_mask, ref_w, ref_logits = block._route(x)
    _same(mask, w, ref_mask, ref_w, strict=tie_heavy)
    assert w.dtype == ref_w.dtype and logits.dtype == ref_logits.dtype


class _CaptureExecutor:
    def dispatch_local(self, layer, hidden, mask, weights, router_logits=None):
        self.args = (mask, weights, router_logits)
        self.hidden = hidden

    def wait_dispatch_local(self):
        return torch.zeros_like(self.hidden)


@pytest.mark.parametrize("norm", [True, False])
@pytest.mark.parametrize("tie_heavy", [False, True])
def test_qwen3_moe_block(norm, tie_heavy, torch_path):
    from transformers import Qwen3MoeConfig

    from moe_store.wrappers.qwen import Qwen3MoEBlock

    cfg = Qwen3MoeConfig(
        hidden_size=256, num_experts=128, num_experts_per_tok=8,
        moe_intermediate_size=32, norm_topk_prob=norm,
    )  # fmt: skip
    block = Qwen3MoEBlock(cfg).to("cuda", torch.bfloat16)
    block.gate.weight.data = _gate(128, 256, tie_heavy)
    block.expert_executor = _CaptureExecutor()
    x = _inputs(40, 256, tie_heavy).view(1, 40, 256)
    block(x)
    mask, w, _ = block.expert_executor.args
    with torch_path:
        block(x)
    ref_mask, ref_w, _ = block.expert_executor.args
    _same(mask, w, ref_mask, ref_w, strict=tie_heavy)


@pytest.mark.parametrize(
    "n_group,topk_group", [(1, 1), (8, 4)], ids=["glm5", "grouped"]
)
@pytest.mark.parametrize("tie_heavy", [False, True])
@pytest.mark.parametrize("T", [1, 300])
def test_glm_moe_dsa_route(T, tie_heavy, n_group, topk_group):
    from moe_store.wrappers import glm_moe_dsa

    if not glm_moe_dsa._GLM_AVAILABLE:
        pytest.skip("transformers without GLM-MoE-DSA")
    from transformers import GlmMoeDsaConfig

    cfg = GlmMoeDsaConfig(
        hidden_size=256, n_routed_experts=64, num_experts_per_tok=8,
        n_group=n_group, topk_group=topk_group, norm_topk_prob=True,
        routed_scaling_factor=2.5, moe_intermediate_size=32,
        n_shared_experts=1,
    )  # fmt: skip
    block = glm_moe_dsa.SyncGlmMoeDsaMoEBlock(cfg).to("cuda", torch.bfloat16)
    block.gate.weight.data = _gate(64, 256, tie_heavy)
    g = torch.Generator(device="cuda").manual_seed(3)
    bias = torch.randint(-2, 3, (64,), device="cuda", generator=g) * 0.25
    block.gate.e_score_correction_bias = bias.float()
    x = _inputs(T, 256, tie_heavy)
    mask, w, _ = block._route_fused(x)
    idx, wts = block._route(x)
    ref_mask = torch.zeros_like(mask).scatter_(1, idx, True)
    ref_w = torch.zeros_like(w).scatter_(1, idx, wts)
    _same(mask, w, ref_mask, ref_w, strict=tie_heavy)


@pytest.mark.parametrize("tie_heavy", [False, True])
@pytest.mark.parametrize("T", [1, 200])
def test_deepseek_v4_routing_masks(T, tie_heavy, torch_path):
    from moe_store.wrappers.deepseek_v4.sync_moe_block import (
        SyncDeepSeekV4MoEBlock,
    )

    cfg = SimpleNamespace(
        num_experts_per_tok=6, n_routed_experts=64, hidden_size=256,
        routed_scaling_factor=1.5, norm_topk_prob=True, num_hash_layers=0,
        vocab_size=16, moe_intermediate_size=32,
    )  # fmt: skip
    try:
        block = SyncDeepSeekV4MoEBlock(cfg, layer_idx=3)
    except AttributeError as exc:
        pytest.skip(f"config needs more fields: {exc}")
    block = block.to("cuda")
    block.gate_weight.data = _gate(64, 256, tie_heavy)
    g = torch.Generator(device="cuda").manual_seed(5)
    block.gate_bias.data = (
        torch.randint(-2, 3, (64,), device="cuda", generator=g) * 0.25
    ).float()
    x = _inputs(T, 256, tie_heavy).view(1, T, 256)
    mask, w = block.compute_routing_masks(x)
    with torch_path:
        ref_mask, ref_w = block.compute_routing_masks(x)
    _same(mask, w, ref_mask, ref_w, strict=tie_heavy)
