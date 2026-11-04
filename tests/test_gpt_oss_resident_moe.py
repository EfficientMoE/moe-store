# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""GPT-OSS resident MoE block: grouped engine op vs. per-expert fallback."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from moe_store import hooks
from moe_store.wrappers.gpt_oss import SyncGptOssMLP


def _config(experts=8, top_k=2, hidden=16, inter=8):
    return SimpleNamespace(
        hidden_size=hidden,
        intermediate_size=inter,
        num_local_experts=experts,
        num_experts_per_tok=top_k,
    )


def _bf16_block(seed=0, **kw):
    torch.manual_seed(seed)
    block = SyncGptOssMLP(_config(**kw))
    for p in block.parameters():
        if p.numel():
            torch.nn.init.normal_(p, std=0.2)
    return block


def _original_loop(block, hidden_states):
    """The pre-#313 resident loop: every expert, one `.any()` each."""
    x = hidden_states.reshape(-1, hidden_states.shape[-1])
    logits = block.router(x)
    w = F.softmax(logits, dim=-1, dtype=torch.float32)
    w, sel = torch.topk(w, block.top_k, dim=-1)
    w = (w / w.sum(-1, keepdim=True)).to(x.dtype)
    mask = torch.zeros(x.shape[0], block.num_experts, dtype=torch.bool)
    mask.scatter_(1, sel, True)
    wmask = torch.zeros(x.shape[0], block.num_experts, dtype=x.dtype)
    wmask.scatter_(1, sel, w)
    out = torch.zeros_like(x)
    for e in range(block.num_experts):
        tm = mask[:, e]
        if not tm.any():
            continue
        out[tm] += wmask[tm, e].unsqueeze(-1) * block._expert_forward(x[tm], e)
    return out.view_as(hidden_states), logits


@pytest.fixture
def no_grouped_op(monkeypatch):
    monkeypatch.setattr(hooks.ops, "fused_moe_mxfp4", None)


@pytest.mark.parametrize("tokens", [1, 3, 17])
def test_fallback_matches_original_loop(no_grouped_op, tokens):
    block = _bf16_block()
    x = torch.randn(1, tokens, 16)
    with torch.no_grad():
        out, logits = block(x)
        ref, ref_logits = _original_loop(block, x)
    torch.testing.assert_close(logits, ref_logits)
    torch.testing.assert_close(out, ref)


def test_fallback_visits_only_routed_experts(no_grouped_op, monkeypatch):
    block = _bf16_block(experts=32, top_k=2)
    seen = []
    forward = block._expert_forward

    def spy(x, e):
        seen.append(e)
        return forward(x, e)

    monkeypatch.setattr(block, "_expert_forward", spy)
    with torch.no_grad():
        block(torch.randn(1, 1, 16))
    assert len(seen) == 2 and seen == sorted(seen)


def test_grouped_op_requires_mxfp4_on_activation_device(monkeypatch):
    calls = []
    monkeypatch.setattr(
        hooks.ops, "fused_moe_mxfp4", lambda *a, **k: calls.append(a)
    )
    block = _bf16_block()
    x = torch.randn(4, 16)
    # BF16 experts and CPU activations always take the per-expert loop.
    assert not block._can_use_grouped_mxfp4(x)
    with torch.no_grad():
        block(x.view(1, 4, 16))
    assert calls == []


def test_grouped_op_receives_resident_tensors(monkeypatch):
    received = {}

    def fake(x, ids, w, *weights, alpha, limit):
        received.update(x=x, ids=ids, w=w, weights=weights)
        received.update(alpha=alpha, limit=limit)
        return torch.zeros_like(x)

    monkeypatch.setattr(hooks.ops, "fused_moe_mxfp4", fake)
    block = _bf16_block()
    x = torch.randn(3, 16)
    ids = torch.tensor([[0, 1], [2, 3], [4, 5]])
    w = torch.full((3, 2), 0.5)
    out = block._grouped_mxfp4_forward(x, ids, w)
    assert out.shape == x.shape
    assert received["ids"] is ids and received["w"] is w
    assert received["x"].dtype == torch.bfloat16
    ex = block.experts
    assert received["weights"] == (
        ex.gate_up_proj,
        ex.gate_up_proj_scales,
        ex.gate_up_proj_bias,
        ex.down_proj,
        ex.down_proj_scales,
        ex.down_proj_bias,
    )
    assert received["alpha"] == block.alpha
    assert received["limit"] == block.swiglu_limit
