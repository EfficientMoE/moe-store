"""CPU regressions for the Qwen3 and DeepSeek-V2 token-packing guards.

Stale metadata with a different sequence count must leave Q/K/V unpacked.
Use real attention projections and RoPE with a causal CPU reference backend;
no checkpoint downloads or GPU paged-attention kernels are needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers.models.deepseek_v2.configuration_deepseek_v2 import (
    DeepseekV2Config,
)
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2RotaryEmbedding,
)
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeRotaryEmbedding,
)

from moe_store.wrappers.deepseek_v2_paged_attention import (
    DeepseekV2PagedAttention,
)
from moe_store.wrappers.qwen3_paged_attention import Qwen3PagedAttention


class _UnpackedCausalBackend:
    """Use the actual input geometry, independently of stale metadata."""

    def __init__(self, batch_size: int, q_len: int):
        self.batch_size = batch_size
        self.q_len = q_len
        self.calls: list[dict] = []

    def forward(
        self,
        query,
        key,
        value,
        attention_metadata=None,
        scale=None,
        layer_idx=0,
    ):
        self.calls.append(
            {
                "q": query.clone(),
                "k": key.clone(),
                "v": value.clone(),
                "metadata": attention_metadata,
                "layer_idx": layer_idx,
            }
        )
        for tokens in (query, key, value):
            assert tokens.shape[0] == self.batch_size * self.q_len

        groups = query.shape[1] // key.shape[1]
        outputs = []
        for row in range(self.batch_size):
            start = row * self.q_len
            stop = start + self.q_len
            q = query[start:stop].transpose(0, 1)
            k = key[start:stop].repeat_interleave(groups, dim=1).transpose(0, 1)
            v = (
                value[start:stop]
                .repeat_interleave(groups, dim=1)
                .transpose(0, 1)
            )
            scores = (q @ k.transpose(-1, -2)) * scale
            mask = torch.triu(
                torch.ones(self.q_len, self.q_len, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(mask, float("-inf"))
            outputs.append((torch.softmax(scores, dim=-1) @ v).transpose(0, 1))
        return torch.cat(outputs, dim=0)


def _metadata(query_lengths: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        lengths=SimpleNamespace(
            query_lengths=torch.tensor(query_lengths, dtype=torch.int32)
        )
    )


@pytest.fixture(params=["qwen3", "deepseek-v2"])
def attention(request):
    torch.manual_seed(5)
    common = dict(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        attention_bias=False,
        attention_dropout=0.0,
    )
    if request.param == "qwen3":
        config = Qwen3MoeConfig(num_key_value_heads=2, **common)
        cls = Qwen3PagedAttention
        rotary = Qwen3MoeRotaryEmbedding(config)
    else:
        config = DeepseekV2Config(
            q_lora_rank=None,
            kv_lora_rank=8,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            **common,
        )
        cls = DeepseekV2PagedAttention
        rotary = DeepseekV2RotaryEmbedding(config)
    cls.clear_paged_context()
    try:
        yield cls(config, layer_idx=3).eval(), config, rotary
    finally:
        cls.clear_paged_context()


@pytest.mark.parametrize(
    "stale_lengths",
    [[2], [2, 3, 2]],
    ids=["too-few-sequences", "too-many-sequences"],
)
def test_stale_metadata_with_wrong_sequence_count_falls_back_unpacked(
    attention, stale_lengths
) -> None:
    shim, config, rotary = attention
    bsz, q_len = 2, 3
    hidden = torch.randn(bsz, q_len, config.hidden_size)
    positions = torch.arange(q_len).unsqueeze(0).expand(bsz, -1)
    position_embeddings = rotary(hidden, positions)

    control = _UnpackedCausalBackend(bsz, q_len)
    backend = _UnpackedCausalBackend(bsz, q_len)
    stale_metadata = _metadata(stale_lengths)
    with torch.no_grad():
        shim.set_paged_context(control, _metadata([q_len] * bsz))
        expected, _ = shim(
            hidden_states=hidden,
            position_embeddings=position_embeddings,
            attention_mask=None,
        )
        shim.set_paged_context(backend, stale_metadata)
        actual, weights = shim(
            hidden_states=hidden,
            position_embeddings=position_embeddings,
            attention_mask=None,
        )

    assert weights is None
    assert actual.shape == (bsz, q_len, config.hidden_size)
    assert len(control.calls) == len(backend.calls) == 1
    assert backend.calls[0]["layer_idx"] == 3
    assert backend.calls[0]["metadata"] is stale_metadata
    for name in ("q", "k", "v"):
        assert backend.calls[0][name].shape[0] == bsz * q_len
        torch.testing.assert_close(
            backend.calls[0][name], control.calls[0][name], rtol=0, atol=0
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
