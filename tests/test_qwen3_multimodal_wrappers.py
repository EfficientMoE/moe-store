# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Sync wrapper equivalence against the HF reference sparse blocks.

The reference blocks hold experts batched as 3D parameters
(``gate_up_proj [E, 2I, H]``, ``down_proj [E, H, I]``); the Sync wrappers
hold per-expert MLPs matching the v2 store layout. With weights copied
across, the local (engine-less) forward must reproduce the reference
output exactly.
"""

import pytest
import torch

NUM_EXPERTS = 4
HIDDEN = 32
INTERMEDIATE = 16


def _copy_batched_experts(wrapper, ref_experts):
    with torch.no_grad():
        for idx, mlp in enumerate(wrapper.experts):
            gate, up = ref_experts.gate_up_proj[idx].chunk(2, dim=0)
            mlp.gate_proj.weight.copy_(gate)
            mlp.up_proj.weight.copy_(up)
            mlp.down_proj.weight.copy_(ref_experts.down_proj[idx])


def _vl_text_config():
    from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
        Qwen3VLMoeTextConfig,
    )

    return Qwen3VLMoeTextConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        moe_intermediate_size=INTERMEDIATE,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
    )


def _omni_text_config():
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeTextConfig,
    )

    return Qwen3OmniMoeTextConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        moe_intermediate_size=INTERMEDIATE,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
    )


@pytest.mark.parametrize("model", ["vl", "omni"])
def test_sync_block_matches_reference(model):
    torch.manual_seed(43)
    if model == "vl":
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
            Qwen3VLMoeTextSparseMoeBlock,
        )

        from moe_store.wrappers import SyncQwen3VLMoeTextSparseMoeBlock

        config = _vl_text_config()
        reference = Qwen3VLMoeTextSparseMoeBlock(config)
        wrapper = SyncQwen3VLMoeTextSparseMoeBlock(config)
    else:
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeThinkerTextSparseMoeBlock,
        )

        from moe_store.wrappers import (
            SyncQwen3OmniMoeThinkerTextSparseMoeBlock,
        )

        config = _omni_text_config()
        reference = Qwen3OmniMoeThinkerTextSparseMoeBlock(config)
        wrapper = SyncQwen3OmniMoeThinkerTextSparseMoeBlock(config)

    with torch.no_grad():
        for param in reference.parameters():
            param.normal_()
    wrapper.gate.load_state_dict(reference.gate.state_dict())
    _copy_batched_experts(wrapper, reference.experts)

    hidden = torch.randn(2, 3, HIDDEN)
    expected = reference(hidden)
    if isinstance(expected, tuple):
        expected = expected[0]
    actual = wrapper(hidden)

    assert actual.shape == hidden.shape
    torch.testing.assert_close(
        actual, expected.view_as(actual), rtol=1e-4, atol=1e-4
    )
