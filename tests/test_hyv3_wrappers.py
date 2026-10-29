# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import json
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
try:
    from transformers.models.hy_v3.configuration_hy_v3 import HYV3Config
    from transformers.models.hy_v3.modeling_hy_v3 import HYV3MoE
except ImportError:
    pytest.skip("transformers build lacks hy_v3", allow_module_level=True)

from moe_store.wrappers.hy_v3 import SyncHYV3MoEBlock

FIXTURE = Path(__file__).parent / "fixtures" / "hy_v3" / "config.json"


@pytest.fixture()
def tiny_config() -> HYV3Config:
    raw = json.loads(FIXTURE.read_text())
    raw.pop("quantization_config", None)
    return HYV3Config(**raw)


def test_block_param_names_match_renamed_checkpoint_layout(tiny_config):
    block = SyncHYV3MoEBlock(tiny_config)
    names = {n for n, _ in block.named_parameters()}
    # After transformers' hy_v3 WeightRenaming entries the on-disk keys
    # "mlp.router.gate.weight" / "mlp.expert_bias" / "mlp.shared_mlp.*"
    # become "mlp.gate.weight" / "mlp.e_score_correction_bias" /
    # "mlp.shared_experts.*"; routed expert keys are untouched.
    assert "gate.weight" in names
    assert "e_score_correction_bias" in names  # Parameter, NOT a buffer
    assert "experts.0.gate_proj.weight" in names
    assert "experts.7.down_proj.weight" in names
    assert "shared_experts.up_proj.weight" in names


def test_hy_v3_weight_renamings_cover_block_layout(tiny_config):
    from transformers.conversion_mapping import (
        get_checkpoint_conversion_mapping,
    )

    entries = [
        e
        for e in (get_checkpoint_conversion_mapping("hy_v3") or [])
        if type(e).__name__ == "WeightRenaming"
    ]

    def rename(key):
        for e in entries:
            renamed, matched = e.rename_source_key(key)
            if matched is not None:
                key = renamed
        return key

    assert (
        rename("model.layers.2.mlp.router.gate.weight")
        == "model.layers.2.mlp.gate.weight"
    )
    assert (
        rename("model.layers.2.mlp.expert_bias")
        == "model.layers.2.mlp.e_score_correction_bias"
    )
    assert (
        rename("model.layers.2.mlp.shared_mlp.gate_proj.weight")
        == "model.layers.2.mlp.shared_experts.gate_proj.weight"
    )
    # routed expert keys must pass through unchanged
    key = "model.layers.2.mlp.experts.0.gate_proj.weight"
    assert rename(key) == key


def test_routing_parity_with_stock_hyv3_moe(tiny_config):
    torch.manual_seed(0)
    stock = HYV3MoE(tiny_config)
    block = SyncHYV3MoEBlock(tiny_config)
    # copy router weight and give both the same non-zero bias
    bias = torch.randn(tiny_config.num_experts) * 0.1
    with torch.no_grad():
        block.gate.weight.copy_(stock.gate.weight)
        block.e_score_correction_bias.copy_(bias)
        stock.e_score_correction_bias.copy_(bias)

    hidden = torch.randn(5, tiny_config.hidden_size)
    ref_logits, ref_w, ref_idx = stock.gate(
        hidden, stock.e_score_correction_bias
    )
    idx, w, logits = block._route(hidden)
    assert torch.equal(idx, ref_idx)
    assert torch.allclose(w, ref_w)
    assert torch.allclose(logits, ref_logits)
