# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""MiniMax-M3 (minimax_m3_vl) registry, config parsing, and Sync MoE block.

Registration is guarded on the transformers ``MiniMaxM3SparseForConditional
Generation`` class; the parsing and wrapper tests skip when it is absent so
the suite stays green on transformers builds without minimax_m3_vl.
"""

import warnings

import pytest
import torch

from moe_store.parsing.hf_config import parse_expert_id, parse_moe_param
from moe_store.registry.constants import (
    MODEL_MAPPING_NAMES,
    MODEL_MAPPING_TYPES,
    parse_expert_type,
)

_HAS_MINIMAX_M3 = "minimaxm3" in MODEL_MAPPING_NAMES


def _full_config():
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
        MiniMaxM3VLConfig,
    )

    cfg = MiniMaxM3VLConfig()
    cfg.architectures = ["MiniMaxM3SparseForConditionalGeneration"]
    return cfg


def _tiny_text_config():
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
        MiniMaxM3VLTextConfig,
    )

    return MiniMaxM3VLTextConfig(
        num_local_experts=8,
        num_experts_per_tok=2,
        hidden_size=64,
        intermediate_size=32,
        shared_intermediate_size=32,
        num_hidden_layers=2,
        vocab_size=256,
    )


@pytest.mark.skipif(
    not _HAS_MINIMAX_M3, reason="transformers lacks MiniMaxM3 classes"
)
def test_registry_maps_minimaxm3_to_expert_type_5():
    assert MODEL_MAPPING_TYPES["minimaxm3"] == 5
    assert (
        MODEL_MAPPING_NAMES["minimaxm3"].__name__
        == "MiniMaxM3SparseForConditionalGeneration"
    )


@pytest.mark.skipif(
    not _HAS_MINIMAX_M3, reason="transformers lacks MiniMaxM3 classes"
)
def test_parse_expert_type_minimaxm3():
    assert parse_expert_type(_full_config()) == 5


def test_fail_fast_when_unregistered():
    if _HAS_MINIMAX_M3:
        pytest.skip("registered in this environment")
    from transformers import PretrainedConfig

    cfg = PretrainedConfig()
    cfg.architectures = ["MiniMaxM3SparseForConditionalGeneration"]
    with pytest.raises(RuntimeError, match="minimaxm3"):
        parse_expert_type(cfg)


@pytest.mark.skipif(
    not _HAS_MINIMAX_M3, reason="transformers lacks MiniMaxM3 classes"
)
def test_parse_moe_param_reads_nested_text_config():
    num_layers, num_experts, num_encoder_layers = parse_moe_param(
        _full_config()
    )
    assert num_layers == 60
    assert num_experts == 128
    assert num_encoder_layers == 0


@pytest.mark.skipif(
    not _HAS_MINIMAX_M3, reason="transformers lacks MiniMaxM3 classes"
)
@pytest.mark.parametrize(
    "name,expected_layer,expected_expert",
    [
        ("model.language_model.layers.7.mlp.experts.0.gate_proj.weight", 7, 0),
        (
            "model.language_model.layers.59.mlp.experts.127.down_proj.weight",
            59,
            127,
        ),
    ],
)
def test_parse_expert_id_routed(name, expected_layer, expected_expert):
    layer_id, expert_id = parse_expert_id(name, _full_config())
    assert layer_id == expected_layer
    assert expert_id == expected_expert


@pytest.mark.skipif(
    not _HAS_MINIMAX_M3, reason="transformers lacks MiniMaxM3 classes"
)
@pytest.mark.parametrize(
    "name",
    [
        "model.language_model.layers.0.mlp.shared_experts.gate_up_proj.weight",
        "model.language_model.layers.0.mlp.gate.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.embed_tokens.weight",
        "model.vision_tower.blocks.0.attn.proj.weight",
        "model.multi_modal_projector.linear_1.weight",
        "lm_head.weight",
    ],
)
def test_parse_expert_id_non_expert(name):
    _, expert_id = parse_expert_id(name, _full_config())
    assert expert_id is None


@pytest.mark.skipif(
    not _HAS_MINIMAX_M3, reason="transformers lacks MiniMaxM3 classes"
)
def test_sync_block_param_layout():
    from moe_store.wrappers import SyncMiniMaxM3VLSparseMoeBlock

    cfg = _tiny_text_config()
    block = SyncMiniMaxM3VLSparseMoeBlock(cfg)
    param_names = {name for name, _ in block.named_parameters()}
    assert "gate.weight" in param_names
    assert "experts.0.gate_proj.weight" in param_names
    assert "experts.0.up_proj.weight" in param_names
    assert "experts.7.down_proj.weight" in param_names
    assert "shared_experts.gate_up_proj.weight" in param_names
    assert block.expert_executor is None


@pytest.mark.skipif(
    not _HAS_MINIMAX_M3, reason="transformers lacks MiniMaxM3 classes"
)
def test_sync_block_matches_hf_block():
    warnings.filterwarnings("ignore")
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3VLSparseMoeBlock,
    )

    from moe_store.wrappers import SyncMiniMaxM3VLSparseMoeBlock

    cfg = _tiny_text_config()
    torch.manual_seed(0)

    hf_block = MiniMaxM3VLSparseMoeBlock(cfg).eval()
    for p in hf_block.parameters():
        torch.nn.init.normal_(p, std=0.05)

    sync_block = SyncMiniMaxM3VLSparseMoeBlock(cfg).eval()
    sd = dict(hf_block.state_dict())
    gate_up = sd.pop("experts.gate_up_proj")
    down = sd.pop("experts.down_proj")
    for e in range(cfg.num_local_experts):
        gate_w, up_w = gate_up[e].chunk(2, dim=0)
        sd[f"experts.{e}.gate_proj.weight"] = gate_w.contiguous()
        sd[f"experts.{e}.up_proj.weight"] = up_w.contiguous()
        sd[f"experts.{e}.down_proj.weight"] = down[e].contiguous()
    missing, unexpected = sync_block.load_state_dict(sd, strict=False)
    assert missing == [], f"missing: {missing}"
    assert unexpected == [], f"unexpected: {unexpected}"

    x = torch.randn(2, 6, cfg.hidden_size)
    with torch.no_grad():
        out_hf = hf_block(x)
        out_sync = sync_block(x)

    assert isinstance(out_sync, torch.Tensor)
    assert out_sync.shape == (2, 6, cfg.hidden_size)
    assert torch.allclose(out_hf, out_sync, rtol=1e-3, atol=1e-4)
