# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import json
from pathlib import Path

import pytest
from transformers import PretrainedConfig

import moe_store.registry.constants as constants
from moe_store.parsing.hf_config import parse_expert_id, parse_moe_param

FIXTURE = Path(__file__).parent / "fixtures" / "hy_v3" / "config.json"


@pytest.fixture()
def hy3_config() -> PretrainedConfig:
    return PretrainedConfig.from_dict(json.loads(FIXTURE.read_text()))


def test_hyv3_registration_is_guarded() -> None:
    model_class = constants.HYV3ForCausalLM
    assert ("hyv3" in constants.MODEL_MAPPING_NAMES) is (
        model_class is not None
    )
    if model_class is not None:
        assert constants.MODEL_MAPPING_NAMES["hyv3"] is model_class
        assert constants.MODEL_MAPPING_TYPES["hyv3"] == 5


def test_parse_expert_type_resolves_hyv3(hy3_config) -> None:
    if constants.HYV3ForCausalLM is None:
        pytest.skip("transformers build lacks HYV3ForCausalLM")
    assert constants.parse_expert_type(hy3_config) == 5


def test_parse_moe_param_hyv3(hy3_config) -> None:
    assert parse_moe_param(hy3_config) == (4, 8, 0)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("model.layers.2.mlp.experts.171.down_proj.weight", (2, 171)),
        # scale members of a real expert parse identically to the weight.
        ("model.layers.2.mlp.experts.0.gate_proj.weight_scale", (2, 0)),
        ("model.layers.2.mlp.experts.0.up_proj.input_scale", (2, 0)),
        ("model.layers.1.mlp.experts.7.up_proj.weight", (1, 7)),
        # MTP layer (index == num_hidden_layers) must NOT be offloaded.
        ("model.layers.4.mlp.experts.0.gate_proj.weight", (None, None)),
        # shared expert, router, bias, and dense layer-0 MLP are not experts.
        ("model.layers.2.mlp.shared_mlp.gate_proj.weight", (None, None)),
        ("model.layers.2.mlp.router.gate.weight", (None, None)),
        ("model.layers.2.mlp.expert_bias", (None, None)),
        ("model.layers.0.mlp.gate_proj.weight", (None, None)),
    ],
)
def test_parse_expert_id_hyv3(hy3_config, name, expected) -> None:
    assert parse_expert_id(name, hy3_config) == expected
