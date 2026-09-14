# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import json
from pathlib import Path

import pytest
from transformers import PretrainedConfig

import moe_store.registry.constants as constants
from moe_store.parsing.hf_config import parse_expert_id, parse_moe_param

FIXTURE = (
    Path(__file__).parent / "fixtures" / "deepseek_v41_flash" / "config.json"
)


@pytest.fixture()
def v41_flash_config() -> PretrainedConfig:
    return PretrainedConfig.from_dict(json.loads(FIXTURE.read_text()))


def test_deepseekv41_registration_is_guarded() -> None:
    model_class = constants.DeepseekV41ForCausalLM
    assert ("deepseekv41" in constants.MODEL_MAPPING_NAMES) is (
        model_class is not None
    )
    if model_class is not None:
        assert constants.MODEL_MAPPING_NAMES["deepseekv41"] is model_class
        assert constants.MODEL_MAPPING_TYPES["deepseekv41"] == 5


def test_longest_registry_key_selects_v41_before_v4(monkeypatch) -> None:
    monkeypatch.setitem(constants.MODEL_MAPPING_NAMES, "deepseekv4", object)
    monkeypatch.setitem(constants.MODEL_MAPPING_NAMES, "deepseekv41", object)
    monkeypatch.setitem(constants.MODEL_MAPPING_TYPES, "deepseekv4", 5)
    monkeypatch.setitem(constants.MODEL_MAPPING_TYPES, "deepseekv41", 41)
    config = PretrainedConfig(
        architectures=["DeepseekV41ForCausalLM"],
    )

    assert constants.parse_expert_type(config) == 41


def test_parse_moe_param_reads_nested_text_config(
    v41_flash_config: PretrainedConfig,
) -> None:
    assert parse_moe_param(v41_flash_config) == (40, 384, 0)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("model.layers.0.ffn.experts.0.w1.weight", (0, 0)),
        ("model.layers.39.ffn.experts.383.w3.weight", (39, 383)),
        (
            "model.language_model.layers.14.ffn.experts.42.w1.scale",
            (14, 42),
        ),
        ("model.layers.3.mlp.experts.0.gate_proj.weight", (None, None)),
        ("model.layers.3.ffn.shared_experts.w1.weight", (None, None)),
    ],
)
def test_parse_expert_id_uses_v4_layout(
    v41_flash_config: PretrainedConfig,
    name: str,
    expected: tuple[int | None, int | None],
) -> None:
    assert parse_expert_id(name, v41_flash_config) == expected
