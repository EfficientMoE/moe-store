# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from transformers import PretrainedConfig

import moe_store.registry.constants as constants


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
