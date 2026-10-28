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
