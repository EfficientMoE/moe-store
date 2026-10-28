# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

HIDDEN = 64
INTER = 32
LAYERS = 2
EXPERTS = 4


@pytest.fixture()
def tiny_v41_ckpt(tmp_path: Path) -> Path:
    tensors = {}
    for layer in range(LAYERS):
        for expert in range(EXPERTS):
            prefix = f"layers.{layer}.ffn.experts.{expert}"
            # w1/w3: [inter, hidden/2] packed, scale [inter, hidden/32]
            tensors[f"{prefix}.w1.weight"] = torch.zeros(
                INTER, HIDDEN // 2, dtype=torch.int8
            )
            tensors[f"{prefix}.w1.scale"] = torch.full(
                (INTER, HIDDEN // 32), 127, dtype=torch.uint8
            ).view(torch.float8_e8m0fnu)
            tensors[f"{prefix}.w2.weight"] = torch.zeros(
                HIDDEN, INTER // 2, dtype=torch.int8
            )
            tensors[f"{prefix}.w2.scale"] = torch.full(
                (HIDDEN, INTER // 32), 127, dtype=torch.uint8
            ).view(torch.float8_e8m0fnu)
            tensors[f"{prefix}.w3.weight"] = torch.zeros(
                INTER, HIDDEN // 2, dtype=torch.int8
            )
            tensors[f"{prefix}.w3.scale"] = torch.full(
                (INTER, HIDDEN // 32), 127, dtype=torch.uint8
            ).view(torch.float8_e8m0fnu)
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, str(tmp_path / shard))
    index = {"weight_map": {k: shard for k in tensors}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    config = {
        "model_type": "deepseek_v41",
        "architectures": ["DeepseekV41ForCausalLM"],
        "text_config": {
            "model_type": "deepseek_v41_text",
            "num_hidden_layers": LAYERS,
            "n_routed_experts": EXPERTS,
            "hidden_size": HIDDEN,
            "moe_intermediate_size": INTER,
            "swiglu_limit": 10.0,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


def test_v41_indexer_reads_nested_text_config(tiny_v41_ckpt: Path) -> None:
    from moe_store.wrappers.deepseek_v41 import (
        DeepSeekV41ExpertTensorIndexer,
    )

    indexer = DeepSeekV41ExpertTensorIndexer(str(tiny_v41_ckpt))
    assert indexer.num_hidden_layers == LAYERS
    assert indexer.n_routed_experts == EXPERTS
    assert indexer.hidden_size == HIDDEN
    assert indexer.intermediate_size == INTER
    assert indexer.swiglu_limit == 10.0


def test_v41_bundle_has_six_tensors_with_fp4_metadata(
    tiny_v41_ckpt: Path,
) -> None:
    from moe_store.wrappers.deepseek_v41 import (
        DeepSeekV41ExpertTensorIndexer,
    )

    indexer = DeepSeekV41ExpertTensorIndexer(str(tiny_v41_ckpt))
    bundle = indexer.bundle(1, 3)
    assert bundle.num_tensors == 6
    assert bundle.quant_format == "fp4_e2m1"
    assert bundle.scale_format == "ue8m0"
    assert bundle.block_size == 32
    w1 = bundle.part("w1", "weight")
    assert w1.key == "layers.1.ffn.experts.3.w1.weight"
    assert w1.shape == (INTER, HIDDEN // 2)
    loaded = indexer.load_bundle_tensors(bundle)
    assert len(loaded) == 6
