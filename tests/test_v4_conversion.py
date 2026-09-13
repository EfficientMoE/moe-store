# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Converter coverage for DeepSeek-V4 native checkpoint naming
(``layers.N.ffn.experts.M.*``); the Path-B loader switch to v2 stores is
gated on the V4 container harness and tracked separately."""

import json

import torch
from safetensors.torch import save_file

from moe_store.convert.convert import convert_checkpoint
from moe_store.convert.writer import read_member_tensor
from moe_store.index import read_index

NUM_LAYERS = 2
NUM_EXPERTS = 4


def _write_v4_style_checkpoint(ckpt_dir):
    ckpt_dir.mkdir()
    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "num_hidden_layers": NUM_LAYERS,
        "n_routed_experts": NUM_EXPERTS,
        "hidden_size": 16,
        "intermediate_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "vocab_size": 32,
        "torch_dtype": "bfloat16",
    }
    (ckpt_dir / "config.json").write_text(json.dumps(config))
    torch.manual_seed(17)
    state = {"embed.weight": torch.randn(32, 16, dtype=torch.bfloat16)}
    for layer in range(NUM_LAYERS):
        state[f"layers.{layer}.attn.q.weight"] = torch.randn(
            16, 16, dtype=torch.bfloat16
        )
        for expert in range(NUM_EXPERTS):
            for slot in ("w1", "w2", "w3"):
                key = f"layers.{layer}.ffn.experts.{expert}.{slot}.weight"
                state[key] = torch.randn(8, 16, dtype=torch.bfloat16)
    save_file(state, str(ckpt_dir / "model.safetensors"))
    return state


def test_v4_native_naming_groups_per_expert(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    state = _write_v4_style_checkpoint(ckpt_dir)

    convert_checkpoint(str(ckpt_dir), str(store_dir))
    index = read_index(store_dir)

    experts = [g for g in index.groups if g.is_expert]
    assert len(experts) == NUM_LAYERS * NUM_EXPERTS
    for group in experts:
        slots = [m.name.rsplit(".", 2)[-2] for m in group.members]
        assert slots == ["w1", "w2", "w3"]
        for member in group.members:
            loaded = read_member_tensor(store_dir, group, member)
            assert torch.equal(loaded, state[member.name]), member.name
