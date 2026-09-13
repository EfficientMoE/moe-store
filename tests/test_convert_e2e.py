# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import json

import torch
from safetensors.torch import save_file

from moe_store.convert.convert import (
    convert_checkpoint,
    dump_inspect,
    inspect_store,
)
from moe_store.index import read_index

NUM_LAYERS = 2
NUM_EXPERTS = 4
HIDDEN = 32
INTERMEDIATE = 16


def _write_tiny_mixtral_checkpoint(ckpt_dir):
    ckpt_dir.mkdir()
    config = {
        "architectures": ["MixtralForCausalLM"],
        "model_type": "mixtral",
        "num_hidden_layers": NUM_LAYERS,
        "num_local_experts": NUM_EXPERTS,
        "num_experts_per_tok": 2,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": 128,
    }
    (ckpt_dir / "config.json").write_text(json.dumps(config))

    torch.manual_seed(11)
    state = {"model.embed_tokens.weight": torch.randn(128, HIDDEN)}
    for layer in range(NUM_LAYERS):
        prefix = f"model.layers.{layer}"
        state[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(
            HIDDEN, HIDDEN
        )
        state[f"{prefix}.block_sparse_moe.gate.weight"] = torch.randn(
            NUM_EXPERTS, HIDDEN
        )
        for expert in range(NUM_EXPERTS):
            eprefix = f"{prefix}.block_sparse_moe.experts.{expert}"
            state[f"{eprefix}.w1.weight"] = torch.randn(
                INTERMEDIATE, HIDDEN, dtype=torch.bfloat16
            )
            state[f"{eprefix}.w2.weight"] = torch.randn(
                HIDDEN, INTERMEDIATE, dtype=torch.bfloat16
            )
            state[f"{eprefix}.w3.weight"] = torch.randn(
                INTERMEDIATE, HIDDEN, dtype=torch.bfloat16
            )
    state["lm_head.weight"] = torch.randn(128, HIDDEN)
    save_file(state, str(ckpt_dir / "model.safetensors"))
    return state


def test_convert_tiny_mixtral_end_to_end(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    state = _write_tiny_mixtral_checkpoint(ckpt_dir)

    convert_checkpoint(str(ckpt_dir), str(store_dir))

    index = read_index(store_dir)
    experts = [g for g in index.groups if g.is_expert]
    assert len(experts) == NUM_LAYERS * NUM_EXPERTS
    for group in experts:
        slots = [m.name.rsplit(".", 2)[-2] for m in group.members]
        assert slots == ["w1", "w2", "w3"]

    from moe_store.convert.writer import read_member_tensor

    group = index.expert_group(layer_id=1, expert_id=3)
    for member in group.members:
        loaded = read_member_tensor(store_dir, group, member)
        assert torch.equal(loaded, state[member.name])

    summary = inspect_store(str(store_dir))
    assert summary["version"] == 2
    assert summary["model_type"] == "mixtral"
    assert summary["expert_groups"] == NUM_LAYERS * NUM_EXPERTS
    assert summary["experts_per_layer"] == NUM_EXPERTS

    text = dump_inspect(str(store_dir), verbose=True)
    assert "version=2" in text
    assert text.count("expert layer=") == NUM_LAYERS * NUM_EXPERTS
