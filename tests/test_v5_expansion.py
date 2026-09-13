# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import json

import torch
from safetensors.torch import save_file

from moe_store.convert.convert import convert_checkpoint
from moe_store.convert.writer import read_member_tensor
from moe_store.index import read_index

NUM_LAYERS = 2
NUM_EXPERTS = 4
HIDDEN = 32
INTERMEDIATE = 16


def _write_v5_batched_checkpoint(ckpt_dir, *, mixtral: bool):
    ckpt_dir.mkdir()
    if mixtral:
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
    else:
        config = {
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
            "num_hidden_layers": NUM_LAYERS,
            "num_experts": NUM_EXPERTS,
            "num_experts_per_tok": 2,
            "hidden_size": HIDDEN,
            "intermediate_size": INTERMEDIATE,
            "moe_intermediate_size": INTERMEDIATE,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "vocab_size": 128,
        }
    (ckpt_dir / "config.json").write_text(json.dumps(config))

    torch.manual_seed(23)
    state = {"model.embed_tokens.weight": torch.randn(128, HIDDEN)}
    for layer in range(NUM_LAYERS):
        prefix = f"model.layers.{layer}"
        state[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(HIDDEN, HIDDEN)
        state[f"{prefix}.mlp.gate.weight"] = torch.randn(NUM_EXPERTS, HIDDEN)
        state[f"{prefix}.mlp.experts.gate_up_proj"] = torch.randn(
            NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, dtype=torch.bfloat16
        )
        state[f"{prefix}.mlp.experts.down_proj"] = torch.randn(
            NUM_EXPERTS, HIDDEN, INTERMEDIATE, dtype=torch.bfloat16
        )
    state["lm_head.weight"] = torch.randn(128, HIDDEN)
    save_file(state, str(ckpt_dir / "model.safetensors"))
    return state


def _convert(tmp_path, *, mixtral: bool):
    ckpt_dir = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    state = _write_v5_batched_checkpoint(ckpt_dir, mixtral=mixtral)
    convert_checkpoint(str(ckpt_dir), str(store_dir))
    return state, store_dir


def test_qwen_v5_expands_to_slot_ordered_expert_groups(tmp_path):
    state, store_dir = _convert(tmp_path, mixtral=False)
    index = read_index(store_dir)
    experts = [g for g in index.groups if g.is_expert]
    assert len(experts) == NUM_LAYERS * NUM_EXPERTS
    for group in experts:
        slots = [m.name.rsplit(".", 2)[-2] for m in group.members]
        assert slots == ["gate_proj", "up_proj", "down_proj"]

    group = index.expert_group(layer_id=1, expert_id=2)
    source = state["model.layers.1.mlp.experts.gate_up_proj"][2]
    gate_expected, up_expected = source.chunk(2, dim=0)
    down_expected = state["model.layers.1.mlp.experts.down_proj"][2]
    gate, up, down = (
        read_member_tensor(store_dir, group, m) for m in group.members
    )
    assert torch.equal(gate, gate_expected.contiguous())
    assert torch.equal(up, up_expected.contiguous())
    assert torch.equal(down, down_expected.contiguous())


def test_mixtral_v5_renames_block_and_uses_w_slots(tmp_path):
    state, store_dir = _convert(tmp_path, mixtral=True)
    index = read_index(store_dir)
    experts = [g for g in index.groups if g.is_expert]
    assert len(experts) == NUM_LAYERS * NUM_EXPERTS
    for group in experts:
        names = [m.name for m in group.members]
        assert all(".block_sparse_moe.experts." in n for n in names)
        slots = [m.name.rsplit(".", 2)[-2] for m in group.members]
        assert slots == ["w1", "w2", "w3"]

    all_names = [m.name for _, m in index.iter_members()]
    assert any(n.endswith(".block_sparse_moe.gate.weight") for n in all_names)
    assert not any(".mlp.gate.weight" in n for n in all_names)

    group = index.expert_group(layer_id=0, expert_id=1)
    source = state["model.layers.0.mlp.experts.gate_up_proj"][1]
    gate_expected, up_expected = source.chunk(2, dim=0)
    down_expected = state["model.layers.0.mlp.experts.down_proj"][1]
    by_slot = {
        m.name.rsplit(".", 2)[-2]: read_member_tensor(store_dir, group, m)
        for m in group.members
    }
    assert torch.equal(by_slot["w1"], gate_expected.contiguous())
    assert torch.equal(by_slot["w3"], up_expected.contiguous())
    assert torch.equal(by_slot["w2"], down_expected.contiguous())
