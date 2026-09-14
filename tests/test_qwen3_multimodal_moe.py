# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Converter coverage for the Qwen3 multimodal MoE family.

Qwen3-VL-MoE ships v5 batched expert tensors under the
``model.language_model.*`` prefix next to a resident vision tower
(``model.visual.*``). Qwen3-Omni-MoE ships per-expert tensors for both
the thinker and the talker; only the thinker experts are offloadable,
the talker/vision/audio stacks stay resident as dense groups.
"""

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


def _write_qwen3_vl_moe_checkpoint(ckpt_dir):
    ckpt_dir.mkdir()
    config = {
        "architectures": ["Qwen3VLMoeForConditionalGeneration"],
        "model_type": "qwen3_vl_moe",
        "text_config": {
            "num_hidden_layers": NUM_LAYERS,
            "num_experts": NUM_EXPERTS,
            "num_experts_per_tok": 2,
            "hidden_size": HIDDEN,
            "intermediate_size": INTERMEDIATE,
            "moe_intermediate_size": INTERMEDIATE,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "vocab_size": 128,
        },
        "torch_dtype": "bfloat16",
    }
    (ckpt_dir / "config.json").write_text(json.dumps(config))

    torch.manual_seed(29)
    state = {
        "model.visual.patch_embed.proj.weight": torch.randn(
            HIDDEN, 3, dtype=torch.bfloat16
        ),
        "model.language_model.embed_tokens.weight": torch.randn(
            128, HIDDEN, dtype=torch.bfloat16
        ),
    }
    for layer in range(NUM_LAYERS):
        prefix = f"model.language_model.layers.{layer}"
        state[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(
            HIDDEN, HIDDEN, dtype=torch.bfloat16
        )
        state[f"{prefix}.mlp.gate.weight"] = torch.randn(
            NUM_EXPERTS, HIDDEN, dtype=torch.bfloat16
        )
        state[f"{prefix}.mlp.experts.gate_up_proj"] = torch.randn(
            NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, dtype=torch.bfloat16
        )
        state[f"{prefix}.mlp.experts.down_proj"] = torch.randn(
            NUM_EXPERTS, HIDDEN, INTERMEDIATE, dtype=torch.bfloat16
        )
    state["lm_head.weight"] = torch.randn(128, HIDDEN, dtype=torch.bfloat16)
    save_file(state, str(ckpt_dir / "model.safetensors"))
    return state


def test_qwen3_vl_moe_expands_batched_experts(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    state = _write_qwen3_vl_moe_checkpoint(ckpt_dir)

    convert_checkpoint(str(ckpt_dir), str(store_dir))
    index = read_index(store_dir)

    experts = [g for g in index.groups if g.is_expert]
    assert len(experts) == NUM_LAYERS * NUM_EXPERTS
    for group in experts:
        names = [m.name for m in group.members]
        assert all(".language_model.layers." in n for n in names)
        slots = [m.name.rsplit(".", 2)[-2] for m in group.members]
        assert slots == ["gate_proj", "up_proj", "down_proj"]

    group = index.expert_group(layer_id=1, expert_id=2)
    source = state["model.language_model.layers.1.mlp.experts.gate_up_proj"][2]
    gate_expected, up_expected = source.chunk(2, dim=0)
    down_expected = state[
        "model.language_model.layers.1.mlp.experts.down_proj"
    ][2]
    gate, up, down = (
        read_member_tensor(store_dir, group, m) for m in group.members
    )
    assert torch.equal(gate, gate_expected.contiguous())
    assert torch.equal(up, up_expected.contiguous())
    assert torch.equal(down, down_expected.contiguous())


def test_qwen3_vl_moe_keeps_vision_tower_dense(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    _write_qwen3_vl_moe_checkpoint(ckpt_dir)

    convert_checkpoint(str(ckpt_dir), str(store_dir))
    index = read_index(store_dir)

    dense_names = [
        m.name for g in index.groups if not g.is_expert for m in g.members
    ]
    assert "model.visual.patch_embed.proj.weight" in dense_names
    assert not any(".experts." in n for n in dense_names)


def _write_qwen3_omni_moe_checkpoint(ckpt_dir):
    ckpt_dir.mkdir()
    config = {
        "architectures": ["Qwen3OmniMoeForConditionalGeneration"],
        "model_type": "qwen3_omni_moe",
        "thinker_config": {
            "text_config": {
                "num_hidden_layers": NUM_LAYERS,
                "num_experts": NUM_EXPERTS,
                "num_experts_per_tok": 2,
                "hidden_size": HIDDEN,
                "intermediate_size": INTERMEDIATE,
                "moe_intermediate_size": INTERMEDIATE,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "vocab_size": 128,
            },
        },
        "torch_dtype": "bfloat16",
    }
    (ckpt_dir / "config.json").write_text(json.dumps(config))

    torch.manual_seed(31)
    state = {
        "thinker.model.embed_tokens.weight": torch.randn(
            128, HIDDEN, dtype=torch.bfloat16
        ),
        "thinker.visual.patch_embed.proj.weight": torch.randn(
            HIDDEN, 3, dtype=torch.bfloat16
        ),
        "code2wav.decoder.proj.weight": torch.randn(
            HIDDEN, HIDDEN, dtype=torch.bfloat16
        ),
    }
    for layer in range(NUM_LAYERS):
        prefix = f"thinker.model.layers.{layer}"
        state[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(
            HIDDEN, HIDDEN, dtype=torch.bfloat16
        )
        state[f"{prefix}.mlp.gate.weight"] = torch.randn(
            NUM_EXPERTS, HIDDEN, dtype=torch.bfloat16
        )
        for expert in range(NUM_EXPERTS):
            eprefix = f"{prefix}.mlp.experts.{expert}"
            state[f"{eprefix}.gate_proj.weight"] = torch.randn(
                INTERMEDIATE, HIDDEN, dtype=torch.bfloat16
            )
            state[f"{eprefix}.up_proj.weight"] = torch.randn(
                INTERMEDIATE, HIDDEN, dtype=torch.bfloat16
            )
            state[f"{eprefix}.down_proj.weight"] = torch.randn(
                HIDDEN, INTERMEDIATE, dtype=torch.bfloat16
            )
    # Talker experts stay resident: same naming shape, different prefix.
    for expert in range(2):
        eprefix = f"talker.model.layers.0.mlp.experts.{expert}"
        state[f"{eprefix}.gate_proj.weight"] = torch.randn(
            INTERMEDIATE, HIDDEN, dtype=torch.bfloat16
        )
        state[f"{eprefix}.up_proj.weight"] = torch.randn(
            INTERMEDIATE, HIDDEN, dtype=torch.bfloat16
        )
        state[f"{eprefix}.down_proj.weight"] = torch.randn(
            HIDDEN, INTERMEDIATE, dtype=torch.bfloat16
        )
    save_file(state, str(ckpt_dir / "model.safetensors"))
    return state


def test_qwen3_omni_moe_offloads_thinker_experts_only(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    state = _write_qwen3_omni_moe_checkpoint(ckpt_dir)

    convert_checkpoint(str(ckpt_dir), str(store_dir))
    index = read_index(store_dir)

    experts = [g for g in index.groups if g.is_expert]
    assert len(experts) == NUM_LAYERS * NUM_EXPERTS
    for group in experts:
        names = [m.name for m in group.members]
        assert all(n.startswith("thinker.model.layers.") for n in names)
        slots = [m.name.rsplit(".", 2)[-2] for m in group.members]
        assert slots == ["gate_proj", "up_proj", "down_proj"]
        for member in group.members:
            loaded = read_member_tensor(store_dir, group, member)
            assert torch.equal(loaded, state[member.name]), member.name

    dense_names = [
        m.name for g in index.groups if not g.is_expert for m in g.members
    ]
    assert any(n.startswith("talker.") for n in dense_names)
    assert any(n.startswith("code2wav.") for n in dense_names)
    assert not any(
        n.startswith("talker.")
        for g in experts
        for n in [m.name for m in g.members]
    )
