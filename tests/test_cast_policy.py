# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import json

import torch
from safetensors.torch import save_file

from moe_store.convert.convert import convert_checkpoint
from moe_store.convert.writer import read_member_tensor
from moe_store.index import read_index


def _write_fp32_checkpoint(ckpt_dir):
    ckpt_dir.mkdir()
    config = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "num_hidden_layers": 1,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "hidden_size": 8,
        "intermediate_size": 4,
        "moe_intermediate_size": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "vocab_size": 16,
        "torch_dtype": "bfloat16",
    }
    (ckpt_dir / "config.json").write_text(json.dumps(config))
    torch.manual_seed(5)
    state = {
        "model.embed_tokens.weight": torch.randn(16, 8, dtype=torch.float32),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(
            8, 8, dtype=torch.float32
        ),
    }
    for expert in range(2):
        for slot in ("gate_proj", "up_proj", "down_proj"):
            key = f"model.layers.0.mlp.experts.{expert}.{slot}.weight"
            state[key] = torch.randn(4, 8, dtype=torch.float32)
    save_file(state, str(ckpt_dir / "model.safetensors"))
    return state


def test_fp32_checkpoint_casts_to_config_dtype(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    state = _write_fp32_checkpoint(ckpt_dir)

    convert_checkpoint(str(ckpt_dir), str(store_dir))
    index = read_index(store_dir)

    for group, member in index.iter_members():
        assert member.dtype == "bfloat16", member.name
        loaded = read_member_tensor(store_dir, group, member)
        assert loaded.dtype == torch.bfloat16
        expected = state[member.name].to(torch.bfloat16)
        assert torch.equal(loaded, expected), member.name


def _write_glm_fp8_like_checkpoint(ckpt_dir):
    ckpt_dir.mkdir()
    config = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 1,
        "n_routed_experts": 2,
        "hidden_size": 8,
        "intermediate_size": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "vocab_size": 16,
        "torch_dtype": "bfloat16",
        "quantization_config": {"quant_method": "fp8", "fmt": "e4m3"},
    }
    (ckpt_dir / "config.json").write_text(json.dumps(config))
    torch.manual_seed(6)
    state = {
        "model.embed_tokens.weight": torch.randn(16, 8, dtype=torch.float32),
    }
    for expert in range(2):
        base = f"model.layers.0.mlp.experts.{expert}"
        for slot in ("gate_proj", "up_proj", "down_proj"):
            state[f"{base}.{slot}.weight"] = torch.randn(
                4, 8, dtype=torch.float32
            ).to(torch.float8_e4m3fn)
            state[f"{base}.{slot}.weight_scale_inv"] = torch.randn(
                1, 1, dtype=torch.float32
            )
    save_file(state, str(ckpt_dir / "model.safetensors"))
    return state


def test_glm_fp8_experts_keep_raw_dtype_and_scales_share_group(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    state = _write_glm_fp8_like_checkpoint(ckpt_dir)

    convert_checkpoint(str(ckpt_dir), str(store_dir))
    index = read_index(store_dir)

    experts = [g for g in index.groups if g.is_expert]
    assert len(experts) == 2
    for group in experts:
        names = [m.name for m in group.members]
        assert len(names) == 6, names
        assert sum(n.endswith("_scale_inv") for n in names) == 3
        for member in group.members:
            if member.name.endswith("_scale_inv"):
                assert member.dtype == "float32"
            else:
                assert member.dtype == "float8_e4m3fn"
            loaded = read_member_tensor(store_dir, group, member)
            original = state[member.name]
            assert loaded.dtype == original.dtype
            assert torch.equal(
                loaded.view(torch.uint8), original.view(torch.uint8)
            ), member.name

    dense = [g for g in index.groups if not g.is_expert]
    for group in dense:
        for member in group.members:
            assert member.dtype == "bfloat16", member.name
