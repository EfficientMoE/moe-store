# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import json

import torch
from safetensors.torch import save_file

from moe_store.convert.convert import convert_checkpoint
from moe_store.convert.writer import read_member_tensor
from moe_store.index import read_index
from moe_store.registry.slots import GPT_OSS_EXPERT_FIELDS

NUM_EXPERTS = 2
OUT = 8
N_BLOCKS = 2
HIDDEN = N_BLOCKS * 32


def _write_gpt_oss_checkpoint(ckpt_dir):
    ckpt_dir.mkdir()
    config = {
        "architectures": ["GptOssForCausalLM"],
        "model_type": "gpt_oss",
        "num_hidden_layers": 1,
        "num_local_experts": NUM_EXPERTS,
        "num_experts_per_tok": 1,
        "hidden_size": HIDDEN,
        "intermediate_size": OUT,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "vocab_size": 16,
        "torch_dtype": "bfloat16",
        "quantization_config": {"quant_method": "mxfp4"},
    }
    (ckpt_dir / "config.json").write_text(json.dumps(config))
    torch.manual_seed(9)
    prefix = "model.layers.0.mlp.experts"
    state = {
        "model.embed_tokens.weight": torch.randn(
            16, HIDDEN, dtype=torch.bfloat16
        ),
        f"{prefix}.gate_up_proj_blocks": torch.randint(
            0, 256, (NUM_EXPERTS, 2 * OUT, N_BLOCKS, 16), dtype=torch.uint8
        ),
        f"{prefix}.gate_up_proj_scales": torch.randint(
            110, 140, (NUM_EXPERTS, 2 * OUT, N_BLOCKS), dtype=torch.uint8
        ),
        f"{prefix}.gate_up_proj_bias": torch.randn(
            NUM_EXPERTS, 2 * OUT, dtype=torch.bfloat16
        ),
        f"{prefix}.down_proj_blocks": torch.randint(
            0, 256, (NUM_EXPERTS, HIDDEN, N_BLOCKS // 2, 16), dtype=torch.uint8
        ),
        f"{prefix}.down_proj_scales": torch.randint(
            110, 140, (NUM_EXPERTS, HIDDEN, N_BLOCKS // 2), dtype=torch.uint8
        ),
        f"{prefix}.down_proj_bias": torch.randn(
            NUM_EXPERTS, HIDDEN, dtype=torch.bfloat16
        ),
    }
    save_file(state, str(ckpt_dir / "model.safetensors"))
    return state


def test_gpt_oss_packed_experts_expand_per_expert(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    state = _write_gpt_oss_checkpoint(ckpt_dir)

    convert_checkpoint(str(ckpt_dir), str(store_dir))
    index = read_index(store_dir)

    experts = [g for g in index.groups if g.is_expert]
    assert len(experts) == NUM_EXPERTS
    for group in experts:
        fields = [m.name.rsplit(".", 1)[-1] for m in group.members]
        assert fields == list(GPT_OSS_EXPERT_FIELDS)
        for member in group.members:
            loaded = read_member_tensor(store_dir, group, member)
            field = member.name.rsplit(".", 1)[-1]
            source = state[f"model.layers.0.mlp.experts.{field}"][
                group.expert_id
            ]
            if field.endswith("_blocks"):
                source = source.reshape(source.shape[0], -1)
            assert loaded.dtype == source.dtype, member.name
            assert torch.equal(loaded, source.contiguous()), member.name
