# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Every architecture in MODEL_MAPPING_NAMES must convert, not just parse.

Catches registry/parsing drift: dbrx, jamba, and opt were registered but
parse_moe_param raised "Unsupported architecture" for them. DBRX ships
fused per-layer expert tensors (``ffn.experts.mlp.{w1,v1,w2}`` shaped
``[E*ffn, d]``) that expand per expert; Jamba routes experts only on
layers with ``L % expert_layer_period == expert_layer_offset``; OPT is
dense.
"""

import json

import torch
from safetensors.torch import save_file

from moe_store.convert.convert import convert_checkpoint
from moe_store.convert.writer import read_member_tensor
from moe_store.index import read_index
from moe_store.parsing.hf_config import parse_moe_param
from moe_store.registry.constants import MODEL_MAPPING_NAMES

HIDDEN = 32
FFN = 16
NUM_EXPERTS = 4


class _Cfg:
    def __init__(self, arch, **kwargs):
        self.architectures = [arch]
        self.__dict__.update(kwargs)


_MATRIX_CONFIGS = {
    "nllb": _Cfg(
        "NllbMoeForConditionalGeneration",
        encoder_sparse_step=2,
        decoder_sparse_step=2,
        encoder_layers=4,
        decoder_layers=4,
        num_experts=NUM_EXPERTS,
    ),
    "mixtral": _Cfg(
        "MixtralForCausalLM", num_hidden_layers=2, num_local_experts=4
    ),
    "opt": _Cfg("OPTForCausalLM", num_hidden_layers=2),
    "deepseek_v3": _Cfg(
        "DeepseekV3ForCausalLM", num_hidden_layers=2, n_routed_experts=4
    ),
    "deepseek": _Cfg(
        "DeepseekV2ForCausalLM", num_hidden_layers=2, n_routed_experts=4
    ),
    "gptoss": _Cfg(
        "GptOssForCausalLM", num_hidden_layers=2, num_local_experts=4
    ),
    "qwen3": _Cfg("Qwen3MoeForCausalLM", num_hidden_layers=2, num_experts=4),
    "dbrx": _Cfg(
        "DbrxForCausalLM",
        n_layers=2,
        ffn_config=type("F", (), {"moe_num_experts": NUM_EXPERTS})(),
    ),
    "olmoe": _Cfg("OlmoeForCausalLM", num_hidden_layers=2, num_experts=4),
    "jamba": _Cfg(
        "JambaForCausalLM",
        num_hidden_layers=4,
        num_experts=NUM_EXPERTS,
        expert_layer_period=2,
        expert_layer_offset=1,
    ),
}


def test_every_registered_arch_parses_moe_params():
    missing = []
    for key in MODEL_MAPPING_NAMES:
        cfg = _MATRIX_CONFIGS.get(key)
        if cfg is None:
            continue
        try:
            parse_moe_param(cfg)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{key}: {exc}")
    assert not missing, f"registry archs that cannot parse: {missing}"


def _write_dbrx_checkpoint(ckpt_dir):
    ckpt_dir.mkdir()
    config = {
        "architectures": ["DbrxForCausalLM"],
        "model_type": "dbrx",
        "d_model": HIDDEN,
        "n_layers": 2,
        "n_heads": 4,
        "max_seq_len": 64,
        "vocab_size": 64,
        "ffn_config": {
            "ffn_hidden_size": FFN,
            "moe_num_experts": NUM_EXPERTS,
            "moe_top_k": 2,
        },
        "attn_config": {},
        "torch_dtype": "bfloat16",
    }
    (ckpt_dir / "config.json").write_text(json.dumps(config))
    torch.manual_seed(47)
    state = {
        "transformer.wte.weight": torch.randn(64, HIDDEN, dtype=torch.bfloat16)
    }
    for layer in range(2):
        prefix = f"transformer.blocks.{layer}"
        state[f"{prefix}.norm_attn_norm.attn.Wqkv.weight"] = torch.randn(
            3 * HIDDEN, HIDDEN, dtype=torch.bfloat16
        )
        state[f"{prefix}.ffn.router.layer.weight"] = torch.randn(
            NUM_EXPERTS, HIDDEN, dtype=torch.bfloat16
        )
        for part in ("w1", "v1", "w2"):
            state[f"{prefix}.ffn.experts.mlp.{part}"] = torch.randn(
                NUM_EXPERTS * FFN, HIDDEN, dtype=torch.bfloat16
            )
    save_file(state, str(ckpt_dir / "model.safetensors"))
    return state


def test_dbrx_fused_experts_expand_per_expert(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    state = _write_dbrx_checkpoint(ckpt_dir)

    convert_checkpoint(str(ckpt_dir), str(store_dir))
    index = read_index(store_dir)

    experts = [g for g in index.groups if g.is_expert]
    assert len(experts) == 2 * NUM_EXPERTS
    group = index.expert_group(layer_id=1, expert_id=2)
    slots = [m.name.rsplit(".", 1)[-1] for m in group.members]
    assert slots == ["w1", "v1", "w2"]
    for member, part in zip(group.members, ("w1", "v1", "w2")):
        source = state[f"transformer.blocks.1.ffn.experts.mlp.{part}"]
        expected = source[2 * FFN : 3 * FFN]
        loaded = read_member_tensor(store_dir, group, member)
        assert torch.equal(loaded, expected.contiguous()), member.name

    dense_names = [
        m.name for g in index.groups if not g.is_expert for m in g.members
    ]
    assert any("router.layer.weight" in n for n in dense_names)


def _write_jamba_checkpoint(ckpt_dir):
    ckpt_dir.mkdir()
    config = {
        "architectures": ["JambaForCausalLM"],
        "model_type": "jamba",
        "hidden_size": HIDDEN,
        "intermediate_size": FFN,
        "num_hidden_layers": 4,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_tok": 2,
        "expert_layer_period": 2,
        "expert_layer_offset": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 64,
        "torch_dtype": "bfloat16",
    }
    (ckpt_dir / "config.json").write_text(json.dumps(config))
    torch.manual_seed(53)
    state = {
        "model.embed_tokens.weight": torch.randn(
            64, HIDDEN, dtype=torch.bfloat16
        )
    }
    for layer in range(4):
        prefix = f"model.layers.{layer}"
        if layer % 2 == 1:
            state[f"{prefix}.feed_forward.router.weight"] = torch.randn(
                NUM_EXPERTS, HIDDEN, dtype=torch.bfloat16
            )
            for expert in range(NUM_EXPERTS):
                eprefix = f"{prefix}.feed_forward.experts.{expert}"
                for slot, shape in (
                    ("gate_proj", (FFN, HIDDEN)),
                    ("up_proj", (FFN, HIDDEN)),
                    ("down_proj", (HIDDEN, FFN)),
                ):
                    state[f"{eprefix}.{slot}.weight"] = torch.randn(
                        *shape, dtype=torch.bfloat16
                    )
        else:
            state[f"{prefix}.feed_forward.gate_proj.weight"] = torch.randn(
                FFN, HIDDEN, dtype=torch.bfloat16
            )
    save_file(state, str(ckpt_dir / "model.safetensors"))
    return state


def test_jamba_expert_layers_remap_to_dense_layer_ids(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    state = _write_jamba_checkpoint(ckpt_dir)

    convert_checkpoint(str(ckpt_dir), str(store_dir))
    index = read_index(store_dir)

    experts = [g for g in index.groups if g.is_expert]
    assert len(experts) == 2 * NUM_EXPERTS
    assert {g.layer_id for g in experts} == {0, 1}
    group = index.expert_group(layer_id=1, expert_id=3)
    assert all(
        m.name.startswith("model.layers.3.feed_forward.experts.3.")
        for m in group.members
    )
    for member in group.members:
        loaded = read_member_tensor(store_dir, group, member)
        assert torch.equal(loaded, state[member.name]), member.name


def test_opt_converts_dense_only(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    config = {
        "architectures": ["OPTForCausalLM"],
        "model_type": "opt",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 2,
        "ffn_dim": FFN,
        "num_attention_heads": 4,
        "vocab_size": 64,
        "max_position_embeddings": 64,
        "torch_dtype": "bfloat16",
    }
    (ckpt_dir / "config.json").write_text(json.dumps(config))
    state = {
        "model.decoder.embed_tokens.weight": torch.randn(
            64, HIDDEN, dtype=torch.bfloat16
        ),
        "model.decoder.layers.0.fc1.weight": torch.randn(
            FFN, HIDDEN, dtype=torch.bfloat16
        ),
    }
    save_file(state, str(ckpt_dir / "model.safetensors"))

    convert_checkpoint(str(ckpt_dir), str(tmp_path / "store"))
    index = read_index(tmp_path / "store")
    assert not any(g.is_expert for g in index.groups)
