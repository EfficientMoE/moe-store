# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Multi-component H3 pipeline conversion (issue #222 Task 0).

A tiny synthetic MiniMax-H3-like modular pipeline (root modular_model_index +
``transformer``/``vae``/``audio_vae``/``text_encoder`` components) converts
into one store with: per-block non-AdaLN groups, per-block AdaLN bundle
groups, and one group per VAE/encoder shard; every member reconstructs
byte-exact. Variant sub-pipelines (``FL2VA``, own model_index) and
weight-free folders (``tokenizer``) are excluded. Single-component pipeline
behavior is pinned separately in test_dense_conversion.py.
"""

import json

import torch
from safetensors.torch import save_file

from moe_store.convert.convert import convert_checkpoint, dump_inspect
from moe_store.convert.writer import read_member_tensor
from moe_store.index import read_index

HIDDEN = 16
NUM_BLOCKS = 2


def _write_h3_modular_pipeline(root):
    root.mkdir()
    (root / "modular_model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3ModularPipeline"})
    )
    torch.manual_seed(7)
    states = {}

    transformer = root / "transformer"
    transformer.mkdir()
    (transformer / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "MiniMaxH3Transformer3DModel",
                "_diffusers_version": "0.36.0",
                "num_layers": NUM_BLOCKS,
            }
        )
    )
    state = {"proj_in.weight": torch.randn(HIDDEN, 8, dtype=torch.bfloat16)}
    for block in range(NUM_BLOCKS):
        prefix = f"transformer_blocks.{block}"
        state[f"{prefix}.adaln_proj.linear.weight"] = torch.randn(
            6 * HIDDEN, HIDDEN, dtype=torch.bfloat16
        )
        state[f"{prefix}.adaln_proj.linear.bias"] = torch.randn(
            6 * HIDDEN, dtype=torch.bfloat16
        )
        state[f"{prefix}.attn.to_q.weight"] = torch.randn(
            HIDDEN, HIDDEN, dtype=torch.bfloat16
        )
        state[f"{prefix}.attn.to_out.0.weight"] = torch.randn(
            HIDDEN, HIDDEN, dtype=torch.bfloat16
        )
        state[f"{prefix}.ff.net.0.proj.weight"] = torch.randn(
            HIDDEN, HIDDEN, dtype=torch.bfloat16
        )
        state[f"{prefix}.norm1.weight"] = torch.randn(
            HIDDEN, dtype=torch.bfloat16
        )
    save_file(state, str(transformer / "diffusion_pytorch_model.safetensors"))
    states["transformer"] = state

    for component, class_name, tensors in (
        (
            "vae",
            "AutoencoderKLMiniMaxH3",
            {
                "encoder.conv_in.weight": torch.randn(
                    HIDDEN, 3, dtype=torch.bfloat16
                ),
                "decoder.conv_out.weight": torch.randn(
                    3, HIDDEN, dtype=torch.bfloat16
                ),
            },
        ),
        (
            "audio_vae",
            "AutoencoderKLMiniMaxH3Audio",
            {
                "encoder.proj.weight": torch.randn(
                    HIDDEN, HIDDEN, dtype=torch.bfloat16
                )
            },
        ),
    ):
        comp_dir = root / component
        comp_dir.mkdir()
        (comp_dir / "config.json").write_text(
            json.dumps(
                {"_class_name": class_name, "_diffusers_version": "0.36.0"}
            )
        )
        save_file(
            tensors, str(comp_dir / "diffusion_pytorch_model.safetensors")
        )
        states[component] = tensors

    text_encoder = root / "text_encoder"
    text_encoder.mkdir()
    (text_encoder / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "hidden_size": HIDDEN,
                "vocab_size": 32,
            }
        )
    )
    shard1 = {
        "model.embed_tokens.weight": torch.randn(
            32, HIDDEN, dtype=torch.bfloat16
        )
    }
    shard2 = {"lm_head.weight": torch.randn(32, HIDDEN, dtype=torch.bfloat16)}
    save_file(shard1, str(text_encoder / "model-00001-of-00002.safetensors"))
    save_file(shard2, str(text_encoder / "model-00002-of-00002.safetensors"))
    (text_encoder / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    name: f"model-{idx:05d}-of-00002.safetensors"
                    for idx, shard in ((1, shard1), (2, shard2))
                    for name in shard
                }
            }
        )
    )
    states["text_encoder"] = {**shard1, **shard2}

    variant = root / "FL2VA"
    (variant / "transformer").mkdir(parents=True)
    (variant / "model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3Pipeline"})
    )
    (variant / "transformer" / "config.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3Transformer3DModel"})
    )
    save_file(
        {"decoy.weight": torch.randn(2, 2, dtype=torch.bfloat16)},
        str(variant / "transformer" / "model.safetensors"),
    )

    tokenizer = root / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")

    return states


def _stage_members(index):
    stage_names = [stage.name for stage in index.stages]
    members = {}
    for group in index.groups:
        stage = stage_names[group.group_id & 0xFFFFFFFF]
        members.setdefault(stage, set()).update(
            member.name for member in group.members
        )
    return members


def _expected_names(states):
    return {
        f"{component}.{name}"
        for component, tensors in states.items()
        for name in tensors
    }


def test_h3_pipeline_groups_blocks_adaln_and_shards(tmp_path):
    root = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    states = _write_h3_modular_pipeline(root)

    convert_checkpoint(str(root), str(store_dir))
    index = read_index(store_dir)

    assert index.model_type == "MiniMaxH3ModularPipeline"
    assert not any(group.is_expert for group in index.groups)

    names = {member.name for _, member in index.iter_members()}
    assert names == _expected_names(states)
    assert not any(name.startswith("FL2VA.") for name in names)

    stages = _stage_members(index)
    for block in range(NUM_BLOCKS):
        prefix = f"transformer_blocks.{block}"
        assert stages[f"transformer.{prefix}.adaln"] == {
            f"transformer.{prefix}.adaln_proj.linear.weight",
            f"transformer.{prefix}.adaln_proj.linear.bias",
        }
        assert stages[f"transformer.{prefix}"] == {
            f"transformer.{prefix}.attn.to_q.weight",
            f"transformer.{prefix}.attn.to_out.0.weight",
            f"transformer.{prefix}.ff.net.0.proj.weight",
            f"transformer.{prefix}.norm1.weight",
        }
    assert stages["vae.diffusion_pytorch_model"] == {
        f"vae.{name}" for name in states["vae"]
    }
    assert stages["audio_vae.diffusion_pytorch_model"] == {
        f"audio_vae.{name}" for name in states["audio_vae"]
    }
    assert stages["text_encoder.model-00001-of-00002"] == {
        "text_encoder.model.embed_tokens.weight"
    }
    assert stages["text_encoder.model-00002-of-00002"] == {
        "text_encoder.lm_head.weight"
    }


def test_h3_pipeline_members_reconstruct_byte_exact(tmp_path):
    root = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    states = _write_h3_modular_pipeline(root)

    convert_checkpoint(str(root), str(store_dir))
    index = read_index(store_dir)

    checked = 0
    for group in index.groups:
        for member in group.members:
            component, _, name = member.name.partition(".")
            loaded = read_member_tensor(store_dir, group, member)
            assert torch.equal(loaded, states[component][name]), member.name
            checked += 1
    assert checked == len(_expected_names(states))


def test_h3_pipeline_preserves_source_dtypes(tmp_path):
    root = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    root.mkdir()
    (root / "modular_model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3ModularPipeline"})
    )
    for component in ("transformer", "vae"):
        comp = root / component
        comp.mkdir()
        (comp / "config.json").write_text(
            json.dumps(
                {
                    "_class_name": (
                        "MiniMaxH3Transformer3DModel"
                        if component == "transformer"
                        else "AutoencoderKLMiniMaxH3"
                    )
                }
            )
        )
        save_file(
            {
                "proj_in.weight": torch.randn(HIDDEN, 8, dtype=torch.float32),
                "block.weight": torch.randn(
                    HIDDEN, HIDDEN, dtype=torch.bfloat16
                ),
            },
            str(comp / "diffusion_pytorch_model.safetensors"),
        )

    convert_checkpoint(str(root), str(store_dir))
    index = read_index(store_dir)

    dtypes = {member.name: member.dtype for _, member in index.iter_members()}
    assert dtypes["transformer.proj_in.weight"] == "float32"
    assert dtypes["transformer.block.weight"] == "bfloat16"
    assert dtypes["vae.proj_in.weight"] == "float32"


def test_h3_pipeline_inspect_names_groups(tmp_path):
    root = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    _write_h3_modular_pipeline(root)

    convert_checkpoint(str(root), str(store_dir))
    text = dump_inspect(str(store_dir), verbose=True)

    assert "version=2" in text
    assert "stage=transformer.transformer_blocks.0.adaln" in text
    assert "stage=transformer.transformer_blocks.0\n" in text or text.endswith(
        "stage=transformer.transformer_blocks.0"
    )
    assert "stage=vae.diffusion_pytorch_model" in text
    assert "stage=audio_vae.diffusion_pytorch_model" in text
    assert "stage=text_encoder.model-00001-of-00002" in text
