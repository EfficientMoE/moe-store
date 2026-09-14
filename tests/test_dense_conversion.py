# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Converter coverage for dense diffusion checkpoints (no MoE experts).

Qwen-Image-2512 ships a diffusers pipeline (``model_index.json`` +
``transformer/`` with a ``_class_name`` config); MiniMax-H3 ships a
modular pipeline (``modular_model_index.json``) with task-variant
subfolders (``FL2VA/transformer``, ``Ref2VA/transformer``). Every tensor
lands in a dense group; there are no expert groups.
"""

import json

import pytest
import torch
from safetensors.torch import save_file

from moe_store.convert.convert import convert_checkpoint, inspect_store
from moe_store.convert.writer import read_member_tensor
from moe_store.index import read_index

NUM_BLOCKS = 2
HIDDEN = 32


def _write_qwen_image_transformer(component_dir, *, weights_name):
    component_dir.mkdir(parents=True)
    config = {
        "_class_name": "QwenImageTransformer2DModel",
        "_diffusers_version": "0.36.0",
        "attention_head_dim": 8,
        "num_attention_heads": 4,
        "num_layers": NUM_BLOCKS,
        "in_channels": 16,
        "out_channels": 4,
        "patch_size": 2,
    }
    (component_dir / "config.json").write_text(json.dumps(config))

    torch.manual_seed(37)
    state = {"img_in.weight": torch.randn(HIDDEN, 16, dtype=torch.bfloat16)}
    for block in range(NUM_BLOCKS):
        prefix = f"transformer_blocks.{block}"
        state[f"{prefix}.attn.to_q.weight"] = torch.randn(
            HIDDEN, HIDDEN, dtype=torch.bfloat16
        )
        state[f"{prefix}.img_mlp.net.0.proj.weight"] = torch.randn(
            HIDDEN, HIDDEN, dtype=torch.bfloat16
        )
    state["proj_out.weight"] = torch.randn(4, HIDDEN, dtype=torch.bfloat16)
    save_file(state, str(component_dir / weights_name))
    return state


def _write_qwen_image_pipeline(root):
    root.mkdir()
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "QwenImagePipeline"})
    )
    return _write_qwen_image_transformer(
        root / "transformer",
        weights_name="diffusion_pytorch_model.safetensors",
    )


def test_dense_pipeline_auto_descends_into_transformer(tmp_path):
    root = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    state = _write_qwen_image_pipeline(root)

    convert_checkpoint(str(root), str(store_dir))
    index = read_index(store_dir)

    assert not any(g.is_expert for g in index.groups)
    assert index.model_type == "QwenImageTransformer2DModel"
    names = {m.name for _, m in index.iter_members()}
    assert names == set(state)
    for group in index.groups:
        for member in group.members:
            loaded = read_member_tensor(store_dir, group, member)
            assert torch.equal(loaded, state[member.name]), member.name

    summary = inspect_store(str(store_dir))
    assert summary["expert_groups"] == 0
    assert summary["dense_groups"] == len(index.groups)


def test_dense_component_dir_converts_directly(tmp_path):
    component = tmp_path / "ckpt" / "transformer"
    store_dir = tmp_path / "store"
    state = _write_qwen_image_transformer(
        component, weights_name="diffusion_pytorch_model.safetensors"
    )

    convert_checkpoint(str(component), str(store_dir))
    index = read_index(store_dir)

    assert not any(g.is_expert for g in index.groups)
    assert {m.name for _, m in index.iter_members()} == set(state)


def _write_minimax_h3_pipeline(root):
    root.mkdir()
    (root / "modular_model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3Pipeline"})
    )
    states = {}
    for variant in ("FL2VA", "Ref2VA"):
        component = root / variant / "transformer"
        component.mkdir(parents=True)
        config = {
            "_class_name": "MiniMaxH3Transformer3DModel",
            "_diffusers_version": "0.36.0",
            "num_layers": NUM_BLOCKS,
            "num_attention_heads": 4,
            "attention_head_dim": 8,
        }
        (component / "config.json").write_text(json.dumps(config))
        torch.manual_seed(41)
        state = {
            "video_patch_proj.weight": torch.randn(
                HIDDEN, 16, dtype=torch.bfloat16
            )
        }
        for block in range(NUM_BLOCKS):
            prefix = f"blocks.{block}"
            state[f"{prefix}.attn.qkv_proj.weight"] = torch.randn(
                3 * HIDDEN, HIDDEN, dtype=torch.bfloat16
            )
            state[f"{prefix}.mlp.fc1.weight"] = torch.randn(
                HIDDEN, HIDDEN, dtype=torch.bfloat16
            )
        save_file(state, str(component / "model.safetensors"))
        states[variant] = state
    return states


def test_minimax_h3_requires_explicit_subfolder(tmp_path):
    root = tmp_path / "ckpt"
    _write_minimax_h3_pipeline(root)

    with pytest.raises(ValueError, match="FL2VA/transformer"):
        convert_checkpoint(str(root), str(tmp_path / "store"))


def test_minimax_h3_converts_selected_variant(tmp_path):
    root = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    states = _write_minimax_h3_pipeline(root)

    convert_checkpoint(str(root), str(store_dir), subfolder="FL2VA/transformer")
    index = read_index(store_dir)

    assert not any(g.is_expert for g in index.groups)
    assert index.model_type == "MiniMaxH3Transformer3DModel"
    assert {m.name for _, m in index.iter_members()} == set(states["FL2VA"])


def test_cli_convert_accepts_subfolder(tmp_path):
    from moe_store.cli import main

    root = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    _write_minimax_h3_pipeline(root)

    rc = main(
        [
            "convert",
            str(root),
            str(store_dir),
            "--subfolder",
            "FL2VA/transformer",
        ]
    )
    assert rc == 0
    assert read_index(store_dir).model_type == "MiniMaxH3Transformer3DModel"
