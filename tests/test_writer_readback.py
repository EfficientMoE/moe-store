# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import re

import torch

from moe_store.convert.planner import TensorSpec, dtype_token, plan_layout
from moe_store.convert.writer import (
    read_group_bytes,
    read_member_tensor,
    write_store,
)
from moe_store.index import read_index


def _make_state_dict() -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    state = {"model.embed_tokens.weight": torch.randn(64, 32)}
    for layer in range(2):
        state[f"model.layers.{layer}.self_attn.q_proj.weight"] = torch.randn(
            32, 32
        )
        for expert in range(3):
            for slot in ("gate_proj", "up_proj", "down_proj"):
                key = f"model.layers.{layer}.mlp.experts.{expert}.{slot}.weight"
                state[key] = torch.randn(16, 32, dtype=torch.bfloat16)
    state["lm_head.weight"] = torch.randn(64, 32)
    return state


def _expert_of(name):
    match = re.search(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.", name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def _plan_and_write(tmp_path):
    state = _make_state_dict()
    specs = [
        TensorSpec(
            name,
            tensor.numel() * tensor.element_size(),
            dtype_token(tensor.dtype),
            tuple(tensor.shape),
        )
        for name, tensor in state.items()
    ]
    index = plan_layout(
        specs,
        _expert_of,
        model_type="synthetic",
        checkpoint_name="synthetic/writer-test",
    )
    store_dir = tmp_path / "store"
    write_store(index, lambda name: state[name], store_dir)
    return state, store_dir


def test_every_member_reads_back_byte_exact(tmp_path):
    state, store_dir = _plan_and_write(tmp_path)
    index = read_index(store_dir)
    seen = set()
    for group, member in index.iter_members():
        loaded = read_member_tensor(store_dir, group, member)
        original = state[member.name]
        assert loaded.dtype == original.dtype
        assert tuple(loaded.shape) == tuple(original.shape)
        assert torch.equal(loaded, original), member.name
        seen.add(member.name)
    assert seen == set(state)


def test_expert_group_is_single_read(tmp_path):
    state, store_dir = _plan_and_write(tmp_path)
    index = read_index(store_dir)
    group = index.expert_group(layer_id=1, expert_id=2)
    blob = read_group_bytes(store_dir, group)
    assert len(blob) == group.total_size
    for member in group.members:
        raw = blob[member.rel_offset : member.rel_offset + member.size]
        original = state[member.name]
        expected = (
            original.contiguous().view(torch.uint8).view(-1).numpy().tobytes()
        )
        assert raw == expected, member.name


def test_conversion_is_atomic_on_failure(tmp_path):
    state, _ = _plan_and_write(tmp_path)

    def broken_loader(name):
        raise RuntimeError("simulated shard corruption")

    specs = [
        TensorSpec(
            name,
            tensor.numel() * tensor.element_size(),
            dtype_token(tensor.dtype),
            tuple(tensor.shape),
        )
        for name, tensor in state.items()
    ]
    index = plan_layout(
        specs,
        _expert_of,
        model_type="synthetic",
        checkpoint_name="synthetic/atomic-test",
    )
    target = tmp_path / "broken-store"
    try:
        write_store(index, broken_loader, target)
    except RuntimeError:
        pass
    assert not target.exists()
    assert not list(tmp_path.glob("broken-store.tmp-*"))
