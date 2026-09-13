# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""G1-G5 property tests over synthetic per-architecture checkpoints.

Each synthetic checkpoint mirrors the real tensor-name patterns matched by
``parse_expert_id`` so the planner is exercised exactly as in conversion.
"""

import pytest

from moe_store.convert.planner import TensorSpec, plan_layout
from moe_store.index import (
    GROUP_ALIGNMENT,
    KIND_EXPERT,
    MEMBER_ALIGNMENT,
    validate_index,
)

ARCH_CASES = {
    "mixtral": {
        "expert_name": (
            "model.layers.{L}.block_sparse_moe.experts.{E}.{S}.weight"
        ),
        "slots": ("w1", "w2", "w3"),
        "regex_probe": "model.layers.0.block_sparse_moe.experts.0.w1.weight",
    },
    "qwen3_deepseek_olmoe": {
        "expert_name": "model.layers.{L}.mlp.experts.{E}.{S}.weight",
        "slots": ("gate_proj", "up_proj", "down_proj"),
        "regex_probe": "model.layers.0.mlp.experts.0.gate_proj.weight",
    },
    "gpt_oss_quantized": {
        "expert_name": "model.layers.{L}.mlp.experts.{E}.{S}",
        "slots": (
            "gate_up_proj_blocks",
            "gate_up_proj_scales",
            "gate_up_proj_bias",
            "down_proj_blocks",
            "down_proj_scales",
            "down_proj_bias",
        ),
        "regex_probe": "model.layers.0.mlp.experts.0.gate_up_proj_blocks",
    },
}

NUM_LAYERS = 2
NUM_EXPERTS = 4


def _synthetic_specs(case) -> list[TensorSpec]:
    specs = [
        TensorSpec("model.embed_tokens.weight", 4096, "bfloat16", (64, 32)),
    ]
    for layer in range(NUM_LAYERS):
        specs.append(
            TensorSpec(
                f"model.layers.{layer}.self_attn.q_proj.weight",
                2048,
                "bfloat16",
                (32, 32),
            )
        )
        specs.append(
            TensorSpec(
                f"model.layers.{layer}.input_layernorm.weight",
                64,
                "bfloat16",
                (32,),
            )
        )
        for expert in range(NUM_EXPERTS):
            for slot_idx, slot in enumerate(case["slots"]):
                name = case["expert_name"].format(L=layer, E=expert, S=slot)
                size = 1000 + 8 * slot_idx
                specs.append(TensorSpec(name, size, "uint8", (size,)))
    specs.append(TensorSpec("lm_head.weight", 4096, "bfloat16", (64, 32)))
    return specs


def _expert_of_factory(case):
    import re

    pattern = re.compile(r"layers\.(\d+)\.\S*experts\.(\d+)\.")

    def expert_of(name):
        match = pattern.search(name)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None

    return expert_of


@pytest.fixture(params=sorted(ARCH_CASES), name="case")
def _case(request):
    return ARCH_CASES[request.param]


def _plan(case, partition_size=None):
    kwargs = {}
    if partition_size is not None:
        kwargs["partition_size"] = partition_size
    return plan_layout(
        _synthetic_specs(case),
        _expert_of_factory(case),
        model_type="synthetic",
        checkpoint_name="synthetic/test",
        **kwargs,
    )


def test_planner_passes_validation(case):
    validate_index(_plan(case))


def test_one_group_per_expert(case):
    index = _plan(case)
    experts = [g for g in index.groups if g.kind == KIND_EXPERT]
    assert len(experts) == NUM_LAYERS * NUM_EXPERTS
    keys = {(g.layer_id, g.expert_id) for g in experts}
    assert len(keys) == NUM_LAYERS * NUM_EXPERTS


def test_g3_slot_order_preserved_never_sorted(case):
    index = _plan(case)
    for group in index.groups:
        if group.kind != KIND_EXPERT:
            continue
        got_slots = []
        for member in group.members:
            for slot in case["slots"]:
                if (
                    f".{slot}"
                    in member.name.replace(f".experts.{group.expert_id}.", ".")
                    or member.name.endswith(slot)
                    or f".{slot}." in member.name
                ):
                    got_slots.append(slot)
                    break
        assert got_slots == list(case["slots"]), (
            f"slot order broken for layer={group.layer_id} "
            f"expert={group.expert_id}: {got_slots}"
        )


def test_g4_quantized_members_share_group(case):
    if "blocks" not in "".join(case["slots"]):
        pytest.skip("non-quantized arch")
    index = _plan(case)
    for group in index.groups:
        if group.kind != KIND_EXPERT:
            continue
        names = [m.name for m in group.members]
        assert any("blocks" in n for n in names)
        assert any("scales" in n for n in names)


def test_g1_partition_roll_keeps_groups_whole(case):
    tiny_partition = 65536
    index = _plan(case, partition_size=tiny_partition)
    for group in index.groups:
        assert group.offset + group.total_size <= tiny_partition
    assert len({g.file_id for g in index.groups}) > 1


def test_g2_alignment(case):
    index = _plan(case)
    for group in index.groups:
        assert group.offset % GROUP_ALIGNMENT == 0
        for member in group.members:
            assert member.rel_offset % MEMBER_ALIGNMENT == 0


def test_g5_dense_ids_in_group_order(case):
    index = _plan(case)
    next_id = 0
    for group in index.groups:
        for member in group.members:
            assert member.tensor_id == next_id
            next_id += 1


def test_group_read_covers_all_members(case):
    index = _plan(case)
    for group in index.groups:
        for member in group.members:
            assert member.rel_offset + member.size <= group.total_size
