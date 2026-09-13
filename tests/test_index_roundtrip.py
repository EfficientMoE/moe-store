# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import pytest

from moe_store.index import (
    GROUP_ALIGNMENT,
    KIND_DENSE,
    KIND_EXPERT,
    GroupMeta,
    MemberMeta,
    StageMeta,
    StoreIndex,
    read_index,
    validate_index,
    write_index,
)


def _tiny_index() -> StoreIndex:
    groups = [
        GroupMeta(
            group_id=(0 << 32) | 0,
            kind=KIND_DENSE,
            layer_id=-1,
            expert_id=-1,
            file_id=0,
            offset=0,
            total_size=GROUP_ALIGNMENT,
            members=(
                MemberMeta(0, "model.embed.weight", 0, 128, "bfloat16", (8, 8)),
                MemberMeta(1, "model.norm.weight", 192, 16, "bfloat16", (8,)),
            ),
        ),
        GroupMeta(
            group_id=(0xFFFFFFFF << 32) | 1,
            kind=KIND_EXPERT,
            layer_id=0,
            expert_id=0,
            file_id=0,
            offset=GROUP_ALIGNMENT,
            total_size=GROUP_ALIGNMENT,
            members=(
                MemberMeta(
                    2, "l0.experts.0.w1.weight", 0, 64, "bfloat16", (4, 8)
                ),
                MemberMeta(
                    3, "l0.experts.0.w2.weight", 64, 64, "bfloat16", (8, 4)
                ),
                MemberMeta(
                    4, "l0.experts.0.w3.weight", 128, 64, "bfloat16", (4, 8)
                ),
            ),
        ),
    ]
    return StoreIndex(
        model_type="mixtral",
        checkpoint_name="tiny/test",
        stages=[
            StageMeta("model.embed", False, 1),
            StageMeta("l0.experts", True, 1),
        ],
        groups=groups,
    )


def test_roundtrip_bytes_equal(tmp_path):
    index = _tiny_index()
    write_index(index, tmp_path)
    loaded = read_index(tmp_path)
    assert loaded == index


def test_rejects_bad_magic(tmp_path):
    write_index(_tiny_index(), tmp_path)
    path = tmp_path / "store_index"
    data = bytearray(path.read_bytes())
    data[:8] = b"BADMAGIC"
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="magic"):
        read_index(tmp_path)


def test_validate_rejects_unaligned_group():
    index = _tiny_index()
    bad = index.groups[1]
    index.groups[1] = GroupMeta(
        bad.group_id,
        bad.kind,
        bad.layer_id,
        bad.expert_id,
        bad.file_id,
        bad.offset + 17,
        bad.total_size,
        bad.members,
    )
    with pytest.raises(ValueError, match="G2"):
        validate_index(index)


def test_validate_rejects_partition_straddle():
    index = _tiny_index()
    index.partition_size = GROUP_ALIGNMENT
    with pytest.raises(ValueError, match="G1"):
        validate_index(index)


def test_validate_rejects_nondense_tensor_ids():
    index = _tiny_index()
    good = index.groups[0]
    index.groups[0] = GroupMeta(
        good.group_id,
        good.kind,
        good.layer_id,
        good.expert_id,
        good.file_id,
        good.offset,
        good.total_size,
        (
            good.members[0],
            MemberMeta(7, "model.norm.weight", 192, 16, "bfloat16", (8,)),
        ),
    )
    with pytest.raises(ValueError, match="G5"):
        validate_index(index)
