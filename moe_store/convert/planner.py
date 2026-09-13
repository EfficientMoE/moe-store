# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Arch-aware layout planner: tensors -> groups -> on-disk placement.

Grouping follows the engine topology convention (one group per topology
node): expert tensors group by ``(layer_id, expert_id)`` from
``parse_expert_id`` preserving checkpoint order (invariant G3); every other
tensor groups by its module prefix (``name.rsplit('.', 1)[0]`` after
stripping a trailing numeric component, matching the engine's stage
derivation). Placement enforces G1/G2/G5 and is validated again by
``moe_store.index.validate_index``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from moe_store.index import (
    DEFAULT_PARTITION_SIZE,
    END_OF_PIPELINE,
    GROUP_ALIGNMENT,
    KIND_DENSE,
    KIND_EXPERT,
    MEMBER_ALIGNMENT,
    GroupMeta,
    MemberMeta,
    StageMeta,
    StoreIndex,
)

_TORCH_DTYPE_TOKENS = {
    "torch.float32": "float32",
    "torch.float16": "float16",
    "torch.bfloat16": "bfloat16",
    "torch.float8_e4m3fn": "float8_e4m3fn",
    "torch.uint8": "uint8",
    "torch.int8": "int8",
    "torch.int32": "int32",
    "torch.int64": "int64",
}


@dataclass(frozen=True)
class TensorSpec:
    """Input to the planner: one checkpoint tensor's metadata."""

    name: str
    nbytes: int
    dtype: str
    shape: tuple[int, ...]


def dtype_token(torch_dtype) -> str:
    token = _TORCH_DTYPE_TOKENS.get(str(torch_dtype))
    if token is None:
        raise ValueError(f"unsupported dtype {torch_dtype}")
    return token


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _dense_stage_name(name: str) -> str:
    matches = [m for m in re.finditer(r"\d", name)]
    if matches:
        return name[: matches[-1].start() + 1]
    return name.rsplit(".", 1)[0]


def plan_layout(
    tensors: Iterable[TensorSpec],
    expert_of,
    *,
    model_type: str,
    checkpoint_name: str,
    partition_size: int = DEFAULT_PARTITION_SIZE,
    slot_rank=None,
) -> StoreIndex:
    """Build a validated v2 StoreIndex.

    ``expert_of(name) -> (layer_id, expert_id) | (None, None)`` is
    typically ``functools.partial(parse_expert_id, config=config)``.
    ``slot_rank(name) -> int | None`` optionally supplies the canonical
    G3 slot rank; expert members sort stably by it (scan order breaks
    ties and is the fallback), normalizing v4 and v5 checkpoints to the
    same positional layout.
    """
    expert_groups: dict[tuple[int, int], list[TensorSpec]] = {}
    stage_order: list[tuple[str, bool, object]] = []
    dense_groups: dict[str, list[TensorSpec]] = {}
    expert_stage_of_layer: dict[int, str] = {}

    for spec in tensors:
        layer_id, expert_id = expert_of(spec.name)
        if layer_id is not None and expert_id is not None:
            key = (layer_id, expert_id)
            if key not in expert_groups:
                expert_groups[key] = []
            expert_groups[key].append(spec)
            if layer_id not in expert_stage_of_layer:
                stage_name = spec.name.split(f".{expert_id}.")[0]
                expert_stage_of_layer[layer_id] = stage_name
                stage_order.append((stage_name, True, layer_id))
        else:
            stage_name = _dense_stage_name(spec.name)
            if stage_name not in dense_groups:
                dense_groups[stage_name] = []
                stage_order.append((stage_name, False, stage_name))
            dense_groups[stage_name].append(spec)

    if not stage_order:
        raise ValueError("no tensors to plan")

    stages: list[StageMeta] = []
    groups: list[GroupMeta] = []
    tensor_id = 0
    file_id = 0
    file_offset = 0

    def place_group(
        specs: list[TensorSpec],
        *,
        stage_idx: int,
        group_idx: int,
        is_last_stage: bool,
        kind: int,
        layer_id: int,
        expert_id: int,
    ) -> GroupMeta:
        nonlocal tensor_id, file_id, file_offset
        members: list[MemberMeta] = []
        cursor = 0
        for spec in specs:
            rel = _align(cursor, MEMBER_ALIGNMENT)
            members.append(
                MemberMeta(
                    tensor_id,
                    spec.name,
                    rel,
                    spec.nbytes,
                    spec.dtype,
                    spec.shape,
                )
            )
            tensor_id += 1
            cursor = rel + spec.nbytes
        total_size = _align(cursor, GROUP_ALIGNMENT)
        if total_size > partition_size:
            raise ValueError(
                f"group of {len(specs)} tensors ({total_size} bytes) exceeds "
                f"partition size {partition_size}; raise --partition-size"
            )
        if file_offset + total_size > partition_size:
            file_id += 1
            file_offset = 0
        high = END_OF_PIPELINE if is_last_stage else group_idx
        group = GroupMeta(
            group_id=(high << 32) | stage_idx,
            kind=kind,
            layer_id=layer_id,
            expert_id=expert_id,
            file_id=file_id,
            offset=file_offset,
            total_size=total_size,
            members=tuple(members),
        )
        file_offset += total_size
        return group

    num_stages = len(stage_order)
    for stage_idx, (stage_name, is_sparse, stage_key) in enumerate(stage_order):
        is_last = stage_idx == num_stages - 1
        if is_sparse:
            layer_id = stage_key
            expert_ids = sorted(
                eid for (lid, eid) in expert_groups if lid == layer_id
            )
            for group_idx, expert_id in enumerate(expert_ids):
                members = expert_groups[(layer_id, expert_id)]
                if slot_rank is not None:
                    ranked = [
                        (slot_rank(spec.name), idx)
                        for idx, spec in enumerate(members)
                    ]
                    if all(rank is not None for rank, _ in ranked):
                        members = [members[idx] for _, idx in sorted(ranked)]
                groups.append(
                    place_group(
                        members,
                        stage_idx=stage_idx,
                        group_idx=group_idx,
                        is_last_stage=is_last,
                        kind=KIND_EXPERT,
                        layer_id=layer_id,
                        expert_id=expert_id,
                    )
                )
            stages.append(StageMeta(stage_name, True, len(expert_ids)))
        else:
            groups.append(
                place_group(
                    dense_groups[stage_key],
                    stage_idx=stage_idx,
                    group_idx=0,
                    is_last_stage=is_last,
                    kind=KIND_DENSE,
                    layer_id=-1,
                    expert_id=-1,
                )
            )
            stages.append(StageMeta(stage_name, False, 1))

    index = StoreIndex(
        model_type=model_type,
        checkpoint_name=checkpoint_name,
        partition_size=partition_size,
        stages=stages,
        groups=groups,
    )
    from moe_store.index import validate_index

    validate_index(index)
    return index
