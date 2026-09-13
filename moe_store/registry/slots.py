# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Canonical per-architecture expert slot order (invariant G3).

Slot order defines the positional member layout of every expert group in
the v2 store, and therefore the positional ``param_`` order the engine's
fused expert kernels read. It equals the v4 per-expert checkpoint
serialization order, so converting a v4 checkpoint keeps its scan order
and converting a v5 batched checkpoint is normalized to the same layout.
"""

from __future__ import annotations

import re

GPT_OSS_EXPERT_FIELDS = (
    "gate_up_proj_blocks",
    "gate_up_proj_scales",
    "gate_up_proj_bias",
    "down_proj_blocks",
    "down_proj_scales",
    "down_proj_bias",
)

_SLOT_TABLES: dict[str, tuple[str, ...]] = {
    "mixtral": ("w1", "w2", "w3"),
    "gate_up_down": ("gate_proj", "up_proj", "down_proj"),
    "gpt_oss": GPT_OSS_EXPERT_FIELDS,
    "nllb": ("fc1", "fc2"),
}

_EXPERT_SEGMENT = re.compile(r"\bexperts?[._](?:expert_)?(\d+)\.(.+)$")


def slot_table_for_arch(arch: str) -> tuple[str, ...] | None:
    arch = arch.lower()
    if "mixtral" in arch:
        return _SLOT_TABLES["mixtral"]
    if "gpt_oss" in arch or "gptoss" in arch:
        return _SLOT_TABLES["gpt_oss"]
    if "nllb" in arch:
        return _SLOT_TABLES["nllb"]
    if any(
        key in arch
        for key in ("deepseek", "qwen", "olmoe", "glm", "jamba", "dbrx")
    ):
        return _SLOT_TABLES["gate_up_down"]
    return None


def member_slot_rank(arch: str, member_name: str) -> int | None:
    table = slot_table_for_arch(arch)
    if table is None:
        return None
    match = _EXPERT_SEGMENT.search(member_name)
    if match is None:
        return None
    tail = match.group(2)
    first = tail.split(".", 1)[0]
    for rank, slot in enumerate(table):
        if first == slot or tail == slot:
            return rank
    return None
