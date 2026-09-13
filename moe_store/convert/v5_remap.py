# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Lazy per-expert expansion of transformers-v5 batched expert tensors.

Mirrors ``_remap_v5_batched_experts`` in MoE-Infinity's model_offload:
v5 checkpoints store experts batched as 3D ``...experts.gate_up_proj``
``[E, 2*inter, H]`` and ``...experts.down_proj`` ``[E, *, *]`` tensors.
The store is per-expert, so these expand into virtual per-expert tensor
names; slicing happens at load time instead of materializing the whole
state dict. Mixtral renames ``.mlp`` -> ``.block_sparse_moe`` with
``w1/w3/w2`` names (w1=gate, w3=up, w2=down) and relocates the router
gate key; other architectures keep ``gate_proj/up_proj/down_proj``.
GPT-OSS keeps its native batched MXFP4 layout and is excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from moe_store.convert.planner import TensorSpec


@dataclass(frozen=True)
class _VirtualMember:
    source: str
    expert_idx: int
    part: str


def _arch_name(config) -> str:
    archs = getattr(config, "architectures", None) or [""]
    name = (archs[0] or "").lower()
    if not name:
        name = (getattr(config, "model_type", "") or "").lower()
    return name


class V5Expansion:
    def __init__(self, config, spec_of: Callable[[str], TensorSpec]):
        arch = _arch_name(config)
        self._enabled = "gpt_oss" not in arch and "gptoss" not in arch
        self._is_mixtral = "mixtral" in arch
        if self._is_mixtral:
            self._gate, self._up, self._down = "w1", "w3", "w2"
        else:
            self._gate, self._up, self._down = (
                "gate_proj",
                "up_proj",
                "down_proj",
            )
        self._spec_of = spec_of
        self._virtual: dict[str, _VirtualMember] = {}
        self._renamed: dict[str, str] = {}

    def _out_block(self, experts_prefix: str) -> str:
        block_prefix = experts_prefix[: -len(".experts")]
        if self._is_mixtral and block_prefix.endswith(".mlp"):
            return block_prefix[: -len(".mlp")] + ".block_sparse_moe"
        return block_prefix

    def expand_names(self, names: list[str]) -> list[str]:
        if not self._enabled:
            return names
        out: list[str] = []
        for name in names:
            if name.endswith("experts.gate_up_proj"):
                spec = self._spec_of(name)
                if len(spec.shape) != 3:
                    out.append(name)
                    continue
                block = self._out_block(name[: -len(".gate_up_proj")])
                for expert in range(spec.shape[0]):
                    base = f"{block}.experts.{expert}"
                    gate = f"{base}.{self._gate}.weight"
                    up = f"{base}.{self._up}.weight"
                    self._virtual[gate] = _VirtualMember(name, expert, "gate")
                    self._virtual[up] = _VirtualMember(name, expert, "up")
                    out.extend((gate, up))
            elif name.endswith("experts.down_proj"):
                spec = self._spec_of(name)
                if len(spec.shape) != 3:
                    out.append(name)
                    continue
                block = self._out_block(name[: -len(".down_proj")])
                for expert in range(spec.shape[0]):
                    down = f"{block}.experts.{expert}.{self._down}.weight"
                    self._virtual[down] = _VirtualMember(name, expert, "down")
                    out.append(down)
            elif (
                self._is_mixtral
                and name.endswith(".mlp.gate.weight")
                and any(
                    n.endswith("experts.gate_up_proj")
                    and n.startswith(name[: -len(".gate.weight")])
                    for n in names
                )
            ):
                renamed = (
                    name[: -len(".mlp.gate.weight")]
                    + ".block_sparse_moe.gate.weight"
                )
                self._renamed[renamed] = name
                out.append(renamed)
            else:
                out.append(name)
        return out

    def spec(self, name: str) -> TensorSpec | None:
        member = self._virtual.get(name)
        if member is None:
            source = self._renamed.get(name)
            if source is None:
                return None
            src = self._spec_of(source)
            return TensorSpec(name, src.nbytes, src.dtype, src.shape)
        src = self._spec_of(member.source)
        expert_count, *rest = src.shape
        if member.part == "down":
            shape = tuple(rest)
        else:
            shape = (rest[0] // 2, *rest[1:])
        nbytes = src.nbytes // expert_count
        if member.part != "down":
            nbytes //= 2
        return TensorSpec(name, nbytes, src.dtype, shape)

    def load(
        self, name: str, base_load: Callable[[str], torch.Tensor]
    ) -> torch.Tensor | None:
        member = self._virtual.get(name)
        if member is None:
            source = self._renamed.get(name)
            if source is None:
                return None
            return base_load(source)
        source = base_load(member.source)
        row = source[member.expert_idx]
        if member.part == "down":
            return row.contiguous()
        gate, up = row.chunk(2, dim=0)
        return (gate if member.part == "gate" else up).contiguous()
