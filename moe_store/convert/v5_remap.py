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


class GptOssExpansion:
    """Lazy per-expert expansion of GPT-OSS packed MXFP4 expert tensors,
    mirroring the engine's legacy ``_expand_gpt_oss_packed_experts``:
    each of the six packed components ``[E, ...]`` expands independently
    per expert, and 3D ``_blocks`` payloads ``[out, n_blocks, 16]``
    collapse to 2D ``[out, n_blocks*16]`` as the fused MoEMLP path
    expects."""

    def __init__(self, config, spec_of: Callable[[str], TensorSpec]):
        from moe_store.registry.slots import GPT_OSS_EXPERT_FIELDS

        self._enabled = getattr(config, "model_type", "") == "gpt_oss"
        self._fields = set(GPT_OSS_EXPERT_FIELDS)
        self._num_experts = (
            int(getattr(config, "num_local_experts", 0)) if self._enabled else 0
        )
        self._spec_of = spec_of
        self._virtual: dict[str, tuple[str, int]] = {}

    def expand_names(self, names: list[str]) -> list[str]:
        if not self._enabled:
            return names
        out: list[str] = []
        for name in names:
            field = name.rsplit(".", 1)[-1]
            if ".mlp.experts." not in name or field not in self._fields:
                out.append(name)
                continue
            spec = self._spec_of(name)
            if spec.shape[0] != self._num_experts:
                raise ValueError(
                    f"{name} has {spec.shape[0]} experts; expected "
                    f"{self._num_experts}"
                )
            prefix = name.rsplit(".", 1)[0]
            for expert in range(self._num_experts):
                virtual = f"{prefix}.{expert}.{field}"
                self._virtual[virtual] = (name, expert)
                out.append(virtual)
        return out

    def spec(self, name: str) -> TensorSpec | None:
        entry = self._virtual.get(name)
        if entry is None:
            return None
        source, _expert = entry
        src = self._spec_of(source)
        shape = tuple(src.shape[1:])
        if name.endswith("_blocks") and len(shape) == 3:
            shape = (shape[0], shape[1] * shape[2])
        return TensorSpec(name, src.nbytes // src.shape[0], src.dtype, shape)

    def load(
        self, name: str, base_load: Callable[[str], torch.Tensor]
    ) -> torch.Tensor | None:
        entry = self._virtual.get(name)
        if entry is None:
            return None
        source, expert = entry
        view = base_load(source)[expert]
        if name.endswith("_blocks") and view.dim() == 3:
            view = view.reshape(view.shape[0], -1)
        return view.contiguous()
