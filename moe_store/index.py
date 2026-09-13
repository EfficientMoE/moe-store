# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Binary reader/writer for the v2 store index (docs/store-format-v2.md §4).

This module is the Python reference implementation of the format; the C++
reader in ``csrc/store`` must stay byte-compatible with it.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterator

MAGIC = b"MOESTOR2"
VERSION = 2
DEFAULT_PARTITION_SIZE = 10 * 1024 * 1024 * 1024
GROUP_ALIGNMENT = 4096
MEMBER_ALIGNMENT = 4096
MEMBER_ALIGNMENT_MIN = 64
END_OF_PIPELINE = 0xFFFFFFFF

INDEX_FILE_NAME = "store_index"
DATA_FILE_PREFIX = "store_data_"

DTYPE_TOKENS = (
    "float32",
    "float16",
    "bfloat16",
    "float8_e4m3fn",
    "uint8",
    "int8",
    "int32",
    "int64",
)

KIND_DENSE = 0
KIND_EXPERT = 1


@dataclass(frozen=True)
class MemberMeta:
    tensor_id: int
    name: str
    rel_offset: int
    size: int
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class GroupMeta:
    group_id: int
    kind: int
    layer_id: int
    expert_id: int
    file_id: int
    offset: int
    total_size: int
    members: tuple[MemberMeta, ...]

    @property
    def is_expert(self) -> bool:
        return self.kind == KIND_EXPERT


@dataclass(frozen=True)
class StageMeta:
    name: str
    is_sparse: bool
    num_groups: int


@dataclass
class StoreIndex:
    model_type: str
    checkpoint_name: str
    partition_size: int = DEFAULT_PARTITION_SIZE
    stages: list[StageMeta] = field(default_factory=list)
    groups: list[GroupMeta] = field(default_factory=list)

    @property
    def num_tensors(self) -> int:
        return sum(len(g.members) for g in self.groups)

    def group_of_tensor(self, tensor_id: int) -> GroupMeta:
        for group in self.groups:
            for member in group.members:
                if member.tensor_id == tensor_id:
                    return group
        raise KeyError(tensor_id)

    def expert_group(self, layer_id: int, expert_id: int) -> GroupMeta:
        for group in self.groups:
            if (
                group.kind == KIND_EXPERT
                and group.layer_id == layer_id
                and group.expert_id == expert_id
            ):
                return group
        raise KeyError((layer_id, expert_id))

    def iter_members(self) -> Iterator[tuple[GroupMeta, MemberMeta]]:
        for group in self.groups:
            for member in group.members:
                yield group, member


def data_file_name(file_id: int) -> str:
    return f"{DATA_FILE_PREFIX}{file_id}"


def _write_str(buf: BinaryIO, value: str) -> None:
    raw = value.encode("utf-8")
    if len(raw) > 0xFFFF:
        raise ValueError("string too long for index encoding")
    buf.write(struct.pack("<H", len(raw)))
    buf.write(raw)


def _read_str(buf: BinaryIO) -> str:
    (length,) = struct.unpack("<H", buf.read(2))
    return buf.read(length).decode("utf-8")


def write_index(index: StoreIndex, store_dir: str | Path) -> Path:
    validate_index(index)
    buf = io.BytesIO()
    buf.write(MAGIC)
    buf.write(struct.pack("<II", VERSION, 0))
    buf.write(struct.pack("<Q", index.partition_size))
    _write_str(buf, index.model_type)
    _write_str(buf, index.checkpoint_name)
    buf.write(
        struct.pack(
            "<IQQ", len(index.stages), len(index.groups), index.num_tensors
        )
    )
    for stage in index.stages:
        _write_str(buf, stage.name)
        buf.write(struct.pack("<BI", int(stage.is_sparse), stage.num_groups))
    for group in index.groups:
        buf.write(
            struct.pack(
                "<QBiiIqqI",
                group.group_id,
                group.kind,
                group.layer_id,
                group.expert_id,
                group.file_id,
                group.offset,
                group.total_size,
                len(group.members),
            )
        )
        for member in group.members:
            buf.write(struct.pack("<I", member.tensor_id))
            _write_str(buf, member.name)
            buf.write(struct.pack("<qq", member.rel_offset, member.size))
            _write_str(buf, member.dtype)
            buf.write(struct.pack("<B", len(member.shape)))
            for dim in member.shape:
                buf.write(struct.pack("<q", dim))

    path = Path(store_dir) / INDEX_FILE_NAME
    path.write_bytes(buf.getvalue())
    return path


def read_index(store_dir: str | Path) -> StoreIndex:
    path = Path(store_dir) / INDEX_FILE_NAME
    buf = io.BytesIO(path.read_bytes())
    magic = buf.read(8)
    if magic != MAGIC:
        raise ValueError(f"not a moe-store v2 index: bad magic {magic!r}")
    version, _flags = struct.unpack("<II", buf.read(8))
    if version != VERSION:
        raise ValueError(f"unsupported index version {version}")
    (partition_size,) = struct.unpack("<Q", buf.read(8))
    model_type = _read_str(buf)
    checkpoint_name = _read_str(buf)
    num_stages, num_groups, num_tensors = struct.unpack("<IQQ", buf.read(20))

    stages = []
    for _ in range(num_stages):
        name = _read_str(buf)
        is_sparse, stage_groups = struct.unpack("<BI", buf.read(5))
        stages.append(StageMeta(name, bool(is_sparse), stage_groups))

    groups = []
    for _ in range(num_groups):
        (
            group_id,
            kind,
            layer_id,
            expert_id,
            file_id,
            offset,
            total_size,
            num_members,
        ) = struct.unpack("<QBiiIqqI", buf.read(41))
        members = []
        for _ in range(num_members):
            (tensor_id,) = struct.unpack("<I", buf.read(4))
            name = _read_str(buf)
            rel_offset, size = struct.unpack("<qq", buf.read(16))
            dtype = _read_str(buf)
            (ndim,) = struct.unpack("<B", buf.read(1))
            shape = struct.unpack(f"<{ndim}q", buf.read(8 * ndim))
            members.append(
                MemberMeta(tensor_id, name, rel_offset, size, dtype, shape)
            )
        groups.append(
            GroupMeta(
                group_id,
                kind,
                layer_id,
                expert_id,
                file_id,
                offset,
                total_size,
                tuple(members),
            )
        )

    index = StoreIndex(
        model_type=model_type,
        checkpoint_name=checkpoint_name,
        partition_size=partition_size,
        stages=stages,
        groups=groups,
    )
    if index.num_tensors != num_tensors:
        raise ValueError("index tensor count mismatch")
    validate_index(index)
    return index


def validate_index(index: StoreIndex) -> None:
    """Enforce invariants G1, G2, and G5 (docs/store-format-v2.md §3)."""
    expected_tensor_id = 0
    for group in index.groups:
        if group.offset % GROUP_ALIGNMENT != 0:
            raise ValueError(f"G2 violation: group {group.group_id:#x} offset")
        if group.offset + group.total_size > index.partition_size:
            raise ValueError(
                f"G1 violation: group {group.group_id:#x} straddles partition"
            )
        if not group.members:
            raise ValueError(f"empty group {group.group_id:#x}")
        cursor = 0
        for member in group.members:
            if member.rel_offset % MEMBER_ALIGNMENT_MIN != 0:
                raise ValueError(
                    f"G2 violation: member {member.name} rel_offset"
                )
            if member.rel_offset < cursor:
                raise ValueError(f"member overlap in group {group.group_id:#x}")
            cursor = member.rel_offset + member.size
            if member.tensor_id != expected_tensor_id:
                raise ValueError(
                    "G5 violation: tensor ids not dense in group order "
                    f"(expected {expected_tensor_id}, got {member.tensor_id} "
                    f"for {member.name})"
                )
            expected_tensor_id += 1
            if member.dtype not in DTYPE_TOKENS:
                raise ValueError(f"unknown dtype token {member.dtype!r}")
        if cursor > group.total_size:
            raise ValueError(
                f"members exceed total_size in group {group.group_id:#x}"
            )
