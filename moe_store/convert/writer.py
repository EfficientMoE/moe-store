# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Write partition files + index for a planned layout (spec §6).

Conversion is atomic per store: everything is written to a sibling
``<store_dir>.tmp-<pid>`` directory and renamed into place on success.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable

import torch

from moe_store.index import StoreIndex, data_file_name, write_index


def write_store(
    index: StoreIndex,
    load_tensor: Callable[[str], torch.Tensor],
    store_dir: str | Path,
) -> Path:
    """``load_tensor(name)`` returns the CPU tensor for a member name."""
    store_dir = Path(store_dir)
    tmp_dir = store_dir.with_name(f"{store_dir.name}.tmp-{os.getpid()}")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    try:
        handles: dict[int, object] = {}

        def file_for(file_id: int):
            if file_id not in handles:
                handles[file_id] = open(tmp_dir / data_file_name(file_id), "wb")
            return handles[file_id]

        for group in index.groups:
            f = file_for(group.file_id)
            for member in group.members:
                tensor = load_tensor(member.name)
                tensor = tensor.detach().contiguous().cpu()
                raw = tensor.view(torch.uint8).view(-1).numpy().tobytes()
                if len(raw) != member.size:
                    raise ValueError(
                        f"size mismatch for {member.name}: planned "
                        f"{member.size}, got {len(raw)}"
                    )
                f.seek(group.offset + member.rel_offset)
                f.write(raw)
            data_end = group.offset + max(
                m.rel_offset + m.size for m in group.members
            )
            end = group.offset + group.total_size
            if end > data_end:
                f.seek(end - 1)
                f.write(b"\x00")

        for f in handles.values():
            f.flush()
            os.fsync(f.fileno())
            f.close()

        write_index(index, tmp_dir)

        if store_dir.exists():
            shutil.rmtree(store_dir)
        os.rename(tmp_dir, store_dir)
        return store_dir
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def read_group_bytes(store_dir: str | Path, group) -> bytes:
    """One-read group fetch; the reference implementation of the reader
    contract (spec §5)."""
    path = Path(store_dir) / data_file_name(group.file_id)
    with open(path, "rb") as f:
        f.seek(group.offset)
        return f.read(group.total_size)


def read_member_tensor(store_dir: str | Path, group, member) -> torch.Tensor:
    blob = read_group_bytes(store_dir, group)
    raw = blob[member.rel_offset : member.rel_offset + member.size]
    dtype = getattr(torch, member.dtype)
    tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    return tensor.view(dtype).view(*member.shape)
