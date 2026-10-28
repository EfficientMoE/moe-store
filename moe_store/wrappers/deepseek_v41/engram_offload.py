# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Host-offloaded engram embedding tables for DeepSeek-V4.1-Flash.

Each table is fp8-e4m3 [N, 256] with ue8m0 scales [N, 8] (group 32 along
the embedding dim) and ~98 GB on disk — far too large to stay resident.
Engram lookups touch only a handful of rows per token (hash-id tensors of
shape [B, L, n_hash_cols]), so the table lives in (ideally pinned) host
memory and ``gather`` moves just the requested rows to the compute device
and dequantizes there, matching the reference
``ParallelEngramEmbedding.forward`` dequant exactly.
"""

from __future__ import annotations

import torch

__all__ = ["EngramHostStore"]


class EngramHostStore:
    def __init__(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        device: torch.device,
        pin: bool = True,
    ):
        if weight.dtype != torch.float8_e4m3fn:
            raise ValueError(f"expected fp8_e4m3 table, got {weight.dtype}")
        if scale.dtype != torch.float8_e8m0fnu:
            raise ValueError(f"expected ue8m0 scales, got {scale.dtype}")
        if weight.shape[0] != scale.shape[0]:
            raise ValueError("weight/scale row count mismatch")
        if weight.shape[1] % scale.shape[1] != 0:
            raise ValueError("embedding dim not divisible by scale groups")
        self.device = torch.device(device)
        self.group = weight.shape[1] // scale.shape[1]
        if pin and self.device.type == "cuda":
            weight = weight.pin_memory()
            scale = scale.pin_memory()
        self._weight = weight
        self._scale = scale

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        layer_id: int,
        device: torch.device,
        pin: bool = True,
    ) -> "EngramHostStore":
        import json
        import os

        from safetensors import safe_open

        index_path = os.path.join(ckpt_path, "model.safetensors.index.json")
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        wkey = f"layers.{layer_id}.engram.embed.weight"
        skey = f"layers.{layer_id}.engram.embed.scale"
        tensors = {}
        for key in (wkey, skey):
            shard = os.path.join(ckpt_path, weight_map[key])
            with safe_open(shard, framework="pt", device="cpu") as f:
                tensors[key] = f.get_tensor(key)
        return cls(tensors[wkey], tensors[skey], device, pin=pin)

    def gather(
        self, indices: torch.Tensor, dtype: torch.dtype = torch.bfloat16
    ) -> torch.Tensor:
        """Dequantized rows for ``indices`` of any shape: returns
        ``[*indices.shape, dim]`` on ``self.device``."""
        flat = indices.reshape(-1).cpu()
        rows = self._weight.index_select(0, flat)
        srows = self._scale.index_select(0, flat)
        rows = rows.to(self.device, non_blocking=True)
        srows = srows.to(self.device, non_blocking=True)
        n, dim = rows.shape
        out = rows.to(torch.float32).view(
            n, dim // self.group, self.group
        ) * srows.to(torch.float32)[..., None]
        return out.view(*indices.shape, dim).to(dtype)
