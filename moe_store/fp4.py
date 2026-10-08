# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""MXFP4 (E2M1 + per-32 E8M0 scales) quant/dequant for the FP4 group
format (fp4_format_version 1).

Layout matches the fused weight-only GEMM consumers (Omni-Infinity
``omni_infinity/kernels``): two FP4 values per uint8 along K (low
nibble = even K index, high nibble = odd), sign in nibble bit 3, and
one uint8 E8M0 scale (power-of-two exponent ``byte - 127``) per
``MXFP4_BLOCK`` consecutive K elements.
"""

from __future__ import annotations

import torch

MXFP4_BLOCK = 32

# E2M1 magnitudes (sign lives in nibble bit 3) and the midpoints between
# adjacent magnitudes used for nearest-value rounding.
_MXFP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_MXFP4_MIDPOINTS = torch.tensor(
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32
)
_MXFP4_LUT = torch.tensor(
    [float(m) for m in _MXFP4_MAGNITUDES]
    + [-float(m) for m in _MXFP4_MAGNITUDES],
    dtype=torch.float32,
)


def quant_mxfp4(
    weight: torch.Tensor, block_size: int = MXFP4_BLOCK
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a [N, K] weight to MXFP4 (E2M1 nibbles, E8M0 scales).

    Returns ``(packed [N, K//2] uint8, scales [N, K//block_size] uint8)``.
    Quantize from the ORIGINAL hi-precision weight.
    """
    if weight.dim() != 2:
        raise ValueError("MXFP4 expert weights must be 2D")
    n, k = weight.shape
    if block_size % 2 != 0 or k % block_size != 0:
        raise ValueError(
            f"MXFP4 requires K divisible by {block_size}: got K={k}"
        )
    w = weight.detach().to(torch.float32)
    tiny = torch.finfo(torch.float32).tiny
    blocks = w.reshape(n, k // block_size, block_size)
    amax = blocks.abs().amax(dim=2)  # [N, K//block_size]
    e = torch.ceil(torch.log2(amax.clamp_min(tiny) / 6.0))
    e = torch.where(amax > 0, e, torch.zeros_like(e))
    e = e.clamp(-126.0, 127.0)
    # Guard against log2 rounding leaving amax just above 6 * 2^e.
    bump = amax > 6.0 * torch.exp2(e)
    e = torch.where(bump, (e + 1).clamp(max=127.0), e)
    scale = torch.exp2(e)
    normalized = blocks / scale.unsqueeze(2)
    mids = _MXFP4_MIDPOINTS.to(w.device)
    mag_idx = torch.bucketize(normalized.abs(), mids, right=True)
    nibble = torch.where(normalized < 0, mag_idx + 8, mag_idx)
    nib = nibble.reshape(n, k // 2, 2).to(torch.uint8)
    packed = nib[:, :, 0] | (nib[:, :, 1] << 4)
    scales = (e + 127.0).to(torch.uint8)
    return packed.contiguous(), scales.contiguous()


def dequant_mxfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    block_size: int = MXFP4_BLOCK,
) -> torch.Tensor:
    """Expand ``(packed [N, K//2], scales [N, K//block_size])`` to [N, K]."""
    n, half_k = packed.shape
    lo = (packed & 0x0F).long()
    hi = (packed >> 4).long()
    nib = torch.stack((lo, hi), dim=2).reshape(n, half_k * 2)
    values = _MXFP4_LUT.to(packed.device)[nib]
    exponents = scales.to(torch.float32) - 127.0
    expanded = torch.exp2(exponents).repeat_interleave(block_size, dim=1)
    return (values * expanded).to(dtype)
