# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Fused MoE routing for the offload wrappers (MoE-Infinity #315).

``moe_infinity.kernel.router.fused_route`` computes the gate logits on tensor
cores, the (grouped) top-k selection and renormalisation, the dense
``[T, E]`` mask / weights the expert executor consumes and the per-layer
active-expert flags (pinned host memory) in one launch, with expert selection
identical to the torch code it replaces. :func:`route` returns ``None`` when
the fused path does not apply (CPU tensors, MoE-Infinity / Triton missing,
``MOE_STORE_FUSED_ROUTER=0``) so callers keep their torch path.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

_FUSED = None
_FUSED_TRIED = False


def _load():
    global _FUSED, _FUSED_TRIED
    if not _FUSED_TRIED:
        _FUSED_TRIED = True
        if os.environ.get("MOE_STORE_FUSED_ROUTER", "1") != "0":
            try:
                from moe_infinity.kernel.router import fused_route

                _FUSED = fused_route
            except Exception:  # not installed / no Triton / old version
                _FUSED = None
    return _FUSED


def route(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    top_k: int,
    *,
    scoring: str = "softmax",
    bias: Optional[torch.Tensor] = None,
    n_group: int = 1,
    topk_group: int = 1,
    renormalize: bool = True,
    scaling: float = 1.0,
    eps: float = 0.0,
    weights_dtype: Optional[torch.dtype] = None,
    return_logits: bool = False,
):
    """``(router_mask, routing_weights, logits | None)`` or ``None``."""
    if not (hidden_states.is_cuda and gate_weight.is_cuda):
        return None
    fused = _load()
    if fused is None:
        return None
    out = fused(
        hidden_states,
        gate_weight,
        top_k,
        scoring=scoring,
        bias=bias,
        n_group=n_group,
        topk_group=topk_group,
        renormalize=renormalize,
        scaling=scaling,
        eps=eps,
        weights_dtype=weights_dtype,
        return_logits=return_logits,
        active_flags=True,
    )
    return out.router_mask, out.routing_weights, out.logits
