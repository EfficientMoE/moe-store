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
``MOE_STORE_FUSED_ROUTER=0``, or a GPU other than the RTX PRO 6000 the
kernels were tuned on: ``moe_infinity.kernel.router.router_supported``) so
callers keep their torch path.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

_MAKE = None
_LOAD_TRIED = False


def _load():
    global _MAKE, _LOAD_TRIED
    if not _LOAD_TRIED:
        _LOAD_TRIED = True
        if os.environ.get("MOE_STORE_FUSED_ROUTER", "1") != "0":
            try:
                from moe_infinity.kernel.router import (
                    make_router,
                    router_supported,
                )

                _MAKE = (make_router, router_supported)
            except Exception:  # not installed / no Triton / old version
                _MAKE = None
    return _MAKE


def route(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    top_k: int,
    *,
    owner: Optional[object] = None,
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
    """``(router_mask, routing_weights, logits | None)`` or ``None``.

    The router is built once and cached on ``owner`` (the MoE block); it is
    rebuilt when the gate weight / bias storage changes (reload, device move).
    """
    if not (hidden_states.is_cuda and gate_weight.is_cuda):
        return None
    make = _load()
    if make is None:
        return None
    key = (
        gate_weight.data_ptr(),
        gate_weight.dtype,
        None if bias is None else bias.data_ptr(),
        top_k,
        return_logits,
    )
    cached = (
        getattr(owner, "_moe_fused_router", None) if owner is not None else None
    )
    if cached is not None and cached[0] == key:
        router = cached[1]
    else:
        make_router, supported = make
        # The fused kernels are tuned for the RTX PRO 6000 (hardware gate);
        # elsewhere the caller keeps its own torch routing.
        router = None
        if supported(gate_weight.device):
            router = make_router(
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
                compact=False,
            )
        if owner is not None:
            owner._moe_fused_router = (key, router)
    if router is None:
        return None
    out = router(hidden_states)
    return out[2], out[3], out[4]
