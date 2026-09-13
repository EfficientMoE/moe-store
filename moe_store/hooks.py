# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Engine-hooks boundary consumed by model integration code.

``EngineHooks`` is the narrow surface model wrappers and the offload
runtime use to talk to the offloading engine (acquire/release tensors,
blocking fetches, prefetch hints). Model-side code moving to the moe-store
repository must depend only on this protocol, never on ``_store`` directly.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

import torch


@runtime_checkable
class EngineHooks(Protocol):
    def begin(
        self, request_id: int, tensor: torch.Tensor, tensor_id: int
    ) -> None: ...

    def end(
        self, request_id: int, tensor: torch.Tensor, tensor_id: int
    ) -> None: ...

    def fetch_tensors(
        self, request_id: int, tensor_ids: Sequence[int]
    ) -> None: ...

    def prefetch_tensors(
        self, tensor_ids: Sequence[int], priority: int
    ) -> None: ...


@dataclass
class EngineOps:
    """Engine-provided operations the model wrappers call at runtime.

    MoE-Infinity fills these via ``register_engine_ops`` at startup;
    wrappers resolve them lazily through ``require_op`` so moe-store
    imports without the engine installed.
    """

    topk_softmax: Callable | None = None
    fused_mxfp4_gemm: Callable | None = None
    union_experts_from_mask: Callable | None = None
    nvtx_phase: Callable = nullcontext
    route_ahead_ctx: Any | None = None
    v4_fp4_ext: Any | None = None


ops = EngineOps()


def register_engine_ops(**kwargs: Any) -> None:
    for name, value in kwargs.items():
        if not hasattr(ops, name):
            raise ValueError(f"unknown engine op {name!r}")
        setattr(ops, name, value)


def require_op(name: str):
    value = getattr(ops, name)
    if value is None:
        raise RuntimeError(
            f"engine op {name!r} is not registered; the serving engine must "
            "call moe_store.hooks.register_engine_ops() before model "
            "execution"
        )
    return value
