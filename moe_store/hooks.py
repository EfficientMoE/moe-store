# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Engine-hooks boundary consumed by model integration code.

``EngineHooks`` is the narrow surface model wrappers and the offload
runtime use to talk to the offloading engine (acquire/release tensors,
blocking fetches, prefetch hints). Model-side code moving to the moe-store
repository must depend only on this protocol, never on ``_store`` directly.
"""

from typing import Protocol, Sequence, runtime_checkable

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
