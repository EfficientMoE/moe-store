# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Hardware gate of the DeepSeek-V4 grouped native FP4 MoE (CPU only).

`v4_moe_forward_sm120_pro6000` uses sm_120a block-scaled MMA and is tuned for
the RTX PRO 6000. The engine registers its SM120 + RTX_PRO_6000 gate as
`ops.v4_moe_sm120_pro6000_supported`; when the gate fails or is missing, the
patched MoE forward must take the per-expert Path B (`v4_expert_forward`).
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from moe_store import hooks
from moe_store.wrappers.deepseek_v4 import official_offload_adapter as adapter


class _Ext:
    def v4_moe_forward_sm120_pro6000(self, *args):
        raise AssertionError("not called: forward is stubbed")


class _Store:
    device = torch.device("cpu")

    def prefetch(self, layer_id, ids):
        pass

    def get(self, layer_id, e):
        return {}


class _MoE(nn.Module):
    def __init__(self, dim=8, n_routed=4):
        super().__init__()
        self.dim = dim
        self.n_routed_experts = n_routed
        self.experts_start_idx, self.experts_end_idx = 0, n_routed
        self.experts = [None] * n_routed
        self.shared_experts = lambda x: torch.zeros_like(x)

        def gate(x, input_ids):
            T = x.shape[0]
            return torch.ones(T, 2), torch.tensor([[0, 1]] * T)

        self.gate = gate


@pytest.fixture
def engine(monkeypatch):
    def use(gate):
        monkeypatch.setattr(hooks.ops, "v4_fp4_ext", _Ext())
        monkeypatch.setattr(hooks.ops, "v4_moe_sm120_pro6000_supported", gate)
        monkeypatch.setattr(adapter, "_V4_FP4", None)

    return use


def _patched_path(monkeypatch):
    calls = []
    monkeypatch.setattr(
        adapter,
        "_moe_forward_grouped",
        lambda x, *a: calls.append("grouped") or torch.zeros(x.shape),
    )
    monkeypatch.setattr(
        adapter,
        "_expert_forward_native",
        lambda x, *a: calls.append("path_b") or torch.zeros(x.shape),
    )
    moe = _MoE()
    model = SimpleNamespace(layers=[SimpleNamespace(ffn=moe)])
    official = SimpleNamespace(world_size=1)
    adapter.patch_moe_with_offload(model, _Store(), official, use_native=True)
    moe.forward(torch.zeros(3, moe.dim), torch.zeros(3, dtype=torch.long))
    return calls


@pytest.mark.parametrize(
    "gate, path",
    [
        (None, "path_b"),  # engine without the hardware gate
        (lambda device: False, "path_b"),  # e.g. H100 or RTX 5090
        (lambda device: True, "grouped"),  # RTX PRO 6000
    ],
)
def test_dispatch_follows_hardware_gate(engine, monkeypatch, gate, path):
    engine(gate)
    calls = _patched_path(monkeypatch)
    assert calls and set(calls) == {path}


def test_gate_receives_store_device(engine):
    seen = []
    engine(lambda device: seen.append(device) or False)
    assert adapter.grouped_native_available("cuda:1") is False
    assert seen == ["cuda:1"]
