"""Off the RTX PRO 6000 the fused router is not built and the wrappers keep
their own torch routing (MoE-Infinity #315 hardware gate)."""

from types import SimpleNamespace

import torch

from moe_store.wrappers import fused_routing


class _FakeCuda(SimpleNamespace):
    is_cuda = True
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16

    def data_ptr(self):
        return 1234


def test_unsupported_gpu_keeps_torch_routing(monkeypatch):
    built = []

    def make_router(*a, **k):
        built.append(1)
        raise AssertionError("fused router built on an unsupported GPU")

    monkeypatch.setattr(fused_routing, "_LOAD_TRIED", True)
    monkeypatch.setattr(
        fused_routing, "_MAKE", (make_router, lambda device: False)
    )
    owner = SimpleNamespace()
    x = w = _FakeCuda()
    assert fused_routing.route(x, w, 2, owner=owner) is None
    # the decision is cached on the block: no rebuild per call
    assert fused_routing.route(x, w, 2, owner=owner) is None
    assert owner._moe_fused_router[1] is None
    assert not built


def test_supported_gpu_builds_router_once(monkeypatch):
    calls = []

    def make_router(*a, **k):
        calls.append(k["active_flags"])
        return lambda h: ("ids", "tw", "mask", "dw", None, "flags")

    monkeypatch.setattr(fused_routing, "_LOAD_TRIED", True)
    monkeypatch.setattr(fused_routing, "_MAKE", (make_router, lambda d: True))
    owner = SimpleNamespace()
    x = w = _FakeCuda()
    assert fused_routing.route(x, w, 2, owner=owner) == ("mask", "dw", None)
    assert fused_routing.route(x, w, 2, owner=owner) == ("mask", "dw", None)
    assert calls == [True]
