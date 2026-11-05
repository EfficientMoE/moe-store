# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""DeepSeek-V4 Path B grouped native call (`_moe_forward_grouped`).

Checks the adapter plumbing (active-expert set, pointer table, masking of
experts outside this rank's range) against the per-expert native forward on
tiny random FP4 experts. Needs a CUDA sm_120 GPU and the MoE-Infinity
`_v4_fp4` extension with `v4_moe_forward_sm120_pro6000` on a device that
passes its hardware gate (RTX PRO 6000); skipped otherwise.
"""

import pytest
import torch


def _ext():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] < 12:
        pytest.skip("sm_120a block-scaled MMA required")
    try:
        from moe_infinity import _v4_fp4
    except ImportError:
        pytest.skip("moe_infinity._v4_fp4 not available")
    if not hasattr(_v4_fp4, "v4_moe_forward_sm120_pro6000"):
        pytest.skip("_v4_fp4 without v4_moe_forward_sm120_pro6000")
    if not _v4_fp4.v4_moe_supported(torch.cuda.current_device()):
        pytest.skip("grouped FP4 MoE is gated to the RTX PRO 6000")
    return _v4_fp4


class _Store:
    def __init__(self, experts):
        self.experts = experts
        self.prefetched = []

    def prefetch(self, layer_id, ids):
        self.prefetched.append(list(ids))

    def get(self, layer_id, e):
        return self.experts[e]


def _expert(H, I, gen):
    def w(n, k):
        packed = torch.randint(
            0, 256, (n, k // 2), generator=gen, dtype=torch.uint8
        )
        scale = torch.randint(
            118, 124, (n, k // 32), generator=gen, dtype=torch.uint8
        )
        return (
            packed.cuda().view(torch.float4_e2m1fn_x2),
            scale.cuda().view(torch.float8_e8m0fnu),
        )

    out = {}
    for name, (n, k) in (("w1", (I, H)), ("w3", (I, H)), ("w2", (H, I))):
        out[f"{name}.weight"], out[f"{name}.scale"] = w(n, k)
    return out


@pytest.mark.parametrize(
    "default_cuda, act_block", [(False, 128), (True, 128), (False, 32)]
)
def test_grouped_matches_per_expert_native(default_cuda, act_block):
    """``default_cuda``: the official DeepSeek generate path runs with
    ``torch.set_default_device("cuda")``; host staging must stay on CPU."""
    ext = _ext()
    from moe_store.wrappers.deepseek_v4 import official_offload_adapter as ad

    ad._V4_FP4 = ext
    gen = torch.Generator().manual_seed(0)
    E, H, I, T, k = 16, 256, 256, 7, 4
    start, end = 4, 12  # this rank owns experts 4..11
    experts = {e: _expert(H, I, gen) for e in range(E)}
    store = _Store(experts)
    x = torch.randn(T, H, generator=gen).bfloat16().cuda()
    indices = torch.rand(T, E, generator=gen).topk(k, -1).indices.cuda()
    weights = torch.ones(T, k, device="cuda")

    prev = torch.get_default_device()
    if default_cuda:
        torch.set_default_device("cuda")
    try:
        y = ad._moe_forward_grouped(
            x, weights, indices, store, 0, start, end, E, 10.0, act_block
        )
    finally:
        torch.set_default_device(prev)
    active = sorted({e for e in indices.flatten().tolist() if start <= e < end})
    assert store.prefetched == [active]

    ref = torch.zeros(T, H, dtype=torch.float32, device="cuda")
    for e in active:
        idx, _ = torch.where(indices == e)
        b = experts[e]
        out = ext.v4_expert_forward_fp4_sm120_pro6000(
            x[idx].contiguous(),
            *(b[n].view(torch.uint8) for n in ("w1.weight", "w1.scale")),
            *(b[n].view(torch.uint8) for n in ("w2.weight", "w2.scale")),
            *(b[n].view(torch.uint8) for n in ("w3.weight", "w3.scale")),
            10.0,
            act_block,
        )
        ref[idx] += out.float()
    torch.testing.assert_close(y, ref)


def test_grouped_no_local_experts_returns_zeros():
    ext = _ext()
    from moe_store.wrappers.deepseek_v4 import official_offload_adapter as ad

    ad._V4_FP4 = ext
    x = torch.randn(3, 256, device="cuda").bfloat16()
    indices = torch.tensor([[0, 1], [1, 2], [2, 0]], device="cuda")
    y = ad._moe_forward_grouped(
        x, torch.ones(3, 2, device="cuda"), indices, _Store({}), 0, 4, 8, 8, 0.0
    )
    assert torch.count_nonzero(y) == 0
