# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Path B: offloaded DeepSeek-V4.1-Flash on the official reference code.

Follows wrappers/deepseek_v4/official_offload_adapter.py. The expert host
store is reused unchanged because V4.1 keeps the V4 expert key layout and
FP4/UE8M0 tensor format. dspark MTP modules (``mtp.*``) get the same
null-expert construction (they share the reference ``MoE`` class) and their
weights stay unloaded — generation never calls ``forward_spec``. Engram
tables are handled separately by ``EngramHostStore`` (engram_offload.py).

Every reference-API assumption below is verified against the pinned
revision recorded in OFFICIAL_PIN.md ("Verified reference API").
"""

from __future__ import annotations

import json

import torch

from moe_store.wrappers.deepseek_v4.official_offload_adapter import (
    OfficialExpertHostStore,
    _expert_forward,
    _expert_forward_native,
    _load_native_fp4,
    resolve_use_native,
)

__all__ = [
    "make_expert_store",
    "patch_moe_with_offload_v41",
    "load_offloaded_v41_flash",
]


def make_expert_store(
    ckpt_path: str,
    device: torch.device,
    max_resident_experts: int = 16,
    shard_file: str = None,
) -> OfficialExpertHostStore:
    """V4.1 uses the V4 store unchanged: same key prefix, same 6 members."""
    return OfficialExpertHostStore(
        ckpt_path,
        device,
        max_resident_experts=max_resident_experts,
        shard_file=shard_file,
    )


def patch_moe_with_offload_v41(
    model, store, official_module, use_native=None
):
    use_native = resolve_use_native(use_native, store.device)
    if use_native:
        _load_native_fp4()

    def make_forward(moe, layer_id):
        gate = moe.gate
        shared = moe.shared_experts
        dim = moe.dim
        n_routed = moe.n_routed_experts
        swiglu_limit = getattr(shared, "swiglu_limit", 0.0)
        start_idx = moe.experts_start_idx
        end_idx = moe.experts_end_idx

        def forward(x, image_mask=None):
            # Verified @ pin: MoE.forward(x, image_mask) flattens the mask
            # before Gate.forward(x, image_mask); None keeps text routing.
            shape = x.size()
            x = x.view(-1, dim)
            weights, indices = gate(
                x, None if image_mask is None else image_mask.flatten()
            )
            y = torch.zeros_like(x, dtype=torch.float32)
            counts = torch.bincount(
                indices.flatten(), minlength=n_routed
            ).tolist()
            active = [e for e in range(start_idx, end_idx) if counts[e] > 0]
            store.prefetch(layer_id, active)
            for e in active:
                idx, top = torch.where(indices == e)
                bundle = store.get(layer_id, e)
                if use_native:
                    out = _expert_forward_native(
                        x[idx], bundle, swiglu_limit
                    )
                else:
                    out = _expert_forward(
                        x[idx], bundle, swiglu_limit, official_module
                    )
                y[idx] += out.float() * weights[idx, top, None].float()
            if official_module.world_size > 1:
                official_module.dist.all_reduce(y)
            y += shared(x)
            return y.type_as(x).view(shape)

        return forward

    expert_layer = 0
    for block in model.layers:
        if hasattr(block, "ffn") and hasattr(block.ffn, "gate"):
            moe = block.ffn
            moe.forward = make_forward(moe, expert_layer)
            for e in range(len(moe.experts)):
                moe.experts[e] = None
            expert_layer += 1
    return model


def load_offloaded_v41_flash(
    official_module,
    ckpt_path: str,
    config_path: str,
    device: torch.device,
    max_resident_experts: int = 16,
    use_native: bool = None,
    text_only: bool = True,
    tokenizer=None,
):
    """Build the official model with null routed experts, then patch.

    ``config_path`` must point at the *inference* config
    (``inference/config.json`` at the pinned revision), whose flat keys
    construct ``ModelArgs`` — not the HF root config. ``tokenizer`` is
    required when the config enables engram layers (it feeds
    ``NgramHashState``). The null-expert ``MoE.__init__`` swap also covers
    the dspark MTP blocks, which reuse the same MoE class.
    """
    M = official_module
    with open(config_path) as f:
        margs = M.ModelArgs(**json.load(f))
    margs.max_batch_size = 1

    orig_moe_init = M.MoE.__init__

    def _null_routed_moe_init(self, layer_id, args):
        import torch.nn as nn

        orig_expert = M.Expert

        class _NullExpert(nn.Module):
            def __init__(self, *a, **k):
                super().__init__()

            def forward(self, x, weights=None):
                return x

        M.Expert = _NullExpert
        try:
            orig_moe_init(self, layer_id, args)
        finally:
            M.Expert = orig_expert
        for i in range(len(self.experts)):
            self.experts[i] = None
        self.shared_experts = orig_expert(
            args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit
        )

    M.MoE.__init__ = _null_routed_moe_init
    try:
        with torch.device(device):
            model = M.Transformer(margs, tokenizer)
    finally:
        M.MoE.__init__ = orig_moe_init
    torch.set_grad_enabled(False)

    store = make_expert_store(
        ckpt_path, device, max_resident_experts=max_resident_experts
    )
    patch_moe_with_offload_v41(model, store, M, use_native=use_native)
    torch.cuda.empty_cache()
    _load_non_expert_weights_v41(
        model, ckpt_path, device, text_only=text_only
    )
    return model, store


def _load_non_expert_weights_v41(model, ckpt_path, device, text_only=True):
    """Resident load of everything except routed experts, engram tables,
    mtp.* (unused for generation), and vision.* when text_only."""
    import os
    from glob import glob

    from safetensors import safe_open

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    shards = sorted(glob(os.path.join(ckpt_path, "*.safetensors")))
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            for k in f.keys():
                if ".ffn.experts." in k and k.startswith("layers."):
                    continue  # offloaded routed experts
                if ".engram.embed." in k:
                    continue  # offloaded by EngramHostStore
                if k.startswith("mtp."):
                    continue  # dspark MTP unused for generation
                if text_only and k.startswith(
                    ("vision.", "aligner.", "image_")
                ):
                    continue
                if k.endswith(".scale"):
                    wp = params.get(k[:-6] + ".weight")
                    if (
                        wp is not None
                        and getattr(wp, "scale", None) is not None
                    ):
                        t = f.get_tensor(k).to(device)
                        if wp.scale.shape == t.shape:
                            wp.scale.copy_(t)
                    continue
                target = params.get(k)
                if target is None:
                    target = buffers.get(k)
                if target is None:
                    continue
                t = f.get_tensor(k).to(device)
                if target.shape == t.shape:
                    if target.dtype == torch.float4_e2m1fn_x2:
                        target.view(torch.uint8).copy_(t.view(torch.uint8))
                    else:
                        target.copy_(t)
    return model
