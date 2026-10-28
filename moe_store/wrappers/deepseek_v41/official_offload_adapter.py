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
    "sparse_attn_eager",
    "maybe_patch_sparse_attn",
]

# Dynamic shared memory the reference tilelang sparse_attn kernel requests
# for the released shapes (h=64 padded heads, d=512): observed
# cudaFuncSetAttribute failure at this size on SM120 workstation GPUs
# (~99 KB opt-in limit; the kernel targets datacenter-class ~227 KB).
_SPARSE_ATTN_SMEM_BYTES = 141_312


def sparse_attn_eager(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Exact-math eager replacement for the reference ``sparse_attn``.

    Shapes (verified @ pin): q [b, s, h, d]; kv [b, n, d] (one shared KV
    row per position); topk_idxs [b, s, topk] int32 with -1 for invalid
    slots; attn_sink [h]. Matches the tilelang kernel's online softmax:
    the sink joins the denominator only, and rows whose indices are all
    invalid produce an all-zero output.
    """
    b, s, h, d = q.shape
    idx = topk_idxs.long()
    valid = idx >= 0
    safe = idx.clamp_min(0)
    rows = kv.gather(
        1, safe.reshape(b, -1, 1).expand(-1, -1, d)
    ).reshape(b, s, -1, d)
    scores = (
        torch.einsum("bshd,bstd->bsht", q.float(), rows.float())
        * softmax_scale
    )
    scores = scores.masked_fill(~valid[:, :, None, :], float("-inf"))
    # Finite lower bound mirrors the kernel's -1e30 fill so all-invalid
    # rows stay NaN-free and come out zero.
    smax = scores.amax(-1, keepdim=True).clamp_min(-1e30)
    num = torch.exp(scores - smax)
    den = num.sum(-1) + torch.exp(
        attn_sink.float().reshape(1, 1, h) - smax.squeeze(-1)
    )
    out = torch.einsum("bsht,bstd->bshd", num, rows.float())
    return (out / den.unsqueeze(-1)).to(q.dtype)


def maybe_patch_sparse_attn(official_module, device) -> bool:
    """Swap in ``sparse_attn_eager`` when the GPU cannot host the reference
    kernel's dynamic shared memory request. Returns True when patched."""
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(device)
    smem_optin = getattr(props, "shared_memory_per_block_optin", 0)
    if smem_optin >= _SPARSE_ATTN_SMEM_BYTES:
        return False
    official_module.sparse_attn = sparse_attn_eager
    return True


class V41ExpertHostStore(OfficialExpertHostStore):
    """V4 store with hub-shard dtype normalization.

    The V4.1 hub checkpoint stores packed FP4 expert weights as int8;
    the reference ``linear()`` dispatches on ``torch.float4_e2m1fn_x2``,
    so host-cached weights are re-viewed on load (the view keeps pinned
    memory and byte layout; V4's converted shards carried the fp4 dtype
    natively, hence no view there).
    """

    def _load_host(self, layer_id, expert_id):
        tensors = super()._load_host(layer_id, expert_id)
        for name in ("w1.weight", "w2.weight", "w3.weight"):
            t = tensors[name]
            if t.dtype == torch.int8:
                tensors[name] = t.view(torch.float4_e2m1fn_x2)
        return tensors


def make_expert_store(
    ckpt_path: str,
    device: torch.device,
    max_resident_experts: int = 16,
    shard_file: str = None,
) -> OfficialExpertHostStore:
    """V4.1 reuses the V4 store: same key prefix, same 6 members; only the
    hub int8->fp4 dtype view is layered on top."""
    return V41ExpertHostStore(
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
    offload_engram: bool = True,
):
    """Build the official model with null routed experts, then patch.

    ``config_path`` must point at the *inference* config
    (``inference/config.json`` at the pinned revision), whose flat keys
    construct ``ModelArgs`` — not the HF root config. ``tokenizer`` is
    required when the config enables engram layers (it feeds
    ``NgramHashState``). The null-expert ``MoE.__init__`` swap also covers
    the dspark MTP blocks, which reuse the same MoE class.

    With ``offload_engram`` the ~98 GB-per-table engram embeddings are
    never allocated on device: ``ParallelEngramEmbedding.__init__`` is
    swapped to a parameter-free shim during construction (the reference
    would otherwise allocate the full fp8 tables as parameters), and after
    the resident weight load each engram layer's ``embed.forward`` is
    rebound to an ``EngramHostStore`` row gather.
    """
    M = official_module
    with open(config_path) as f:
        margs = M.ModelArgs(**json.load(f))
    margs.max_batch_size = 1
    maybe_patch_sparse_attn(M, device)

    orig_moe_init = M.MoE.__init__
    orig_engram_embed_init = M.ParallelEngramEmbedding.__init__

    def _null_engram_embed_init(self, num_embeddings, dim):
        import torch.nn as nn

        nn.Module.__init__(self)
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.part_num_embeddings = num_embeddings
        self.vocab_start_idx = 0
        self.vocab_end_idx = num_embeddings
        self.register_parameter("weight", None)
        self.register_parameter("scale", None)

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
    if offload_engram:
        M.ParallelEngramEmbedding.__init__ = _null_engram_embed_init
    try:
        # Verified @ pin: official generate.py sets the bf16 default dtype
        # before building; without it bf16-less params (embed, norms) come
        # up fp32 and act_quant rejects the activations.
        with M.set_dtype(torch.bfloat16), torch.device(device):
            model = M.Transformer(margs, tokenizer)
    finally:
        M.MoE.__init__ = orig_moe_init
        M.ParallelEngramEmbedding.__init__ = orig_engram_embed_init
    torch.set_grad_enabled(False)

    store = make_expert_store(
        ckpt_path, device, max_resident_experts=max_resident_experts
    )
    patch_moe_with_offload_v41(model, store, M, use_native=use_native)
    torch.cuda.empty_cache()
    _load_non_expert_weights_v41(
        model, ckpt_path, device, text_only=text_only
    )
    if offload_engram:
        _attach_engram_offload(model, margs, ckpt_path, device)
    return model, store


def _attach_engram_offload(model, margs, ckpt_path, device):
    """Rebind each engram layer's embedding lookup to a host row store.

    Verified @ pin: ``model.layers[L].engram.embed.forward(indices)``
    returns dequantized bf16 rows ``[*indices.shape, head_dim]``; for mp1
    no vocab masking applies. ``EngramHostStore.gather`` reproduces the
    reference dequant (fp8 rows x ue8m0 group-32 scales) exactly.
    """
    from .engram_offload import EngramHostStore

    for layer_id in getattr(margs, "engram_layer_ids", ()):
        engram = model.layers[layer_id].engram
        if engram is None:
            continue
        host_store = EngramHostStore.from_checkpoint(
            ckpt_path, layer_id, device
        )
        engram.embed._moe_engram_store = host_store

        def _lookup(indices, _store=host_store):
            return _store.gather(indices, dtype=torch.bfloat16)

        engram.embed.forward = _lookup
    return model


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
