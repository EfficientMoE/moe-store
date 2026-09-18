# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Sync MoE block for MiniMax-M3 (``minimax_m3_vl``).

MiniMax-M3 is a vision-language MoE: 128 routed experts + 1 shared expert with
top-4 *sigmoid* routing (plus an ``e_score_correction_bias``) and a SwiGLU-OAI
activation ``(up + 1) * g * sigmoid(alpha * g)`` where ``g = gate.clamp(max=
limit)`` and ``up`` is clamped to ``[-limit, limit]``. The HF
``MiniMaxM3VLSparseMoeBlock`` packs the routed experts into batched 3D
``experts.gate_up_proj`` ``[E, 2*inter, H]`` / ``experts.down_proj``
``[E, H, inter]`` parameters (the transformers-v5 layout).

The offload engine indexes experts one-per-``(layer, expert)``, so this Sync
block replaces the packed ``MiniMaxM3VLExperts`` with a per-expert
``nn.ModuleList`` whose parameter names
(``experts.<i>.gate_proj/up_proj/down_proj.weight``) match the per-expert store
layout produced by ``moe_store.convert.v5_remap.V5Expansion`` (the packed
``gate_up_proj`` is chunked on dim 0 into ``gate_proj`` and ``up_proj``). The
router (``MiniMaxM3VLTopKRouter``) and the shared expert
(``MiniMaxM3VLDenseMLP``) reuse the HF classes unchanged, so their checkpoint
keys (``gate.weight``, ``gate.e_score_correction_bias``,
``shared_experts.{gate_up_proj,down_proj}.weight``) load verbatim.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3VLDenseMLP,
        MiniMaxM3VLTopKRouter,
    )

    _MINIMAX_M3_AVAILABLE = True
except ImportError:
    _MINIMAX_M3_AVAILABLE = False
    MiniMaxM3VLDenseMLP = MiniMaxM3VLTopKRouter = None


class _SyncMiniMaxM3VLExpertMLP(nn.Module):
    """A single routed expert with split gate/up projections.

    Replicates one expert of the packed ``MiniMaxM3VLExperts`` forward:
    ``gate = gate_proj(x)``, ``up = up_proj(x)``, then the SwiGLU-OAI gate
    ``(up.clamp(-limit, limit) + 1) * g * sigmoid(alpha * g)`` with
    ``g = gate.clamp(max=limit)``, followed by ``down_proj``. The
    ``gate_proj``/``up_proj`` split (rather than a fused ``gate_up_proj``)
    matches the per-expert store layout, where the packed
    ``experts.gate_up_proj`` row is chunked on dim 0 into gate then up.
    """

    def __init__(self, config):
        super().__init__()
        inter = config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, config.hidden_size, bias=False)
        self.swiglu_alpha = config.swiglu_alpha
        self.swiglu_limit = config.swiglu_limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        gate = gate.clamp(max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        glu = gate * torch.sigmoid(gate * self.swiglu_alpha)
        return self.down_proj((up + 1.0) * glu)


class SyncMiniMaxM3VLSparseMoeBlock(nn.Module):
    archer_config = None
    layer_id: Optional[int] = None

    def __init__(self, config):
        if not _MINIMAX_M3_AVAILABLE:
            raise ImportError(
                "transformers with the minimax_m3_vl model is required for "
                "MiniMax-M3 support"
            )
        super().__init__()
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor

        self.gate = MiniMaxM3VLTopKRouter(config)
        self.experts = nn.ModuleList(
            [_SyncMiniMaxM3VLExpertMLP(config) for _ in range(self.num_experts)]
        )
        self.shared_experts = MiniMaxM3VLDenseMLP(
            config, intermediate_size=config.shared_intermediate_size
        )

        self.expert_executor = None
        self.expert_prefetcher = None
        self.expert_tracer = None
        self.expert_predictor = None
        self.archer_engine = None
        self.lib = None

    def _route(self, hidden_flat):
        # MiniMaxM3VLTopKRouter returns (router_logits, top_k_weights,
        # top_k_index); weights are sigmoid-scored, top-k selected, then
        # normalized to sum to 1 per token (routed_scaling_factor is applied
        # later in forward(), matching the HF block).
        router_logits, top_k_weights, top_k_index = self.gate(hidden_flat)
        num_tokens = hidden_flat.shape[0]
        router_mask = torch.zeros(
            num_tokens,
            self.num_experts,
            dtype=torch.bool,
            device=hidden_flat.device,
        )
        router_mask.scatter_(1, top_k_index, True)
        routing_weights_mask = torch.zeros(
            num_tokens,
            self.num_experts,
            dtype=top_k_weights.dtype,
            device=hidden_flat.device,
        )
        routing_weights_mask.scatter_(1, top_k_index, top_k_weights)
        return router_mask, routing_weights_mask, router_logits

    def _local_experts(self, hidden_flat, router_mask, routing_weights_mask):
        final_hidden = torch.zeros_like(hidden_flat)
        for expert_idx in range(self.num_experts):
            token_mask = router_mask[:, expert_idx]
            if not token_mask.any():
                continue
            expert_out = self.experts[expert_idx](hidden_flat[token_mask])
            weight = routing_weights_mask[token_mask, expert_idx].unsqueeze(-1)
            final_hidden[token_mask] += (weight * expert_out).to(
                final_hidden.dtype
            )
        return final_hidden

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)

        router_mask, routing_weights_mask, router_logits = self._route(
            hidden_flat
        )

        if self.expert_executor is not None:
            self.expert_executor.dispatch_local(
                self.layer_id,
                hidden_flat,
                router_mask,
                routing_weights_mask,
                router_logits=router_logits,
            )
            expert_output = self.expert_executor.wait_dispatch_local()
        else:
            expert_output = self._local_experts(
                hidden_flat, router_mask, routing_weights_mask
            )

        # Routed output is scaled, then the shared expert is added (order and
        # scaling match MiniMaxM3VLSparseMoeBlock.forward exactly).
        expert_output = (
            expert_output.view(-1, hidden_dim) * self.routed_scaling_factor
        )
        shared_output = self.shared_experts(hidden_flat)
        expert_output = expert_output + shared_output
        return expert_output.view(batch_size, sequence_length, hidden_dim).to(
            hidden_states.dtype
        )
