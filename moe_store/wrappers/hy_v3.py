"""Sync MoE block for Hy3 (hy_v3).

Router semantics come straight from the HF HYV3TopKRouter (sigmoid scoring,
expert-bias used for selection only, route_norm renormalization,
router_scaling_factor), reused as-is so parity is structural. The batched HF
HYV3Experts module (gate_up_proj [E, 2*inter, H]) is replaced with a
per-expert ModuleList whose parameter names (experts.<i>.gate_proj/up_proj/
down_proj) match the per-expert checkpoint layout the offload engine indexes.

Naming: parameters follow the post-WeightRenaming module namespace
(gate.weight, e_score_correction_bias, shared_experts.*) so the resident
loader's checkpoint-key renamer maps on-disk keys (mlp.router.gate.weight,
mlp.expert_bias, mlp.shared_mlp.*) onto them. e_score_correction_bias is an
nn.Parameter here — HF keeps it as a buffer, but the resident loader only
materializes parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from transformers.models.hy_v3.modeling_hy_v3 import (
        HYV3MLP,
        HYV3TopKRouter,
    )

    _HYV3_AVAILABLE = True
except ImportError:
    _HYV3_AVAILABLE = False
    HYV3MLP = HYV3TopKRouter = None

from moe_store.wrappers.glm_moe_dsa import SyncGlmMoeDsaMoEBlock


class SyncHYV3MoEBlock(SyncGlmMoeDsaMoEBlock):
    def __init__(self, config):
        if not _HYV3_AVAILABLE:
            raise ImportError(
                "transformers >= 5.6 is required for hy_v3 support"
            )
        nn.Module.__init__(self)
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        self.gate = HYV3TopKRouter(config)
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(config.num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        self.experts = nn.ModuleList(
            [
                HYV3MLP(
                    config,
                    intermediate_size=config.moe_intermediate_size,
                )
                for _ in range(config.num_experts)
            ]
        )
        self.shared_experts = HYV3MLP(
            config,
            intermediate_size=config.moe_intermediate_size
            * config.num_shared_experts,
        )

    def _route(self, hidden_flat: torch.Tensor):
        if self.e_score_correction_bias.device != hidden_flat.device:
            self.e_score_correction_bias.data = (
                self.e_score_correction_bias.data.to(hidden_flat.device)
            )
        # HYV3TopKRouter applies sigmoid, bias-for-selection, route_norm,
        # and router_scaling_factor internally and returns
        # (router_logits, topk_weights, topk_indices).
        router_logits, topk_weights, topk_indices = self.gate(
            hidden_flat, self.e_score_correction_bias
        )
        return topk_indices, topk_weights, router_logits
