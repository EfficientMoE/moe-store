# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from moe_store.wrappers.deepseek_v4.routing import sqrtsoftplus

__all__ = ["topk_route_v41"]


def topk_route_v41(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    bias_vl: torch.Tensor,
    vision_token_mask: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float = 1.0,
    norm_topk_prob: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """V4 noaux-tc routing with V4.1's modality-dependent selection bias.

    ``vision_token_mask`` is a bool tensor of shape [num_tokens]; True rows
    select with ``bias_vl``, False rows with ``bias``. As in V4, the bias
    affects expert *selection* only; returned weights come from the
    sqrtsoftplus scores of the selected experts.
    """
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    logits = F.linear(flat.float(), gate_weight.float())
    scores = sqrtsoftplus(logits)
    per_token_bias = torch.where(
        vision_token_mask.reshape(-1, 1),
        bias_vl.float().unsqueeze(0),
        bias.float().unsqueeze(0),
    )
    select = scores + per_token_bias
    indices = torch.topk(select, top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    if norm_topk_prob:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights * routed_scaling_factor, indices
