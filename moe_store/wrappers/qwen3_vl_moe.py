from typing import Optional

import torch
import torch.nn as nn
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeTextMLP,
    Qwen3VLMoeTextTopKRouter,
)


class SyncQwen3VLMoeTextSparseMoeBlock(nn.Module):
    archer_config = None
    layer_id: Optional[int] = None

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        self.gate = Qwen3VLMoeTextTopKRouter(config)
        self.experts = nn.ModuleList(
            [
                Qwen3VLMoeTextMLP(
                    config, intermediate_size=config.moe_intermediate_size
                )
                for _ in range(self.num_experts)
            ]
        )

        self.expert_executor = None
        self.expert_prefetcher = None
        self.expert_tracer = None
        self.expert_predictor = None
        self.archer_engine = None
        self.lib = None

    def _route(self, hidden_flat):
        router_logits, routing_weights, selected_experts = self.gate(
            hidden_flat
        )
        num_tokens = hidden_flat.shape[0]
        router_mask = torch.zeros(
            num_tokens,
            self.num_experts,
            dtype=torch.bool,
            device=hidden_flat.device,
        )
        router_mask.scatter_(1, selected_experts, True)
        routing_weights_mask = torch.zeros(
            num_tokens,
            self.num_experts,
            dtype=routing_weights.dtype,
            device=hidden_flat.device,
        )
        routing_weights_mask.scatter_(1, selected_experts, routing_weights)
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

        return expert_output.view(batch_size, sequence_length, hidden_dim).to(
            hidden_states.dtype
        )
