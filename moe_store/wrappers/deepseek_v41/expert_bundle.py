# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""DeepSeek-V4.1-Flash expert tensor indexing.

Routed experts use the exact V4-Flash on-disk format (FP4 E2M1 packed two
values per byte along K, UE8M0 scales with group size 32):

    layers.{L}.ffn.experts.{E}.w1.weight  int8     [inter, hidden/2]
    layers.{L}.ffn.experts.{E}.w1.scale   f8_e8m0  [inter, hidden/32]
    layers.{L}.ffn.experts.{E}.w2.weight  int8     [hidden, inter/2]
    layers.{L}.ffn.experts.{E}.w2.scale   f8_e8m0  [hidden, inter/32]
    layers.{L}.ffn.experts.{E}.w3.weight  int8     [inter, hidden/2]
    layers.{L}.ffn.experts.{E}.w3.scale   f8_e8m0  [inter, hidden/32]

Verified shapes for deepseek-ai/DeepSeek-V4.1-Flash (hidden=5120, inter=2304):
    w1/w3.weight int8 [2304, 2560]   scale f8_e8m0 [2304, 160]
    w2.weight    int8 [5120, 1152]   scale f8_e8m0 [5120, 72]

The only difference from V4-Flash is the config: model dims live under the
nested ``text_config`` object. ``mtp.{K}.ffn.experts.*`` (dspark MTP) and
``layers.{L}.engram.*`` are NOT routed experts and are never enumerated here.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

from moe_store.wrappers.deepseek_v4.expert_bundle import (
    DeepSeekV4ExpertTensorIndexer,
)

__all__ = ["DeepSeekV41ExpertTensorIndexer"]


class DeepSeekV41ExpertTensorIndexer(DeepSeekV4ExpertTensorIndexer):
    """V4 indexer with V4.1's nested ``text_config`` dimension lookup."""

    @staticmethod
    def _load_config(ckpt_dir: str, config: Optional[object]) -> Dict:
        if config is not None:
            get = (
                config.get
                if isinstance(config, dict)
                else lambda k, d=None: getattr(config, k, d)
            )
            text = get("text_config") or config
            tget = (
                text.get
                if isinstance(text, dict)
                else lambda k, d=None: getattr(text, k, d)
            )
            cfg = {
                "num_hidden_layers": tget("num_hidden_layers"),
                "n_routed_experts": tget("n_routed_experts"),
                "hidden_size": tget("hidden_size"),
                "moe_intermediate_size": tget("moe_intermediate_size"),
                "swiglu_limit": tget("swiglu_limit", 0.0),
            }
            if all(
                cfg[k] is not None
                for k in (
                    "num_hidden_layers",
                    "n_routed_experts",
                    "hidden_size",
                    "moe_intermediate_size",
                )
            ):
                return cfg
        cfg_path = os.path.join(ckpt_dir, "config.json")
        with open(cfg_path) as f:
            raw = json.load(f)
        text = raw.get("text_config", raw)
        return {
            "num_hidden_layers": text["num_hidden_layers"],
            "n_routed_experts": text["n_routed_experts"],
            "hidden_size": text["hidden_size"],
            "moe_intermediate_size": text["moe_intermediate_size"],
            "swiglu_limit": text.get("swiglu_limit", 0.0),
        }
