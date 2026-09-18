# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""DeepSeek-V4-Flash-Vision-Exp tensor classification.

Vision-Exp keeps the V4-Flash text backbone and adds a resident vision encoder,
an aligner, learned image-token embeddings, and MTP/nextn draft layers under a
separate top-level ``mtp.<i>.`` prefix. Each MTP layer has routed experts of its
own, which must stay out of the text-layer host store. See
``VISION_EXP_NOTES.md`` for the checkpoint inventory behind these rules.
"""

import re
from enum import Enum

_EXPERT_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.")

_VISION_PREFIXES = ("vision.", "aligner.")
_IMAGE_TOKEN_KEYS = ("image_start", "image_end", "image_newline", "image_pad")


class TensorClass(str, Enum):
    ROUTED_EXPERT = "routed_expert"
    RESIDENT_TEXT = "resident_text"
    RESIDENT_VISION = "resident_vision"
    MTP_NEXTN = "mtp_nextn"


def is_vision_exp_config(config) -> bool:
    return getattr(config, "vision_n_layers", None) is not None


def _text_layer_count(config) -> int:
    # HF-shaped configs expose num_hidden_layers; the native AST-extracted
    # ModelArgs exposes num_layers (issue #7). Missing both must stay loud.
    n_layers = getattr(config, "num_hidden_layers", None)
    if n_layers is None:
        n_layers = config.num_layers
    return int(n_layers)


def classify_vision_exp_tensor(name: str, config) -> TensorClass:
    if name.startswith(_VISION_PREFIXES) or name in _IMAGE_TOKEN_KEYS:
        return TensorClass.RESIDENT_VISION
    if name.startswith("mtp."):
        return TensorClass.MTP_NEXTN
    expert_match = _EXPERT_RE.match(name)
    if expert_match is not None and int(
        expert_match.group(1)
    ) < _text_layer_count(config):
        return TensorClass.ROUTED_EXPERT
    return TensorClass.RESIDENT_TEXT


def should_skip_resident_load(
    name: str, config, text_only: bool = True
) -> bool:
    if not is_vision_exp_config(config):
        return ".ffn.experts." in name
    tensor_class = classify_vision_exp_tensor(name, config)
    if tensor_class is TensorClass.ROUTED_EXPERT:
        return True
    return text_only and tensor_class is TensorClass.MTP_NEXTN
