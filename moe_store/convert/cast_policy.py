# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Conversion-time dtype policy, mirroring the engine's legacy
``_cast_state_dict_tensors``: quantized payloads (MXFP4 blocks/scales,
GLM blockwise-FP8 weights + ``_scale_inv``, GPTQ packed tensors, and any
tensor the quantization config protects) keep their raw dtype; everything
else casts to the model compute dtype so the store holds exactly what the
engine expects to load."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from moe_store.convert.planner import TensorSpec, dtype_token
from moe_store.parsing.gptq import is_gptq_packed_tensor, is_gptq_quantized
from moe_store.parsing.hf_config import resolve_config_dtype
from moe_store.parsing.mxfp4 import is_mxfp4_quantized
from moe_store.parsing.quantization import (
    detect_quantization,
    should_cast_tensor,
)


def _has_fp8_blockwise(config: object) -> bool:
    qcfg = getattr(config, "quantization_config", None)
    if qcfg is None:
        return False
    if isinstance(qcfg, dict):
        method = qcfg.get("quant_method", "") or qcfg.get("fmt", "")
    else:
        method = getattr(qcfg, "quant_method", "") or getattr(qcfg, "fmt", "")
    return "fp8" in str(method).lower()


@dataclass(frozen=True)
class CastPolicy:
    compute_dtype: torch.dtype
    is_gptq: bool
    is_mxfp4: bool
    is_glm_fp8: bool
    quant_info: object

    @classmethod
    def from_config(cls, config, ckpt_dir: str) -> "CastPolicy":
        compute_dtype = resolve_config_dtype(config) or torch.bfloat16
        arch = (getattr(config, "architectures", None) or [""])[0]
        try:
            is_mxfp4 = is_mxfp4_quantized(config)
        except Exception:
            is_mxfp4 = False
        return cls(
            compute_dtype=compute_dtype,
            is_gptq=is_gptq_quantized(config),
            is_mxfp4=is_mxfp4,
            is_glm_fp8=(
                ("GlmMoeDsa" in arch or "Glm5Next" in arch)
                and _has_fp8_blockwise(config)
            ),
            quant_info=detect_quantization(config, ckpt_dir),
        )

    def keeps_raw_dtype(self, name: str, raw_dtype: torch.dtype) -> bool:
        if self.is_mxfp4 and (
            name.endswith("_blocks") or name.endswith("_scales")
        ):
            return True
        if self.is_glm_fp8 and (
            name.endswith("_scale_inv") or raw_dtype == torch.float8_e4m3fn
        ):
            return True
        if self.is_gptq and is_gptq_packed_tensor(name):
            return True
        if not should_cast_tensor(name, self.quant_info):
            return True
        return not raw_dtype.is_floating_point

    def target_dtype(self, name: str, raw_dtype: torch.dtype) -> torch.dtype:
        if self.keeps_raw_dtype(name, raw_dtype):
            return raw_dtype
        return self.compute_dtype

    def apply_to_spec(self, spec: TensorSpec, raw_dtype: torch.dtype):
        target = self.target_dtype(spec.name, raw_dtype)
        if target == raw_dtype:
            return spec
        itemsize = torch.empty((), dtype=target).element_size()
        numel = 1
        for dim in spec.shape:
            numel *= dim
        return TensorSpec(
            spec.name, numel * itemsize, dtype_token(target), spec.shape
        )

    def apply_to_tensor(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        target = self.target_dtype(name, tensor.dtype)
        if target == tensor.dtype:
            return tensor
        return tensor.to(target)
