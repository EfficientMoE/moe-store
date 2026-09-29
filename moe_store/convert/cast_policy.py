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
from typing import Callable

import torch

from moe_store.convert.planner import TensorSpec, dtype_token
from moe_store.fp8 import FP8_BLOCK, quant_fp8_blockwise
from moe_store.parsing.gptq import is_gptq_packed_tensor, is_gptq_quantized
from moe_store.parsing.hf_config import resolve_config_dtype
from moe_store.parsing.mxfp4 import is_mxfp4_quantized
from moe_store.parsing.quantization import (
    detect_quantization,
    should_cast_tensor,
)

_ALREADY_QUANTIZED_METHODS = ("gptq", "awq", "mxfp4", "fp8", "fp4")


def reject_already_quantized_source(config: object, quant_info: object) -> None:
    qcfg = getattr(config, "quantization_config", None)
    if isinstance(qcfg, dict):
        configured = qcfg.get("quant_method") or qcfg.get("fmt")
    else:
        configured = getattr(qcfg, "quant_method", None) or getattr(
            qcfg, "fmt", None
        )
    detected = getattr(quant_info, "method", None)
    for method in (configured, detected):
        normalized = str(method or "").lower().replace("-", "").replace("_", "")
        if any(token in normalized for token in _ALREADY_QUANTIZED_METHODS):
            raise ValueError(
                "cannot quantize experts from an already-quantized source "
                f"checkpoint ({method})"
            )


class FP8ExpertQuantizer:
    def __init__(self, expert_of: Callable, slot_rank: Callable):
        self._expert_of = expert_of
        self._slot_rank = slot_rank
        self._weight_specs: dict[str, TensorSpec] = {}
        self._scale_sources: dict[str, str] = {}
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def prepare(
        self, names: list[str], spec_of: Callable[[str], TensorSpec]
    ) -> list[str]:
        expanded: list[str] = []
        for name in names:
            layer_id, expert_id = self._expert_of(name)
            is_expert_slot = (
                layer_id is not None
                and expert_id is not None
                and self._slot_rank(name) is not None
            )
            if not is_expert_slot:
                expanded.append(name)
                continue
            if not name.endswith(".weight"):
                expanded.append(name)
                continue
            spec = spec_of(name)
            if len(spec.shape) != 2:
                raise ValueError(
                    f"FP8 blockwise expert weights must be 2D: {name} has "
                    f"shape {spec.shape}"
                )
            if spec.dtype not in ("bfloat16", "float16"):
                raise ValueError(
                    "FP8 expert quantization requires bf16/fp16 expert "
                    f"weights: {name} has dtype {spec.dtype}"
                )
            scale_name = f"{name}_scale_inv"
            # The scale shares the weight's registry slot rank; insertion
            # order is the stable tie-break that keeps each pair adjacent.
            self._weight_specs[name] = spec
            self._scale_sources[scale_name] = name
            expanded.extend((name, scale_name))
        return expanded

    def spec(self, name: str) -> TensorSpec | None:
        source = self._scale_sources.get(name)
        if source is not None:
            rows, cols = self._weight_specs[source].shape
            shape = (
                (rows + FP8_BLOCK - 1) // FP8_BLOCK,
                (cols + FP8_BLOCK - 1) // FP8_BLOCK,
            )
            return TensorSpec(name, shape[0] * shape[1] * 4, "float32", shape)
        weight = self._weight_specs.get(name)
        if weight is None:
            return None
        rows, cols = weight.shape
        return TensorSpec(name, rows * cols, "float8_e4m3fn", weight.shape)

    def load(
        self, name: str, base_load: Callable[[str], torch.Tensor]
    ) -> torch.Tensor | None:
        source = self._scale_sources.get(name, name)
        if source not in self._weight_specs:
            return None
        if source not in self._cache:
            self._cache[source] = quant_fp8_blockwise(base_load(source))
        weight, scale = self._cache[source]
        if name in self._scale_sources:
            del self._cache[source]
            return scale
        return weight

    @property
    def num_quantized_weights(self) -> int:
        return len(self._weight_specs)


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
