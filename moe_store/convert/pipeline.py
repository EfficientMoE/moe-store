# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Multi-component diffusers-pipeline conversion (MiniMax-H3 class).

A modular pipeline repo (``model_index.json`` / ``modular_model_index.json``)
with two or more weight-bearing component subfolders (e.g. H3's
``transformer``, ``transformer_ref``, ``vae``, ``audio_vae``,
``text_encoder``) converts into ONE v2 store covering every component, with
tensor names prefixed ``<component>.``. Group semantics (issue #222 Task 0):

- H3 transformer components (``MiniMaxH3Transformer3DModel``): one group per
  ``transformer_blocks.N`` holding the block's non-AdaLN weights, plus one
  ``...transformer_blocks.N.adaln`` group per block holding the AdaLN branch
  bundle (the host-cacheable unit for Omni-Infinity); tensors outside the
  blocks fall back to the default module-prefix rule.
- Every other component (VAEs, text encoder): one group per safetensors
  shard, so a whole component streams with sequential shard-sized reads.

Single-component pipelines (e.g. Qwen-Image) keep the historical behavior of
descending into ``transformer/`` alone; variant sub-pipelines (H3's
``FL2VA``/``Ref2VA``, which carry their own model_index) are not components
and still require ``--subfolder``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from tqdm import tqdm

from moe_store.convert.planner import TensorSpec, plan_layout
from moe_store.convert.writer import write_store
from moe_store.index import DEFAULT_PARTITION_SIZE, StoreIndex

PIPELINE_INDEX_FILES = ("modular_model_index.json", "model_index.json")
_H3_TRANSFORMER_CLASSES = frozenset({"MiniMaxH3Transformer3DModel"})
_BLOCK_RE = re.compile(r"^(transformer_blocks\.\d+)\.([^.]+)")


def has_pipeline_index(root: Path) -> bool:
    return any((root / name).is_file() for name in PIPELINE_INDEX_FILES)


def pipeline_class_name(root: Path) -> str:
    for name in PIPELINE_INDEX_FILES:
        path = root / name
        if path.is_file():
            class_name = json.loads(path.read_text()).get("_class_name")
            if class_name:
                return str(class_name)
    return "diffusers_pipeline"


def weight_bearing_components(root: Path) -> list[str]:
    components = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if has_pipeline_index(child):
            continue
        if any(child.glob("*.safetensors")):
            components.append(child.name)
    return components


def is_multi_component_pipeline(root: Path) -> bool:
    return (
        has_pipeline_index(root) and len(weight_bearing_components(root)) >= 2
    )


def _component_arch(config) -> str:
    archs = getattr(config, "architectures", None) or [""]
    return archs[0] or str(getattr(config, "model_type", "") or "")


def _component_stage(
    component: str, arch: str, name: str, shard_stem: str
) -> str | None:
    if arch in _H3_TRANSFORMER_CLASSES:
        match = _BLOCK_RE.match(name)
        if match is None:
            return None
        block, first_module = match.groups()
        if first_module.startswith("adaln"):
            return f"{component}.{block}.adaln"
        return f"{component}.{block}"
    return f"{component}.{shard_stem}"


def convert_pipeline_checkpoint(
    checkpoint: str,
    root: Path,
    store_dir: str,
    *,
    partition_size: int = DEFAULT_PARTITION_SIZE,
) -> StoreIndex:
    from moe_store.convert.cast_policy import CastPolicy
    from moe_store.convert.convert import _load_config, _ShardedCheckpoint

    specs: list[TensorSpec] = []
    stage_of: dict[str, str] = {}
    sources: dict[str, tuple[_ShardedCheckpoint, CastPolicy, str]] = {}

    for component in weight_bearing_components(root):
        component_dir = root / component
        config = _load_config(component_dir)
        arch = _component_arch(config)
        shards = _ShardedCheckpoint(component_dir)
        policy = CastPolicy.from_config(config, str(component_dir))
        for name in tqdm(shards.names(), desc=f"scan {component}"):
            raw = shards.spec(name)
            spec = policy.apply_to_spec(
                TensorSpec(
                    f"{component}.{name}", raw.nbytes, raw.dtype, raw.shape
                ),
                getattr(torch, raw.dtype),
            )
            specs.append(spec)
            stage = _component_stage(
                component, arch, name, shards.file_stem(name)
            )
            if stage is not None:
                stage_of[spec.name] = stage
            sources[spec.name] = (shards, policy, name)

    def load(full_name: str) -> torch.Tensor:
        shards, policy, name = sources[full_name]
        return policy.apply_to_tensor(name, shards.load(name))

    index = plan_layout(
        specs,
        lambda _name: (None, None),
        model_type=pipeline_class_name(root),
        checkpoint_name=str(checkpoint),
        partition_size=partition_size,
        dense_stage_of=stage_of.get,
    )
    write_store(index, load, store_dir)
    return index
