# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from __future__ import annotations

import functools
import json
from pathlib import Path

import torch
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoConfig

from moe_store.checkpoints import get_checkpoint_paths
from moe_store.convert.planner import TensorSpec, dtype_token, plan_layout
from moe_store.convert.writer import write_store
from moe_store.index import DEFAULT_PARTITION_SIZE, StoreIndex
from moe_store.parsing.hf_config import parse_expert_id

_PIPELINE_INDEX_FILES = ("model_index.json", "modular_model_index.json")


def _download_root(checkpoint: str) -> Path:
    path = Path(checkpoint)
    if not path.exists():
        from huggingface_hub import snapshot_download

        path = Path(
            snapshot_download(
                checkpoint, allow_patterns=["*.safetensors*", "*.json", "*.py"]
            )
        )
    return path


def _resolve_checkpoint_dir(
    checkpoint: str,
    subfolder: str | None = None,
    root: Path | None = None,
) -> Path:
    path = root if root is not None else _download_root(checkpoint)
    if subfolder is not None:
        sub = path / subfolder
        if not sub.is_dir():
            raise ValueError(
                f"subfolder {subfolder!r} not found under {checkpoint}"
            )
        return sub
    if any((path / name).is_file() for name in _PIPELINE_INDEX_FILES):
        transformer = path / "transformer"
        if transformer.is_dir():
            return transformer
        candidates = sorted(
            str(p.relative_to(path))
            for p in path.glob("*/transformer")
            if p.is_dir()
        )
        raise ValueError(
            f"{checkpoint} is a multi-component pipeline without a top-level "
            f"transformer/ component; pass --subfolder, e.g. one of "
            f"{candidates}"
        )
    return path


def _load_config(ckpt_dir: Path):
    config_path = ckpt_dir / "config.json"
    if config_path.is_file():
        raw = json.loads(config_path.read_text())
        class_name = raw.get("_class_name")
        if class_name:
            # diffusers component config: no `architectures`/`model_type`
            # keys and not loadable through AutoConfig. Normalize into a
            # PretrainedConfig so the parsing/cast helpers see the usual
            # attribute surface.
            from transformers import PretrainedConfig

            attrs = {
                key: value
                for key, value in raw.items()
                if not key.startswith("_")
            }
            config = PretrainedConfig(**attrs)
            config.architectures = [class_name]
            config.model_type = class_name
            return config
    return AutoConfig.from_pretrained(ckpt_dir, trust_remote_code=True)


def _no_expert(name: str):
    return (None, None)


class _ShardedCheckpoint:
    def __init__(self, ckpt_dir: Path):
        self._files = get_checkpoint_paths(str(ckpt_dir))
        self._location: dict[str, str] = {}
        self._handles: dict[str, object] = {}
        for file in self._files:
            if not file.endswith(".safetensors"):
                raise ValueError(
                    "moe-store convert supports safetensors checkpoints only; "
                    f"got {file}"
                )
            with safe_open(file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    self._location[name] = file

    def names(self) -> list[str]:
        ordered: list[str] = []
        for file in self._files:
            with safe_open(file, framework="pt", device="cpu") as f:
                ordered.extend(f.keys())
        return ordered

    def file_stem(self, name: str) -> str:
        return Path(self._location[name]).stem

    def spec(self, name: str) -> TensorSpec:
        with safe_open(self._location[name], framework="pt", device="cpu") as f:
            sl = f.get_slice(name)
            shape = tuple(sl.get_shape())
            dtype = str(sl.get_dtype()).lower()
        dtype_map = {
            "f32": "float32",
            "f16": "float16",
            "bf16": "bfloat16",
            "f8_e4m3": "float8_e4m3fn",
            "u8": "uint8",
            "i8": "int8",
            "i32": "int32",
            "i64": "int64",
        }
        token = dtype_map.get(dtype, dtype)
        itemsize = torch.empty((), dtype=getattr(torch, token)).element_size()
        nbytes = itemsize
        for dim in shape:
            nbytes *= dim
        return TensorSpec(name=name, nbytes=nbytes, dtype=token, shape=shape)

    def load(self, name: str) -> torch.Tensor:
        with safe_open(self._location[name], framework="pt", device="cpu") as f:
            return f.get_tensor(name)


def convert_checkpoint(
    checkpoint: str,
    store_dir: str,
    *,
    partition_size: int = DEFAULT_PARTITION_SIZE,
    subfolder: str | None = None,
) -> StoreIndex:
    root = _download_root(checkpoint)
    if subfolder is None:
        from moe_store.convert.pipeline import (
            convert_pipeline_checkpoint,
            is_multi_component_pipeline,
        )

        if is_multi_component_pipeline(root):
            return convert_pipeline_checkpoint(
                checkpoint, root, store_dir, partition_size=partition_size
            )
    ckpt_dir = _resolve_checkpoint_dir(checkpoint, subfolder, root=root)
    config = _load_config(ckpt_dir)
    shards = _ShardedCheckpoint(ckpt_dir)

    from moe_store.convert.cast_policy import CastPolicy
    from moe_store.convert.v5_remap import (
        DbrxExpansion,
        GptOssExpansion,
        V5Expansion,
    )
    from moe_store.registry.constants import is_dense_architecture
    from moe_store.registry.slots import member_slot_rank

    policy = CastPolicy.from_config(config, str(ckpt_dir))
    expansion = V5Expansion(config, shards.spec)
    gpt_oss = GptOssExpansion(config, shards.spec)
    dbrx = DbrxExpansion(config, shards.spec)
    arch = (getattr(config, "architectures", None) or [""])[0] or getattr(
        config, "model_type", ""
    )
    if is_dense_architecture(arch):
        expert_of = _no_expert
    else:
        expert_of = functools.partial(parse_expert_id, config=config)

    names = dbrx.expand_names(
        gpt_oss.expand_names(expansion.expand_names(shards.names()))
    )

    def spec_of(name: str) -> TensorSpec:
        raw = (
            expansion.spec(name)
            or gpt_oss.spec(name)
            or dbrx.spec(name)
            or shards.spec(name)
        )
        return policy.apply_to_spec(raw, getattr(torch, raw.dtype))

    def load(name: str) -> torch.Tensor:
        tensor = expansion.load(name, shards.load)
        if tensor is None:
            tensor = gpt_oss.load(name, shards.load)
        if tensor is None:
            tensor = dbrx.load(name, shards.load)
        if tensor is None:
            tensor = shards.load(name)
        return policy.apply_to_tensor(name, tensor)

    specs = [spec_of(name) for name in tqdm(names, desc="scan")]

    index = plan_layout(
        specs,
        lambda name: expert_of(name),
        model_type=getattr(config, "model_type", "unknown"),
        checkpoint_name=str(checkpoint),
        partition_size=partition_size,
        slot_rank=functools.partial(member_slot_rank, arch),
    )
    write_store(index, load, store_dir)
    return index


def inspect_store(store_dir: str) -> dict:
    from moe_store.index import read_index

    index = read_index(store_dir)
    experts = [g for g in index.groups if g.is_expert]
    dense = [g for g in index.groups if not g.is_expert]
    return {
        "version": 2,
        "model_type": index.model_type,
        "checkpoint": index.checkpoint_name,
        "partition_size": index.partition_size,
        "stages": len(index.stages),
        "groups": len(index.groups),
        "expert_groups": len(experts),
        "dense_groups": len(dense),
        "tensors": index.num_tensors,
        "layers": len({g.layer_id for g in experts}),
        "experts_per_layer": (
            len(
                {
                    g.expert_id
                    for g in experts
                    if g.layer_id == experts[0].layer_id
                }
            )
            if experts
            else 0
        ),
    }


def dump_inspect(store_dir: str, *, verbose: bool = False) -> str:
    from moe_store.index import read_index

    summary = inspect_store(store_dir)
    lines = [f"{key}={value}" for key, value in summary.items()]
    if verbose:
        index = read_index(store_dir)
        stage_names = [stage.name for stage in index.stages]
        for group in index.groups:
            kind = "expert" if group.is_expert else "dense"
            stage_idx = group.group_id & 0xFFFFFFFF
            stage = (
                stage_names[stage_idx] if stage_idx < len(stage_names) else "?"
            )
            lines.append(
                f"group {group.group_id:#018x} {kind} layer={group.layer_id} "
                f"expert={group.expert_id} file={group.file_id} "
                f"offset={group.offset} size={group.total_size} "
                f"members={len(group.members)} stage={stage}"
            )
    return "\n".join(lines)
