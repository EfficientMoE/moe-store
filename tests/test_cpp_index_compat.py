# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Cross-language byte-compatibility: the C++ index reader must parse
Python-written stores identically (docs/store-format-v2.md is the
contract; this test is its enforcement)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from moe_store.convert.planner import TensorSpec, dtype_token, plan_layout
from moe_store.convert.writer import write_store
from moe_store.index import read_index

REPO_ROOT = Path(__file__).resolve().parents[1]

DUMP_MAIN = r"""
#include "index_v2.h"
#include <cstdio>

int main(int argc, char** argv) {
  auto index = moe_store::ReadIndex(argv[1]);
  std::printf("{\"model_type\":\"%s\",\"partition_size\":%llu,"
              "\"stages\":%zu,\"groups\":%zu,\"tensors\":%zu,",
              index.model_type.c_str(),
              (unsigned long long)index.partition_size,
              index.stages.size(), index.groups.size(), index.NumTensors());
  std::printf("\"group_list\":[");
  bool first = true;
  for (const auto& g : index.groups) {
    if (!first) std::printf(",");
    first = false;
    std::printf("{\"id\":%llu,\"kind\":%d,\"layer\":%d,\"expert\":%d,"
                "\"file\":%u,\"offset\":%lld,\"size\":%lld,\"members\":[",
                (unsigned long long)g.group_id, (int)g.kind, g.layer_id,
                g.expert_id, g.file_id, (long long)g.offset,
                (long long)g.total_size);
    bool mfirst = true;
    for (const auto& m : g.members) {
      if (!mfirst) std::printf(",");
      mfirst = false;
      std::printf("{\"tid\":%u,\"name\":\"%s\",\"rel\":%lld,\"size\":%lld,"
                  "\"dtype\":\"%s\"}",
                  m.tensor_id, m.name.c_str(), (long long)m.rel_offset,
                  (long long)m.size, m.dtype.c_str());
    }
    std::printf("]}");
  }
  std::printf("]}\n");
  return 0;
}
"""


def _build_store(tmp_path: Path) -> Path:
    torch.manual_seed(3)
    state = {"model.embed_tokens.weight": torch.randn(16, 8)}
    for layer in range(2):
        for expert in range(3):
            for slot in ("gate_proj", "up_proj", "down_proj"):
                key = f"model.layers.{layer}.mlp.experts.{expert}.{slot}.weight"
                state[key] = torch.randn(4, 8, dtype=torch.bfloat16)
    specs = [
        TensorSpec(
            name,
            t.numel() * t.element_size(),
            dtype_token(t.dtype),
            tuple(t.shape),
        )
        for name, t in state.items()
    ]

    import re

    def expert_of(name):
        match = re.search(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.", name)
        return (
            (int(match.group(1)), int(match.group(2)))
            if match
            else (None, None)
        )

    index = plan_layout(
        specs,
        expert_of,
        model_type="synthetic",
        checkpoint_name="synthetic/cpp-compat",
    )
    store_dir = tmp_path / "store"
    write_store(index, lambda name: state[name], store_dir)
    return store_dir


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not available")
def test_cpp_reader_matches_python_reader(tmp_path):
    store_dir = _build_store(tmp_path)

    main_cc = tmp_path / "dump_index.cc"
    main_cc.write_text(DUMP_MAIN)
    binary = tmp_path / "dump_index"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-I",
            str(REPO_ROOT / "moe_store" / "csrc" / "store"),
            str(REPO_ROOT / "moe_store" / "csrc" / "store" / "index_v2.cc"),
            str(main_cc),
            "-o",
            str(binary),
        ],
        check=True,
    )
    dumped = json.loads(
        subprocess.run(
            [str(binary), str(store_dir)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )

    index = read_index(store_dir)
    assert dumped["model_type"] == index.model_type
    assert dumped["partition_size"] == index.partition_size
    assert dumped["stages"] == len(index.stages)
    assert dumped["groups"] == len(index.groups)
    assert dumped["tensors"] == index.num_tensors

    assert len(dumped["group_list"]) == len(index.groups)
    for got, expected in zip(dumped["group_list"], index.groups):
        assert got["id"] == expected.group_id
        assert got["kind"] == expected.kind
        assert got["layer"] == expected.layer_id
        assert got["expert"] == expected.expert_id
        assert got["file"] == expected.file_id
        assert got["offset"] == expected.offset
        assert got["size"] == expected.total_size
        assert len(got["members"]) == len(expected.members)
        for gm, em in zip(got["members"], expected.members):
            assert gm["tid"] == em.tensor_id
            assert gm["name"] == em.name
            assert gm["rel"] == em.rel_offset
            assert gm["size"] == em.size
            assert gm["dtype"] == em.dtype


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not available")
def test_cpp_reader_rejects_corrupted_index(tmp_path):
    store_dir = _build_store(tmp_path)
    index_path = store_dir / "store_index"
    data = bytearray(index_path.read_bytes())
    data[0] = 0x58
    index_path.write_bytes(bytes(data))

    main_cc = tmp_path / "dump_index.cc"
    main_cc.write_text(DUMP_MAIN)
    binary = tmp_path / "dump_index"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-I",
            str(REPO_ROOT / "moe_store" / "csrc" / "store"),
            str(REPO_ROOT / "moe_store" / "csrc" / "store" / "index_v2.cc"),
            str(main_cc),
            "-o",
            str(binary),
        ],
        check=True,
    )
    result = subprocess.run(
        [str(binary), str(store_dir)], capture_output=True, text=True
    )
    assert result.returncode != 0
