# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

HIDDEN = 64
INTER = 32
LAYERS = 2
EXPERTS = 4


@pytest.fixture()
def tiny_v41_ckpt(tmp_path: Path) -> Path:
    tensors = {}
    for layer in range(LAYERS):
        for expert in range(EXPERTS):
            prefix = f"layers.{layer}.ffn.experts.{expert}"
            # w1/w3: [inter, hidden/2] packed, scale [inter, hidden/32]
            tensors[f"{prefix}.w1.weight"] = torch.zeros(
                INTER, HIDDEN // 2, dtype=torch.int8
            )
            tensors[f"{prefix}.w1.scale"] = torch.full(
                (INTER, HIDDEN // 32), 127, dtype=torch.uint8
            ).view(torch.float8_e8m0fnu)
            tensors[f"{prefix}.w2.weight"] = torch.zeros(
                HIDDEN, INTER // 2, dtype=torch.int8
            )
            tensors[f"{prefix}.w2.scale"] = torch.full(
                (HIDDEN, INTER // 32), 127, dtype=torch.uint8
            ).view(torch.float8_e8m0fnu)
            tensors[f"{prefix}.w3.weight"] = torch.zeros(
                INTER, HIDDEN // 2, dtype=torch.int8
            )
            tensors[f"{prefix}.w3.scale"] = torch.full(
                (INTER, HIDDEN // 32), 127, dtype=torch.uint8
            ).view(torch.float8_e8m0fnu)
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, str(tmp_path / shard))
    index = {"weight_map": {k: shard for k in tensors}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    config = {
        "model_type": "deepseek_v41",
        "architectures": ["DeepseekV41ForCausalLM"],
        "text_config": {
            "model_type": "deepseek_v41_text",
            "num_hidden_layers": LAYERS,
            "n_routed_experts": EXPERTS,
            "hidden_size": HIDDEN,
            "moe_intermediate_size": INTER,
            "swiglu_limit": 10.0,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


def test_v41_indexer_reads_nested_text_config(tiny_v41_ckpt: Path) -> None:
    from moe_store.wrappers.deepseek_v41 import (
        DeepSeekV41ExpertTensorIndexer,
    )

    indexer = DeepSeekV41ExpertTensorIndexer(str(tiny_v41_ckpt))
    assert indexer.num_hidden_layers == LAYERS
    assert indexer.n_routed_experts == EXPERTS
    assert indexer.hidden_size == HIDDEN
    assert indexer.intermediate_size == INTER
    assert indexer.swiglu_limit == 10.0


def test_v41_bundle_has_six_tensors_with_fp4_metadata(
    tiny_v41_ckpt: Path,
) -> None:
    from moe_store.wrappers.deepseek_v41 import (
        DeepSeekV41ExpertTensorIndexer,
    )

    indexer = DeepSeekV41ExpertTensorIndexer(str(tiny_v41_ckpt))
    bundle = indexer.bundle(1, 3)
    assert bundle.num_tensors == 6
    assert bundle.quant_format == "fp4_e2m1"
    assert bundle.scale_format == "ue8m0"
    assert bundle.block_size == 32
    w1 = bundle.part("w1", "weight")
    assert w1.key == "layers.1.ffn.experts.3.w1.weight"
    assert w1.shape == (INTER, HIDDEN // 2)
    loaded = indexer.load_bundle_tensors(bundle)
    assert len(loaded) == 6


def test_topk_route_v41_matches_v4_for_all_text_tokens() -> None:
    from moe_store.wrappers.deepseek_v4 import topk_route
    from moe_store.wrappers.deepseek_v41 import topk_route_v41

    torch.manual_seed(0)
    hidden = torch.randn(5, HIDDEN)
    gate_w = torch.randn(EXPERTS * 4, HIDDEN)
    bias = torch.randn(EXPERTS * 4)
    bias_vl = torch.randn(EXPERTS * 4)
    mask = torch.zeros(5, dtype=torch.bool)  # all text

    w_ref, i_ref = topk_route(
        hidden,
        gate_w,
        bias,
        top_k=3,
        routed_scaling_factor=1.5,
        norm_topk_prob=True,
    )
    w, i = topk_route_v41(
        hidden,
        gate_w,
        bias,
        bias_vl,
        mask,
        top_k=3,
        routed_scaling_factor=1.5,
        norm_topk_prob=True,
    )
    assert torch.equal(i.sort(dim=-1).values, i_ref.sort(dim=-1).values)
    torch.testing.assert_close(
        w.gather(1, i.argsort(dim=-1)), w_ref.gather(1, i_ref.argsort(dim=-1))
    )


def test_topk_route_v41_vision_tokens_use_bias_vl() -> None:
    from moe_store.wrappers.deepseek_v41 import topk_route_v41

    torch.manual_seed(1)
    n_experts = 8
    hidden = torch.randn(4, HIDDEN)
    gate_w = torch.randn(n_experts, HIDDEN)
    # bias strongly prefers experts {0,1}; bias_vl strongly prefers {6,7}
    bias = torch.full((n_experts,), -100.0)
    bias[:2] = 100.0
    bias_vl = torch.full((n_experts,), -100.0)
    bias_vl[6:] = 100.0
    mask = torch.tensor([False, True, False, True])

    _, indices = topk_route_v41(
        hidden,
        gate_w,
        bias,
        bias_vl,
        mask,
        top_k=2,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
    )
    assert set(indices[0].tolist()) == {0, 1}
    assert set(indices[1].tolist()) == {6, 7}
    assert set(indices[2].tolist()) == {0, 1}
    assert set(indices[3].tolist()) == {6, 7}


def test_topk_route_v41_weights_exclude_selection_bias() -> None:
    from moe_store.wrappers.deepseek_v41 import topk_route_v41

    torch.manual_seed(2)
    hidden = torch.randn(3, HIDDEN)
    gate_w = torch.randn(8, HIDDEN)
    bias = torch.randn(8) * 100  # bias shifts selection only
    mask = torch.zeros(3, dtype=torch.bool)
    weights, _ = topk_route_v41(
        hidden,
        gate_w,
        bias,
        bias,
        mask,
        top_k=2,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
    )
    # normalized sqrtsoftplus scores are positive and sum to 1 per token
    torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones(3), atol=1e-5, rtol=1e-5
    )
    assert (weights > 0).all()


def test_v41_adapter_reuses_v4_host_store(tiny_v41_ckpt: Path) -> None:
    from moe_store.wrappers.deepseek_v4.official_offload_adapter import (
        OfficialExpertHostStore,
    )
    from moe_store.wrappers.deepseek_v41.official_offload_adapter import (
        make_expert_store,
    )

    store = make_expert_store(
        str(tiny_v41_ckpt),
        device=torch.device("cpu"),
        max_resident_experts=2,
    )
    assert isinstance(store, OfficialExpertHostStore)
    bundle = store.get(1, 3)
    assert set(bundle) == {
        "w1.weight",
        "w1.scale",
        "w2.weight",
        "w2.scale",
        "w3.weight",
        "w3.scale",
    }
    assert store.is_resident(1, 3)


def test_engram_host_store_gathers_and_dequants_rows() -> None:
    from moe_store.wrappers.deepseek_v41 import EngramHostStore

    torch.manual_seed(3)
    rows, dim, group = 64, 256, 32
    reference = torch.randn(rows, dim)
    # Build an fp8/ue8m0 quantized table: per 32-wide group, scale = 2^ceil
    scales_f = reference.view(rows, dim // group, group).abs().amax(-1)
    exponents = torch.clamp(scales_f.log2().ceil(), -126, 127)
    scale_e8m0 = (exponents + 127).to(torch.uint8).view(
        torch.float8_e8m0fnu
    )
    scaled = reference.view(rows, dim // group, group) / exponents.exp2()[
        ..., None
    ]
    table_fp8 = scaled.view(rows, dim).to(torch.float8_e4m3fn)

    store = EngramHostStore(
        weight=table_fp8, scale=scale_e8m0, device=torch.device("cpu")
    )
    idx = torch.tensor([0, 63, 7, 7])
    out = store.gather(idx, dtype=torch.float32)
    assert out.shape == (4, dim)
    expected = (
        table_fp8[idx].to(torch.float32).view(4, dim // group, group)
        * scale_e8m0[idx].to(torch.float32)[..., None]
    ).view(4, dim)
    torch.testing.assert_close(out, expected)
    # quantization round-trip stays within fp8 e4m3 relative tolerance
    torch.testing.assert_close(out, reference[idx], atol=0.15, rtol=0.15)

    # the reference engram passes [B, L, n_hash_cols] index tensors
    nd_idx = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    nd_out = store.gather(nd_idx, dtype=torch.float32)
    assert nd_out.shape == (2, 2, 2, dim)
    torch.testing.assert_close(
        nd_out[1, 0, 1],
        store.gather(torch.tensor([5]), dtype=torch.float32)[0],
    )
