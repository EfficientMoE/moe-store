import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import moe_store.index as store_index
from moe_store.cli import main
from moe_store.convert.convert import convert_checkpoint
from moe_store.convert.writer import read_group_bytes, read_member_tensor
from moe_store.fp8 import dequant_fp8_blockwise
from moe_store.index import GROUP_ALIGNMENT, MEMBER_ALIGNMENT, read_index

MAX_FP8_RELATIVE_ERROR = 0.05


def test_cli_forwards_quantize_experts(monkeypatch, capsys):
    calls = []

    def fake_convert(checkpoint, store_dir, **kwargs):
        calls.append((checkpoint, store_dir, kwargs))
        return type("Index", (), {"groups": (), "num_tensors": 0})()

    monkeypatch.setattr(
        "moe_store.convert.convert.convert_checkpoint", fake_convert
    )

    assert main(["convert", "checkpoint", "store"]) == 0
    assert calls[-1][2]["quantize_experts"] is None

    assert (
        main(
            [
                "convert",
                "checkpoint",
                "store",
                "--quantize-experts",
                "fp8",
            ]
        )
        == 0
    )
    assert calls[-1][2]["quantize_experts"] == "fp8"
    capsys.readouterr()


def _write_bf16_moe_checkpoint(
    ckpt_dir,
    *,
    expert_shape=(129, 130),
    expert_dtype=torch.bfloat16,
    deterministic=False,
):
    ckpt_dir.mkdir()
    config = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "num_hidden_layers": 1,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "hidden_size": 130,
        "intermediate_size": 129,
        "moe_intermediate_size": 129,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "vocab_size": 16,
        "torch_dtype": "bfloat16",
    }
    (ckpt_dir / "config.json").write_text(json.dumps(config))
    torch.manual_seed(23)
    state = {
        "model.embed_tokens.weight": torch.randn(16, 130, dtype=torch.bfloat16),
        "model.layers.0.mlp.gate.weight": torch.randn(
            2, 130, dtype=torch.bfloat16
        ),
        "model.layers.0.mlp.shared_expert.gate_proj.weight": torch.randn(
            *expert_shape, dtype=expert_dtype
        ),
    }
    for expert in range(2):
        for slot in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.layers.0.mlp.experts.{expert}.{slot}.weight"
            state[name] = torch.randn(*expert_shape, dtype=expert_dtype)
    if deterministic:
        state = {
            name: torch.arange(tensor.numel(), dtype=torch.int32)
            .remainder(31)
            .sub(15)
            .reshape(tensor.shape)
            .to(tensor.dtype)
            for name, tensor in state.items()
        }
    save_file(state, str(ckpt_dir / "model.safetensors"))
    return state


@pytest.mark.parametrize("expert_dtype", [torch.bfloat16, torch.float16])
def test_convert_quantizes_only_routed_expert_weights(tmp_path, expert_dtype):
    ckpt_dir = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    state = _write_bf16_moe_checkpoint(ckpt_dir, expert_dtype=expert_dtype)

    convert_checkpoint(str(ckpt_dir), str(store_dir), quantize_experts="fp8")
    index = read_index(store_dir)

    for group in (group for group in index.groups if group.is_expert):
        assert len(group.members) == 6
        for weight, scale in zip(
            group.members[::2], group.members[1::2], strict=True
        ):
            assert weight.name.endswith(".weight")
            assert scale.name == f"{weight.name}_scale_inv"
            assert weight.dtype == "float8_e4m3fn"
            assert scale.dtype == "float32"
            assert weight.shape == (129, 130)
            assert scale.shape == (2, 2)

    stored = {
        member.name: read_member_tensor(store_dir, group, member)
        for group, member in index.iter_members()
    }
    for name in (
        "model.embed_tokens.weight",
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.shared_expert.gate_proj.weight",
    ):
        assert torch.equal(stored[name], state[name].to(torch.bfloat16))


def test_convert_rejects_non_2d_expert_weight(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    _write_bf16_moe_checkpoint(ckpt_dir, expert_shape=(2, 3, 4))

    with pytest.raises(ValueError, match="expert weights must be 2D"):
        convert_checkpoint(
            str(ckpt_dir), str(store_dir), quantize_experts="fp8"
        )


def test_convert_does_not_quantize_fp32_expert_source(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    _write_bf16_moe_checkpoint(ckpt_dir, expert_dtype=torch.float32)

    convert_checkpoint(str(ckpt_dir), str(store_dir), quantize_experts="fp8")
    index = read_index(store_dir)

    for group in (group for group in index.groups if group.is_expert):
        assert len(group.members) == 3
        assert all(member.dtype == "bfloat16" for member in group.members)


def test_convert_rejects_unknown_expert_quantization_mode(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    _write_bf16_moe_checkpoint(ckpt_dir)

    with pytest.raises(ValueError, match="unsupported expert quantization"):
        convert_checkpoint(
            str(ckpt_dir),
            str(tmp_path / "store"),
            quantize_experts="int4",
        )


@pytest.mark.parametrize("method", ["gptq", "awq", "mxfp4", "fp8", "fp4"])
def test_convert_rejects_already_quantized_source(tmp_path, method):
    ckpt_dir = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    _write_bf16_moe_checkpoint(ckpt_dir)
    config_path = ckpt_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"] = {"quant_method": method}
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="already-quantized source"):
        convert_checkpoint(
            str(ckpt_dir), str(store_dir), quantize_experts="fp8"
        )


def test_fp8_convert_writes_store_metadata(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    _write_bf16_moe_checkpoint(ckpt_dir)

    convert_checkpoint(str(ckpt_dir), str(store_dir), quantize_experts="fp8")

    assert store_index.read_store_meta(store_dir) == {
        "store_format_version": 1,
        "quantize_experts": "fp8",
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
        },
    }


def test_default_convert_does_not_write_store_metadata(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    _write_bf16_moe_checkpoint(ckpt_dir)

    convert_checkpoint(str(ckpt_dir), str(store_dir))

    assert store_index.read_store_meta(store_dir) is None


def test_fp8_conversion_is_atomic_when_metadata_write_fails(
    tmp_path, monkeypatch
):
    ckpt_dir = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    _write_bf16_moe_checkpoint(ckpt_dir)
    original_write_text = Path.write_text

    def fail_metadata_write(path, *args, **kwargs):
        if path.name == store_index.STORE_META_FILE_NAME:
            raise RuntimeError("simulated metadata write failure")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_metadata_write)

    with pytest.raises(RuntimeError, match="metadata write failure"):
        convert_checkpoint(
            str(ckpt_dir), str(store_dir), quantize_experts="fp8"
        )
    assert not store_dir.exists()


def test_fp8_store_satisfies_g1_g4_and_reconstructs_from_one_read(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    _write_bf16_moe_checkpoint(ckpt_dir)
    convert_checkpoint(str(ckpt_dir), str(store_dir), quantize_experts="fp8")
    index = read_index(store_dir)

    for group in (group for group in index.groups if group.is_expert):
        assert group.offset + group.total_size <= index.partition_size  # G1
        assert group.offset % GROUP_ALIGNMENT == 0  # G2
        blob = read_group_bytes(store_dir, group)
        slots = []
        for weight, scale in zip(
            group.members[::2], group.members[1::2], strict=True
        ):
            assert weight.rel_offset % MEMBER_ALIGNMENT == 0  # G2
            assert scale.rel_offset % MEMBER_ALIGNMENT == 0
            slot = weight.name.rsplit(".", 2)[-2]
            slots.append(slot)
            assert scale.name == f"{weight.name}_scale_inv"  # G4
            for member in (weight, scale):
                raw = blob[member.rel_offset : member.rel_offset + member.size]
                reconstructed = (
                    torch.frombuffer(bytearray(raw), dtype=torch.uint8)
                    .view(getattr(torch, member.dtype))
                    .view(*member.shape)
                )
                expected = read_member_tensor(store_dir, group, member)
                assert torch.equal(reconstructed, expected)
        assert slots == ["gate_proj", "up_proj", "down_proj"]  # G3


def test_stored_fp8_experts_round_trip_with_recorded_tolerance(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    state = _write_bf16_moe_checkpoint(ckpt_dir)
    convert_checkpoint(str(ckpt_dir), str(store_dir), quantize_experts="fp8")
    index = read_index(store_dir)

    for group in (group for group in index.groups if group.is_expert):
        for weight_meta, scale_meta in zip(
            group.members[::2], group.members[1::2], strict=True
        ):
            weight = read_member_tensor(store_dir, group, weight_meta)
            scale = read_member_tensor(store_dir, group, scale_meta)
            restored = dequant_fp8_blockwise(weight, scale)
            original = state[weight_meta.name]
            relative_error = (
                restored.float() - original.float()
            ).abs().max() / original.float().abs().max()
            assert relative_error <= MAX_FP8_RELATIVE_ERROR

    for group in (group for group in index.groups if not group.is_expert):
        for member in group.members:
            stored = read_member_tensor(store_dir, group, member)
            original = state[member.name].to(stored.dtype)
            assert torch.equal(
                stored.contiguous().view(torch.uint8),
                original.contiguous().view(torch.uint8),
            )


def test_default_convert_matches_frozen_pre_fp8_store_bytes(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _write_bf16_moe_checkpoint(Path("ckpt"), deterministic=True)

    convert_checkpoint("ckpt", "store")

    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path("store").iterdir())
    }
    assert hashes == {
        "store_data_0": (
            "4b4aa7a7938004c2253ca72f081863fc99d4b628bda75440ea3b86c69b37d5e1"
        ),
        "store_index": (
            "58e41d4e5cb454d8efda1d5bdfda734250ab69285e5cab4ec890ecfabe015a70"
        ),
    }


def test_cli_rejects_gptq_source_for_fp8_expert_quantization(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    _write_bf16_moe_checkpoint(ckpt_dir)
    config_path = ckpt_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"] = {"quant_method": "gptq"}
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="already-quantized source"):
        main(
            [
                "convert",
                str(ckpt_dir),
                str(tmp_path / "store"),
                "--quantize-experts",
                "fp8",
            ]
        )
