import json

import pytest
import torch

import moe_store.index as store_index
from moe_store.cli import main
from moe_store.convert.cast_policy import FP4ExpertQuantizer
from moe_store.convert.convert import convert_checkpoint
from moe_store.convert.planner import TensorSpec
from moe_store.convert.writer import read_member_tensor
from moe_store.fp4 import dequant_mxfp4, quant_mxfp4
from moe_store.index import read_index
from tests.test_fp8_convert import _write_bf16_moe_checkpoint

# Per-32 block amax lands in (3, 6] * 2^e and the widest E2M1 gap is 2
# (between 4 and 6), so elementwise error <= 2^e <= block amax / 3 --
# EXCEPT when the E8M0 clamp e >= -126 engages (block amax < 3 * 2^-126):
# there the scale floors at 2^-126 and the error is bounded by half the
# smallest LUT step at that scale, 0.25 * 2^-126 = 2^-128.
MAX_MXFP4_RELATIVE_ERROR = 1.0 / 3.0 + 1e-3
CLAMPED_SCALE_ERROR_FLOOR = 2.0**-128

EXPERT_SHAPE = (129, 128)


def test_cli_forwards_quantize_experts_fp4(monkeypatch, capsys):
    calls = []

    def fake_convert(checkpoint, store_dir, **kwargs):
        calls.append((checkpoint, store_dir, kwargs))
        return type("Index", (), {"groups": (), "num_tensors": 0})()

    monkeypatch.setattr(
        "moe_store.convert.convert.convert_checkpoint", fake_convert
    )

    assert (
        main(
            [
                "convert",
                "checkpoint",
                "store",
                "--quantize-experts",
                "fp4",
            ]
        )
        == 0
    )
    assert calls[-1][2]["quantize_experts"] == "fp4"
    capsys.readouterr()


@pytest.mark.parametrize("expert_dtype", [torch.bfloat16, torch.float16])
def test_convert_quantizes_only_routed_expert_weights(tmp_path, expert_dtype):
    ckpt_dir = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    state = _write_bf16_moe_checkpoint(
        ckpt_dir, expert_shape=EXPERT_SHAPE, expert_dtype=expert_dtype
    )

    convert_checkpoint(str(ckpt_dir), str(store_dir), quantize_experts="fp4")
    index = read_index(store_dir)

    for group in (group for group in index.groups if group.is_expert):
        assert len(group.members) == 6
        for weight, scale in zip(
            group.members[::2], group.members[1::2], strict=True
        ):
            assert weight.name.endswith(".weight")
            assert scale.name == f"{weight.name}_scales"
            assert weight.dtype == "uint8"
            assert scale.dtype == "uint8"
            assert weight.shape == (129, 64)
            assert scale.shape == (129, 4)

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
    _write_bf16_moe_checkpoint(ckpt_dir, expert_shape=(2, 4, 32))

    with pytest.raises(ValueError, match="expert weights must be 2D"):
        convert_checkpoint(
            str(ckpt_dir), str(tmp_path / "store"), quantize_experts="fp4"
        )


def test_convert_rejects_fp32_expert_source(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    _write_bf16_moe_checkpoint(
        ckpt_dir, expert_shape=EXPERT_SHAPE, expert_dtype=torch.float32
    )

    with pytest.raises(ValueError, match="requires bf16/fp16 expert weights"):
        convert_checkpoint(
            str(ckpt_dir), str(store_dir), quantize_experts="fp4"
        )

    assert not store_dir.exists()


def test_convert_rejects_inner_dim_not_divisible_by_block(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    _write_bf16_moe_checkpoint(ckpt_dir, expert_shape=(129, 130))

    with pytest.raises(ValueError, match="divisible by 32"):
        convert_checkpoint(
            str(ckpt_dir), str(store_dir), quantize_experts="fp4"
        )

    assert not store_dir.exists()


def test_convert_preserves_expert_biases(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    state = _write_bf16_moe_checkpoint(
        ckpt_dir, expert_shape=EXPERT_SHAPE, include_expert_bias=True
    )

    convert_checkpoint(str(ckpt_dir), str(store_dir), quantize_experts="fp4")
    index = read_index(store_dir)
    stored = {
        member.name: read_member_tensor(store_dir, group, member)
        for group, member in index.iter_members()
    }

    bias_names = [name for name in state if name.endswith(".bias")]
    assert bias_names
    for name in bias_names:
        assert stored[name].dtype == torch.bfloat16
        assert torch.equal(stored[name], state[name])


def test_fp4_quantizer_releases_weight_after_scale_load():
    name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    scale_name = f"{name}_scales"
    tensor = torch.randn(129, 128, dtype=torch.bfloat16)
    spec = TensorSpec(name, tensor.numel() * 2, "bfloat16", tensor.shape)
    quantizer = FP4ExpertQuantizer(lambda _: (0, 0), lambda _: 0)

    assert quantizer.prepare([name], lambda _: spec) == [name, scale_name]
    assert quantizer.load(name, lambda _: tensor) is not None
    assert len(quantizer._cache) == 1

    assert quantizer.load(scale_name, lambda _: tensor) is not None
    assert quantizer._cache == {}


def test_convert_rejects_already_quantized_source_for_fp4(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    _write_bf16_moe_checkpoint(ckpt_dir, expert_shape=EXPERT_SHAPE)
    config_path = ckpt_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"] = {"quant_method": "mxfp4"}
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="already-quantized source"):
        convert_checkpoint(
            str(ckpt_dir), str(tmp_path / "store"), quantize_experts="fp4"
        )


def test_fp4_convert_writes_store_metadata(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    _write_bf16_moe_checkpoint(ckpt_dir, expert_shape=EXPERT_SHAPE)

    convert_checkpoint(str(ckpt_dir), str(store_dir), quantize_experts="fp4")

    assert store_index.read_store_meta(store_dir) == {
        "store_format_version": 1,
        "quantize_experts": "fp4",
        "quantization_config": {
            "quant_method": "mxfp4",
            "fp4_format_version": 1,
            "fmt": "e2m1",
            "scale_fmt": "e8m0",
            "weight_block_size": [1, 32],
            "pack_dim": 1,
            "pack_factor": 2,
        },
    }


def test_stored_fp4_experts_round_trip_with_recorded_tolerance(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    state = _write_bf16_moe_checkpoint(ckpt_dir, expert_shape=EXPERT_SHAPE)
    convert_checkpoint(str(ckpt_dir), str(store_dir), quantize_experts="fp4")
    index = read_index(store_dir)

    checked = 0
    for group in (group for group in index.groups if group.is_expert):
        for weight_meta, scale_meta in zip(
            group.members[::2], group.members[1::2], strict=True
        ):
            packed = read_member_tensor(store_dir, group, weight_meta)
            scales = read_member_tensor(store_dir, group, scale_meta)
            restored = dequant_mxfp4(packed, scales, dtype=torch.float32)
            original = state[weight_meta.name].float()
            block_amax = (
                original.reshape(original.shape[0], -1, 32)
                .abs()
                .amax(dim=2)
                .repeat_interleave(32, dim=1)
            )
            err = (restored - original).abs()
            bound = (block_amax * MAX_MXFP4_RELATIVE_ERROR).clamp_min(
                CLAMPED_SCALE_ERROR_FLOOR
            )
            assert (err <= bound).all()
            checked += 1
    assert checked == 6

    for group in (group for group in index.groups if not group.is_expert):
        for member in group.members:
            stored = read_member_tensor(store_dir, group, member)
            original = state[member.name].to(stored.dtype)
            assert torch.equal(
                stored.contiguous().view(torch.uint8),
                original.contiguous().view(torch.uint8),
            )


def test_subnormal_block_clamps_scale_floor():
    weight = torch.full((1, 32), 2.0**-128, dtype=torch.bfloat16)

    packed, scales = quant_mxfp4(weight)
    restored = dequant_mxfp4(packed, scales, dtype=torch.float32)

    assert scales.item() == 1
    err = (restored - weight.float()).abs()
    assert err.max().item() > weight.float().abs().amax().item() / 3
    assert (err <= CLAMPED_SCALE_ERROR_FLOOR).all()


def test_quant_dequant_round_trips_exact_lut_values():
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    signs = torch.tensor([1.0, -1.0])
    base = (lut.repeat(8)[:32] * signs.repeat(16)).reshape(1, 32)
    weight = torch.cat([base * 2.0**e for e in (-3, 0, 5)], dim=0)

    packed, scales = quant_mxfp4(weight)
    restored = dequant_mxfp4(packed, scales, dtype=torch.float32)

    assert torch.equal(restored, weight)
