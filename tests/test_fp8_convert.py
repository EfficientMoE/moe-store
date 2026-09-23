from moe_store.cli import main


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
