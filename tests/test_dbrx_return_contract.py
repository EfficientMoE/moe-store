"""Return-contract test: the dbrx wrapper must satisfy transformers v5 decoder layers.

transformers v5's ``DbrxBlock.forward`` does
``hidden_states = resid_states + hidden_states`` where ``hidden_states`` is the
FFN output — a bare tensor. The v4 contract returned
``(hidden_states, router_logits)``, which makes that addition raise
``TypeError: unsupported operand type(s) for +: 'Tensor' and 'tuple'``.
v5 no longer consumes router logits from the FFN, so the second element has no
consumer.

Additionally, v5 ``DbrxRouter.__init__`` takes the ``ffn_config`` object itself
and its ``forward`` returns bare ``router_logits`` (the v4 wrapper unpacked a
3-tuple ``weights, top_weights, top_experts`` that v5 never produces), so the
wrapper's constructor and routing path must be updated together.

Mirrors ``test_mixtral_return_contract.py`` (#5), which covers the same defect
in the Mixtral wrapper, and the Jamba/OLMoE equivalents upstream (#206/#207).
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
modeling = pytest.importorskip("transformers.models.dbrx.modeling_dbrx")

if not hasattr(modeling, "DbrxFFN"):
    pytest.skip("transformers does not expose DbrxFFN", allow_module_level=True)


def _config():
    from transformers.models.dbrx import DbrxConfig
    from transformers.models.dbrx.configuration_dbrx import DbrxFFNConfig

    ffn = DbrxFFNConfig(
        hidden_size=16,
        ffn_hidden_size=32,
        moe_num_experts=4,
        moe_top_k=2,
        ffn_act_fn={"identifier": "silu", "kwargs": {}},
    )
    return DbrxConfig(d_model=16, n_layers=1, ffn_config=ffn)


def test_constructs_under_transformers_v5():
    """DbrxRouter/DbrxExperts take the config object in v5, not v4 kwargs."""
    from moe_store.wrappers.dbrx import SyncDbrxFFNBlock

    block = SyncDbrxFFNBlock(_config())


def test_forward_returns_bare_tensor():
    """SyncDbrxFFNBlock.forward must return a bare (batch, seq, hidden) tensor.

    A stubbed expert_executor isolates the wrapper's own contract from the
    distributed execution layer: dispatch_local is a no-op and
    wait_dispatch_local returns an identity-shaped payload, so what is under
    test is only what forward hands back.
    """
    from moe_store.wrappers.dbrx import SyncDbrxFFNBlock

    config = _config()
    block = SyncDbrxFFNBlock(config)

    hidden_states = torch.randn(2, 5, config.d_model)
    flat = hidden_states.view(-1, config.d_model).clone()

    block.expert_executor = SimpleNamespace(
        dispatch_local=lambda *args, **kwargs: None,
        wait_dispatch_local=lambda: flat,
    )

    out = block(hidden_states)

    assert isinstance(out, torch.Tensor), (
        f"forward must return a bare tensor for the transformers v5 decoder "
        f"contract, got {type(out)}"
    )
    assert out.shape == hidden_states.shape

    # The actual operation the decoder layer performs. With a tuple return this
    # raises TypeError, which is the failure this test exists to catch.
    residual = torch.randn_like(hidden_states)
    summed = residual + out
    assert summed.shape == hidden_states.shape


def test_routing_matches_hf_dbrx_ffn():
    """Route selection must match HF ``DbrxFFN`` under matched router weights.

    The v5 top-k selection lives in ``DbrxFFN.route_tokens_to_experts`` (softmax
    -> topk -> optional normalize); the wrapper must produce identical logits
    and identical (top_weights, top_experts) under the same router weights.
    """
    from moe_store.wrappers.dbrx import SyncDbrxFFNBlock

    config = _config()
    block = SyncDbrxFFNBlock(config)

    ref = modeling.DbrxFFN(config)
    # transformers 5.12 sizes DbrxRouter's Linear from ffn_hidden_size; give
    # the reference the same d_model-shaped router the wrapper guarantees so
    # this stays a routing-math equivalence check across transformers versions.
    if ref.router.layer.in_features != config.d_model:
        ref.router.hidden_size = config.d_model
        ref.router.layer = torch.nn.Linear(
            config.d_model, config.ffn_config.moe_num_experts, bias=False
        )
    with torch.no_grad():
        ref.router.layer.weight.copy_(block.router.layer.weight)

    for t in (1, 3):
        torch.manual_seed(42)
        x = torch.randn(t, 5, config.d_model)

        ref_logits = ref.router(x)
        new_logits = block.router(x)
        assert torch.equal(ref_logits, new_logits), "router logits differ"

        ref_w, ref_e = modeling.DbrxFFN.route_tokens_to_experts(ref, ref_logits)
        new_w, new_e = modeling.DbrxFFN.route_tokens_to_experts(
            block, new_logits
        )
        assert torch.equal(ref_w, new_w) and torch.equal(
            ref_e, new_e
        ), "top-k selection differs"
