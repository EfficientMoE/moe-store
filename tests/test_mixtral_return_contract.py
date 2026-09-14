"""Return-contract test: the wrapper must satisfy transformers v5 decoder layers.

transformers v5's ``MixtralDecoderLayer.forward`` does
``hidden_states = self.mlp(hidden_states)`` and then
``hidden_states = residual + hidden_states`` — a bare tensor. The v4 contract
returned ``(hidden_states, router_logits)``, which makes that addition raise
``TypeError: unsupported operand type(s) for +: 'Tensor' and 'tuple'``.
v5 no longer consumes router logits from the block (experts were batched into
``MixtralExperts``), so the second element has no consumer.

A registration-layout test cannot see this: the expert weights are laid out
correctly either way, and the failure is one frame further along, in the
decoder layer. This test can, and it fails on the tuple return.

Mirrors ``test_jamba_return_contract.py`` (#206) and
``test_olmoe_return_contract.py`` (#207), which cover the same defect in the
Jamba and OLMoE wrappers. Both ``forward`` return sites are covered here:
the dispatched (executor) path and the ``is_gptq`` local-experts path.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
modeling = pytest.importorskip("transformers.models.mixtral.modeling_mixtral")

if not hasattr(modeling, "MixtralExperts"):
    pytest.skip("transformers does not expose MixtralExperts", allow_module_level=True)


def _config():
    from transformers import MixtralConfig

    return MixtralConfig(
        hidden_size=16,
        intermediate_size=40,
        num_local_experts=2,
        num_experts_per_tok=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=1,
        architectures=["MixtralForCausalLM"],
    )


def test_forward_returns_bare_tensor():
    """SyncMixtralSparseMoeBlock.forward must return a bare (batch, seq, hidden) tensor.

    A stubbed expert_executor isolates the wrapper's own contract from the
    distributed execution layer: dispatch_local is a no-op and
    wait_dispatch_local returns an identity-shaped payload, so what is under
    test is only what forward hands back.
    """
    from moe_store.wrappers.mixtral import SyncMixtralSparseMoeBlock

    config = _config()
    block = SyncMixtralSparseMoeBlock(config)

    hidden_states = torch.randn(2, 5, config.hidden_size)
    flat = hidden_states.view(-1, config.hidden_size).clone()

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


def test_forward_gptq_path_returns_bare_tensor():
    """The is_gptq branch must return a bare tensor too.

    This path runs the local per-expert loop (no executor), so no stubbing is
    needed — it exercises the wrapper's own second return site directly.
    """
    from moe_store.wrappers.mixtral import SyncMixtralSparseMoeBlock

    config = _config()
    block = SyncMixtralSparseMoeBlock(config)
    block.is_gptq = True

    hidden_states = torch.randn(2, 5, config.hidden_size)

    out = block(hidden_states)

    assert isinstance(out, torch.Tensor), (
        f"gptq-path forward must return a bare tensor for the transformers "
        f"v5 decoder contract, got {type(out)}"
    )
    assert out.shape == hidden_states.shape

    residual = torch.randn_like(hidden_states)
    summed = residual + out
    assert summed.shape == hidden_states.shape
