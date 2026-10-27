"""Unit tests for the ERA5 evaluator's per-task probe pooling."""

import pytest
import torch
import torch.nn.functional as F

from olmoearth_pretrain.train.callbacks.era5_evaluator_callback import _probe_input


def test_pooled_is_the_encoder_pooled_embedding() -> None:
    """The default reads the encoder's pooled vector unchanged."""
    out = {"pooled": torch.randn(3, 8), "tokens": torch.randn(3, 10, 8)}
    assert torch.equal(_probe_input(out, "pooled"), out["pooled"])


def test_last4_is_layernormed_mean_of_last_four_tokens() -> None:
    """``last4`` averages the final 4 tokens and LayerNorms the result."""
    tokens = torch.randn(2, 10, 6)
    expected = F.layer_norm(tokens[:, -4:].mean(dim=1), (6,))
    got = _probe_input({"pooled": torch.zeros(2, 6), "tokens": tokens}, "last4")
    assert got.shape == (2, 6)
    assert torch.allclose(got, expected)
    # Earlier tokens do not matter.
    tokens2 = tokens.clone()
    tokens2[:, :-4] = 100.0
    got2 = _probe_input({"pooled": torch.zeros(2, 6), "tokens": tokens2}, "last4")
    assert torch.allclose(got, got2)


def test_unknown_pooling_raises() -> None:
    """A typo in the registry fails loudly."""
    with pytest.raises(ValueError):
        _probe_input(
            {"pooled": torch.zeros(1, 2), "tokens": torch.zeros(1, 4, 2)}, "cls"
        )
