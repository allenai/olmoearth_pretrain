"""Unit tests for lat/lon encoding support in CompositeEncodings."""

import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_vit import CompositeEncodings


def _make_composite_encodings(
    use_latlon_encoding: bool = False,
    latlon_dropout_rate: float = 0.0,
) -> CompositeEncodings:
    """Create a CompositeEncodings instance for testing."""
    return CompositeEncodings(
        embedding_size=16,
        supported_modalities=[Modality.SENTINEL2_L2A],
        max_sequence_length=12,
        learnable_channel_embeddings=False,
        use_latlon_encoding=use_latlon_encoding,
        latlon_dropout_rate=latlon_dropout_rate,
    )


def _make_tokens_and_timestamps(
    B: int = 2, H: int = 2, W: int = 2, T: int = 3
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Create dummy tokens, timestamps, and latlon for testing."""
    num_bandsets = Modality.SENTINEL2_L2A.num_band_sets
    tokens = {
        "sentinel2_l2a": torch.randn(B, H, W, T, num_bandsets, 16),
    }
    days = torch.randint(0, 25, (B, T, 1), dtype=torch.long)
    months = torch.randint(0, 12, (B, T, 1), dtype=torch.long)
    years = torch.randint(2018, 2020, (B, T, 1), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=-1)
    latlon = torch.tensor([[45.0, 90.0], [-30.0, -60.0]][:B])
    return tokens, timestamps, latlon


def test_latlon_encoding_changes_output() -> None:
    """CompositeEncodings with use_latlon_encoding=True produces different output."""
    ce_without = _make_composite_encodings(use_latlon_encoding=False)
    ce_with = _make_composite_encodings(use_latlon_encoding=True)

    tokens, timestamps, latlon = _make_tokens_and_timestamps()

    out_without = ce_without.forward(tokens, timestamps, patch_size=4, latlon=latlon)
    out_with = ce_with.forward(tokens, timestamps, patch_size=4, latlon=latlon)

    assert not torch.allclose(
        out_without["sentinel2_l2a"], out_with["sentinel2_l2a"]
    ), "Latlon encoding should change the output"


def test_latlon_none_is_noop() -> None:
    """When latlon is None, output matches the no-encoding case."""
    ce_with = _make_composite_encodings(use_latlon_encoding=True)
    ce_without = _make_composite_encodings(use_latlon_encoding=False)

    tokens, timestamps, _ = _make_tokens_and_timestamps()

    out_with_none = ce_with.forward(tokens, timestamps, patch_size=4, latlon=None)
    out_without = ce_without.forward(tokens, timestamps, patch_size=4, latlon=None)

    assert torch.allclose(
        out_with_none["sentinel2_l2a"], out_without["sentinel2_l2a"]
    ), "With latlon=None, output should match no-latlon-encoding case"


def test_latlon_dropout_rate_1_zeroes_encoding() -> None:
    """With latlon_dropout_rate=1.0 in training mode, latlon encoding is fully dropped."""
    ce_with_dropout = _make_composite_encodings(
        use_latlon_encoding=True, latlon_dropout_rate=1.0
    )
    ce_without = _make_composite_encodings(use_latlon_encoding=False)

    ce_with_dropout.train()
    ce_without.train()

    tokens, timestamps, latlon = _make_tokens_and_timestamps()

    out_dropout = ce_with_dropout.forward(
        tokens, timestamps, patch_size=4, latlon=latlon
    )
    out_without = ce_without.forward(tokens, timestamps, patch_size=4, latlon=None)

    assert torch.allclose(out_dropout["sentinel2_l2a"], out_without["sentinel2_l2a"]), (
        "Full dropout should match no-latlon-encoding output"
    )


def test_latlon_no_dropout_in_eval() -> None:
    """In eval mode, dropout rate is ignored and encoding is always applied."""
    ce = _make_composite_encodings(use_latlon_encoding=True, latlon_dropout_rate=1.0)
    ce_without = _make_composite_encodings(use_latlon_encoding=False)

    ce.eval()
    ce_without.eval()

    tokens, timestamps, latlon = _make_tokens_and_timestamps()

    out_eval = ce.forward(tokens, timestamps, patch_size=4, latlon=latlon)
    out_without = ce_without.forward(tokens, timestamps, patch_size=4, latlon=None)

    # In eval mode, even with dropout_rate=1.0, the encoding should be applied
    assert not torch.allclose(
        out_eval["sentinel2_l2a"], out_without["sentinel2_l2a"]
    ), (
        "In eval mode, latlon encoding should always be applied regardless of dropout rate"
    )


def test_latlon_dropout_rate_0_always_applies() -> None:
    """With latlon_dropout_rate=0.0, encoding is always applied in training mode."""
    ce = _make_composite_encodings(use_latlon_encoding=True, latlon_dropout_rate=0.0)
    ce_without = _make_composite_encodings(use_latlon_encoding=False)

    ce.train()
    ce_without.train()

    tokens, timestamps, latlon = _make_tokens_and_timestamps()

    out_with = ce.forward(tokens, timestamps, patch_size=4, latlon=latlon)
    out_without = ce_without.forward(tokens, timestamps, patch_size=4, latlon=None)

    assert not torch.allclose(
        out_with["sentinel2_l2a"], out_without["sentinel2_l2a"]
    ), "With dropout_rate=0, encoding should always be applied"
