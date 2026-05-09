"""Tests for static_temporal encoding mode and get_static_temporal_encoding."""

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.encodings import (
    get_static_temporal_encoding,
)
from olmoearth_pretrain.nn.flexi_vit import CompositeEncodings


def _make_ce(
    mode: str = "static_temporal", embedding_size: int = 768
) -> CompositeEncodings:
    return CompositeEncodings(
        embedding_size=embedding_size,
        supported_modalities=[Modality.SENTINEL2_L2A],
        max_sequence_length=12,
        timestamp_encoding_mode=mode,
    )


def _make_timestamps(b: int = 2, t: int = 4) -> torch.Tensor:
    days = torch.randint(1, 28, (b, t, 1), dtype=torch.long)
    months = torch.randint(0, 12, (b, t, 1), dtype=torch.long)
    years = torch.randint(2018, 2023, (b, t, 1), dtype=torch.long)
    return torch.cat([days, months, years], dim=-1)


def test_get_static_temporal_encoding_shape() -> None:
    """Output shape should be (B, T, encoding_dim)."""
    ts = _make_timestamps(3, 5)
    enc = get_static_temporal_encoding(ts, 64)
    assert enc.shape == (3, 5, 64)


def test_get_static_temporal_encoding_deterministic() -> None:
    """Same input should produce same output."""
    ts = torch.tensor([[[15, 6, 2021], [1, 0, 2020]]])
    a = get_static_temporal_encoding(ts, 32)
    b = get_static_temporal_encoding(ts, 32)
    assert torch.allclose(a, b)


def test_get_static_temporal_encoding_different_dates_differ() -> None:
    """Different dates should produce different encodings."""
    ts = torch.tensor([[[1, 0, 2020], [1, 6, 2021]]])
    enc = get_static_temporal_encoding(ts, 32)
    assert not torch.allclose(enc[0, 0], enc[0, 1])


def test_get_static_temporal_encoding_odd_dim_raises() -> None:
    """Odd encoding_dim should raise AssertionError."""
    ts = _make_timestamps(1, 1)
    with pytest.raises(AssertionError, match="encoding_dim must be even"):
        get_static_temporal_encoding(ts, 33)


def test_invalid_timestamp_encoding_mode_raises() -> None:
    """Invalid mode should raise ValueError."""
    with pytest.raises(ValueError):
        _make_ce("nonexistent_mode")


def test_static_temporal_no_pos_embed() -> None:
    """static_temporal mode should not create pos_embed or month_embed."""
    ce = _make_ce("static_temporal")
    assert ce.pos_embed is None
    assert ce.month_embed is None


def test_legacy_has_pos_embed() -> None:
    """Legacy mode should still create pos_embed and month_embed."""
    ce = _make_ce("legacy")
    assert ce.pos_embed is not None
    assert ce.month_embed is not None


def test_static_temporal_forward_shape() -> None:
    """Forward pass should preserve token shape."""
    ce = _make_ce("static_temporal", embedding_size=16)
    B, H, W, T = 2, 2, 2, 4
    tokens = torch.randn(B, H, W, T, 3, 16)
    timestamps = _make_timestamps(B, T)
    out = ce.forward({"sentinel2_l2a": tokens}, timestamps, patch_size=4)
    assert out["sentinel2_l2a"].shape == tokens.shape


def test_static_temporal_same_date_same_encoding() -> None:
    """Same calendar date in different slots should get identical temporal encoding."""
    ce = _make_ce("static_temporal", embedding_size=16)
    B, H, W, T = 1, 2, 2, 3
    tokens = torch.zeros(B, H, W, T, 3, 16)
    ts = torch.tensor([[[15, 6, 2021]] * T])
    out = ce.forward({"sentinel2_l2a": tokens}, ts, patch_size=4)
    result = out["sentinel2_l2a"]
    n = ce.embedding_dim_per_embedding_type
    # static_temporal occupies the [n:2n] slot only.
    assert torch.allclose(
        result[0, 0, 0, 0, 0, n : 2 * n],
        result[0, 0, 0, 1, 0, n : 2 * n],
    )


def test_static_temporal_differs_from_legacy() -> None:
    """static_temporal and legacy should produce different temporal embeddings."""
    ce_st = _make_ce("static_temporal", embedding_size=16)
    ce_lg = _make_ce("legacy", embedding_size=16)
    B, H, W, T = 1, 2, 2, 4
    tokens = torch.zeros(B, H, W, T, 3, 16)
    ts = torch.tensor([[[1, 0, 2020], [15, 6, 2020], [1, 0, 2021], [15, 6, 2021]]])
    out_st = ce_st.forward({"sentinel2_l2a": tokens}, ts, patch_size=4)
    out_lg = ce_lg.forward({"sentinel2_l2a": tokens}, ts, patch_size=4)
    n = ce_st.embedding_dim_per_embedding_type
    # Compare on the static_temporal slot [n:2n]; legacy writes time-index here
    # and month into [2n:3n], so the two should differ in the [n:2n] slot alone.
    assert not torch.allclose(
        out_st["sentinel2_l2a"][..., n : 2 * n],
        out_lg["sentinel2_l2a"][..., n : 2 * n],
    )
