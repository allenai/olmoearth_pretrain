"""Tests for static_spatial encoding mode: static per-token lat/lon spatial + legacy temporal."""

import torch

from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.nn.flexi_vit import CompositeEncodings


def _make_static_spatial_ce(
    embedding_size: int = 768,
    supported_modalities: list[ModalitySpec] | None = None,
    max_sequence_length: int = 12,
    spatial_dim_fraction: float = 0.5,
    temporal_dim_fraction: float = 0.25,
) -> CompositeEncodings:
    if supported_modalities is None:
        supported_modalities = [Modality.SENTINEL2_L2A, Modality.LATLON]
    return CompositeEncodings(
        embedding_size=embedding_size,
        supported_modalities=supported_modalities,
        max_sequence_length=max_sequence_length,
        timestamp_encoding_mode="static_spatial",
        spatial_dim_fraction=spatial_dim_fraction,
        temporal_dim_fraction=temporal_dim_fraction,
    )


def _make_timestamps(b: int = 2, t: int = 4) -> torch.Tensor:
    days = torch.randint(1, 28, (b, t, 1), dtype=torch.long)
    months = torch.randint(0, 12, (b, t, 1), dtype=torch.long)
    years = torch.randint(2018, 2023, (b, t, 1), dtype=torch.long)
    return torch.cat([days, months, years], dim=-1)


def test_static_spatial_creates_with_legacy_temporal() -> None:
    """static_spatial should have pos_embed and month_embed (legacy temporal)."""
    ce = _make_static_spatial_ce()
    assert ce.timestamp_encoding_mode == "static_spatial"
    assert ce.pos_embed is not None
    assert ce.month_embed is not None
    assert ce.timestamp_mlp is None
    assert ce.latlon_mlp is None


def test_static_spatial_dimension_layout() -> None:
    """Verify the 50/25/25 layout matches STATIC mode."""
    ce = _make_static_spatial_ce(embedding_size=768)
    assert ce.spatial_dim == 384  # 768 * 0.5, divisible by 6
    assert ce.temporal_dim == 192  # 768 * 0.25, even
    assert ce.channel_dim == 192  # remainder


def test_static_spatial_pos_embed_shape() -> None:
    """pos_embed and month_embed should be sized to half the temporal dim."""
    ce = _make_static_spatial_ce(embedding_size=768)
    half_t = ce.temporal_dim // 2
    assert ce.pos_embed.shape == (12, half_t)
    assert ce.month_embed.weight.shape[1] == half_t


def test_static_spatial_output_shape() -> None:
    """Output shape should match input."""
    ce = _make_static_spatial_ce(embedding_size=48)
    B, T, H, W = 2, 4, 2, 2
    num_bandsets = 3
    tokens = torch.randn(B, H, W, T, num_bandsets, 48)
    timestamps = _make_timestamps(B, T)
    latlon = torch.tensor([[45.0, 10.0], [30.0, -90.0]])

    per_modality_tokens = {"sentinel2_l2a": tokens}
    out = ce.forward(per_modality_tokens, timestamps, patch_size=4, latlon=latlon)
    assert out["sentinel2_l2a"].shape == tokens.shape


def test_static_spatial_spatial_varies_with_latlon() -> None:
    """Different lat/lon should produce different spatial encodings."""
    ce = _make_static_spatial_ce(embedding_size=48)
    B, T, H, W = 1, 2, 2, 2
    num_bandsets = 3
    tokens = torch.zeros(B, H, W, T, num_bandsets, 48)
    timestamps = _make_timestamps(B, T)

    per_modality_tokens = {"sentinel2_l2a": tokens}
    out_a = ce.forward(
        per_modality_tokens,
        timestamps,
        patch_size=4,
        latlon=torch.tensor([[45.0, 10.0]]),
    )
    out_b = ce.forward(
        per_modality_tokens,
        timestamps,
        patch_size=4,
        latlon=torch.tensor([[-30.0, 120.0]]),
    )
    s_dim = ce.spatial_dim
    assert not torch.allclose(
        out_a["sentinel2_l2a"][..., :s_dim],
        out_b["sentinel2_l2a"][..., :s_dim],
    )


def test_static_spatial_temporal_uses_legacy_embeddings() -> None:
    """Temporal slice should differ between time steps (legacy time-index + month)."""
    ce = _make_static_spatial_ce(embedding_size=48)
    B, T, H, W = 2, 4, 2, 2
    num_bandsets = 3
    tokens = torch.zeros(B, H, W, T, num_bandsets, 48)
    timestamps = torch.tensor(
        [[[1, 0, 2020], [15, 6, 2020], [1, 0, 2021], [15, 6, 2021]]] * B
    )
    latlon = torch.tensor([[45.0, 10.0], [30.0, -90.0]])

    per_modality_tokens = {"sentinel2_l2a": tokens}
    out = ce.forward(per_modality_tokens, timestamps, patch_size=4, latlon=latlon)
    result = out["sentinel2_l2a"]

    s_dim = ce.spatial_dim
    t_dim = ce.temporal_dim
    # Different time slots → different temporal encoding
    assert not torch.allclose(
        result[0, 0, 0, 0, 0, s_dim : s_dim + t_dim],
        result[0, 0, 0, 1, 0, s_dim : s_dim + t_dim],
    )


def test_static_spatial_same_timeslot_same_temporal() -> None:
    """Legacy temporal: same slot index → same time-index encoding regardless of date.

    The pos_embed is indexed by slot position, not calendar time, so slots 0
    across two batches with different dates should share the time-index part.
    """
    ce = _make_static_spatial_ce(embedding_size=48)
    B, T, H, W = 2, 2, 2, 2
    num_bandsets = 3
    tokens = torch.zeros(B, H, W, T, num_bandsets, 48)
    ts = torch.tensor([[[1, 0, 2020], [15, 6, 2021]], [[10, 3, 2019], [20, 9, 2022]]])
    latlon = torch.tensor([[45.0, 10.0], [45.0, 10.0]])

    per_modality_tokens = {"sentinel2_l2a": tokens}
    out = ce.forward(per_modality_tokens, ts, patch_size=4, latlon=latlon)
    result = out["sentinel2_l2a"]

    s_dim = ce.spatial_dim
    half_t = ce.temporal_dim // 2
    # Slot 0 time-index encoding should be identical across batches
    assert torch.allclose(
        result[0, 0, 0, 0, 0, s_dim : s_dim + half_t],
        result[1, 0, 0, 0, 0, s_dim : s_dim + half_t],
    )


def test_static_spatial_latlon_none_fallback() -> None:
    """When latlon is None, spatial should still have local-freq content."""
    ce = _make_static_spatial_ce(embedding_size=768)
    B, T, H, W = 2, 4, 4, 4
    num_bandsets = 3
    tokens = torch.zeros(B, H, W, T, num_bandsets, 768)
    timestamps = _make_timestamps(B, T)

    per_modality_tokens = {"sentinel2_l2a": tokens}
    out = ce.forward(per_modality_tokens, timestamps, patch_size=4, latlon=None)

    s_dim = ce.spatial_dim
    spatial = out["sentinel2_l2a"][..., :s_dim]
    assert spatial.abs().sum() > 0
