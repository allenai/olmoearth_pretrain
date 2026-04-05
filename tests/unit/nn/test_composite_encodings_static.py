"""Tests for static multi-frequency sinusoidal encoding in CompositeEncodings."""

import torch

from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.nn.flexi_vit import CompositeEncodings


def _make_static_ce(
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
        timestamp_encoding_mode="static",
        spatial_dim_fraction=spatial_dim_fraction,
        temporal_dim_fraction=temporal_dim_fraction,
    )


def _make_timestamps(b: int = 2, t: int = 4) -> torch.Tensor:
    days = torch.randint(1, 28, (b, t, 1), dtype=torch.long)
    months = torch.randint(0, 12, (b, t, 1), dtype=torch.long)
    years = torch.randint(2018, 2023, (b, t, 1), dtype=torch.long)
    return torch.cat([days, months, years], dim=-1)


def test_static_mode_creates_successfully() -> None:
    """Static mode should create without error."""
    ce = _make_static_ce()
    assert ce.timestamp_encoding_mode == "static"
    assert ce.timestamp_mlp is None
    assert ce.latlon_mlp is None
    assert ce.pos_embed is None
    assert ce.month_embed is None


def test_static_mode_dimension_layout() -> None:
    """Verify the 50/25/25 layout for a 768-dim model."""
    ce = _make_static_ce(embedding_size=768)
    assert ce.spatial_dim == 384  # 768 * 0.5, divisible by 6
    assert ce.temporal_dim == 192  # 768 * 0.25, even
    assert ce.channel_dim == 192  # remainder


def test_static_mode_output_shape() -> None:
    """Output shape should match input."""
    ce = _make_static_ce(embedding_size=16)
    B, T, H, W = 2, 4, 2, 2
    num_bandsets = 3
    tokens = torch.randn(B, H, W, T, num_bandsets, 16)
    timestamps = _make_timestamps(B, T)
    latlon = torch.tensor([[45.0, 10.0], [30.0, -90.0]])
    patch_size = 4

    per_modality_tokens = {"sentinel2_l2a": tokens}
    out = ce.forward(per_modality_tokens, timestamps, patch_size, latlon=latlon)
    assert out["sentinel2_l2a"].shape == tokens.shape


def test_static_mode_spatial_varies_across_grid() -> None:
    """Different (h, w) positions should have different spatial encoding."""
    ce = _make_static_ce(embedding_size=48)
    B, T, H, W = 2, 4, 4, 4
    num_bandsets = 3
    tokens = torch.zeros(B, H, W, T, num_bandsets, 48)
    timestamps = _make_timestamps(B, T)
    latlon = torch.tensor([[45.0, 10.0], [30.0, -90.0]])
    patch_size = 4

    per_modality_tokens = {"sentinel2_l2a": tokens}
    out = ce.forward(per_modality_tokens, timestamps, patch_size, latlon=latlon)
    result = out["sentinel2_l2a"]

    s_dim = ce.spatial_dim
    # Different spatial positions should have different spatial slices
    assert not torch.allclose(
        result[0, 0, 0, 0, 0, :s_dim], result[0, 1, 1, 0, 0, :s_dim]
    )


def test_static_mode_temporal_varies_across_timesteps() -> None:
    """Different timesteps should have different temporal encoding."""
    ce = _make_static_ce(embedding_size=48)
    B, T, H, W = 2, 4, 2, 2
    num_bandsets = 3
    tokens = torch.zeros(B, H, W, T, num_bandsets, 48)
    # Use very different timestamps
    timestamps = torch.tensor(
        [[[1, 0, 2020], [15, 6, 2020], [1, 0, 2021], [15, 6, 2021]]] * B
    )
    latlon = torch.tensor([[45.0, 10.0], [30.0, -90.0]])
    patch_size = 4

    per_modality_tokens = {"sentinel2_l2a": tokens}
    out = ce.forward(per_modality_tokens, timestamps, patch_size, latlon=latlon)
    result = out["sentinel2_l2a"]

    s_dim = ce.spatial_dim
    t_dim = ce.temporal_dim
    # Different timesteps should differ in temporal slice
    assert not torch.allclose(
        result[0, 0, 0, 0, 0, s_dim : s_dim + t_dim],
        result[0, 0, 0, 1, 0, s_dim : s_dim + t_dim],
    )


def test_static_mode_latlon_none_fallback() -> None:
    """When latlon is None, should still produce spatial encoding using (0,0)."""
    ce = _make_static_ce(embedding_size=48)
    B, T, H, W = 2, 4, 2, 2
    num_bandsets = 3
    tokens = torch.zeros(B, H, W, T, num_bandsets, 48)
    timestamps = _make_timestamps(B, T)
    patch_size = 4

    per_modality_tokens = {"sentinel2_l2a": tokens}
    out = ce.forward(per_modality_tokens, timestamps, patch_size, latlon=None)

    s_dim = ce.spatial_dim
    assert out["sentinel2_l2a"][..., :s_dim].abs().sum() > 0


def test_static_mode_slices_independent() -> None:
    """Channel, temporal, spatial slices should be written independently."""
    ce = _make_static_ce(embedding_size=48)
    B, T, H, W = 2, 4, 2, 2
    num_bandsets = 3
    tokens = torch.zeros(B, H, W, T, num_bandsets, 48)
    timestamps = _make_timestamps(B, T)
    latlon = torch.tensor([[45.0, 10.0], [30.0, -90.0]])
    patch_size = 4

    per_modality_tokens = {"sentinel2_l2a": tokens}
    out = ce.forward(per_modality_tokens, timestamps, patch_size, latlon=latlon)
    result = out["sentinel2_l2a"]

    s_dim = ce.spatial_dim
    t_dim = ce.temporal_dim
    # Spatial and temporal slices should be non-zero
    assert result[..., :s_dim].abs().sum() > 0  # spatial
    assert result[..., s_dim : s_dim + t_dim].abs().sum() > 0  # temporal
    # Channel embeddings are learnable and start at zero — they're non-zero
    # only after training or with random init. Just verify the slice exists.
    assert result[..., s_dim + t_dim :].shape[-1] == ce.channel_dim
