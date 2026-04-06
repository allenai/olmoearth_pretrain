"""Tests for static_temporal mode: legacy layout with multi-frequency temporal encodings."""

import torch

from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.nn.flexi_vit import CompositeEncodings


def _make_ce(
    embedding_size: int = 64,
    supported_modalities: list[ModalitySpec] | None = None,
    max_sequence_length: int = 12,
) -> CompositeEncodings:
    if supported_modalities is None:
        supported_modalities = [Modality.SENTINEL2_L2A, Modality.LATLON]
    return CompositeEncodings(
        embedding_size=embedding_size,
        supported_modalities=supported_modalities,
        max_sequence_length=max_sequence_length,
        timestamp_encoding_mode="static_temporal",
    )


def _make_timestamps(b: int = 2, t: int = 4) -> torch.Tensor:
    days = torch.randint(1, 28, (b, t, 1), dtype=torch.long)
    months = torch.randint(0, 12, (b, t, 1), dtype=torch.long)
    years = torch.randint(2018, 2023, (b, t, 1), dtype=torch.long)
    return torch.cat([days, months, years], dim=-1)


def test_static_temporal_builds_without_time_index_embeddings() -> None:
    """static_temporal should not allocate 1D time position or month embeddings."""
    ce = _make_ce(embedding_size=256)
    assert ce.timestamp_encoding_mode == "static_temporal"
    assert ce.pos_embed is None
    assert ce.month_embed is None


def test_static_temporal_differs_from_legacy_for_same_calendar_time() -> None:
    """Legacy uses slot index + month table; static_temporal uses fractional year freqs."""
    mods = [Modality.SENTINEL2_L2A, Modality.LATLON]
    embedding = 64
    ce_legacy = CompositeEncodings(
        embedding_size=embedding,
        supported_modalities=mods,
        max_sequence_length=12,
        timestamp_encoding_mode="legacy",
    )
    ce_st = CompositeEncodings(
        embedding_size=embedding,
        supported_modalities=mods,
        max_sequence_length=12,
        timestamp_encoding_mode="static_temporal",
    )
    B, T, H, W, num_bandsets = 2, 4, 2, 2, 3
    tokens = torch.zeros(B, H, W, T, num_bandsets, embedding)
    # Same calendar date at every timestep (differs from legacy which encodes index 0..T-1)
    ts = torch.tensor([[[15, 5, 2021]] * T] * B, dtype=torch.long)
    latlon = torch.tensor([[10.0, 20.0], [15.0, -30.0]])
    patch_size = 4

    out_l = ce_legacy.forward(
        {"sentinel2_l2a": tokens.clone()}, ts, patch_size, latlon=latlon
    )["sentinel2_l2a"]
    out_s = ce_st.forward(
        {"sentinel2_l2a": tokens.clone()}, ts, patch_size, latlon=latlon
    )["sentinel2_l2a"]
    assert not torch.allclose(out_l, out_s)


def test_static_temporal_forward_shape() -> None:
    """Forward pass preserves token layout for multitemporal spatial input."""
    ce = _make_ce(embedding_size=32)
    B, T, H, W, num_bandsets = 1, 3, 2, 2, 3
    tokens = torch.randn(B, H, W, T, num_bandsets, 32)
    timestamps = _make_timestamps(B, T)
    latlon = torch.tensor([[0.0, 0.0]])
    out = ce.forward(
        {"sentinel2_l2a": tokens}, timestamps, patch_size=8, latlon=latlon
    )["sentinel2_l2a"]
    assert out.shape == tokens.shape
