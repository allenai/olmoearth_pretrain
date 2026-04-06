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


def test_static_temporal_same_date_same_encoding_across_slots() -> None:
    """Identical timestamps at different slots should get identical temporal encoding.

    This is the key advantage over legacy, where temporal encoding depends on
    the slot index (pos_embed[0] != pos_embed[1]) even when both slots have
    the same calendar date.
    """
    ce = _make_ce(embedding_size=64)
    B, T, H, W, num_bandsets = 2, 4, 2, 2, 3
    tokens = torch.zeros(B, H, W, T, num_bandsets, 64)
    ts = torch.tensor([[[15, 5, 2021]] * T] * B, dtype=torch.long)
    latlon = torch.tensor([[10.0, 20.0], [15.0, -30.0]])
    out = ce.forward({"sentinel2_l2a": tokens}, ts, patch_size=4, latlon=latlon)[
        "sentinel2_l2a"
    ]
    n = ce.embedding_dim_per_embedding_type
    temporal_slice = out[..., n : n * 3]
    # All T slots should have the same temporal encoding
    for slot in range(1, T):
        assert torch.allclose(
            temporal_slice[:, :, :, 0, :, :],
            temporal_slice[:, :, :, slot, :, :],
        ), f"Slot 0 and slot {slot} should match for identical timestamps"


def test_static_temporal_different_dates_differ() -> None:
    """Different calendar dates should produce different temporal encoding."""
    ce = _make_ce(embedding_size=64)
    B, H, W, num_bandsets = 1, 2, 2, 3
    T = 2
    tokens = torch.zeros(B, H, W, T, num_bandsets, 64)
    ts = torch.tensor([[[15, 0, 2021], [15, 6, 2021]]], dtype=torch.long)
    latlon = torch.tensor([[10.0, 20.0]])
    out = ce.forward({"sentinel2_l2a": tokens}, ts, patch_size=4, latlon=latlon)[
        "sentinel2_l2a"
    ]
    n = ce.embedding_dim_per_embedding_type
    temporal_slice = out[..., n : n * 3]
    assert not torch.allclose(
        temporal_slice[:, :, :, 0, :, :],
        temporal_slice[:, :, :, 1, :, :],
    )


def test_static_temporal_uses_legacy_spatial_encoding() -> None:
    """static_temporal should use the same spatial quarter as legacy."""
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
    B, T, H, W, num_bandsets = 1, 3, 2, 2, 3
    tokens = torch.zeros(B, H, W, T, num_bandsets, embedding)
    ts = _make_timestamps(B, T)
    latlon = torch.tensor([[10.0, 20.0]])
    out_l = ce_legacy.forward(
        {"sentinel2_l2a": tokens.clone()}, ts, patch_size=4, latlon=latlon
    )["sentinel2_l2a"]
    out_s = ce_st.forward(
        {"sentinel2_l2a": tokens.clone()}, ts, patch_size=4, latlon=latlon
    )["sentinel2_l2a"]
    n = ce_legacy.embedding_dim_per_embedding_type
    # Spatial quarter (3n:4n) should be identical between legacy and static_temporal
    assert torch.allclose(out_l[..., 3 * n : 4 * n], out_s[..., 3 * n : 4 * n])
    # Channel quarter (0:n) should be identical (both use zero-init learnable)
    assert torch.allclose(out_l[..., :n], out_s[..., :n])


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
