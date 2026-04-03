"""Tests for learned per-token latlon encoding in CompositeEncodings."""

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.nn.flexi_vit import CompositeEncodings


@pytest.fixture
def supported_modalities() -> list[ModalitySpec]:
    """Modalities used for testing."""
    return [Modality.SENTINEL2_L2A, Modality.LATLON]


@pytest.fixture
def embedding_size() -> int:
    """Embedding dimension for testing."""
    return 16


@pytest.fixture
def max_sequence_length() -> int:
    """Max sequence length for testing."""
    return 12


@pytest.fixture
def timestamps() -> torch.Tensor:
    """Random timestamps tensor for testing."""
    B, T = 2, 4
    days = torch.randint(1, 28, (B, T, 1), dtype=torch.long)
    months = torch.randint(0, 12, (B, T, 1), dtype=torch.long)
    years = torch.randint(2018, 2023, (B, T, 1), dtype=torch.long)
    return torch.cat([days, months, years], dim=-1)


@pytest.fixture
def latlon() -> torch.Tensor:
    """Tile center lat/lon for testing."""
    return torch.tensor([[45.0, 10.0], [30.0, -90.0]])


def _make_composite_encodings(
    embedding_size: int,
    supported_modalities: list[ModalitySpec],
    max_sequence_length: int,
    use_learned_latlon_encoding: bool = False,
    latlon_hidden_dim: int = 64,
) -> CompositeEncodings:
    return CompositeEncodings(
        embedding_size=embedding_size,
        supported_modalities=supported_modalities,
        max_sequence_length=max_sequence_length,
        use_learned_latlon_encoding=use_learned_latlon_encoding,
        latlon_hidden_dim=latlon_hidden_dim,
    )


def test_learned_latlon_creates_successfully(
    embedding_size: int,
    supported_modalities: list[ModalitySpec],
    max_sequence_length: int,
) -> None:
    """Learned latlon mode should create without error and have an MLP."""
    ce = _make_composite_encodings(
        embedding_size, supported_modalities, max_sequence_length, True
    )
    assert ce.use_learned_latlon_encoding is True
    assert ce.latlon_mlp is not None


def test_learned_latlon_has_learnable_params(
    embedding_size: int,
    supported_modalities: list[ModalitySpec],
    max_sequence_length: int,
) -> None:
    """The learned latlon MLP should have trainable parameters."""
    ce = _make_composite_encodings(
        embedding_size, supported_modalities, max_sequence_length, True
    )
    assert ce.latlon_mlp is not None
    mlp_params = list(ce.latlon_mlp.parameters())
    assert len(mlp_params) > 0
    assert all(p.requires_grad for p in mlp_params)


def test_learned_latlon_writes_only_to_spatial_slice(
    embedding_size: int,
    supported_modalities: list[ModalitySpec],
    max_sequence_length: int,
    timestamps: torch.Tensor,
    latlon: torch.Tensor,
) -> None:
    """Learned latlon should only modify [3n:4n], not channel or temporal slices."""
    B, T = timestamps.shape[:2]
    H, W = 2, 2
    num_bandsets = 3
    tokens = torch.zeros(B, H, W, T, num_bandsets, embedding_size)
    patch_size = 4

    legacy_ce = _make_composite_encodings(
        embedding_size, supported_modalities, max_sequence_length, False
    )
    learned_ce = _make_composite_encodings(
        embedding_size, supported_modalities, max_sequence_length, True
    )

    per_modality_tokens = {"sentinel2_l2a": tokens.clone()}
    legacy_out = legacy_ce.forward(per_modality_tokens, timestamps, patch_size)

    per_modality_tokens = {"sentinel2_l2a": tokens.clone()}
    learned_out = learned_ce.forward(
        per_modality_tokens, timestamps, patch_size, latlon=latlon
    )

    n = embedding_size // 4

    # Channel slice [0:n] and temporal slices [n:3n] should be identical
    assert torch.allclose(
        legacy_out["sentinel2_l2a"][..., :n],
        learned_out["sentinel2_l2a"][..., :n],
    )
    assert torch.allclose(
        legacy_out["sentinel2_l2a"][..., n : n * 3],
        learned_out["sentinel2_l2a"][..., n : n * 3],
    )


def test_learned_latlon_spatial_slice_nonzero(
    embedding_size: int,
    supported_modalities: list[ModalitySpec],
    max_sequence_length: int,
    timestamps: torch.Tensor,
    latlon: torch.Tensor,
) -> None:
    """Learned latlon should produce non-zero spatial slice."""
    B, T = timestamps.shape[:2]
    H, W = 2, 2
    num_bandsets = 3
    tokens = torch.zeros(B, H, W, T, num_bandsets, embedding_size)
    patch_size = 4

    ce = _make_composite_encodings(
        embedding_size, supported_modalities, max_sequence_length, True
    )

    per_modality_tokens = {"sentinel2_l2a": tokens.clone()}
    out = ce.forward(per_modality_tokens, timestamps, patch_size, latlon=latlon)

    n = embedding_size // 4
    assert out["sentinel2_l2a"][..., n * 3 :].abs().sum() > 0


def test_learned_latlon_varies_across_grid(
    embedding_size: int,
    supported_modalities: list[ModalitySpec],
    max_sequence_length: int,
    timestamps: torch.Tensor,
    latlon: torch.Tensor,
) -> None:
    """Per-token latlon values should differ across the spatial grid."""
    B, T = timestamps.shape[:2]
    H, W = 4, 4
    num_bandsets = 3
    tokens = torch.zeros(B, H, W, T, num_bandsets, embedding_size)
    patch_size = 4

    ce = _make_composite_encodings(
        embedding_size, supported_modalities, max_sequence_length, True
    )

    per_modality_tokens = {"sentinel2_l2a": tokens.clone()}
    out = ce.forward(per_modality_tokens, timestamps, patch_size, latlon=latlon)

    n = embedding_size // 4
    spatial = out["sentinel2_l2a"][..., n * 3 :]

    # Different spatial positions should have different embeddings
    assert not torch.allclose(spatial[0, 0, 0], spatial[0, 1, 1])
    assert not torch.allclose(spatial[0, 0, 0], spatial[0, 3, 3])


def test_learned_latlon_fallback_when_none(
    embedding_size: int,
    supported_modalities: list[ModalitySpec],
    max_sequence_length: int,
    timestamps: torch.Tensor,
) -> None:
    """When latlon is None, should fall back to (0,0) and still produce spatial encoding."""
    B, T = timestamps.shape[:2]
    H, W = 2, 2
    num_bandsets = 3
    tokens = torch.zeros(B, H, W, T, num_bandsets, embedding_size)
    patch_size = 4

    ce = _make_composite_encodings(
        embedding_size, supported_modalities, max_sequence_length, True
    )

    per_modality_tokens = {"sentinel2_l2a": tokens.clone()}
    out = ce.forward(per_modality_tokens, timestamps, patch_size, latlon=None)

    n = embedding_size // 4
    # Spatial slice should still be non-zero (relative positions from (0,0))
    assert out["sentinel2_l2a"][..., n * 3 :].abs().sum() > 0
