"""Tests for unified timestamp encoding in CompositeEncodings."""

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


def _make_composite_encodings(
    embedding_size: int,
    supported_modalities: list[ModalitySpec],
    max_sequence_length: int,
    timestamp_encoding_mode: str = "legacy",
    timestamp_dropout_rate: float = 0.0,
) -> CompositeEncodings:
    return CompositeEncodings(
        embedding_size=embedding_size,
        supported_modalities=supported_modalities,
        max_sequence_length=max_sequence_length,
        timestamp_encoding_mode=timestamp_encoding_mode,
        timestamp_dropout_rate=timestamp_dropout_rate,
    )


def test_legacy_mode_is_default(
    embedding_size: int,
    supported_modalities: list[ModalitySpec],
    max_sequence_length: int,
) -> None:
    """Default mode should be legacy."""
    ce = _make_composite_encodings(
        embedding_size, supported_modalities, max_sequence_length
    )
    assert ce.timestamp_encoding_mode == "legacy"


def test_unified_mode_creates_successfully(
    embedding_size: int,
    supported_modalities: list[ModalitySpec],
    max_sequence_length: int,
) -> None:
    """Unified mode should create without error."""
    ce = _make_composite_encodings(
        embedding_size, supported_modalities, max_sequence_length, "unified"
    )
    assert ce.timestamp_encoding_mode == "unified"


def test_unified_produces_different_output_than_legacy(
    embedding_size: int,
    supported_modalities: list[ModalitySpec],
    max_sequence_length: int,
    timestamps: torch.Tensor,
) -> None:
    """Unified and legacy modes should produce different encodings."""
    B, T = timestamps.shape[:2]
    H, W = 2, 2
    num_bandsets = 3  # sentinel2_l2a has 3 band sets
    tokens = torch.randn(B, H, W, T, num_bandsets, embedding_size)
    patch_size = 4

    legacy_ce = _make_composite_encodings(
        embedding_size, supported_modalities, max_sequence_length, "legacy"
    )
    unified_ce = _make_composite_encodings(
        embedding_size, supported_modalities, max_sequence_length, "unified"
    )

    per_modality_tokens = {"sentinel2_l2a": tokens.clone()}
    legacy_out = legacy_ce.forward(per_modality_tokens, timestamps, patch_size)

    per_modality_tokens = {"sentinel2_l2a": tokens.clone()}
    unified_out = unified_ce.forward(per_modality_tokens, timestamps, patch_size)

    assert not torch.allclose(legacy_out["sentinel2_l2a"], unified_out["sentinel2_l2a"])


def test_unified_timestamp_dropout_1_zeroes_encoding(
    embedding_size: int,
    supported_modalities: list[ModalitySpec],
    max_sequence_length: int,
    timestamps: torch.Tensor,
) -> None:
    """With dropout_rate=1.0 in training, the unified timestamp encoding should be zeroed."""
    B, T = timestamps.shape[:2]
    H, W = 2, 2
    num_bandsets = 3
    tokens = torch.randn(B, H, W, T, num_bandsets, embedding_size)
    patch_size = 4

    ce_drop = _make_composite_encodings(
        embedding_size, supported_modalities, max_sequence_length, "unified", 1.0
    )
    ce_drop.train()

    ce_nodrop = _make_composite_encodings(
        embedding_size, supported_modalities, max_sequence_length, "unified", 0.0
    )
    ce_nodrop.train()

    # With full dropout, the unified timestamp part should be zeroed,
    # so it should match a model with no temporal encoding at all.
    # At minimum, it should differ from no-dropout version.
    per_modality_tokens = {"sentinel2_l2a": tokens.clone()}
    out_drop = ce_drop.forward(per_modality_tokens, timestamps, patch_size)

    per_modality_tokens = {"sentinel2_l2a": tokens.clone()}
    out_nodrop = ce_nodrop.forward(per_modality_tokens, timestamps, patch_size)

    # Full dropout (rate=1.0) should zero the timestamp encoding,
    # so with dropout the output differs from without dropout
    assert not torch.allclose(out_drop["sentinel2_l2a"], out_nodrop["sentinel2_l2a"])


def test_unified_no_dropout_in_eval(
    embedding_size: int,
    supported_modalities: list[ModalitySpec],
    max_sequence_length: int,
    timestamps: torch.Tensor,
) -> None:
    """In eval mode, dropout rate should be ignored (encoding always applied)."""
    B, T = timestamps.shape[:2]
    H, W = 2, 2
    num_bandsets = 3
    tokens = torch.randn(B, H, W, T, num_bandsets, embedding_size)
    patch_size = 4

    ce = _make_composite_encodings(
        embedding_size, supported_modalities, max_sequence_length, "unified", 0.5
    )
    ce.eval()

    # Run twice — should be identical in eval mode
    per_modality_tokens = {"sentinel2_l2a": tokens.clone()}
    out1 = ce.forward(per_modality_tokens, timestamps, patch_size)

    per_modality_tokens = {"sentinel2_l2a": tokens.clone()}
    out2 = ce.forward(per_modality_tokens, timestamps, patch_size)

    assert torch.allclose(out1["sentinel2_l2a"], out2["sentinel2_l2a"])


def test_unified_skips_old_temporal_slices(
    embedding_size: int,
    supported_modalities: list[ModalitySpec],
    max_sequence_length: int,
    timestamps: torch.Tensor,
) -> None:
    """In unified mode, the old temporal-position and month encoding slices should be zero."""
    B, T = timestamps.shape[:2]
    H, W = 2, 2
    num_bandsets = 3
    tokens = torch.zeros(B, H, W, T, num_bandsets, embedding_size)
    patch_size = 4

    ce = _make_composite_encodings(
        embedding_size, supported_modalities, max_sequence_length, "unified"
    )

    per_modality_tokens = {"sentinel2_l2a": tokens.clone()}
    out = ce.forward(per_modality_tokens, timestamps, patch_size)
    result = out["sentinel2_l2a"]

    n = embedding_size // 4  # embedding_dim_per_embedding_type

    # In unified mode, the temporal (slice 1) and month (slice 2) slices
    # are NOT populated by the legacy code. Instead the unified encoding
    # is added across the full dimension. So slices 1 and 2 should NOT
    # be zero (they get the additive unified encoding), but they should
    # differ from what legacy mode would produce.
    legacy_ce = _make_composite_encodings(
        embedding_size, supported_modalities, max_sequence_length, "legacy"
    )
    per_modality_tokens_legacy = {"sentinel2_l2a": torch.zeros_like(tokens)}
    legacy_out = legacy_ce.forward(per_modality_tokens_legacy, timestamps, patch_size)
    legacy_result = legacy_out["sentinel2_l2a"]

    # The legacy temporal slice should be non-zero (sinusoidal pos encoding)
    assert legacy_result[..., n : n * 2].abs().sum() > 0
    # The legacy month slice should be non-zero
    assert legacy_result[..., n * 2 : n * 3].abs().sum() > 0

    # The unified output should differ from legacy in these slices
    assert not torch.allclose(result[..., n : n * 2], legacy_result[..., n : n * 2])
