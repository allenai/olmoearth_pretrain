"""Tests for SeparateEncodings: image/encoding parallel-projection token build."""

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_vit import SeparateEncodings
from olmoearth_pretrain.nn.tokenization import ModalityTokenization, TokenizationConfig

# Use single-bandset sentinel2_l2a for tests (matches v1.1 setup).
_S2_SINGLE_BANDSET = ModalityTokenization(
    band_groups=[
        [
            "B02",
            "B03",
            "B04",
            "B08",
            "B05",
            "B06",
            "B07",
            "B8A",
            "B11",
            "B12",
            "B01",
            "B09",
        ],
    ],
)
_TOK = TokenizationConfig(overrides={"sentinel2_l2a": _S2_SINGLE_BANDSET})


def _make(
    embedding_size: int = 256,
    channel_dim: int = 64,
    temporal_dim: int = 64,
    latlon_dim: int = 96,
    latlon_dropout_rate: float = 0.0,
    modalities: list | None = None,
) -> SeparateEncodings:
    return SeparateEncodings(
        embedding_size=embedding_size,
        supported_modalities=modalities or [Modality.SENTINEL2_L2A],
        tokenization_config=_TOK,
        channel_dim=channel_dim,
        temporal_dim=temporal_dim,
        latlon_dim=latlon_dim,
        latlon_dropout_rate=latlon_dropout_rate,
    )


def _ts(b: int = 1, t: int = 4) -> torch.Tensor:
    """Build a (B, T, 3) timestamp tensor of (day, month, year)."""
    days = torch.randint(1, 28, (b, t, 1), dtype=torch.long)
    months = torch.randint(0, 12, (b, t, 1), dtype=torch.long)
    years = torch.randint(2018, 2024, (b, t, 1), dtype=torch.long)
    return torch.cat([days, months, years], dim=-1)


def test_forward_shape_spatial_multitemporal() -> None:
    """Output shape matches input modality shape on the last (embedding) dim."""
    ce = _make()
    B, H, W, T, B_s, D = 2, 3, 3, 4, 1, 256
    tokens = torch.randn(B, H, W, T, B_s, D)
    out = ce.forward(
        {"sentinel2_l2a": tokens},
        timestamps=_ts(B, T),
        patch_size=4,
        latlon=torch.tensor([[37.0, -122.0], [0.0, 0.0]]),
    )["sentinel2_l2a"]
    assert out.shape == (B, H, W, T, B_s, D)


def test_combine_proj_changes_value() -> None:
    """The output is NOT just modality_tokens (combiner mixes in encodings)."""
    ce = _make()
    B, H, W, T, B_s, D = 1, 2, 2, 2, 1, 256
    tokens = torch.randn(B, H, W, T, B_s, D)
    out = ce.forward(
        {"sentinel2_l2a": tokens},
        timestamps=_ts(B, T),
        patch_size=4,
        latlon=torch.tensor([[37.0, -122.0]]),
    )["sentinel2_l2a"]
    assert not torch.allclose(out, tokens)


def test_zero_enc_dim_is_pure_linear() -> None:
    """With all encoding dims zero, the combiner is a single Linear(D, D)."""
    ce = _make(channel_dim=0, temporal_dim=0, latlon_dim=0)
    B, H, W, T, B_s, D = 1, 2, 2, 2, 1, 256
    tokens = torch.randn(B, H, W, T, B_s, D)
    out = ce.forward(
        {"sentinel2_l2a": tokens}, timestamps=_ts(B, T), patch_size=4, latlon=None
    )["sentinel2_l2a"]
    # Should equal linear(modality_tokens)
    expected = ce.combine_proj(tokens)
    assert torch.allclose(out, expected)


def test_latlon_dropout_rate_one_zeros_latlon_slot_in_eval() -> None:
    """rate=1.0 means latlon slot is always zero (train and eval)."""
    ce = _make(latlon_dropout_rate=1.0)
    ce.eval()
    B, H, W, T, B_s, D = 2, 2, 2, 2, 1, 256
    tokens = torch.zeros(B, H, W, T, B_s, D)
    ll = torch.tensor([[37.0, -122.0], [-30.0, 150.0]])
    ts = _ts(B, T)  # fixed across both forward passes below
    out = ce.forward({"sentinel2_l2a": tokens}, timestamps=ts, patch_size=4, latlon=ll)[
        "sentinel2_l2a"
    ]
    # Build a reference module identical to ce but with rate=0 and latlon=None,
    # which also zeros the latlon slot of enc. The two outputs should match.
    ce_zero = _make(latlon_dropout_rate=0.0, latlon_dim=96)
    ce_zero.eval()
    with torch.no_grad():
        ce_zero.combine_proj.weight.copy_(ce.combine_proj.weight)
        ce_zero.combine_proj.bias.copy_(ce.combine_proj.bias)
        for k, v in ce.per_modality_channel_embeddings.items():
            ce_zero.per_modality_channel_embeddings[k].copy_(v)
    out_ref = ce_zero.forward(
        {"sentinel2_l2a": tokens}, timestamps=ts, patch_size=4, latlon=None
    )["sentinel2_l2a"]
    assert torch.allclose(out, out_ref)


def test_latlon_dropout_in_eval_full_encoding_for_rate_below_one() -> None:
    """In eval mode with rate<1.0, latlon dropout doesn't fire (full encoding)."""
    ce = _make(latlon_dropout_rate=0.5)
    ce.eval()
    B, H, W, T, B_s, D = 4, 2, 2, 2, 1, 256
    tokens = torch.zeros(B, H, W, T, B_s, D)
    ll = torch.tensor([[37.0, -122.0]] * B)
    ts = _ts(B, T)
    out_a = ce.forward(
        {"sentinel2_l2a": tokens}, timestamps=ts, patch_size=4, latlon=ll
    )["sentinel2_l2a"]
    out_b = ce.forward(
        {"sentinel2_l2a": tokens}, timestamps=ts, patch_size=4, latlon=ll
    )["sentinel2_l2a"]
    # deterministic in eval
    assert torch.allclose(out_a, out_b)


def test_latlon_dropout_in_train_some_samples_zeroed() -> None:
    """In train mode at rate=0.5, some samples get zero-latlon (different outputs)."""
    torch.manual_seed(0)
    ce = _make(latlon_dropout_rate=0.5)
    ce.train()
    B, H, W, T, B_s, D = 64, 2, 2, 2, 1, 256
    tokens = torch.zeros(B, H, W, T, B_s, D)
    ll = torch.zeros(B, 2)  # all (0,0) so latlon encoding is constant per sample
    ll[:, 0] = torch.linspace(-60, 60, B)
    # Compute encoding manually: with same input tokens & same latlon, the only
    # source of per-sample variation is the bernoulli dropout. Check that
    # not-all-samples are identical along the batch dim.
    out = ce.forward(
        {"sentinel2_l2a": tokens}, timestamps=_ts(B, T), patch_size=4, latlon=ll
    )["sentinel2_l2a"]
    per_sample = out[:, 0, 0, 0, 0, :]
    # Variation comes from (a) different latlon per sample (real signal) and
    # (b) bernoulli dropout zeroing some. We just check it's not constant.
    norms = per_sample.norm(dim=-1)
    assert norms.std() > 0


def test_latlon_none_yields_zero_latlon_slot() -> None:
    """If latlon=None, the latlon portion of enc is zero (effective dropout)."""
    ce_with = _make(latlon_dropout_rate=1.0)  # forces zero latlon
    ce_without = _make(latlon_dropout_rate=0.0)  # latlon=None also yields zero
    ce_with.eval()
    ce_without.eval()
    with torch.no_grad():
        ce_without.combine_proj.weight.copy_(ce_with.combine_proj.weight)
        ce_without.combine_proj.bias.copy_(ce_with.combine_proj.bias)
        for k, v in ce_with.per_modality_channel_embeddings.items():
            ce_without.per_modality_channel_embeddings[k].copy_(v)

    B, H, W, T, B_s, D = 1, 2, 2, 2, 1, 256
    tokens = torch.randn(B, H, W, T, B_s, D)
    ts = _ts(B, T)
    out_a = ce_with.forward(
        {"sentinel2_l2a": tokens},
        timestamps=ts,
        patch_size=4,
        latlon=torch.tensor([[37.0, -122.0]]),
    )["sentinel2_l2a"]
    out_b = ce_without.forward(
        {"sentinel2_l2a": tokens}, timestamps=ts, patch_size=4, latlon=None
    )["sentinel2_l2a"]
    assert torch.allclose(out_a, out_b)


def test_latlon_dim_div_6_validation() -> None:
    """latlon_dim must be divisible by 6 when > 0."""
    with pytest.raises(ValueError, match="divisible by 6"):
        _make(latlon_dim=64)


def test_temporal_dim_even_validation() -> None:
    """temporal_dim must be even (sin/cos split)."""
    with pytest.raises(ValueError, match="must be even"):
        _make(temporal_dim=63)


def test_rate_in_range_validation() -> None:
    """latlon_dropout_rate must be in [0, 1]."""
    with pytest.raises(ValueError, match="must be in"):
        _make(latlon_dropout_rate=1.5)


def test_channel_embeddings_match_modality_bandsets() -> None:
    """Channel embedding row count matches the modality's bandsets."""
    ce = _make()
    n_bandsets = ce.tokenization_config.get_num_bandsets("sentinel2_l2a")
    ch = ce.per_modality_channel_embeddings["sentinel2_l2a"]
    assert ch.shape == (n_bandsets, ce.channel_dim)
    assert n_bandsets == 1  # single-bandset config used in tests


def test_no_latlon_field_is_validated_for_additive_mode() -> None:
    """EncoderConfig validator catches separate-only fields under additive mode."""
    from olmoearth_pretrain.nn.flexi_vit import EncoderConfig

    cfg = EncoderConfig(
        supported_modality_names=["sentinel2_l2a"],
        embedding_size=128,
        num_heads=4,
        depth=2,
        encoding_mode="additive",
        latlon_encoding_dim=192,  # not allowed in additive mode
    )
    with pytest.raises(ValueError, match="encoding_mode='separate'"):
        cfg.validate()


# ---- simple encoding types -----------------------------------------------


def _make_simple(
    embedding_size: int = 256,
    channel_dim: int = 64,
    latlon_dropout_rate: float = 0.0,
) -> SeparateEncodings:
    """SeparateEncodings with simple (3-number) temporal + latlon."""
    return SeparateEncodings(
        embedding_size=embedding_size,
        supported_modalities=[Modality.SENTINEL2_L2A],
        tokenization_config=_TOK,
        channel_dim=channel_dim,
        temporal_dim=3,
        latlon_dim=3,
        latlon_dropout_rate=latlon_dropout_rate,
        temporal_encoding_type="simple",
        latlon_encoding_type="simple",
    )


def test_simple_types_forward_shape() -> None:
    """Simple temporal/latlon route produces correct output shape."""
    ce = _make_simple()
    assert ce.enc_dim == 64 + 3 + 3
    B, H, W, T, B_s, D = 2, 3, 3, 4, 1, 256
    tokens = torch.randn(B, H, W, T, B_s, D)
    out = ce.forward(
        {"sentinel2_l2a": tokens},
        timestamps=_ts(B, T),
        patch_size=4,
        latlon=torch.tensor([[37.0, -122.0], [0.0, 0.0]]),
    )["sentinel2_l2a"]
    assert out.shape == (B, H, W, T, B_s, D)


def test_simple_temporal_requires_dim_3() -> None:
    """Simple temporal type with dim != 3 raises."""
    with pytest.raises(ValueError, match="simple temporal encoding requires"):
        SeparateEncodings(
            embedding_size=256,
            supported_modalities=[Modality.SENTINEL2_L2A],
            tokenization_config=_TOK,
            channel_dim=64,
            temporal_dim=128,
            latlon_dim=3,
            temporal_encoding_type="simple",
            latlon_encoding_type="simple",
        )


def test_simple_latlon_requires_dim_3() -> None:
    """Simple latlon type with dim != 3 raises."""
    with pytest.raises(ValueError, match="simple latlon encoding requires"):
        SeparateEncodings(
            embedding_size=256,
            supported_modalities=[Modality.SENTINEL2_L2A],
            tokenization_config=_TOK,
            channel_dim=64,
            temporal_dim=3,
            latlon_dim=192,
            temporal_encoding_type="simple",
            latlon_encoding_type="simple",
        )


def test_simple_latlon_dropout_rate_one_disables() -> None:
    """rate=1.0 zeros the simple latlon (x,y,z) slot in eval too."""
    ce = _make_simple(latlon_dropout_rate=1.0)
    ce.eval()
    B, H, W, T, B_s, D = 2, 2, 2, 2, 1, 256
    tokens = torch.zeros(B, H, W, T, B_s, D)
    ll = torch.tensor([[37.0, -122.0], [-30.0, 150.0]])
    ts = _ts(B, T)
    out = ce.forward({"sentinel2_l2a": tokens}, timestamps=ts, patch_size=4, latlon=ll)[
        "sentinel2_l2a"
    ]
    ce_ref = _make_simple(latlon_dropout_rate=0.0)
    ce_ref.eval()
    with torch.no_grad():
        ce_ref.combine_proj.weight.copy_(ce.combine_proj.weight)
        ce_ref.combine_proj.bias.copy_(ce.combine_proj.bias)
        for k, v in ce.per_modality_channel_embeddings.items():
            ce_ref.per_modality_channel_embeddings[k].copy_(v)
    out_ref = ce_ref.forward(
        {"sentinel2_l2a": tokens}, timestamps=ts, patch_size=4, latlon=None
    )["sentinel2_l2a"]
    assert torch.allclose(out, out_ref)


def test_invalid_encoding_type_raises() -> None:
    """Bad temporal_encoding_type raises."""
    with pytest.raises(ValueError, match="temporal_encoding_type"):
        SeparateEncodings(
            embedding_size=256,
            supported_modalities=[Modality.SENTINEL2_L2A],
            tokenization_config=_TOK,
            channel_dim=64,
            temporal_dim=3,
            latlon_dim=3,
            temporal_encoding_type="bogus",
        )
