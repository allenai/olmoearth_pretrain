"""Integration tests for the encoder class token (encoder + decoder visible)."""

from typing import Any

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.flexi_vit import Encoder, EncoderConfig, Predictor
from olmoearth_pretrain.nn.latent_mim import LatentMIM
from olmoearth_pretrain.nn.utils import unpack_encoder_output

SUPPORTED_MODALITIES = [Modality.SENTINEL2_L2A, Modality.LATLON, Modality.WORLDCOVER]
EMBEDDING_SIZE = 16
DECODER_EMBEDDING_SIZE = 16


def _build_encoder(**overrides: Any) -> Encoder:
    """Build a small Encoder with the class token enabled by default."""
    kwargs: dict[str, Any] = dict(
        supported_modalities=SUPPORTED_MODALITIES,
        embedding_size=EMBEDDING_SIZE,
        max_patch_size=8,
        min_patch_size=1,
        num_heads=2,
        mlp_ratio=2.0,
        max_sequence_length=12,
        depth=2,
        drop_path=0.0,
        use_class_token=True,
    )
    kwargs.update(overrides)
    return Encoder(**kwargs)


def _build_predictor() -> Predictor:
    """Build a small Predictor matching the test encoder."""
    return Predictor(
        supported_modalities=SUPPORTED_MODALITIES,
        encoder_embedding_size=EMBEDDING_SIZE,
        decoder_embedding_size=DECODER_EMBEDDING_SIZE,
        depth=2,
        mlp_ratio=2.0,
        num_heads=2,
        max_sequence_length=12,
    )


def _masked_sample(
    mask_value: int = MaskValue.ONLINE_ENCODER.value, B: int = 2
) -> MaskedOlmoEarthSample:
    """Build a small masked sample with S2 (visible) and worldcover (decode)."""
    H = W = 4
    T = 2
    C = Modality.SENTINEL2_L2A.num_bands
    sentinel2_l2a = torch.randn(B, H, W, T, C)
    sentinel2_l2a_mask = torch.full(
        (B, H, W, T, C), fill_value=mask_value, dtype=torch.long
    )
    latlon = torch.tensor([[37.0, -122.0]] * B)
    latlon_mask = torch.full(
        (B, Modality.LATLON.num_bands),
        fill_value=MaskValue.ONLINE_ENCODER.value,
        dtype=torch.long,
    )
    worldcover = torch.randn(B, H, W, 1, 1)
    worldcover_mask = torch.full(
        (B, H, W, 1, 1), fill_value=MaskValue.DECODER.value, dtype=torch.long
    )
    days = torch.randint(1, 28, (B, T, 1), dtype=torch.long)
    months = torch.randint(0, 12, (B, T, 1), dtype=torch.long)
    years = torch.randint(2018, 2025, (B, T, 1), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=-1)
    return MaskedOlmoEarthSample(
        sentinel2_l2a=sentinel2_l2a,
        sentinel2_l2a_mask=sentinel2_l2a_mask,
        latlon=latlon,
        latlon_mask=latlon_mask,
        worldcover=worldcover,
        worldcover_mask=worldcover_mask,
        timestamps=timestamps,
    )


def test_encoder_emits_class_token() -> None:
    """class_token is a [B, D] output key; project_aggregated is its projection."""
    encoder = _build_encoder()
    x = _masked_sample()
    out = encoder(x, patch_size=4)
    assert "class_token" in out
    assert out["class_token"].shape == (2, EMBEDDING_SIZE)
    assert torch.isfinite(out["class_token"]).all()
    assert out["project_aggregated"].shape == (2, EMBEDDING_SIZE)


def test_encoder_class_token_fast_pass() -> None:
    """The class token is emitted in the eval fast_pass too."""
    encoder = _build_encoder()
    encoder.eval()
    x = _masked_sample()
    with torch.no_grad():
        out = encoder(x, patch_size=4, fast_pass=True)
    assert out["class_token"].shape == (2, EMBEDDING_SIZE)
    assert torch.isfinite(out["class_token"]).all()


def test_class_token_differs_per_sample() -> None:
    """Different inputs produce different class tokens (it attends to content)."""
    encoder = _build_encoder()
    x = _masked_sample()
    out = encoder(x, patch_size=4)
    cls = out["class_token"]
    assert not torch.allclose(cls[0], cls[1])


def test_tokens_and_masks_unaffected_by_class_token() -> None:
    """Patch-token output shapes are identical with and without the class token."""
    torch.manual_seed(0)
    with_cls = _build_encoder(use_class_token=True)
    torch.manual_seed(0)
    without_cls = _build_encoder(use_class_token=False)
    x = _masked_sample()
    out_a = with_cls(x, patch_size=4)["tokens_and_masks"]
    out_b = without_cls(x, patch_size=4)["tokens_and_masks"]
    for modality in out_a.modalities:
        assert getattr(out_a, modality).shape == getattr(out_b, modality).shape


def test_class_token_gradient_flows_from_instance_path() -> None:
    """project_aggregated backprop reaches the class token parameter."""
    encoder = _build_encoder()
    x = _masked_sample()
    out = encoder(x, patch_size=4)
    out["project_aggregated"].sum().backward()
    assert encoder.class_token.grad is not None
    assert encoder.class_token.grad.abs().sum() > 0


def test_class_token_visible_to_decoder() -> None:
    """The decoder reads the class token: token-loss backprop reaches it."""
    encoder = _build_encoder()
    decoder = _build_predictor()
    model = LatentMIM(encoder=encoder, decoder=decoder)
    x = _masked_sample()
    output_dict = model.encoder(x, patch_size=4)
    latent, _, decoder_kwargs = unpack_encoder_output(output_dict)
    assert "class_token" in decoder_kwargs
    decoded = model.decoder(
        latent, timestamps=x.timestamps, patch_size=4, **decoder_kwargs
    )
    # Backprop only through the decoded (DECODER-masked) tokens — the class
    # token receives gradient solely via the cross-attention context path.
    model.encoder.class_token.grad = None
    decoded.worldcover.sum().backward()
    assert model.encoder.class_token.grad is not None
    assert model.encoder.class_token.grad.abs().sum() > 0


def test_latent_mim_forward_with_class_token() -> None:
    """End-to-end LatentMIM forward works with the class token enabled."""
    model = LatentMIM(encoder=_build_encoder(), decoder=_build_predictor())
    x = _masked_sample()
    latent, decoded, pooled, reconstructed, _ = model(x, patch_size=4)
    assert pooled.shape == (2, EMBEDDING_SIZE)
    assert torch.isfinite(pooled).all()
    assert reconstructed is None
    assert torch.isfinite(decoded.worldcover).all()


def test_class_token_with_rope_and_separate_encodings() -> None:
    """The combined-branch configuration: RoPE + separate simple encodings."""
    encoder = _build_encoder(
        spatial_pos_encoding="rope",
        rope_coordinate_scale=0.25,
        encoding_mode="separate",
        channel_encoding_dim=8,
        temporal_encoding_dim=4,
        latlon_encoding_dim=0,
        temporal_encoding_type="simple",
    )
    x = _masked_sample()
    out = encoder(x, patch_size=4)
    assert out["class_token"].shape == (2, EMBEDDING_SIZE)
    assert torch.isfinite(out["class_token"]).all()


def test_class_token_rejects_flash_attn() -> None:
    """Class token + flash attention is explicitly unsupported."""
    with pytest.raises(ValueError, match="use_flash_attn"):
        _build_encoder(use_flash_attn=True)
    config = EncoderConfig(
        [m.name for m in SUPPORTED_MODALITIES],
        use_class_token=True,
        use_flash_attn=True,
    )
    with pytest.raises(ValueError, match="use_flash_attn"):
        config.validate()


def test_class_token_with_register_tokens() -> None:
    """Class token coexists with register tokens (registers popped first)."""
    encoder = _build_encoder(num_register_tokens=2)
    x = _masked_sample()
    out = encoder(x, patch_size=4)
    assert out["class_token"].shape == (2, EMBEDDING_SIZE)
    assert torch.isfinite(out["class_token"]).all()
