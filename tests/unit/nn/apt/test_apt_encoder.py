"""Simple smoke test for APTEncoder and APTEncoderWrapper.

Constructs a minimal encoder and passes synthetic data through
the full forward pass for debugging.
"""

import torch
from einops import repeat

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.apt.apt_encoder import APTEncoder, APTEncoderWrapper
from olmoearth_pretrain.nn.apt.config import APTConfig
from olmoearth_pretrain.train.masking import MaskValue

APT_CONFIG = APTConfig.default_s2_config()
# Align APT base_patch_size with the encoder's patch_size so grid coordinates match
APT_CONFIG.partitioner.base_patch_size = 4
APT_CONFIG.embed.base_patch_size = 4


def _make_encoder(model_size: str = "tiny", patch_size: int = 8):
    """Build an APTEncoder using MODEL_SIZE_ARGS for the given size."""
    args = MODEL_SIZE_ARGS[model_size]
    supported_modalities = [Modality.SENTINEL2_L2A, Modality.LATLON]
    encoder = APTEncoder(
        embedding_size=args["encoder_embedding_size"],
        max_patch_size=patch_size,
        min_patch_size=1,
        num_heads=args["encoder_num_heads"],
        mlp_ratio=args["mlp_ratio"],
        depth=args["encoder_depth"],
        drop_path=0.0,
        supported_modalities=supported_modalities,
        max_sequence_length=256,
        apt_num_scales=APT_CONFIG.partitioner.num_scales,
        apt_base_patch_size=APT_CONFIG.partitioner.base_patch_size,
    )
    return encoder


def _make_sample(
    batch_size: int = 2,
    h: int = 16,
    w: int = 16,
    t: int = 2,
    s2_channels: int = 12,
):
    """Build a synthetic MaskedOlmoEarthSample."""
    sentinel2 = torch.randn(batch_size, h, w, t, s2_channels)
    sentinel2_mask = torch.full(
        (batch_size, h, w, t, s2_channels),
        fill_value=MaskValue.ONLINE_ENCODER.value,
        dtype=torch.long,
    )
    latlon = torch.randn(batch_size, 2)
    latlon_mask = torch.full(
        (batch_size, 2),
        fill_value=MaskValue.ONLINE_ENCODER.value,
        dtype=torch.float32,
    )
    timestamps = torch.tensor([[15, 7, 2023], [15, 8, 2023]], dtype=torch.long)
    timestamps = repeat(timestamps[:t], "t d -> b t d", b=batch_size)

    return MaskedOlmoEarthSample(
        timestamps=timestamps,
        sentinel2_l2a=sentinel2,
        sentinel2_l2a_mask=sentinel2_mask,
        latlon=latlon,
        latlon_mask=latlon_mask,
    )


def test_apt_encoder_wrapper_forward():
    """Smoke test: APTEncoderWrapper forward pass (partitioning + fallback)."""
    encoder = _make_encoder()
    wrapper = APTEncoderWrapper(
        encoder=encoder,
        apt_config=APT_CONFIG,
    )
    sample = _make_sample()
    patch_size = APT_CONFIG.partitioner.base_patch_size

    output = wrapper(sample, patch_size=patch_size)

    assert "tokens_and_masks" in output
    tam = output["tokens_and_masks"]
    assert tam.sentinel2_l2a is not None
    print(f"APTEncoderWrapper output sentinel2_l2a shape: {tam.sentinel2_l2a.shape}")


if __name__ == "__main__":
    print("--- test_apt_encoder_wrapper_forward ---")
    test_apt_encoder_wrapper_forward()
    print("PASSED\n")
