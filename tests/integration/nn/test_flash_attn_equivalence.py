"""Integration test: flash attention vs SDPA produce equivalent outputs through the full model."""

import copy
import logging

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

try:
    import flash_attn  # noqa: F401
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

HAS_CUDA = torch.cuda.is_available()

requires_flash = pytest.mark.skipif(
    not (HAS_CUDA and HAS_FLASH_ATTN),
    reason="Requires CUDA GPU and flash-attn package",
)


@pytest.fixture(autouse=True)
def _gpu_test_setup():
    """Disable deterministic mode (incompatible with CuBLAS) for GPU tests."""
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(True)

MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
    Modality.WORLDCOVER.name,
    Modality.LATLON.name,
]


def _build_model(use_flash_attn: bool) -> LatentMIMConfig:
    return LatentMIMConfig(
        encoder_config=EncoderConfig(
            supported_modality_names=MODALITIES,
            embedding_size=64, max_patch_size=8, num_heads=4, mlp_ratio=1.0,
            depth=2, drop_path=0.0, max_sequence_length=12,
            use_flash_attn=use_flash_attn,
        ),
        decoder_config=PredictorConfig(
            supported_modality_names=MODALITIES,
            encoder_embedding_size=64, decoder_embedding_size=64,
            depth=2, mlp_ratio=1.0, num_heads=4, max_sequence_length=12,
            drop_path=0.0, use_flash_attn=use_flash_attn,
        ),
    )


def _make_fake_batch(device: torch.device, all_encoder: bool = False):
    """Create a MaskedOlmoEarthSample.

    Args:
        device: torch device
        all_encoder: If True, all masks are ONLINE_ENCODER (eval scenario).
            If False, ~50% ONLINE_ENCODER / ~50% DECODER (training scenario).
    """
    B, H, W, T = 2, 8, 8, 3
    s2_C = Modality.SENTINEL2_L2A.num_bands
    s1_C = Modality.SENTINEL1.num_bands
    wc_C = Modality.WORLDCOVER.num_bands
    s2_bs = Modality.SENTINEL2_L2A.num_band_sets
    s1_bs = Modality.SENTINEL1.num_band_sets
    wc_bs = Modality.WORLDCOVER.num_band_sets
    timestamps = torch.randint(0, 12, (B, T, 3), dtype=torch.long, device=device)
    if all_encoder:
        s2_mask = torch.zeros(B, H, W, T, s2_bs, dtype=torch.long, device=device)
        s1_mask = torch.zeros(B, H, W, T, s1_bs, dtype=torch.long, device=device)
    else:
        s2_mask = torch.randint(0, 2, (B, H, W, T, s2_bs), device=device) * MaskValue.DECODER.value
        s1_mask = torch.randint(0, 2, (B, H, W, T, s1_bs), device=device) * MaskValue.DECODER.value
    wc_mask = torch.zeros(B, H, W, 1, wc_bs, dtype=torch.long, device=device)
    ll_mask = torch.zeros(B, 1, dtype=torch.long, device=device)
    return MaskedOlmoEarthSample(
        timestamps=timestamps,
        sentinel2_l2a=torch.randn(B, H, W, T, s2_C, device=device),
        sentinel2_l2a_mask=s2_mask,
        sentinel1=torch.randn(B, H, W, T, s1_C, device=device),
        sentinel1_mask=s1_mask,
        worldcover=torch.randn(B, H, W, 1, wc_C, device=device),
        worldcover_mask=wc_mask,
        latlon=torch.randn(B, 2, device=device),
        latlon_mask=ll_mask,
    )


@requires_flash
@pytest.mark.parametrize("patch_size", [1, 2, 4, 8])
def test_flash_vs_sdpa_train_mode(patch_size: int) -> None:
    """Train mode: mixed masks (ONLINE_ENCODER + DECODER), both paths apply masking."""
    device = torch.device("cuda")
    torch.manual_seed(42)
    model_sdpa = _build_model(use_flash_attn=False).build().to(device=device, dtype=torch.bfloat16).train()
    model_flash = _build_model(use_flash_attn=True).build().to(device=device, dtype=torch.bfloat16)
    model_flash.load_state_dict(copy.deepcopy(model_sdpa.state_dict()))
    model_flash.train()
    torch.manual_seed(42)
    batch = _make_fake_batch(device, all_encoder=False)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        latent_sdpa, decoded_sdpa, pooled_sdpa, _, _ = model_sdpa(batch, patch_size)
        latent_flash, decoded_flash, pooled_flash, _, _ = model_flash(batch, patch_size)
    dec_s = decoded_sdpa.flatten_all_tokens_and_masks()[0]
    dec_f = decoded_flash.flatten_all_tokens_and_masks()[0]
    max_diff = (dec_s - dec_f).abs().max().item()
    mean_diff = (dec_s - dec_f).abs().mean().item()
    logger.info(f"TRAIN patch_size={patch_size}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    assert torch.allclose(dec_s, dec_f, rtol=1e-3, atol=1e-3), (
        f"Train mode patch_size={patch_size}: max_diff={max_diff}, mean_diff={mean_diff}"
    )


@requires_flash
@pytest.mark.parametrize("patch_size", [1, 2, 4, 8])
def test_flash_vs_sdpa_eval_encoder_only(patch_size: int) -> None:
    """Eval mode encoder-only: all tokens are ONLINE_ENCODER (no padding).

    Real eval (PASTIS etc.) only uses the encoder for feature extraction,
    never the decoder. With all-ONLINE_ENCODER masks there are no decoder
    tokens, so the decoder can't run (flash attn crashes on empty queries).
    """
    device = torch.device("cuda")
    torch.manual_seed(42)
    model_sdpa = _build_model(use_flash_attn=False).build().to(device=device, dtype=torch.bfloat16).eval()
    model_flash = _build_model(use_flash_attn=True).build().to(device=device, dtype=torch.bfloat16)
    model_flash.load_state_dict(copy.deepcopy(model_sdpa.state_dict()))
    model_flash.eval()
    torch.manual_seed(42)
    batch = _make_fake_batch(device, all_encoder=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out_sdpa = model_sdpa.encoder(batch, patch_size=patch_size)
        out_flash = model_flash.encoder(batch, patch_size=patch_size)
    sdpa_tokens = out_sdpa["tokens_and_masks"].flatten_all_tokens_and_masks()[0]
    flash_tokens = out_flash["tokens_and_masks"].flatten_all_tokens_and_masks()[0]
    max_diff = (sdpa_tokens - flash_tokens).abs().max().item()
    mean_diff = (sdpa_tokens - flash_tokens).abs().mean().item()
    logger.info(f"EVAL encoder patch_size={patch_size}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    assert torch.allclose(sdpa_tokens, flash_tokens, rtol=1e-3, atol=1e-3), (
        f"Eval encoder patch_size={patch_size}: max_diff={max_diff}, mean_diff={mean_diff}"
    )
