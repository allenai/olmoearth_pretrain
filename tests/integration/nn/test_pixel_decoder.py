"""Integration tests for the pixel-representation decoders (reconstruction + map probe)."""

import logging

import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, OlmoEarthSample
from olmoearth_pretrain.nn.dual_res_encoder import (
    DualResEncoder,
    DualResEncoderConfig,
)
from olmoearth_pretrain.nn.pixel_decoder import (
    PixelMapProbe,
    PixelReconstructionDecoder,
    build_recon_groupings,
)
from olmoearth_pretrain.train.masking import MaskingConfig

logger = logging.getLogger(__name__)

EMB = 32
PIXEL_EMB = 16
PATCH_SIZE = 4
S2 = Modality.SENTINEL2_L2A.name
WC = Modality.WORLDCOVER_ONEHOT.name
SRTM = Modality.SRTM.name
OSM = Modality.OPENSTREETMAP_RASTER.name
WORLDCEREAL = Modality.WORLDCEREAL.name
B, H, W, T = 2, 8, 8, 12
WC_CLASSES = Modality.get(WC).num_bands


def _encoder() -> DualResEncoder:
    return DualResEncoderConfig(
        supported_modality_names=[S2],
        embedding_size=EMB,
        max_patch_size=8,
        min_patch_size=1,
        num_heads=4,
        mlp_ratio=2.0,
        depth=2,
        drop_path=0.0,
        max_sequence_length=T,
        pixel_embedding_size=PIXEL_EMB,
        pixel_num_heads=4,
        pixel_mlp_ratio=2.0,
    ).build()


def _sample(with_maps: bool = False) -> MaskedOlmoEarthSample:
    kwargs = dict(
        sentinel2_l2a=torch.randn(B, H, W, T, Modality.SENTINEL2_L2A.num_bands),
        timestamps=torch.stack(
            [
                torch.randint(1, 28, (B, T)),
                torch.randint(1, 12, (B, T)),
                torch.full((B, T), 2023),
            ],
            dim=-1,
        ),
    )
    if with_maps:
        labels = torch.randint(0, WC_CLASSES, (B, H, W, 1))
        kwargs["worldcover_onehot"] = torch.nn.functional.one_hot(
            labels, WC_CLASSES
        ).float()
        kwargs["srtm"] = torch.randn(B, H, W, 1, 1)
        kwargs["openstreetmap_raster"] = (
            torch.rand(B, H, W, 1, Modality.get(OSM).num_bands) > 0.5
        ).float()
        kwargs["worldcereal"] = (
            torch.rand(B, H, W, 1, Modality.get(WORLDCEREAL).num_bands) > 0.5
        ).float()
    sample = OlmoEarthSample(**kwargs)
    strategy = MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "random_ratio": 1.0,
        }
    ).build()
    return strategy.apply_mask(sample, patch_size=PATCH_SIZE)


def test_build_recon_groupings_only_decodes_with_online_context() -> None:
    """DECODER units are kept only in (patch, bandset) groups that have ONLINE context."""
    torch.manual_seed(0)
    b, g, t, bs = 2, 2, 4, 1
    online = torch.zeros(b, g, g, t, bs, dtype=torch.bool)
    decode = torch.zeros(b, g, g, t, bs, dtype=torch.bool)
    # Patch (0,0): 1 online + 1 decode -> decode kept. Patch (1,1): decode only -> dropped.
    online[0, 0, 0, 0, 0] = True
    decode[0, 0, 0, 1, 0] = True
    decode[0, 1, 1, 2, 0] = True
    rg = build_recon_groupings(online, decode)
    assert rg.num_groups == 1
    assert rg.num_decode == 1  # the patch-(1,1) decode is dropped (no online context)


def test_reconstruction_decoder_trains_encoder_and_heads() -> None:
    """Reconstruction MSE is finite and its grads reach the encoder + decoder heads."""
    model = _encoder()
    dec = PixelReconstructionDecoder(
        supported_modality_names=[S2],
        pixel_embedding_size=PIXEL_EMB,
        num_heads=4,
        mlp_ratio=2.0,
        depth=2,
    )
    x = _sample()
    out = model(x, patch_size=PATCH_SIZE)
    loss = dec(out["pixel_branch"], x, PATCH_SIZE)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in dec.blocks.parameters())
    assert dec.mask_token.grad is not None
    # Grad flows back into the encoder's pixel branch.
    assert any(p.grad is not None for p in model.pixel_self_blocks.parameters())


def test_map_probe_ce_mse_and_bce() -> None:
    """The map probe produces finite CE + MSE + BCE losses and trains through reps."""
    model = _encoder()
    probe = PixelMapProbe(
        pixel_embedding_size=PIXEL_EMB,
        map_targets={WC: "ce", SRTM: "mse", OSM: "bce", WORLDCEREAL: "bce"},
    )
    x = _sample(with_maps=True)
    out = model(x, patch_size=PATCH_SIZE)
    loss = probe(out["pixel_branch"], x, PATCH_SIZE)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in probe.heads.parameters())
    assert any(p.grad is not None for p in model.pixel_self_blocks.parameters())


def test_reconstruction_finite_when_no_decode_targets() -> None:
    """With an all-ONLINE sample (no decode units) the recon loss is finite (keep-alive)."""
    model = _encoder()
    dec = PixelReconstructionDecoder(
        supported_modality_names=[S2], pixel_embedding_size=PIXEL_EMB, depth=1
    )
    x = _sample().unmask()  # no DECODER tokens remain
    out = model(x, patch_size=PATCH_SIZE)
    loss = dec(out["pixel_branch"], x, PATCH_SIZE)
    assert torch.isfinite(loss)
    loss.backward()
    # Keep-alive: heads still receive (zero) gradient so FSDP collectives fire.
    assert all(p.grad is not None for head in dec.heads[S2] for p in head.parameters())
