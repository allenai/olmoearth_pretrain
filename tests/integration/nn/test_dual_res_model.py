"""Integration test for the dual-resolution model wrapper (coarse + pixel heads)."""

import logging

import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, OlmoEarthSample
from olmoearth_pretrain.nn.dual_res_encoder import DualResEncoderConfig
from olmoearth_pretrain.nn.dual_res_model import DualResLatentMIMConfig
from olmoearth_pretrain.nn.flexi_vit import PredictorConfig
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


def _model_config(map_targets: dict[str, str] | None = None) -> DualResLatentMIMConfig:
    encoder_config = DualResEncoderConfig(
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
    )
    decoder_config = PredictorConfig(
        supported_modality_names=[S2],
        encoder_embedding_size=EMB,
        decoder_embedding_size=EMB,
        num_heads=4,
        mlp_ratio=2.0,
        depth=2,
        max_sequence_length=T,
    )
    return DualResLatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        pixel_reconstruction=True,
        pixel_recon_depth=1,
        map_targets=map_targets or {},
        map_num_classes={WC: WC_CLASSES} if map_targets else {},
    )


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
    strategy = MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "random_ratio": 1.0,
        }
    ).build()
    return strategy.apply_mask(OlmoEarthSample(**kwargs), patch_size=PATCH_SIZE)


def test_dual_res_model_returns_pixel_loss() -> None:
    """Forward returns the 6-tuple; the pixel loss is finite and backprops end-to-end."""
    model = _model_config().build()
    x = _sample()

    latent, decoded, pooled, reconstructed, extra, pixel_loss = model(
        x, patch_size=PATCH_SIZE
    )
    assert decoded.sentinel2_l2a is not None
    assert pixel_loss is not None
    assert torch.isfinite(pixel_loss)
    assert "pixel_loss" in extra and "recon" in extra["pixel_loss"]

    (decoded.sentinel2_l2a.sum() + pixel_loss).backward()
    assert model.pixel_reconstruction_decoder is not None
    assert any(
        p.grad is not None for p in model.pixel_reconstruction_decoder.parameters()
    )
    assert any(p.grad is not None for p in model.encoder.pixel_self_blocks.parameters())


def test_dual_res_model_with_map_probe() -> None:
    """With map targets, the returned pixel loss includes the map term and trains."""
    model = _model_config(
        map_targets={WC: "ce", SRTM: "mse", OSM: "bce", WORLDCEREAL: "bce"}
    ).build()
    x = _sample(with_maps=True)

    *_, extra, pixel_loss = model(x, patch_size=PATCH_SIZE)
    assert "map" in extra["pixel_loss"]
    assert torch.isfinite(pixel_loss)
    pixel_loss.backward()
    assert model.pixel_map_probe is not None
    assert any(p.grad is not None for p in model.pixel_map_probe.parameters())
