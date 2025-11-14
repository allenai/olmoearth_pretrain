"""Averaged FLOPs per task."""

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from thop import clever_format, profile

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.models import (
    AnySat,
    Clay,
    Croma,
    GalileoWrapper,
    Panopticon,
    PrithviV2,
    PrithviV2Models,
    Satlas,
    Terramind,
)
from olmoearth_pretrain.evals.models.dinov3.dinov3 import DINOv3, DinoV3Models
from olmoearth_pretrain.nn.flexi_vit import Encoder, PoolingType
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, MaskValue


def count_params(model: torch.nn.Module, trainable_only: bool = True):
    """count_params."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


@contextmanager
def suppress_stdout():
    """suppress_stdout."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def construct_eval_samples() -> list[tuple[MaskedOlmoEarthSample, int, bool]]:
    """construct_eval_samples."""
    # construct a list of bs = 1 samples for all 13 eval tasks
    shapes = [
        # T, H, W, multiplier, spatial pool (for AnySat)
        (1, 120, 120, 1, False),  # bigearthnet
        (1, 32, 32, 1, False),  # so2sat
        (1, 64, 64, 2, False),  # brick-kiln, eurosat
        (12, 1, 1, 2, False),  # breizhcrops, cropharvest togo
        (6, 1, 1, 1, False),  # CropHarvest PRC 6
        (1, 256, 256, 2, True),  # cashew plant, SA crop type
        (12, 64, 64, 1, True),  # PASTIS
        (1, 80, 80, 1, True),  # MADOS
        (12, 4, 4, 1, False),  # Nandi
        (12, 32, 32, 1, False),  # AWF
    ]

    B = 1
    samples = []
    for shape in shapes:
        T, H, W, multiplier, spatial_pool = shape
        samples.append(
            (
                MaskedOlmoEarthSample(
                    timestamps=torch.ones(B, T, 3).long(),
                    sentinel2_l2a=torch.ones(
                        B, H, W, T, Modality.SENTINEL2_L2A.num_bands
                    ),
                    sentinel2_l2a_mask=torch.ones(
                        B, H, W, T, Modality.SENTINEL2_L2A.num_bands
                    ).long()
                    * MaskValue.ONLINE_ENCODER.value,
                ),
                multiplier,
                spatial_pool,
            )
        )
    return samples


def flops_per_model(model, samples: list[MaskedOlmoEarthSample, int, bool]) -> float:
    """flops_per_model."""
    total_macs = []
    for i, multiplier, spatial_pool in samples:
        if isinstance(model, Encoder):
            inputs = (i, min(i.sentinel2_l2a.shape[1], 4))
        elif isinstance(model, AnySat):
            inputs = (i, PoolingType.MEAN, spatial_pool)
            model.patch_size = min(i.sentinel2_l2a.shape[1], 4)
        else:
            inputs = (i,)
        with suppress_stdout():
            macs, _ = profile(model, inputs=inputs)
        for _ in range(multiplier):
            total_macs.append(macs)
    return sum(total_macs) / len(total_macs)


if __name__ == "__main__":
    samples = construct_eval_samples()
    num_tasks = sum([s[1] for s in samples])
    # this is what we say in our figure
    assert num_tasks == 13
    modalities = [
        Modality.SENTINEL2_L2A,
        Modality.SENTINEL1,
        Modality.LANDSAT,
        Modality.WORLDCOVER,
        Modality.SRTM,
        Modality.OPENSTREETMAP_RASTER,
        Modality.WRI_CANOPY_HEIGHT_MAP,
        Modality.CDL,
        Modality.WORLDCEREAL,
    ]

    models = [
        Encoder(  # large encoder
            embedding_size=1024,
            min_patch_size=1,
            max_patch_size=8,
            num_heads=16,
            mlp_ratio=4,
            depth=24,
            drop_path=0.1,
            supported_modalities=modalities,
            max_sequence_length=24,
        ),
        Encoder(  # base encoder
            embedding_size=768,
            min_patch_size=1,
            max_patch_size=8,
            num_heads=12,
            mlp_ratio=4,
            depth=12,
            drop_path=0.1,
            supported_modalities=modalities,
            max_sequence_length=24,
        ),
        Encoder(  # tiny encoder
            embedding_size=192,
            min_patch_size=1,
            max_patch_size=8,
            num_heads=3,
            mlp_ratio=4,
            depth=12,
            drop_path=0.1,
            supported_modalities=modalities,
            max_sequence_length=24,
        ),
        Encoder(  # nano encoder
            embedding_size=128,
            min_patch_size=1,
            max_patch_size=8,
            num_heads=8,
            mlp_ratio=4,
            depth=4,
            drop_path=0.1,
            supported_modalities=modalities,
            max_sequence_length=24,
        ),
        Terramind("base"),
        Terramind("large"),
        # for the models below, the paths will
        # need to point to where the model weights are.
        Satlas("."),
        PrithviV2(".", PrithviV2Models.VIT_300),
        PrithviV2(".", PrithviV2Models.VIT_600),
        Panopticon(),
        GalileoWrapper(
            pretrained_path=Path(
                "/Users/gabrieltseng/Documents/code/presto-v3/data/final_models/nano"
            )
        ),
        GalileoWrapper(
            pretrained_path=Path(
                "/Users/gabrieltseng/Documents/code/presto-v3/data/final_models/tiny"
            )
        ),
        GalileoWrapper(
            pretrained_path=Path(
                "/Users/gabrieltseng/Documents/code/presto-v3/data/final_models/base"
            )
        ),
        DINOv3(DinoV3Models.LARGE_SATELLITE),
        DINOv3(DinoV3Models.FULL_7B_SATELLITE),
        Croma(
            size="base",
            load_directory="/Users/gabrieltseng/Documents/code/presto-v3/data/baseline_weights",
        ),
        Croma(
            size="large",
            load_directory="/Users/gabrieltseng/Documents/code/presto-v3/data/baseline_weights",
        ),
        Clay(load_path="clay/clay-v1.5.ckpt"),
        AnySat(),
    ]

    for model in models:
        fpm = flops_per_model(model, samples)
        print(type(model).__name__, count_params(model), fpm, clever_format(fpm))
