"""Trying to prototype fitting everything into olmo core."""

import logging

from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import WSD
from upath import UPath

from olmoearth_pretrain.data.concat import OlmoEarthConcatDatasetConfig
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.evals.models.prithviv2.prithviv2 import PrithviV2Config
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
    OlmoEarthVisualizeConfig,
)
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1


def build_model_config(common: CommonComponents) -> PrithviV2Config:
    """Build the model config for an experiment."""
    model_config = PrithviV2Config()
    return model_config


def build_train_module_config(
    common: CommonComponents,
) -> LatentMIMTrainModuleConfig:
    """Build the train module config for an experiment."""
    scheduler = WSD(
        decay_steps=0,
        decay_fraction=None,
        warmup_steps=0,
    )
    return LatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02, fused=True),
        rank_microbatch_size=64,  # Can be 256 on titan, needs to be <= 64 (i think) on jupiter
        masking_config=MaskingConfig(
            strategy_config={
                "type": "modality_cross_random",
                "encode_ratio": 0.5,
                "decode_ratio": 0.5,
                "allow_encoding_decoding_same_bandset": True,
                "min_decoded_bandsets": 6,
                "only_decode_modalities": [
                    Modality.OPENSTREETMAP_RASTER.name,
                    Modality.WORLDCOVER.name,
                    Modality.SRTM.name,
                ],
            }
        ),
        loss_config=LossConfig(
            loss_config={
                "type": "patch_discrimination_new",
            }
        ),
        token_exit_cfg={modality: 0 for modality in common.training_modalities},
        max_grad_norm=1.0,
        scheduler=scheduler,
        ema_decay=(1.0, 1.0),
        dp_config=None,  # FSDP is not supported for DINOv2
    )


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Build the dataloader config for an experiment."""
    # things should be set during building

    return OlmoEarthDataLoaderConfig(
        num_workers=0,
        global_batch_size=512,
        token_budget=1500,
        prefetch_factor=4,
        sampled_hw_p_list=list(range(5, 13)),
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        work_dir=common.save_folder,
        seed=3622,
    )


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Build the dataset config for an experiment."""
    dataset_configs = [
        # ENsure this is any existing dataset
        OlmoEarthDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/osmbig/h5py_data_w_missing_timesteps_zstd_3_128_x_4/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/1297928",
            training_modalities=common.training_modalities,
        ),
    ]
    return OlmoEarthConcatDatasetConfig(dataset_configs=dataset_configs)


def build_visualize_config(common: CommonComponents) -> OlmoEarthVisualizeConfig:
    """Build the visualize config for an experiment."""
    return OlmoEarthVisualizeConfig(
        num_samples=None,
        output_dir=str(UPath(common.save_folder) / "visualizations"),
        std_multiplier=2.0,
    )
