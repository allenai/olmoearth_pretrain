"""v1.2 encoder + auxiliary NAIP conditional pix2pix-GAN branch.

This is a single-view variant of the v1.2 baseline (``base.py``): it keeps the
latent-MIM patch-discrimination objective but adds a NAIP generator on top of
the encoder's pooled spatial embedding plus a conditional discriminator. NAIP
(``naip_10``) is a decode-only modality, so the generator must synthesize it
from the (masked) Sentinel-2 / other encode tokens.

Validate before launching::

    python3 scripts/official/v1_2/naip_gan.py dry_run naip_gan local
"""

import logging

import base as v1_2_base
from olmo_core.config import DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import CosWithWarmup
from olmo_core.train.config import TrainerConfig

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
    OlmoEarthVisualizeConfig,
    SubCmd,
    main,
)
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.naip_gan import (
    NaipDiscriminatorConfig,
    NaipGanModelConfig,
    NaipGeneratorConfig,
)
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.naip_gan import NaipGanTrainModuleConfig

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = v1_2_base.MAX_PATCH_SIZE
MIN_PATCH_SIZE = v1_2_base.MIN_PATCH_SIZE

# NAIP is added as a decode-only modality on top of the v1.2 decode targets.
ONLY_DECODE_MODALITIES = [*v1_2_base.ONLY_DECODE_MODALITIES, Modality.NAIP_10.name]

# Generator upsamples the pooled token grid 4x to reach the 2.5 m/px NAIP grid.
NAIP_UPSAMPLE_FACTOR = 4
GENERATOR_HIDDEN_SIZE = 128
DISCRIMINATOR_HIDDEN_SIZE = 64
LAMBDA_ADV = 0.1
LAMBDA_L1 = 10.0
GAN_WARMUP_STEPS = 8000


def _masking_config(tokenization_config=None) -> MaskingConfig:
    return MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "random_ratio": 0.5,
            "only_decode_modalities": ONLY_DECODE_MODALITIES,
        },
        tokenization_config=tokenization_config,
    )


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build common components: v1.2 modalities plus NAIP."""
    config = v1_2_base.build_common_components(
        script, cmd, run_name, cluster, overrides
    )
    config.training_modalities = [
        *config.training_modalities,
        Modality.NAIP_10.name,
    ]
    return config


def build_model_config(common: CommonComponents) -> NaipGanModelConfig:
    """Build the NAIP GAN model config (v1.2 encoder/decoder + generator)."""
    base_model = v1_2_base.build_size_model_config(
        common, "base_shallow_decoder", v1_2_base.PATCH_EMBED_HIDDEN_SIZES
    )
    embedding_size = (
        base_model.encoder_config.output_embedding_size
        or base_model.encoder_config.embedding_size
    )
    generator_config = NaipGeneratorConfig(
        embedding_size=embedding_size,
        hidden_size=GENERATOR_HIDDEN_SIZE,
        out_channels=Modality.NAIP_10.num_bands,
        upsample_factor=NAIP_UPSAMPLE_FACTOR,
    )
    return NaipGanModelConfig(
        encoder_config=base_model.encoder_config,
        decoder_config=base_model.decoder_config,
        generator_config=generator_config,
    )


def build_train_module_config(
    common: CommonComponents,
) -> NaipGanTrainModuleConfig:
    """Build the NAIP GAN train module config."""
    model_size = MODEL_SIZE_ARGS["base_shallow_decoder"]
    embedding_size = model_size["encoder_embedding_size"]
    return NaipGanTrainModuleConfig(
        optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02, fused=False),
        rank_microbatch_size=64,
        masking_config=_masking_config(common.tokenization_config),
        loss_config=LossConfig(
            loss_config={
                "type": "modality_patch_discrimination_masked_negatives_vec",
                "tau": 0.1,
                "same_target_threshold": 0.999,
                "mask_negatives_for_modalities": ONLY_DECODE_MODALITIES,
            }
        ),
        discriminator_config=NaipDiscriminatorConfig(
            embedding_size=embedding_size,
            in_channels=Modality.NAIP_10.num_bands,
            hidden_size=DISCRIMINATOR_HIDDEN_SIZE,
        ),
        disc_optim_config=AdamWConfig(
            lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0, fused=False
        ),
        lambda_adv=LAMBDA_ADV,
        lambda_l1=LAMBDA_L1,
        gan_warmup_steps=GAN_WARMUP_STEPS,
        token_exit_cfg={modality: 0 for modality in common.training_modalities},
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=8000),
        ema_decay=(1.0, 1.0),
        dp_config=DataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Build the dataloader config (single masked view for latent-MIM)."""
    return OlmoEarthDataLoaderConfig(
        num_workers=16,
        global_batch_size=512,
        token_budget=1500,
        prefetch_factor=4,
        sampled_hw_p_list=list(range(1, 13)),
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        work_dir=common.save_folder,
        seed=3622,
        num_masked_views=1,
        masking_config=_masking_config(common.tokenization_config),
    )


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Build the dataset config (a NAIP-containing h5py dataset)."""
    return OlmoEarthDatasetConfig(
        h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_landsat_naip_10_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/1138828",
        training_modalities=common.training_modalities,
    )


WANDB_PROJECT = "2026_07_01_naip_gan"


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Reuse the v1.2 trainer config, overriding the W&B project."""
    trainer_config = v1_2_base.build_trainer_config(common)
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
    return trainer_config


def build_visualize_config(common: CommonComponents) -> OlmoEarthVisualizeConfig:
    """Reuse the v1.2 visualize config."""
    return v1_2_base.build_visualize_config(common)


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )
