"""v1.2 NAIP conditional GAN conditioned on the raw Sentinel-2 input.

Instead of a pooled embedding, the discriminator conditions on the raw
Sentinel-2 temporal stack (``raw_sentinel2``): the full time series of all S2
bands is embedded by non-strided per-timestep convs (3 layers), averaged over
the valid timesteps, then refined by more non-strided convs before fusing with
the NAIP image features. The generator and NAIP image path match
``naip_gan_unmasked_embed.py``; the projection term is off.

Validate before launching::

    python3 scripts/official/v1_2/naip_gan_s2.py dry_run naip_gan_s2 local
"""

import logging

import naip_gan as naip_gan_base

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.train.train_module.naip_gan import NaipGanTrainModuleConfig

logger = logging.getLogger(__name__)

# Non-strided convs applied to each Sentinel-2 timestep before the mean over
# time (input bands -> 128 -> 128 -> 128), then two more after the mean pooling.
DISCRIMINATOR_COND_PRE_POOL_CHANNELS = [128, 128, 128]
DISCRIMINATOR_COND_POST_POOL_CHANNELS = [128, 128]


def build_train_module_config(common: CommonComponents) -> NaipGanTrainModuleConfig:
    """Build the train module config conditioning on the raw Sentinel-2 stack."""
    config = naip_gan_base.build_train_module_config(common)
    disc = config.discriminator_config
    disc.cond_mode = "image"
    disc.cond_in_channels = Modality.SENTINEL2_L2A.num_bands
    disc.cond_image_pre_pool_channels = DISCRIMINATOR_COND_PRE_POOL_CHANNELS
    disc.cond_image_post_pool_channels = DISCRIMINATOR_COND_POST_POOL_CHANNELS
    config.discriminator_cond_source = "raw_sentinel2"
    return config


if __name__ == "__main__":
    main(
        common_components_builder=naip_gan_base.build_common_components,
        model_config_builder=naip_gan_base.build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=naip_gan_base.build_dataset_config,
        dataloader_config_builder=naip_gan_base.build_dataloader_config,
        trainer_config_builder=naip_gan_base.build_trainer_config,
        visualize_config_builder=naip_gan_base.build_visualize_config,
    )
