"""v1.2 NAIP raw-Sentinel-2 projection GAN with perceptual + spectral-norm + BCE.

Same three additions as ``naip_gan_s2_percep_sn_bce.py`` (VGG perceptual loss,
spectrally-normalized discriminator, BCE adversarial loss) but with the
projection-discriminator term enabled (``use_projection=True``). Tests whether
the projection discriminator helps once it is stabilized by spectral norm. Loss
balance matches the ESRGAN reference (L1=1.0, perceptual=1.0, adv=0.1).

Validate before launching::

    python3 scripts/official/v1_2/naip_gan_s2_projdis_percep_sn_bce.py dry_run naip_gan_s2_projdis_percep_sn_bce local
"""

import logging

import naip_gan as naip_gan_base
import naip_gan_s2_percep_sn_bce as percep_base

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.train.train_module.naip_gan import NaipGanTrainModuleConfig

logger = logging.getLogger(__name__)


def build_train_module_config(common: CommonComponents) -> NaipGanTrainModuleConfig:
    """Build the perceptual + SN + BCE config with the projection term on."""
    config = percep_base.build_train_module_config(common)
    config.discriminator_config.use_projection = True
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
