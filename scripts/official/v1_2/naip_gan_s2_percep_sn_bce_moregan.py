"""v1.2 NAIP raw-Sentinel-2 GAN with perceptual + spectral-norm + BCE, stronger GAN.

Same three additions as ``naip_gan_s2_percep_sn_bce.py`` (VGG perceptual loss,
spectrally-normalized discriminator, BCE adversarial loss) but pushes the
adversarial objective harder now that spectral norm stabilizes the
discriminator: the adversarial weight is raised (``lambda_adv=0.5``) while L1
and perceptual stay at 1.0, and the adversarial warmup is shortened so the GAN
term engages sooner.

Validate before launching::

    python3 scripts/official/v1_2/naip_gan_s2_percep_sn_bce_moregan.py dry_run naip_gan_s2_percep_sn_bce_moregan local
"""

import logging

import naip_gan as naip_gan_base
import naip_gan_s2_percep_sn_bce as percep_base

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.train.train_module.naip_gan import NaipGanTrainModuleConfig

logger = logging.getLogger(__name__)

LAMBDA_ADV = 0.5
GAN_WARMUP_STEPS = 2000


def build_train_module_config(common: CommonComponents) -> NaipGanTrainModuleConfig:
    """Build the perceptual + SN + BCE config with a stronger adversarial term."""
    config = percep_base.build_train_module_config(common)
    config.lambda_adv = LAMBDA_ADV
    config.gan_warmup_steps = GAN_WARMUP_STEPS
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
