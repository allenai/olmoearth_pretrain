"""v1.2 NAIP raw-Sentinel-2 GAN, ESRGAN-faithful recipe.

Reference variant of the perceptual + spectral-norm + BCE sweep: raw
Sentinel-2 condition (no projection term), a spectrally-normalized
discriminator, a vanilla BCE adversarial loss, and the ESRGAN loss balance
(``lambda_l1=1.0``, ``lambda_perceptual=1.0``, ``lambda_adv=0.1``). This is the
closest match to the satlas super-resolution / Real-ESRGAN recipe.

Validate before launching::

    python3 scripts/official/v1_2/naip_gan_s2_percep_sn_bce.py dry_run naip_gan_s2_percep_sn_bce local
"""

import logging

import naip_gan as naip_gan_base
import naip_gan_s2 as naip_gan_s2_base

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.train.train_module.naip_gan import NaipGanTrainModuleConfig

logger = logging.getLogger(__name__)

# ESRGAN / satlas loss balance.
LAMBDA_L1 = 1.0
LAMBDA_PERCEPTUAL = 1.0
LAMBDA_ADV = 0.1


def build_train_module_config(common: CommonComponents) -> NaipGanTrainModuleConfig:
    """Build the raw-S2 config with perceptual loss, spectral norm, and BCE."""
    config = naip_gan_s2_base.build_train_module_config(common)
    config.discriminator_config.use_spectral_norm = True
    config.gan_loss_type = "bce"
    config.lambda_l1 = LAMBDA_L1
    config.lambda_perceptual = LAMBDA_PERCEPTUAL
    config.lambda_adv = LAMBDA_ADV
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
