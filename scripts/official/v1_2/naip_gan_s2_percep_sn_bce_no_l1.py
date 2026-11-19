"""v1.2 NAIP raw-Sentinel-2 GAN with perceptual as the sole content anchor.

Same three additions as ``naip_gan_s2_percep_sn_bce.py`` (VGG perceptual loss,
spectrally-normalized discriminator, BCE adversarial loss) but drops the L1
pixel loss (``lambda_l1=0``). This is the corrected "no-L1" recipe: unlike a
bare adversarial-only run, the perceptual loss still supplies a strong
content/alignment gradient (exactly what ESRGAN keeps when its L1 term is
removed), so the generator has a reconstruction anchor. A short adversarial
warmup lets the perceptual term shape the output first.

Validate before launching::

    python3 scripts/official/v1_2/naip_gan_s2_percep_sn_bce_no_l1.py dry_run naip_gan_s2_percep_sn_bce_no_l1 local
"""

import logging

import naip_gan as naip_gan_base
import naip_gan_s2_percep_sn_bce as percep_base

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.train.train_module.naip_gan import NaipGanTrainModuleConfig

logger = logging.getLogger(__name__)

# Perceptual replaces L1 as the content anchor; keep the adversarial weight at
# the ESRGAN reference. A short warmup lets the perceptual term act first.
GAN_WARMUP_STEPS = 2000


def build_train_module_config(common: CommonComponents) -> NaipGanTrainModuleConfig:
    """Build the perceptual + SN + BCE config with L1 disabled."""
    config = percep_base.build_train_module_config(common)
    config.lambda_l1 = 0.0
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
