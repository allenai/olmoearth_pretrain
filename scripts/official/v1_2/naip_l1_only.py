"""v1.2 NAIP reconstruction with L1 loss only (no GAN).

The NAIP GAN experiments (``naip_gan.py`` and its variants) were not converging,
so this config drops the adversarial branch entirely: the NAIP generator on top
of the encoder's pooled spatial embedding is trained by the L1 reconstruction
loss alone (``lambda_adv=0``, no discriminator built, no perceptual loss). The
latent-MIM patch-discrimination SSL objective for the encoder is unchanged.

The generator is also slimmed from the base ``[256, 192, 192]`` per-stage widths
(~11M params) to ``[192, 160, 128]`` (~7M params).

Validate before launching::

    python3 scripts/official/v1_2/naip_l1_only.py dry_run naip_l1_only local
"""

import logging

import naip_gan as naip_gan_base

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.naip_gan import NaipGanModelConfig
from olmoearth_pretrain.train.train_module.naip_gan import NaipGanTrainModuleConfig

logger = logging.getLogger(__name__)

# Slimmer generator (base resolution + one per upsampling stage). Lands at
# ~7M params vs. the base ``[256, 192, 192]`` (~11M), within the 5-10M target.
GENERATOR_HIDDEN_SIZES = [192, 160, 128]


def build_model_config(common: CommonComponents) -> NaipGanModelConfig:
    """Build the NAIP model config with a slimmed generator (no discriminator)."""
    config = naip_gan_base.build_model_config(common)
    config.generator_config.hidden_sizes = GENERATOR_HIDDEN_SIZES
    return config


def build_train_module_config(common: CommonComponents) -> NaipGanTrainModuleConfig:
    """Build the train module config: L1 reconstruction only, no GAN."""
    config = naip_gan_base.build_train_module_config(common)
    # Drop the adversarial branch: no discriminator (or its optimizer) is built,
    # and the generator is trained by L1 alone from step 0.
    config.lambda_adv = 0.0
    config.lambda_perceptual = 0.0
    # L1 is the only NAIP loss now, so it no longer has to out-weigh an
    # adversarial term; lower it from the base 10.0 to 2.0.
    config.lambda_l1 = 2.0
    config.discriminator_config = None
    config.disc_optim_config = None
    config.gan_warmup_steps = 0
    return config


if __name__ == "__main__":
    main(
        common_components_builder=naip_gan_base.build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=naip_gan_base.build_dataset_config,
        dataloader_config_builder=naip_gan_base.build_dataloader_config,
        trainer_config_builder=naip_gan_base.build_trainer_config,
        visualize_config_builder=naip_gan_base.build_visualize_config,
    )
