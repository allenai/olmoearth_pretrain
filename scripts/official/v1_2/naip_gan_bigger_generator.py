"""v1.2 NAIP conditional GAN with a higher-capacity generator.

Identical to ``naip_gan.py`` except the generator uses a per-stage channel
schedule that concentrates capacity at the coarse (post-unpatchify) resolution,
where the semantic-to-texture mapping is decided and compute is cheapest. The
finer upsampling stages keep the original width so high-resolution FLOPs are
essentially unchanged.

At NAIP settings the three stages correspond to 10 m/px, 5 m/px and 2.5 m/px, so
``GENERATOR_HIDDEN_SIZES = [256, 128, 128]`` widens only the 10 m/px base stage.

Validate before launching::

    python3 scripts/official/v1_2/naip_gan_bigger_generator.py dry_run naip_gan_bigger_generator local
"""

import logging

import naip_gan as naip_gan_base

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.naip_gan import NaipGanModelConfig

logger = logging.getLogger(__name__)

# Per-stage generator channel widths (base 10 m/px, then 5 m/px, then 2.5 m/px).
GENERATOR_HIDDEN_SIZES = [256, 128, 128]


def build_model_config(common: CommonComponents) -> NaipGanModelConfig:
    """Build the NAIP GAN model config with the wider coarse-resolution trunk."""
    config = naip_gan_base.build_model_config(common)
    config.generator_config.hidden_sizes = GENERATOR_HIDDEN_SIZES
    return config


if __name__ == "__main__":
    main(
        common_components_builder=naip_gan_base.build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=naip_gan_base.build_train_module_config,
        dataset_config_builder=naip_gan_base.build_dataset_config,
        dataloader_config_builder=naip_gan_base.build_dataloader_config,
        trainer_config_builder=naip_gan_base.build_trainer_config,
        visualize_config_builder=naip_gan_base.build_visualize_config,
    )
