"""v1.2 NAIP unmasked-embedding GAN with a projection discriminator.

Same as ``naip_gan_unmasked_embed.py`` but enables the projection-discriminator
term (``use_projection=True``), so the discriminator's logits reward agreement
between the NAIP image features and the projected condition.

Validate before launching::

    python3 scripts/official/v1_2/naip_gan_unmasked_embed_projdis.py dry_run naip_gan_unmasked_embed_projdis local
"""

import logging

import naip_gan as naip_gan_base

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.train.train_module.naip_gan import NaipGanTrainModuleConfig

logger = logging.getLogger(__name__)


def build_train_module_config(common: CommonComponents) -> NaipGanTrainModuleConfig:
    """Build the train module config with the projection-discriminator term."""
    config = naip_gan_base.build_train_module_config(common)
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
