"""v1.2 NAIP unmasked-embedding projection GAN without the L1 loss.

Same as ``naip_gan_unmasked_embed_projdis.py`` (unmasked-input embedding
condition, projection discriminator) but drops the L1 reconstruction loss
(``lambda_l1=0``), so the generator is trained by the adversarial loss alone
(``lambda_adv=0.1``).

Validate before launching::

    python3 scripts/official/v1_2/naip_gan_unmasked_embed_projdis_no_l1.py dry_run naip_gan_unmasked_embed_projdis_no_l1 local
"""

import logging

import naip_gan as naip_gan_base

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.train.train_module.naip_gan import NaipGanTrainModuleConfig

logger = logging.getLogger(__name__)


def build_train_module_config(common: CommonComponents) -> NaipGanTrainModuleConfig:
    """Build the train module config: projection discriminator, no L1 loss."""
    config = naip_gan_base.build_train_module_config(common)
    config.discriminator_config.use_projection = True
    config.lambda_l1 = 0.0
    # No L1 term, so there is nothing to train the generator during a warmup;
    # start the adversarial loss immediately.
    config.gan_warmup_steps = 0
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
