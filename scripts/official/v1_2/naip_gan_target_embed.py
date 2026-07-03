"""v1.2 NAIP conditional GAN conditioned on the target-encoder embedding.

This is the base ``naip_gan.py`` configuration made explicit: the discriminator
conditions on the target-encoder pooled embedding (``target_pooled``) with the
``[256, 128, 128]`` generator and the condition-aware discriminator (two head
residual blocks, post-unpatchify condition convs, no projection term). It exists
as a named experiment; it reuses the base builders unchanged.

Validate before launching::

    python3 scripts/official/v1_2/naip_gan_target_embed.py dry_run naip_gan_target_embed local
"""

import logging

import naip_gan as naip_gan_base

from olmoearth_pretrain.internal.experiment import main

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    main(
        common_components_builder=naip_gan_base.build_common_components,
        model_config_builder=naip_gan_base.build_model_config,
        train_module_config_builder=naip_gan_base.build_train_module_config,
        dataset_config_builder=naip_gan_base.build_dataset_config,
        dataloader_config_builder=naip_gan_base.build_dataloader_config,
        trainer_config_builder=naip_gan_base.build_trainer_config,
        visualize_config_builder=naip_gan_base.build_visualize_config,
    )
