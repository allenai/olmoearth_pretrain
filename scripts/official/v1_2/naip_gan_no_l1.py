"""v1.2 NAIP conditional GAN with the generator L1 loss disabled.

Identical to ``naip_gan.py`` except the generator is trained with a pure
adversarial objective (``lambda_l1 = 0``) and generated-vs-real NAIP images are
uploaded to W&B every 2000 steps.

Note: the generator only receives gradients once the adversarial phase turns on
(after ``gan_warmup_steps``), since the L1 term is disabled. Lower
``gan_warmup_steps`` if you want the generator to start training sooner.

Validate before launching::

    python3 scripts/official/v1_2/naip_gan_no_l1.py dry_run naip_gan_no_l1 local
"""

import logging

import naip_gan as naip_gan_base

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.train.train_module.naip_gan import NaipGanTrainModuleConfig

logger = logging.getLogger(__name__)

IMAGE_LOG_INTERVAL = 2000


def build_train_module_config(
    common: CommonComponents,
) -> NaipGanTrainModuleConfig:
    """NAIP GAN train module config with L1 disabled and periodic image logging."""
    config = naip_gan_base.build_train_module_config(common)
    config.lambda_l1 = 0.0
    config.image_log_interval = IMAGE_LOG_INTERVAL
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
