"""v1.2 NAIP raw-Sentinel-2 conditional GAN with a stronger adversarial term.

Same as ``naip_gan_s2.py`` (raw Sentinel-2 condition) but rebalances the losses
toward the GAN objective: the L1 reconstruction weight is lowered
(``lambda_l1=2.0``) and the adversarial weight is raised (``lambda_adv=0.5``).

Validate before launching::

    python3 scripts/official/v1_2/naip_gan_s2_moregan.py dry_run naip_gan_s2_moregan local
"""

import logging

import naip_gan as naip_gan_base
import naip_gan_s2 as naip_gan_s2_base

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.train.train_module.naip_gan import NaipGanTrainModuleConfig

logger = logging.getLogger(__name__)

LAMBDA_L1 = 2.0
LAMBDA_ADV = 0.5


def build_train_module_config(common: CommonComponents) -> NaipGanTrainModuleConfig:
    """Build the raw-Sentinel-2 train module config with a stronger GAN term."""
    config = naip_gan_s2_base.build_train_module_config(common)
    config.lambda_l1 = LAMBDA_L1
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
