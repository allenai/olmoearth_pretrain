"""v1.2 NAIP conditional GAN with a bigger generator AND discriminator.

Combines two capacity changes over ``naip_gan.py``:

* Generator: a per-stage channel schedule ``[256, 128, 128]`` that concentrates
  capacity at the coarse (10 m/px) resolution while keeping the finer 5 m/px and
  2.5 m/px stages at the original width (so high-resolution FLOPs are unchanged).
* Discriminator: made to reason about the Sentinel-2-derived conditioning tokens
  rather than only the realism of the NAIP image, via an MLP condition
  projection, a deeper fusion head, a wider/deeper image path, and a
  projection-discriminator inner-product term.

Validate before launching::

    python3 scripts/official/v1_2/naip_gan_big_gen_dis.py dry_run naip_gan_big_gen_dis local
"""

import logging

import naip_gan as naip_gan_base

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.naip_gan import NaipGanModelConfig
from olmoearth_pretrain.train.train_module.naip_gan import NaipGanTrainModuleConfig

logger = logging.getLogger(__name__)

# Per-stage generator channel widths (base 10 m/px, then 5 m/px, then 2.5 m/px).
GENERATOR_HIDDEN_SIZES = [256, 128, 128]

# Bigger, more condition-aware discriminator.
DISCRIMINATOR_HIDDEN_SIZE = 96
DISCRIMINATOR_NUM_IMAGE_LAYERS = 4
DISCRIMINATOR_COND_HIDDEN_SIZE = 512
DISCRIMINATOR_NUM_HEAD_RES_BLOCKS = 2
DISCRIMINATOR_USE_PROJECTION = True


def build_model_config(common: CommonComponents) -> NaipGanModelConfig:
    """Build the NAIP GAN model config with the wider coarse-resolution trunk."""
    config = naip_gan_base.build_model_config(common)
    config.generator_config.hidden_sizes = GENERATOR_HIDDEN_SIZES
    return config


def build_train_module_config(
    common: CommonComponents,
) -> NaipGanTrainModuleConfig:
    """Build the train module config with the bigger, condition-aware discriminator."""
    config = naip_gan_base.build_train_module_config(common)
    disc = config.discriminator_config
    disc.hidden_size = DISCRIMINATOR_HIDDEN_SIZE
    disc.num_image_layers = DISCRIMINATOR_NUM_IMAGE_LAYERS
    disc.cond_hidden_size = DISCRIMINATOR_COND_HIDDEN_SIZE
    disc.num_head_res_blocks = DISCRIMINATOR_NUM_HEAD_RES_BLOCKS
    disc.use_projection = DISCRIMINATOR_USE_PROJECTION
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
