"""``base_faster`` without the instance-level contrastive (InfoNCE) loss.

Ablation of the ``v1.2_add_dsm_canopy_height`` run: identical to
``base_faster.py`` in every respect except that ``contrastive_config`` is
``None``, so the objective is the masked patch-discrimination loss alone.
``ContrastiveLatentMIMTrainModule`` guards every use of the contrastive term on
``self.contrastive_loss is not None``, so setting the config to ``None`` skips
the InfoNCE computation, its contribution to the total loss, and its
``train/<name>`` metric entirely -- rather than merely zeroing its weight.

Note this is a pure objective ablation, not a speedup: the two-view forward
(``masked_batch_a`` / ``masked_batch_b``) belongs to the train module, not to
the contrastive term, so both views are still encoded and the patch
discrimination loss is still averaged over them.
"""

import logging

from base import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_visualize_config,
)
from base_faster import build_model_config
from base_faster import build_train_module_config as _faster_build_train_module_config
from base_faster import build_trainer_config as _faster_build_trainer_config
from olmo_core.train.config import TrainerConfig

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/vnext/2026_07_24_new_maps/base_faster_no_contrastive.py"


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """base_faster train module config with the InfoNCE term removed."""
    config = _faster_build_train_module_config(common)
    config.contrastive_config = None
    return config


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """base_faster trainer config, with loop evals pointed at this module."""
    trainer_config = _faster_build_trainer_config(common)
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    evaluator.beaker_eval_module_path = MODULE_PATH
    return trainer_config


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )
