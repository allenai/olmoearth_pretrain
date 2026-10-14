"""Control: ``v1_3_rc_ld1_pixtgt_pix512`` rebuilt on the merged (main-API) code.

The v1.3 base with per-pixel location tokens: 12 ViT blocks, a 1-layer AttentionPooler
whose location-query patch size is drawn per batch under a 512-query budget (one query
per pixel at eval), the [128, 64] student, map supervision, and pixel-resolution MIM
targets (one uniformly drawn pixel per token cell, shared draw). See ``common.py`` for
the API mapping and ``diff_against_trained.py`` for the field-by-field check.

Trained originally at 9f5a25b02 as ``v1_3_rc_ld1_pixtgt_pix512`` (W&B project
``20260921_perceiver_shapes``), 300 epochs = 667,200 steps.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_model_config,
    build_visualize_config,
)
from common import build_train_module_config as _build_train_module_config  # noqa: E402
from common import build_trainer_config as _build_trainer_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.train.train_module.latent_mim import (  # noqa: E402
    LatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/claude_findings/control_pixtgt.py"


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """The v1.3 train module with pixel-resolution MIM targets."""
    return _build_train_module_config(common, pixel_targets=True)


def build_trainer_config(common: CommonComponents):
    """The rc_ld1 in-loop evals, re-importing THIS module."""
    return _build_trainer_config(common, MODULE_PATH)


def run() -> None:
    """Run the experiment."""
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )


if __name__ == "__main__":
    run()
