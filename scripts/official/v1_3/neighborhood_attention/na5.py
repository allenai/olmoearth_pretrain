"""Run 1: the v1.3 control (``claude_findings/control_pixtgt.py``) with 5x5 neighborhood attention.

THE CHANGE, AND ONLY IT: ``neighborhood_attention_cells = 5`` on the encoder (ViT
self-attention, AttentionPooler read and location-query self-attention) and on the
decoder (cross-attention from masked tokens to the location tokens). Every attention
sees only the box of 5x5 patch cells around the query's own cell (2 cells each side,
shifted inward at the domain edge), across all timesteps and modalities -- the
Searchlight rule (``nn/searchlight.py``) applied in training, at every patch size.
Grids of at most 5 cells are exactly full attention. Sampler, masking, losses and
in-loop evals are the control's; the evals run with NA because it is part of the model.

Searchlight inference for this model: ``neighborhood_attention_size_px = 5 * patch_size``.
Card t-ktw5up; plan in agent/favyen/2026-11-05-favyen-neighborhood-attention-pretrain.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "claude_findings"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_visualize_config,
)
from common import build_model_config as _build_model_config  # noqa: E402
from common import build_train_module_config as _build_train_module_config  # noqa: E402
from common import build_trainer_config as _build_trainer_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from olmoearth_pretrain.train.train_module.latent_mim import (  # noqa: E402
    LatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/neighborhood_attention/na5.py"
NEIGHBORHOOD_ATTENTION_CELLS = 5
WANDB_PROJECT = "20261105_neighborhood_attention"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The control model with 5x5 NA in the encoder, pooler and decoder."""
    config = _build_model_config(common)
    config.encoder_config.neighborhood_attention_cells = NEIGHBORHOOD_ATTENTION_CELLS
    config.decoder_config.neighborhood_attention_cells = NEIGHBORHOOD_ATTENTION_CELLS
    return config


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """The control's train module (pixel-resolution MIM targets)."""
    return _build_train_module_config(common, pixel_targets=True)


def build_trainer_config(common: CommonComponents):
    """The control's in-loop evals, re-importing THIS module."""
    trainer_config = _build_trainer_config(common, MODULE_PATH)
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
    return trainer_config


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
