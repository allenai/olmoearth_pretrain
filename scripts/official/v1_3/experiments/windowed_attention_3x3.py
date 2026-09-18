"""Windowed-attention experiment: v1.3 with 3x3 neighborhood attention in the encoder.

THE CHANGE, AND ONLY IT: ``encoder_config.windowed_attention_size = 3``. Every
encoder self-attention block becomes neighborhood attention over the patch grid:
a spatial patch token attends only to tokens within a 3x3 patch neighborhood
(Chebyshev radius 1), across all timesteps, band sets, and spatial modalities.
Non-spatial tokens attend globally. The register bottleneck reads and the decoder
keep full attention, so the register grid is still where global context is
aggregated. Everything else -- d768 registers, supervision heads, the linear +
LayerNorm student, sampler, data, in-loop evals -- is identical to ``base.py``.

The architecture is baked into this file (not a CLI override) because the in-loop
eval Beaker jobs rebuild the model from ``MODULE_PATH``.
"""

import logging
import sys
from pathlib import Path

# The experiments import the release recipe from the directory above.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_visualize_config,
    set_student_loop_evals,
)
from base import build_model_config as _base_build_model_config  # noqa: E402
from base import build_trainer_config as _base_build_trainer_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/experiments/windowed_attention_3x3.py"

# Side length of the neighborhood each spatial patch token attends over. Odd, >= 3.
WINDOWED_ATTENTION_SIZE = 3


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The v1.3 model with 3x3 neighborhood attention in every encoder block."""
    config = _base_build_model_config(common)
    config.encoder_config.windowed_attention_size = WINDOWED_ATTENTION_SIZE
    return config


def build_trainer_config(common: CommonComponents):
    """Same student in-loop evals as the release run, pointed at this module."""
    trainer_config = _base_build_trainer_config(common)
    return set_student_loop_evals(trainer_config, MODULE_PATH)


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
