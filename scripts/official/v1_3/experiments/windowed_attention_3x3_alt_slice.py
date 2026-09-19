"""Alternating-attention experiment: v1.3 with 3x3 windowed / per-slice encoder blocks.

THE CHANGE, AND ONLY IT: the 12 encoder self-attention blocks alternate between two
restricted attention patterns, both implemented as dense SDPA masks:

* EVEN blocks (0, 2, ..., 10): 3x3 neighborhood attention over the patch grid. A
  spatial patch token attends only to tokens within Chebyshev radius 1, across all
  timesteps, band sets, and spatial modalities (exactly ``windowed_attention_3x3.py``).
* ODD blocks (1, 3, ..., 11): per-slice spatial attention. A spatial patch token
  attends to the full spatial extent of ITS OWN modality at ITS OWN timestep (all
  band sets of that slice), and nothing from other modalities or timesteps. Static
  spatial modalities (no temporal axis) form one slice per modality.

So the two directions of context are split by layer: local mixing across
time/modalities in the even blocks, global spatial mixing within one image in the
odd blocks. Non-spatial tokens (latlon) attend globally and are attended by everyone
in both block types. The register bottleneck reads and the decoder keep full
attention, so the register grid (the last thing to read the per-slice layer 11) is
still where fully global context is aggregated. Everything else -- d768 registers,
supervision heads, the linear + LayerNorm student, sampler, data, in-loop evals -- is
identical to ``base.py``. Neither mask saves compute; this is an inductive-bias arm.

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
from olmoearth_pretrain.nn.flexi_vit import (  # noqa: E402
    NON_WINDOWED_ATTENTION_PER_SLICE,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/experiments/windowed_attention_3x3_alt_slice.py"

# Side length of the neighborhood each spatial patch token attends over in the
# windowed blocks. Odd, >= 3.
WINDOWED_ATTENTION_SIZE = 3
# Every other block starting from the first is windowed; the rest are per-slice.
WINDOWED_LAYER_STRIDE = 2


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The v1.3 model with alternating 3x3-window / per-slice encoder blocks."""
    config = _base_build_model_config(common)
    encoder_config = config.encoder_config
    encoder_config.windowed_attention_size = WINDOWED_ATTENTION_SIZE
    encoder_config.windowed_attention_layers = list(
        range(0, encoder_config.depth, WINDOWED_LAYER_STRIDE)
    )
    encoder_config.non_windowed_attention = NON_WINDOWED_ATTENTION_PER_SLICE
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
