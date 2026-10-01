"""Spatially LOCAL copy of ``pure_perceiver_joint_latentread_rstride_fast.py``.

Same model, sampler, microbatch, seed and evals as the compiled-RoPE random-stride arm
(``v1_3_vit0_rstride_ps8_lb512_fast2_latentread_joint12``); the only config difference
is that the joint blocks' spatial edges become local, in patch cells (Chebyshev):

* token -> latents within ``LOCAL_RADIUS`` (2 = a 5x5 window of cells, i.e. 10x10
  latents at patch size 2 with per-pixel latents) + its own cell's tokens;
* latent -> latents within ``LATENT_RADIUS`` (3 = 7x7 cells) + tokens within
  ``LOCAL_RADIUS`` (replacing ``latent_reads_all``'s read of every token).

Purpose: attention cost linear in image area. On the 16x16x12 eval window it changes
little (~928 -> ~845 G MACs at ps1); on large single-date tiles it removes the
quadratic latent-latent term. Receptive field after 12 blocks: 2 + 11 * 3 = 35 cells,
which still spans the largest 32-cell training grids corner to corner.

W&B project ``20260921_perceiver_shapes``; trained as
``v1_3_vit0_rstride_ps8_lb512_fast2_rad2lat3_latentread_joint12``.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from pure_perceiver_joint_latentread_rstride import (  # noqa: E402
    build_dataloader_config,
    build_train_module_config,
)
from pure_perceiver_joint_latentread_rstride import (  # noqa: E402
    build_trainer_config as _rstride_trainer_config,
)
from pure_perceiver_joint_latentread_rstride_fast import (  # noqa: E402
    build_model_config as _fast_model_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.joint_latent import JointLatentConfig  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = (
    "scripts/official/v1_3/ablations/pure_perceiver_joint_latentread_rstride_local.py"
)

# Patch cells, Chebyshev distance.
LOCAL_RADIUS = 2
LATENT_RADIUS = 3


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The compiled-RoPE random-stride arm with local token-latent / latent-latent edges."""
    config = _fast_model_config(common)
    perceiver = config.encoder_config.perceiver_config
    assert isinstance(perceiver, JointLatentConfig)
    assert perceiver.latent_reads_all and perceiver.compile_rope
    perceiver.local_radius = LOCAL_RADIUS
    perceiver.latent_radius = LATENT_RADIUS
    return config


def build_trainer_config(common: CommonComponents):
    """The random-stride arms' evals, re-importing THIS module."""
    return _rstride_trainer_config(common, module_path=MODULE_PATH)


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
