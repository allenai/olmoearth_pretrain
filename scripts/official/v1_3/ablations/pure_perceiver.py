"""Pure-Perceiver shape ablation: 0 ViT encoder blocks, 12 ``[read -> self-attend]`` layers.

``base.py`` runs 12 ViT blocks over every patch token and then a 4-layer Perceiver
that reads them into the register grid. This arm deletes the ViT: the patch
embeddings (plus channel + month encodings) are read DIRECTLY into the register grid
by a 12-layer Perceiver, so all of the model's depth lives on the spatial grid and
the per-token compute is one K/V projection per read. Everything else is
``base.py``: d768 registers with attention at encoder width, register supervision,
the ``[128, 64]`` student with the MLP back-projection + Gram distillation, the
sampler, the train module and the student in-loop evals.

Why this is a fair swap and what changes:

* The latent-MIM target is unchanged. v1.3 trains with all-zero token exits and a
  projection-only target encoder, so the decoder already predicts patch EMBEDDINGS,
  not ViT-encoded tokens; deleting the encoder blocks does not touch the target.
* Time reaches the reads through 3D RoPE (``read_time_rope``): keys carry each
  token's calendar-day coordinate and the register queries are anchored at the
  window-centre time, so a read sees each token's offset within the window. (The
  first launch of this arm, ``v1_3_vit0_ld12``, had time-blind 2D reads -- month
  embedding only -- and was stopped at ~15k steps in favour of this version.)
* Compute (MACs, attention included, 16x16 / 12 timesteps / S1+S2+L8 / patch size 1):
  base.py 2,449 G vs 244 G here -- 10x fewer, because the ViT is quadratic in the
  9,216 input tokens while the reads are linear in them and the latent blocks see
  only the 256 registers. On single-timestep inputs the register grid is as large as
  the token set and this arm is ~1.2x MORE expensive than base.py (the 13-task
  Figure-1 protocol averages 243 G vs 238 G).

IN-LOOP EVALS: identical to ``base.py`` (student at 128 and 64 dims, 80k interval).

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_vit0_trope_ld12``.
"""

import logging
import sys
from pathlib import Path

# The ablations import the release recipe from the directory above.
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
from v1_2.base import build_trainer_config as _v1_2_build_trainer_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/ablations/pure_perceiver.py"

WANDB_PROJECT = "20260921_perceiver_shapes"
# No ViT blocks between the patch embedding and the Perceiver reads.
ENCODER_DEPTH = 0
# All the depth moves onto the register grid: 12 interleaved [read -> self-attend]
# layers, matching the ViT block count of base.py.
REGISTER_LATENT_DEPTH = 12
# 3D RoPE on the reads (see build_model_config).
READ_TIME_ROPE = True


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """base.py's model with the ViT deleted and a 12-layer Perceiver in its place."""
    config = _base_build_model_config(common)
    config.encoder_config.depth = ENCODER_DEPTH
    perceiver_config = config.encoder_config.perceiver_config
    assert perceiver_config is not None
    perceiver_config.latent_depth = REGISTER_LATENT_DEPTH
    # Reads rotate over (t, row, col): keys carry calendar time, register queries sit
    # at the window-centre time (same anchoring as the joint arm). Without this the
    # reads were time-blind and the tokens' only temporal signal was the month
    # embedding (the first launch, v1_3_vit0_ld12, trained that way and was stopped).
    perceiver_config.read_time_rope = READ_TIME_ROPE
    return config


def build_trainer_config(common: CommonComponents):
    """base.py's student in-loop evals, rebuilt from THIS module, in the shapes project."""
    trainer_config = set_student_loop_evals(
        _v1_2_build_trainer_config(common), MODULE_PATH
    )
    # Baked in rather than passed as a CLI override so the eval Beaker jobs, which
    # rebuild the config from MODULE_PATH, log to the same project.
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
