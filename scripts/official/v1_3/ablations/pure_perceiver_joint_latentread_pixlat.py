"""Joint latentread with PIXEL latents on patch-size <= 4 tokens.

``pure_perceiver_joint_latentread.py`` keeps one latent per patch, so its tokens have
to run at patch size 1 to give per-pixel embeddings, and its encoder cost is set by
the 9,216 patch-size-1 tokens of a 16x16 / 12-timestep / S1+S2+L8 window (~930 G MACs).
This arm lays the latents at PIXEL resolution instead (``pixel_latents=True``) --
one latent per pixel, each at its pixel centre inside the patch frame, carrying the
cell id of its containing patch -- the pixel-register idea of
``favyen/20260917-pixreg-v1_3`` inside the joint latent-token transformer. Tokens can
then run at patch size 4 while the output stays per pixel: 576 tokens + 256 pixel
latents on that window, ~80 G MACs.

Latents keep ``latent_reads_all``: each reads every valid token and every latent;
tokens see every latent and their own patch's tokens. The decoder is unchanged: its
queries are the masked tokens at the batch's patch size and its keys are the pixel
latents. Map supervision predicts one value per pixel latent (``spatial_unfold=1``).

Training shapes follow Favyen's pixel-register recipe: patch embed base 4, patch
sizes 1..4, grids capped at 24 patches, rank microbatch 32. Pixel latents grow with
the sample's pixel AREA, not its tokens, so the sampler's pixel side is also capped
at 64 (``tile_size = 64``, which only bounds the grid; 4,096 latents at most).

In-loop evals: the usual student evals at patch size 1 (where pixel and patch
latents coincide), plus the d128 student at patch size 4 on the same tasks
(``*_ws16_ps4_*_proj128``), which is the configuration the arm exists for.

W&B project ``20260921_perceiver_shapes``; trained as
``v1_3_vit0_pixlat_latentread_joint12``.
"""

import logging
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    STUDENT_LOOP_EVAL_INTERVAL_STEPS,
    aeftrial_loop_eval_tasks,
    build_common_components,
    build_dataset_config,
    build_visualize_config,
    set_student_loop_evals,
)
from base import build_dataloader_config as _base_build_dataloader_config  # noqa: E402
from base import (  # noqa: E402
    build_train_module_config as _base_build_train_module_config,
)
from pure_perceiver import WANDB_PROJECT  # noqa: E402
from pure_perceiver_joint_latentread import (  # noqa: E402
    build_model_config as _latentread_model_config,
)
from v1_2.base import build_trainer_config as _v1_2_build_trainer_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.joint_latent import JointLatentConfig  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = (
    "scripts/official/v1_3/ablations/pure_perceiver_joint_latentread_pixlat.py"
)

# Favyen's pixel-register shapes: patch embed base 4 (also the dataloader's max patch
# size), grids up to 24 patches, microbatch 32.
MAX_PATCH_SIZE = 4
SAMPLED_HW_P_LIST = list(range(1, 17)) + [18, 20, 24]
RANK_MICROBATCH_SIZE = 32
# Pixel side cap: bounds the pixel-latent count at 64 * 64 = 4096 per sample.
MAX_SAMPLE_PIXELS = 64
EVAL_PATCH_SIZE = 4


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Joint latentread with one latent per pixel; per-pixel map supervision."""
    config = _latentread_model_config(common)
    perceiver = config.encoder_config.perceiver_config
    assert isinstance(perceiver, JointLatentConfig) and perceiver.latent_reads_all
    perceiver.pixel_latents = True
    config.encoder_config.max_patch_size = MAX_PATCH_SIZE
    assert config.supervision_head_config is not None
    config.supervision_head_config.spatial_unfold = 1
    return config


def build_dataloader_config(common: CommonComponents):
    """v1.3 sampler at patch sizes 1..4, grids <= 24, samples <= 64 pixels a side."""
    config = _base_build_dataloader_config(common)
    config.max_patch_size = MAX_PATCH_SIZE
    config.sampled_hw_p_list = list(SAMPLED_HW_P_LIST)
    config.tile_size = MAX_SAMPLE_PIXELS
    return config


def build_train_module_config(common: CommonComponents):
    """v1.3 train module at the pixel-register microbatch."""
    config = _base_build_train_module_config(common)
    config.rank_microbatch_size = RANK_MICROBATCH_SIZE
    return config


def build_trainer_config(common: CommonComponents, module_path: str = MODULE_PATH):
    """Student evals at patch size 1, plus the d128 student at patch size 4.

    ``module_path`` is what the eval jobs re-import to rebuild the model; sibling arms
    MUST pass their own path, or their evals silently score this arm's architecture.
    """
    trainer_config = set_student_loop_evals(
        _v1_2_build_trainer_config(common), module_path
    )
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    for name, task in aeftrial_loop_eval_tasks(
        STUDENT_LOOP_EVAL_INTERVAL_STEPS
    ).items():
        assert "_ps1_" in name, name
        ps_name = name.replace("_ps1_", f"_ps{EVAL_PATCH_SIZE}_") + "_proj128"
        evaluator.tasks[ps_name] = replace(
            task,
            patch_size=EVAL_PATCH_SIZE,
            eval_on_student_registers=True,
            eval_student_dim=128,
        )
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
