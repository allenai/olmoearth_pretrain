"""Pure Perceiver with ONE shared K/V projection across its 12 reads.

``pure_perceiver.py`` deletes the ViT and reads the patch embeddings into the register
grid with 12 ``[read -> self-attend]`` layers. Each of those reads projects the same
input tokens through its own K and V layers, and with no ViT in front of them those 12
full-width projections are the arm's dominant per-token cost (131 G of its 244 G MACs
on 16x16 / 12 timesteps / S1+S2+L8 / patch size 1). This arm computes K and V ONCE
(``share_read_kv=True``) and lets every read attend over them: the input is seen
through a single linear map and all depth lives in the queries and latent blocks, the
original Perceiver design. Queries, output projections and MLPs stay per read.

Consequences relative to ``pure_perceiver.py``: MACs on the shape above drop to ~125 G
(0.05x the v1.3 RC), the 11 saved K/V activations over all tokens come off peak memory,
and the per-read input LayerNorms (``per_depth_read_proj``) are replaced by one shared
norm, since a shared projection needs a single shared source. Everything else -- d768
registers, supervision, the ``[128, 64]`` student with MLP back-projection + Gram
distillation, sampler, train module, student in-loop evals -- is unchanged.

Inherits ``read_time_rope`` (3D RoPE on the reads) from ``pure_perceiver.py``.

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_vit0_sharedkv_trope_ld12``
(the time-blind first launch, ``v1_3_vit0_sharedkv_ld12``, was stopped at ~8k steps).
"""

import logging
import sys
from pathlib import Path

# Builds on the pure-Perceiver arm next to it, which imports the release recipe.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_visualize_config,
    set_student_loop_evals,
)
from pure_perceiver import WANDB_PROJECT, build_train_module_config  # noqa: E402
from pure_perceiver import (
    build_model_config as _pure_perceiver_model_config,  # noqa: E402
)
from v1_2.base import build_trainer_config as _v1_2_build_trainer_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/ablations/pure_perceiver_shared_kv.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """pure_perceiver.py's model with one K/V projection shared by all 12 reads."""
    config = _pure_perceiver_model_config(common)
    perceiver_config = config.encoder_config.perceiver_config
    assert perceiver_config is not None
    perceiver_config.share_read_kv = True
    # A shared projection needs a single shared read source.
    perceiver_config.per_depth_read_proj = False
    return config


def build_trainer_config(common: CommonComponents):
    """Student in-loop evals rebuilt from THIS module, in the shapes project."""
    trainer_config = set_student_loop_evals(
        _v1_2_build_trainer_config(common), MODULE_PATH
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
