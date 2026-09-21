"""Joint latent-token attention: no ViT blocks, no reads, one block type.

``pure_perceiver.py`` deletes the ViT and keeps the Perceiver's two block types (reads
+ latent self-attention), so time is only ever processed by a query-driven pooling
and tokens never interact. This arm replaces both with 12 JOINT blocks over the
concatenated ``[tokens ; latents]`` sequence under one structured mask
(``nn/joint_latent.py``): a latent at cell (i, j) attends to all latents and to the
tokens of cell (i, j); a token at cell (i, j) attends to all latents and to the other
tokens of its cell. Tokens therefore interact along time and modality within a cell,
latents carry spatial context, and the latent-to-token edges ARE the read.

Compute: attention is linear in tokens (each query sees ``n_latents + tokens_per_cell``
keys) while every token still passes through the block's linear layers, so on
16x16 / 12 timesteps / S1+S2+L8 / patch size 1 a block costs ~71 G MACs against ~196 G
for a joint ViT block and ~7 G for a shared-K/V [read, latent] pair; the 12-block arm
is ~0.36x the v1.3 RC on that shape and grows cheaper with window size (linear vs
quadratic). Tokens rotate with the encoder's 3D mixed RoPE, so slot-index time is back.
Runs FlexAttention (block-sparse, no quadratic memory) on CUDA.

Everything else -- d768 registers, supervision, the ``[128, 64]`` student with MLP
back-projection + Gram distillation, sampler, train module, student in-loop evals --
is ``base.py``.

W&B project ``20260921_perceiver_shapes``.
"""

import logging
import sys
from pathlib import Path

# Sibling arm + the release recipe one directory up.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    PROJECTION_DIMS,
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_visualize_config,
    set_student_loop_evals,
)
from base import build_model_config as _base_build_model_config  # noqa: E402
from pure_perceiver import WANDB_PROJECT  # noqa: E402
from v1_2.base import build_trainer_config as _v1_2_build_trainer_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.joint_latent import JointLatentConfig  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/ablations/pure_perceiver_joint.py"

# Joint blocks over [tokens ; latents]; matches the ViT block count of base.py.
JOINT_DEPTH = 12
# No latent-only tail: every block is a joint block.
LATENT_ONLY_DEPTH = 0
# Tokens get the block MLP too (the ~71 G/block variant).
TOKEN_MLP = True


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """base.py's model with the ViT + Perceiver replaced by the joint transformer."""
    config = _base_build_model_config(common)
    encoder_config = config.encoder_config
    perceiver = encoder_config.perceiver_config
    assert perceiver is not None
    encoder_config.depth = 0
    encoder_config.perceiver_config = JointLatentConfig(
        register_dim=perceiver.register_dim,
        joint_depth=JOINT_DEPTH,
        latent_only_depth=LATENT_ONLY_DEPTH,
        token_mlp=TOKEN_MLP,
        student_dims=list(PROJECTION_DIMS),
        student_output_norm=True,
    )
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
