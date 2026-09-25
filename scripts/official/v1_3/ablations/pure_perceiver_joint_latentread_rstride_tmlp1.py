"""Random-stride latentread (point latents) with a light CoLT5-style token MLP.

``pure_perceiver_joint_latentread_rstride.py`` with ``token_mlp_ratio=1.0``: in every joint block the tokens get their
OWN MLP of hidden width 1 x 768 (with its own pre-norm), while the latents keep the
block's 4 x 768 MLP -- a light branch for the many tokens and a heavy one for the few
latents, as in CoLT5 (Ainslie et al. 2023). A token's per-block linear cost drops from
12 d^2 to 6 d^2 (Q/K/V/out 4 d^2 + MLP 2 d^2); estimated ~0.77x the per-sample training
MACs of ``pure_perceiver_joint_latentread_rstride.py`` and ~0.60 s/step. Everything else is ``pure_perceiver_joint_latentread_rstride.py``.

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_vit0_rstride_ps8_tmlp1_latentread_joint12``.
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
)
from pure_perceiver_joint_latentread_rstride import (  # noqa: E402
    build_model_config as _base_arm_model_config,
)
from pure_perceiver_joint_latentread_rstride import (  # noqa: E402
    build_train_module_config as _base_arm_train_module_config,
)
from pure_perceiver_joint_latentread_rstride import (  # noqa: E402
    build_trainer_config as _rstride_trainer_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.joint_latent import JointLatentConfig  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = (
    "scripts/official/v1_3/ablations/pure_perceiver_joint_latentread_rstride_tmlp1.py"
)

TOKEN_MLP_RATIO = 1.0
# Pinned to what these arms trained with (the base arms moved to 512 / microbatch 64).
MAX_LATENTS = 2048
RANK_MICROBATCH_SIZE = 32


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """``pure_perceiver_joint_latentread_rstride``'s model with a light token MLP (latents keep the full MLP)."""
    config = _base_arm_model_config(common)
    perceiver = config.encoder_config.perceiver_config
    assert isinstance(perceiver, JointLatentConfig) and perceiver.token_mlp
    perceiver.token_mlp_ratio = TOKEN_MLP_RATIO
    perceiver.max_latents = MAX_LATENTS
    return config


def build_train_module_config(common: CommonComponents):
    """The base arm's train module at the microbatch these arms trained with."""
    config = _base_arm_train_module_config(common)
    config.rank_microbatch_size = RANK_MICROBATCH_SIZE
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
