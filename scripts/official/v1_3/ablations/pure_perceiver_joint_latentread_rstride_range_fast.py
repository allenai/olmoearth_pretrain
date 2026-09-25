"""SPEED copy of ``pure_perceiver_joint_latentread_rstride_range.py``: compiled RoPE.

Same model, sampler, microbatch, seed and evals; the only config difference is
``compile_rope=True`` (the mixed-RoPE op runs through ``torch.compile``). The code
this copy trains with also carries the sync removals that are numerically
equivalent and not flag-gated (masked-mean supervision losses, slice+where token
re-insertion, CPU band-dropout draw, cached month table, on-device non-finite loss
count) -- the original run is pinned to the commit before them.

Purpose: a same-seed loss comparison. It should be faster per step and train
essentially identically to ``v1_3_vit0_rstride_ps8_lb512_latentread_rangerope_joint12``;
curves that separate beyond run-to-run noise would point at the compiled op.

W&B project ``20260921_perceiver_shapes``; trained as
``v1_3_vit0_rstride_ps8_lb512_fast_latentread_rangerope_joint12``.
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
    build_trainer_config as _rstride_trainer_config,
)
from pure_perceiver_joint_latentread_rstride_range import (  # noqa: E402
    build_dataloader_config,
    build_train_module_config,
)
from pure_perceiver_joint_latentread_rstride_range import (  # noqa: E402
    build_model_config as _base_arm_model_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.joint_latent import JointLatentConfig  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = (
    "scripts/official/v1_3/ablations/"
    "pure_perceiver_joint_latentread_rstride_range_fast.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The interval random-stride arm with the compiled RoPE op."""
    config = _base_arm_model_config(common)
    perceiver = config.encoder_config.perceiver_config
    assert isinstance(perceiver, JointLatentConfig)
    assert perceiver.latent_time_range and not perceiver.latent_spatial_range
    perceiver.compile_rope = True
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
