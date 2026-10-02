"""``rc_pixtgt_pix512`` with an INDEPENDENT pixel draw for every token.

``rc_pixtgt_pix512`` draws one pixel per token cell (the ``p x p`` footprint of one
token) and shares it with every token stacked on that cell: all its timesteps, band
sets and modalities predict the same pixel. Here every token draws its own pixel:
per (sample, cell, timestep, modality) for Sentinel-2 / Sentinel-1 / Landsat and per
(sample, cell, modality) for the static maps. v1.3 tokenizes every modality as a
single band set, so this is one draw per token. Same count of queries and targets
(one per masked token); only the correlation between draws changes. See
``olmoearth_pretrain/nn/pixel_targets.py``.

Question: is it better for a cell's masked tokens to predict one pixel's whole time
series (shared), or to spread over more of the cell's pixels in a single step
(independent)?

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_rc_pixtgtind_pix512``.
Its partners are ``v1_3_rc_pixtgt_pix512`` (shared draw) and ``v1_3_rc_pix512`` (patch
targets).
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_visualize_config,
)
from pure_perceiver_mix import build_mix_trainer_config  # noqa: E402
from rc_pix512 import build_model_config  # noqa: E402
from rc_pixtgt_pix512 import (
    build_train_module_config as _pixtgt_build_train_module_config,  # noqa: E402
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.train.train_module.latent_mim import (  # noqa: E402
    LatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/ablations/rc_pixtgtind_pix512.py"


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """rc_pixtgt_pix512's train module with one pixel drawn per token."""
    config = _pixtgt_build_train_module_config(common)
    config.pixel_target_draw = "independent"
    return config


def build_trainer_config(common: CommonComponents):
    """rc_pix512's evals (incl. ps4 per pixel), re-importing THIS module."""
    return build_mix_trainer_config(common, MODULE_PATH, ps4_student_evals=True)


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
