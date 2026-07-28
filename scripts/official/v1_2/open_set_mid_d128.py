r"""Launch script: open-set mid-training on the d128 spatial latent.

Initializes from the finished ``regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1``
pretraining checkpoint (667,200 steps), trains ONLY the open-set probe for the
first 40k steps (backbone frozen), then unfreezes everything at a low backbone
LR. Trains on the full osm_sampling + open-set concat dataset throughout. See
``open_set_mid_base`` for the rationale and knobs.

Usage (from the repo root; the env var is required for the partial checkpoint
load and is propagated to the Beaker job)::

    OE_LOAD_SKIP_MISMATCHED_KEYS=1 \
    python scripts/official/v1_2/open_set_mid_d128.py launch open_set_mid_d128 \
        ai2/jupiter --launch.num_gpus=8
"""

import logging

from base import build_visualize_config
from open_set_base import (
    build_common_components,
    build_dataloader_config,
    build_osm_plus_open_set_dataset_config,
)
from open_set_base import (
    build_model_config as _build_model_config_with_dim,
)
from open_set_mid_base import (
    build_mid_train_module_config,
    build_mid_trainer_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.open_set_latent_mim import OpenSetLatentMIMConfig

logger = logging.getLogger(__name__)

REGISTER_DIM = 128
# Path (relative to the repo root) used by the in-loop Beaker eval jobs to rebuild
# this exact model config when loading a checkpoint.
MODULE_PATH = "scripts/official/v1_2/open_set_mid_d128.py"


def build_model_config(common: CommonComponents) -> OpenSetLatentMIMConfig:
    """d128 spatial latent + map supervision + open-set probe."""
    return _build_model_config_with_dim(common, register_dim=REGISTER_DIM)


def build_dataset_config(common: CommonComponents):
    """Full osm_sampling + open-set concat dataset."""
    return build_osm_plus_open_set_dataset_config(common)


def build_trainer_config(common: CommonComponents):
    """Trainer config: init from the d128 checkpoint, eval jobs point here."""
    return build_mid_trainer_config(common, MODULE_PATH, REGISTER_DIM)


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_mid_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )
