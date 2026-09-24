r"""Launch script: v1.3 recipe + open-set supervised probe, osm_sampling + open-set.

Trains the v1.3 model from scratch on the concatenation of the global
``osm_sampling`` corpus (SSL + map supervision + student only; its H5s carry no
open-set labels) and the open-set supervised dataset (all of the above plus the
probe loss). Plain concatenation, so samples are drawn in proportion to dataset
size (~1.14M osm_sampling vs ~1.4M open-set). See ``open_set_base.py`` for the
shared configuration.

Usage (from the repo root)::

    python scripts/official/v1_3/open_set_osm.py launch open_set_osm ai2/jupiter \\
        --launch.num_gpus=8
"""

import logging

from base import build_visualize_config
from open_set_base import (
    build_common_components,
    build_dataloader_config,
    build_model_config,
    build_osm_plus_open_set_dataset_config,
    build_train_module_config,
)
from open_set_base import build_trainer_config as _build_open_set_trainer_config

from olmoearth_pretrain.internal.experiment import CommonComponents, main

logger = logging.getLogger(__name__)

# Path (relative to the repo root) used by the in-loop Beaker eval jobs to rebuild
# this exact model config when loading a checkpoint.
MODULE_PATH = "scripts/official/v1_3/open_set_osm.py"


def build_dataset_config(common: CommonComponents):
    """osm_sampling + open-set concatenation."""
    return build_osm_plus_open_set_dataset_config(common)


def build_trainer_config(common: CommonComponents):
    """Trainer with the in-loop evals pointed at this module."""
    return _build_open_set_trainer_config(common, MODULE_PATH)


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )
