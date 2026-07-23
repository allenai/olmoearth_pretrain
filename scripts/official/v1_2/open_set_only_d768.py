r"""Launch script: open-set supervised pretraining on the d768 spatial latent.

Trains on the open-set supervised dataset alone (every sample carries the
``open_set`` / ``open_set_regression`` labels). The model is the register-
bottleneck (Perceiver) ``wideread`` recipe with a 768-wide spatial latent grid;
the open-set probe and the map supervision heads (worldcover, srtm, osm raster,
canopy height, cdl, worldcereal) both read that grid.

Usage (from the repo root)::

    python scripts/official/v1_2/open_set_only_d768.py launch open_set_only_d768 \\
        ai2/jupiter --launch.num_gpus=8

Set the open-set H5 directory in ``open_set_base.OPEN_SET_H5_DIR`` (or override
via ``--dataset.h5py_dir=...``) once the H5s are built.
"""

import logging

from base import build_visualize_config
from open_set_base import (
    build_common_components,
    build_dataloader_config,
    build_open_set_dataset_config,
    build_train_module_config,
)
from open_set_base import (
    build_model_config as _build_model_config_with_dim,
)
from open_set_base import (
    build_trainer_config as _build_trainer_config_with_path,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.open_set_latent_mim import OpenSetLatentMIMConfig

logger = logging.getLogger(__name__)

REGISTER_DIM = 768
# Path (relative to the repo root) used by the in-loop Beaker eval jobs to rebuild
# this exact model config when loading a checkpoint.
MODULE_PATH = "scripts/official/v1_2/open_set_only_d768.py"


def build_model_config(common: CommonComponents) -> OpenSetLatentMIMConfig:
    """d768 spatial latent + map supervision + open-set probe."""
    return _build_model_config_with_dim(common, register_dim=REGISTER_DIM)


def build_dataset_config(common: CommonComponents):
    """Open-set supervised dataset only."""
    return build_open_set_dataset_config(common)


def build_trainer_config(common: CommonComponents):
    """Trainer config with the eval jobs pointed at this script."""
    return _build_trainer_config_with_path(common, MODULE_PATH)


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
