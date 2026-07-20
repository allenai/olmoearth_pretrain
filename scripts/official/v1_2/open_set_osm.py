r"""Launch script: osm_sampling + open-set supervised pretraining.

Trains on a concatenation of the global ``osm_sampling`` dataset (self-supervised
only) and the open-set supervised dataset (self-supervised + supervised). The
``osm_sampling`` H5s lack the label layers, so those samples are missing-filled
and contribute only the self-supervised loss, while the open-set samples add the
supervised segmentation + regression signal.

Inherits all v1.2-faster speedups (projection-only target, DDP + bf16, in-loop
evals as separate Beaker jobs).

Usage (from the repo root)::

    python scripts/official/v1_2/open_set_osm.py launch open_set_osm ai2/jupiter \\
        --launch.num_gpus=8

Set the open-set H5 directory in ``open_set_base.OPEN_SET_H5_DIR`` once the H5s
are built.
"""

import logging

from base_faster import build_dataloader_config, build_visualize_config
from open_set_base import (
    build_common_components,
    build_model_config,
    build_osm_plus_open_set_dataset_config,
    build_train_module_config,
)
from open_set_base import (
    build_trainer_config as _build_trainer_config_with_path,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main

logger = logging.getLogger(__name__)

# Path (relative to the repo root) used by the in-loop Beaker eval jobs to rebuild
# this exact model config when loading a checkpoint.
MODULE_PATH = "scripts/official/v1_2/open_set_osm.py"


def build_dataset_config(common: CommonComponents):
    """Concatenated osm_sampling + open-set supervised dataset."""
    return build_osm_plus_open_set_dataset_config(common)


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
