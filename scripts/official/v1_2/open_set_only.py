r"""Launch script: open-set *supervised only* pretraining.

Trains on the open-set supervised dataset alone (every sample carries the
``open_set`` / ``open_set_regression`` labels). Inherits all v1.2-faster speedups
(projection-only target, DDP + bf16, in-loop evals as separate Beaker jobs) and
adds the supervised probe loss.

Usage (from the repo root)::

    python scripts/official/v1_2/open_set_only.py launch open_set_only ai2/jupiter \\
        --launch.num_gpus=8

Set the open-set H5 directory in ``open_set_base.OPEN_SET_H5_DIR`` (or override
via ``--dataset.h5py_dir=...``) once the H5s are built.
"""

import logging

from base_faster import build_visualize_config
from open_set_base import (
    build_common_components,
    build_dataloader_config,
    build_model_config,
    build_open_set_dataset_config,
    build_train_module_config,
)
from open_set_base import (
    build_trainer_config as _build_trainer_config_with_path,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main

logger = logging.getLogger(__name__)

# Path (relative to the repo root) used by the in-loop Beaker eval jobs to rebuild
# this exact model config when loading a checkpoint.
MODULE_PATH = "scripts/official/v1_2/open_set_only.py"


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
