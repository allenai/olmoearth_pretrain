"""ViT-base v1.1 with 2D RoPE, using one-hot WorldCover instead of raw WorldCover.

Identical to scripts/official/v1_1/rope.py (axial 2D RoPE on top of the hidden1
baseline) except that the ``worldcover`` modality -- a single band of raw ESA class
codes -- is replaced with ``worldcover_onehot``, which expands those codes into one
channel per class at load time. WorldCover remains a decode-only target, so the change
only affects how that target is embedded by the (target) encoder, not the loss itself.

The on-disk dataset is unchanged: worldcover_onehot is derived from the stored
``worldcover`` band at read time, so the same h5py_dir is used.
"""

import logging

import base
from base import (
    build_dataset_config,
    build_trainer_config,
    build_visualize_config,
)
from rope import build_model_config

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.internal.experiment import CommonComponents, SubCmd, main

logger = logging.getLogger(__name__)

_RAW_WORLDCOVER = Modality.WORLDCOVER.name
_ONEHOT_WORLDCOVER = Modality.WORLDCOVER_ONEHOT.name


def _swap_worldcover(modalities: list[str]) -> list[str]:
    """Replace raw worldcover with its one-hot variant, preserving order."""
    return [
        _ONEHOT_WORLDCOVER if modality == _RAW_WORLDCOVER else modality
        for modality in modalities
    ]


# WorldCover stays a decode-only target and is excluded as a contrastive negative.
# base.build_train_module_config and base.build_dataloader_config (and base._masking_config)
# read this module-level list at call time, so reassigning it here is enough for the raw
# -> one-hot swap to propagate to the masking and loss configs.
base.ONLY_DECODE_MODALITIES = _swap_worldcover(base.ONLY_DECODE_MODALITIES)


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build common components, swapping raw WorldCover for the one-hot version."""
    config = base.build_common_components(script, cmd, run_name, cluster, overrides)
    config.training_modalities = _swap_worldcover(config.training_modalities)
    return config


def run() -> None:
    """Run the experiment."""
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=base.build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=base.build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )


if __name__ == "__main__":
    run()
