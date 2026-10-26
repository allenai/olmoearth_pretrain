"""Bathymetry-target ablation, treatment arm: tiny_60k plus a decode-only depth target.

The target is satellite-derived bathymetry (Modality.SDB) computed from the S2 L2A
series at load time, with optically deep pixels removed using GEBCO_2026. See
/weka/dfive-default/agent/piperbots/2026-10-26-piperw-bathymetry-target/README.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "official" / "v1_2"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import base  # noqa: E402
import tiny  # noqa: E402
from tiny_60k import run_with  # noqa: E402

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig  # noqa: E402
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents  # noqa: E402
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (  # noqa: E402
    ContrastiveLatentMIMTrainModuleConfig,
)

SDB_GEBCO_PATCHES = "/weka/dfive-default/agent/piperbots/2026-10-26-piperw-bathymetry-target/gebco_patches.npy"
ONLY_DECODE_MODALITIES = base.ONLY_DECODE_MODALITIES + [Modality.SDB.name]


def build_common_components(*args, **kwargs) -> CommonComponents:
    """v1.2 modalities plus sdb."""
    config = base.build_common_components(*args, **kwargs)
    config.training_modalities = config.training_modalities + [Modality.SDB.name]
    return config


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Tiny train module with sdb decoded only and its negatives masked like other maps."""
    config = tiny.build_train_module_config(common)
    config.masking_config.strategy_config["only_decode_modalities"] = (
        ONLY_DECODE_MODALITIES
    )
    config.loss_config.loss_config["mask_negatives_for_modalities"] = (
        ONLY_DECODE_MODALITIES
    )
    return config


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Same dataloader, with sdb decoded only."""
    config = base.build_dataloader_config(common)
    config.masking_config.strategy_config["only_decode_modalities"] = (
        ONLY_DECODE_MODALITIES
    )
    return config


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Same H5 set, with GEBCO patches for the deep-water mask."""
    config = base.build_dataset_config(common)
    config.sdb_gebco_patches_file = SDB_GEBCO_PATCHES
    return config


if __name__ == "__main__":
    run_with(
        common_components_builder=build_common_components,
        train_module_config_builder=build_train_module_config,
        dataloader_config_builder=build_dataloader_config,
        dataset_config_builder=build_dataset_config,
    )
