"""hidden1 + weight_high supervision with SRTM fix + AlphaEarth-style log on optical.

Extends hidden1_supervision_weight_high_fix_srtm_norm.py with two changes layered:

1. SRTM: Presto-style `x / 2000` (replaces stale computed.json stats).
2. Sentinel-2 / Landsat: AlphaEarth Eq. 1 `log(x + 1) / 10` applied per pixel
   (raw → log-compressed [0, ~1] range). Negatives from atmospheric correction
   are clipped to 0 before log so log1p stays defined.

Both overrides live in a dataset subclass instantiated only by this script's
config builder, so the global Normalizer / computed.json is unchanged and
checkpoints from other experiments load unaffected.
"""

import logging
from dataclasses import dataclass
from functools import partial

import numpy as np
from hidden1 import (
    build_common_components,
    build_dataloader_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)
from hidden1 import build_dataset_config as _build_default_dataset_config
from hidden1_supervision import build_model_config as _build_model_config

from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.data.dataset import OlmoEarthDataset, OlmoEarthDatasetConfig
from olmoearth_pretrain.internal.experiment import CommonComponents, main

logger = logging.getLogger(__name__)

WEIGHT_MULTIPLIER = 10.0

LOG_OPTICAL_MODALITIES = {
    Modality.SENTINEL2_L2A.name,
    Modality.LANDSAT.name,
}

build_model_config = partial(_build_model_config, weight_multiplier=WEIGHT_MULTIPLIER)


class OlmoEarthDatasetFixSrtmLogOptical(OlmoEarthDataset):
    """Dataset that fixes SRTM (Presto x/2000) and log-scales S2 / Landsat (AlphaEarth Eq. 1)."""

    def normalize_image(self, modality: ModalitySpec, image: np.ndarray) -> np.ndarray:
        """Normalize the image with SRTM and optical overrides."""
        if modality.name == Modality.SRTM.name:
            return (image.astype(np.float32)) / 2000.0
        if modality.name in LOG_OPTICAL_MODALITIES:
            return np.log1p(np.maximum(image.astype(np.float32), 0.0)) / 10.0
        return super().normalize_image(modality, image)


@dataclass
class OlmoEarthDatasetConfigFixSrtmLogOptical(OlmoEarthDatasetConfig):
    """Dataset config that builds an OlmoEarthDatasetFixSrtmLogOptical instance."""

    def build(self) -> OlmoEarthDatasetFixSrtmLogOptical:
        """Build the dataset with the SRTM and optical normalization overrides."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs["h5py_dir"] = self.h5py_dir_upath
        kwargs["cache_dir"] = (
            self.cache_dir_upath if self.cache_dir is not None else None
        )
        kwargs["dtype"] = self.get_numpy_dtype()
        logger.info(f"OlmoEarthDatasetFixSrtmLogOptical kwargs: {kwargs}")
        return OlmoEarthDatasetFixSrtmLogOptical(**kwargs)


def build_dataset_config(
    common: CommonComponents,
) -> OlmoEarthDatasetConfigFixSrtmLogOptical:
    """Build the dataset config for an experiment."""
    default = _build_default_dataset_config(common)
    return OlmoEarthDatasetConfigFixSrtmLogOptical(
        h5py_dir=default.h5py_dir,
        training_modalities=default.training_modalities,
    )


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
