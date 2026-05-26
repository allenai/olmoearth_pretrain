"""hidden1 + weight_high supervision with SRTM normalization fixed.

The default SRTM normalization in computed.json uses stale stats (mean=677, std=993)
that don't match the actual dataset (mean≈360, std≈516, scanned over 1.6B pixels).
This experiment overrides SRTM normalization with the Presto convention `x / 2000`,
which is stat-free and avoids the asymmetric ±2σ min-max shape that compresses 99%
of pixels into a narrow band.

The override is scoped to a dataset subclass instantiated by this script's config
builder — the global Normalizer / computed.json is untouched, so checkpoints from
other experiments load unaffected.
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

build_model_config = partial(_build_model_config, weight_multiplier=WEIGHT_MULTIPLIER)


class OlmoEarthDatasetFixSrtm(OlmoEarthDataset):
    """Dataset that normalizes SRTM as Presto x/2000 and leaves all other modalities alone."""

    def normalize_image(self, modality: ModalitySpec, image: np.ndarray) -> np.ndarray:
        """Normalize the image, overriding SRTM to use Presto-style x/2000."""
        if modality.name == Modality.SRTM.name:
            return (image.astype(np.float32)) / 2000.0
        return super().normalize_image(modality, image)


@dataclass
class OlmoEarthDatasetConfigFixSrtm(OlmoEarthDatasetConfig):
    """Dataset config that builds an OlmoEarthDatasetFixSrtm instance."""

    def build(self) -> OlmoEarthDatasetFixSrtm:
        """Build the dataset with the SRTM normalization override."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs["h5py_dir"] = self.h5py_dir_upath
        kwargs["cache_dir"] = (
            self.cache_dir_upath if self.cache_dir is not None else None
        )
        kwargs["dtype"] = self.get_numpy_dtype()
        logger.info(f"OlmoEarthDatasetFixSrtm kwargs: {kwargs}")
        return OlmoEarthDatasetFixSrtm(**kwargs)


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfigFixSrtm:
    """Build the dataset config for an experiment."""
    default = _build_default_dataset_config(common)
    return OlmoEarthDatasetConfigFixSrtm(
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
