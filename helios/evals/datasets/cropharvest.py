"""CropHarvest dataset."""

import logging
from pathlib import Path

import numpy as np
import torch
from einops import repeat
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from upath import UPath

from helios.data.dataset import HeliosSample
from helios.train.masking import MaskedHeliosSample

from .constants import (
    EVAL_S1_BAND_NAMES,
    EVAL_S2_BAND_NAMES,
    EVAL_TO_HELIOS_S1_BANDS,
    EVAL_TO_HELIOS_S2_BANDS,
)
from .cropharvest_package.bands import BANDS
from .cropharvest_package.datasets import CropHarvest
from .cropharvest_package.utils import memoized

logger = logging.getLogger("__main__")


CROPHARVEST_DIR = UPath("/weka/dfive-default/presto-eval_sets/cropharvest")


def _s2helios2ch_name(band_name: str) -> str:
    """Transform Helios S2 band name to CropHarvest (and Breizhcrops) S2 band name."""
    band_number = band_name.split(" ")[0]
    if band_number.startswith("0"):
        band_number = band_number[1:]
    return f"B{band_number}"


def _s1helios2ch_name(band_name: str) -> str:
    """Transform Helios S1 band name to CropHarvest (and Breizhcrops) S2 band name."""
    return band_name.upper()


S2_INPUT_TO_OUTPUT_BAND_MAPPING = [
    BANDS.index(_s2helios2ch_name(b)) for b in EVAL_S2_BAND_NAMES
]

S1_INPUT_TO_OUTPUT_BAND_MAPPING = [
    BANDS.index(_s1helios2ch_name(b)) for b in EVAL_S1_BAND_NAMES
]


@memoized
def _get_eval_datasets(root: Path = CROPHARVEST_DIR) -> list:
    return CropHarvest.create_benchmark_datasets(
        root=root, balance_negative_crops=False, normalize=False
    )


def _download_cropharvest_data(root: Path = CROPHARVEST_DIR) -> None:
    if not root.exists():
        root.mkdir()
        CropHarvest(root, download=True)


class CropHarvestDataset(Dataset):
    """CropHarvest dataset for the Togo cropland classification task."""

    START_MONTH = 1
    DEFAULT_SEED = 42  # mirrors nasaharvest/galileo

    """
    TODO:
      1. get the norm stats (S1, S2) from the norm dict
      3. add the timesteps and latlons too
      4. normalize

      Add SRTM to eval constants (but we might not train it?)
    """

    def __init__(
        self,
        cropharvest_dir: UPath,
        country: str,
        split: str,
        partition: str,
        norm_stats_from_pretrained: bool = False,
        norm_method: str = "norm_no_clip",
        visualize_samples: bool = False,
        timesteps: int = 12,
    ) -> None:
        """CropHarvest dataset.

        cropharvest_dir: Where the cropharvest data is (or where it should be downloaded)
        country: Country to use in eval. Choice of ["China", "Togo", "Kenya", "Brazil"]
        split: Split to use
        partition: Partition to use
        norm_stats_from_pretrained: Whether to use normalization stats from pretrained model
        norm_method: Normalization method to use, only when norm_stats_from_pretrained is False
        visualize_samples: Whether to visualize samples
        timesteps: num timesteps to use
        """
        self.timesteps = timesteps

        _download_cropharvest_data(cropharvest_dir)

        evaluation_datasets = _get_eval_datasets()
        evaluation_datasets = [d for d in evaluation_datasets if country in d.id]
        assert len(evaluation_datasets) == 1
        self.dataset: CropHarvest = evaluation_datasets[0]
        assert self.dataset.task.normalize is False

        if split in ["train", "val"]:
            array, latlons, labels = self.dataset.as_array()
            array, val_array, latlons, val_latlons, labels, val_labels = (
                train_test_split(
                    # 0.2 is arbitrary
                    array,
                    latlons,
                    labels,
                    test_size=0.2,
                    random_state=self.DEFAULT_SEED,
                    stratify=labels,
                )
            )
            if split == "train":
                self.array = array
                self.latlons = latlons
                self.labels = labels
            else:
                self.array = val_array
                self.latlons = val_latlons
                self.labels = val_labels
        elif split == "test":
            arrays, latlons, labels = [], [], []
            for _, test_instance in self.dataset.test_data(max_size=10000):
                arrays.append(test_instance.x)
                labels.append(test_instance.y)
                latlons.append(
                    np.stack([test_instance.lats, test_instance.lons], axis=-1)
                )
            self.array = np.concatenate(arrays, axis=0)
            self.labels = np.concatenate(labels, axis=0)
            self.latlons = np.concatenate(latlons, axis=0)
        else:
            raise ValueError(f"Unrecognized split {split}")

    def __len__(self) -> int:
        """Length of the dataset."""
        return self.array.shape[0]

    def __getitem__(self, idx: int) -> tuple[MaskedHeliosSample, torch.Tensor]:
        """Return the sample at idx."""
        x = self.array[idx]
        y = self.labels[idx]
        # latlon = self.latlons[idx]

        x_hw = repeat(x, "t c -> h w t c", w=1, h=1)

        s2 = x_hw[:, :, :, S2_INPUT_TO_OUTPUT_BAND_MAPPING][
            :, :, :, EVAL_TO_HELIOS_S2_BANDS
        ]
        s1 = x_hw[:, :, :, S1_INPUT_TO_OUTPUT_BAND_MAPPING][
            :, :, :, EVAL_TO_HELIOS_S1_BANDS
        ]

        return MaskedHeliosSample.from_heliossample(
            HeliosSample(sentinel1=s1, sentinel2_l2a=s2, timestamps=y)
        ), y
