"""SICKLE dataset class."""

import logging
import random
from pathlib import Path

import einops
import numpy as np
import torch
from torch.utils.data import Dataset

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.evals.datasets.constants import (
    EVAL_L8_BAND_NAMES,
    EVAL_S1_BAND_NAMES,
    EVAL_S2_L2A_BAND_NAMES,
    EVAL_TO_HELIOS_L8_BANDS,
    EVAL_TO_HELIOS_S1_BANDS,
    EVAL_TO_HELIOS_S2_L2A_BANDS,
)
from olmoearth_pretrain.evals.datasets.normalize import normalize_bands
from olmoearth_pretrain.evals.datasets.utils import load_min_max_stats
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)

# TODO: Move this into a worker init function and see if this has to do with the eval memory leak on long runs
torch.multiprocessing.set_sharing_strategy("file_system")
MONTH_TO_INT = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# Minimum number of months of data required for each modality
# The growing season is either 5 or 6 months
# TODO: Use dynamic number of months instead of fixed to 5
MIN_MONTHS = 5

# Original bands from the SICKLE dataset
S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
S1_BANDS = ["VV", "VH"]
L8_BANDS = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10"]
# Landsat 8 bands after imputing missing bands
L8_BANDS_IMPUTED = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"]

S2_BAND_STATS = {
    "01 - Coastal aerosol": {"mean": 2881.0837, "std": 1811.5366},
    "02 - Blue": {"mean": 2863.4873, "std": 1579.1715},
    "03 - Green": {"mean": 2917.7632, "std": 1458.1348},
    "04 - Red": {"mean": 2773.0498, "std": 1375.6265},
    "05 - Vegetation Red Edge": {"mean": 3179.3999, "std": 1428.6418},
    "06 - Vegetation Red Edge": {"mean": 3766.0242, "std": 1330.0535},
    "07 - Vegetation Red Edge": {"mean": 4023.8601, "std": 1321.4177},
    "08 - NIR": {"mean": 4051.9189, "std": 1266.5647},
    "08A - Vegetation Red Edge": {"mean": 4097.6514, "std": 1284.4961},
    "09 - Water vapour": {"mean": 5982.4233, "std": 2571.2117},
    "11 - SWIR": {"mean": 2460.8518, "std": 834.2809},
    "12 - SWIR": {"mean": 1922.3293, "std": 694.0873},
}

S1_BAND_STATS = {
    "vv": {"mean": -9.6199, "std": 3.0528},
    "vh": {"mean": -16.7964, "std": 3.7383},
}

L8_BAND_STATS = {
    "B1": {"mean": 13920.3027, "std": 7094.2832},
    "B2": {"mean": 14240.5117, "std": 6925.8511},
    "B3": {"mean": 15333.0703, "std": 6172.2549},
    "B4": {"mean": 15181.3350, "std": 6173.0508},
    "B5": {"mean": 21489.8750, "std": 4819.3726},
    "B6": {"mean": 15757.5273, "std": 3551.9348},
    "B7": {"mean": 13321.4941, "std": 2948.6643},
    "B8": {"mean": 13321.4941, "std": 2948.6643},
    "B9": {"mean": 13321.4941, "std": 2948.6643},
    "B10": {"mean": 32624.0508, "std": 14472.2197},
    "B11": {"mean": 32624.0508, "std": 14472.2197},
}


class SICKLEDataset(Dataset):
    """SICKLE dataset class."""

    allowed_modalities = [
        Modality.LANDSAT.name,
        Modality.SENTINEL1.name,
        Modality.SENTINEL2_L2A.name,
    ]

    def __init__(
        self,
        path_to_splits: Path,
        split: str = "train",
        partition: str = "default",
        norm_stats_from_pretrained: bool = True,
        norm_method: str = "norm_no_clip",
        input_modalities: list[str] = [],
    ):
        """Init SICKLE dataset.

        Args:
            path_to_splits: Path where .pt objects returned by process_sickle have been saved
            split: Split to use
            partition: Partition to use
            norm_stats_from_pretrained: Whether to use normalization stats from pretrained model
            norm_method: Normalization method to use, only when norm_stats_from_pretrained is False
            input_modalities: List of modalities to use, must be a subset of ["landsat8", "sentinel1", "sentinel2"]
        """
        assert split in ["train", "val", "valid", "test"]
        if split == "valid":
            split = "val"
        if split in ["train", "val"]:
            split_to_load = "train"
        else:
            # the validation set is used as the test set, since
            # the test set has no labels. This means that for us,
            # we will draw the validation set from the training set
            split_to_load = "val"

        assert len(input_modalities) > 0, "input_modalities must be set"
        assert all(
            modality in self.allowed_modalities for modality in input_modalities
        ), f"input_modalities must be a subset of {self.allowed_modalities}"

        self.input_modalities = input_modalities

        # Load min/max stats and merge with band stats
        self.min_max_stats = load_min_max_stats()["sickle"]
        s2_minmax = self.min_max_stats["sentinel2_l2a"]
        s1_minmax = self.min_max_stats["sentinel1"]
        l8_minmax = self.min_max_stats["landsat"]

        merged_s2_stats = {
            band_name: {
                **(
                    {k: S2_BAND_STATS[band_name][k] for k in ("mean", "std")}
                    if band_name in S2_BAND_STATS
                    else {}
                ),
                **(
                    {k: s2_minmax[band_name][k] for k in ("min", "max")}
                    if band_name in s2_minmax
                    else {}
                ),
            }
            for band_name in EVAL_S2_L2A_BAND_NAMES
        }
        merged_s1_stats = {
            band_name: {
                **(
                    {k: S1_BAND_STATS[band_name][k] for k in ("mean", "std")}
                    if band_name in S1_BAND_STATS
                    else {}
                ),
                **(
                    {k: s1_minmax[band_name][k] for k in ("min", "max")}
                    if band_name in s1_minmax
                    else {}
                ),
            }
            for band_name in EVAL_S1_BAND_NAMES
        }
        merged_l8_stats = {
            band_name: {
                **(
                    {k: L8_BAND_STATS[band_name][k] for k in ("mean", "std")}
                    if band_name in L8_BAND_STATS
                    else {}
                ),
                **(
                    {k: l8_minmax[band_name][k] for k in ("min", "max")}
                    if band_name in l8_minmax
                    else {}
                ),
            }
            for band_name in EVAL_L8_BAND_NAMES
        }

        self.s2_means, self.s2_stds, self.s2_mins, self.s2_maxs = self._get_norm_stats(
            merged_s2_stats, EVAL_S2_L2A_BAND_NAMES
        )
        self.s1_means, self.s1_stds, self.s1_mins, self.s1_maxs = self._get_norm_stats(
            merged_s1_stats, EVAL_S1_BAND_NAMES
        )
        self.l8_means, self.l8_stds, self.l8_mins, self.l8_maxs = self._get_norm_stats(
            merged_l8_stats, EVAL_L8_BAND_NAMES
        )

        self.norm_method = norm_method

        self.norm_stats_from_pretrained = norm_stats_from_pretrained
        # If normalize with pretrained stats, we initialize the normalizer here
        if self.norm_stats_from_pretrained:
            from olmoearth_pretrain.data.normalize import Normalizer, Strategy

            self.normalizer_computed = Normalizer(Strategy.COMPUTED)

        self.s2_images_dir = path_to_splits / f"sickle_{split_to_load}" / "s2_images"
        self.s1_images_dir = path_to_splits / f"sickle_{split_to_load}" / "s1_images"
        self.l8_images_dir = path_to_splits / f"sickle_{split_to_load}" / "l8_images"
        self.labels = torch.load(
            path_to_splits / f"sickle_{split_to_load}" / "targets.pt"
        )
        self.months = torch.load(
            path_to_splits / f"sickle_{split_to_load}" / "months.pt"
        )
        self.indices = list(range(self.labels.shape[0]))

        if split in ["train", "val"]:
            # sample 90 % of the training data for training, 10% for val
            num_train = int(len(self.indices) * 0.9)
            random.Random(6012).shuffle(self.indices)
            if split == "train":
                self.indices = self.indices[:num_train]
            elif split == "val":
                self.indices = self.indices[num_train:]

    @staticmethod
    def _get_norm_stats(
        imputed_band_info: dict[str, dict[str, float]],
        band_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        means = []
        stds = []
        mins = []
        maxs = []
        for band_name in band_names:
            assert band_name in imputed_band_info, f"{band_name} not found in band_info"
            means.append(imputed_band_info[band_name]["mean"])  # type: ignore
            stds.append(imputed_band_info[band_name]["std"])  # type: ignore
            mins.append(imputed_band_info[band_name]["min"])  # type: ignore
            maxs.append(imputed_band_info[band_name]["max"])  # type: ignore
        return np.array(means), np.array(stds), np.array(mins), np.array(maxs)

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:
        """Return a single SICKLE data instance."""
        idx = self.indices[idx]

        l8_image = torch.load(self.l8_images_dir / f"{idx}.pt")
        l8_image = einops.rearrange(l8_image, "t c h w -> h w t c")  # (32, 32, 5, 11)
        l8_image = l8_image[:, :, :, EVAL_TO_HELIOS_L8_BANDS]

        s2_image = torch.load(self.s2_images_dir / f"{idx}.pt")
        s2_image = einops.rearrange(s2_image, "t c h w -> h w t c")  # (32, 32, 5, 13)
        s2_image = s2_image[:, :, :, EVAL_TO_HELIOS_S2_L2A_BANDS]

        s1_image = torch.load(self.s1_images_dir / f"{idx}.pt")
        s1_image = einops.rearrange(s1_image, "t c h w -> h w t c")  # (32, 32, 5, 2)
        s1_image = s1_image[:, :, :, EVAL_TO_HELIOS_S1_BANDS]

        labels = self.labels[idx]  # (32, 32)
        months = self.months[idx]  # (5)

        if not self.norm_stats_from_pretrained:
            l8_image = normalize_bands(
                l8_image.numpy(),
                self.l8_means,
                self.l8_stds,
                self.l8_mins,
                self.l8_maxs,
                self.norm_method,
            )
            s2_image = normalize_bands(
                s2_image.numpy(),
                self.s2_means,
                self.s2_stds,
                self.s2_mins,
                self.s2_maxs,
                self.norm_method,
            )
            s1_image = normalize_bands(
                s1_image.numpy(),
                self.s1_means,
                self.s1_stds,
                self.s1_mins,
                self.s1_maxs,
                self.norm_method,
            )
        else:
            l8_image = self.normalizer_computed.normalize(Modality.LANDSAT, l8_image)
            s2_image = self.normalizer_computed.normalize(
                Modality.SENTINEL2_L2A, s2_image
            )
            s1_image = self.normalizer_computed.normalize(Modality.SENTINEL1, s1_image)

        timestamps = []
        for month in months:
            item = int(month)
            item_month, item_year = str(item)[4:], str(item)[:4]
            # NOTE: month is 0-indexed, from 0 to 11
            timestamps.append(
                torch.tensor([1, int(item_month) - 1, int(item_year)], dtype=torch.long)
            )
        timestamps = torch.stack(timestamps)

        # Build sample dict based on requested modalities
        sample_dict = {"timestamps": timestamps}

        if Modality.LANDSAT.name in self.input_modalities:
            sample_dict[Modality.LANDSAT.name] = torch.tensor(l8_image).float()
        if Modality.SENTINEL1.name in self.input_modalities:
            sample_dict[Modality.SENTINEL1.name] = torch.tensor(s1_image).float()
        if Modality.SENTINEL2_L2A.name in self.input_modalities:
            sample_dict[Modality.SENTINEL2_L2A.name] = torch.tensor(s2_image).float()

        if not sample_dict:
            raise ValueError(f"No modalities requested in {self.input_modalities}")

        masked_sample = MaskedOlmoEarthSample.from_olmoearthsample(
            OlmoEarthSample(**sample_dict)
        )

        return masked_sample, labels.long()
