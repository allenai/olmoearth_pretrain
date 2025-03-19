"""PASTIS-R (S2+S1) dataset class."""

import json
from pathlib import Path

import einops
import numpy as np
import torch
import torch.multiprocessing
from torch.utils.data import Dataset
from upath import UPath

from helios.data.constants import Modality
from helios.data.dataset import HeliosSample
from helios.train.masking import MaskedHeliosSample

from .constants import (
    EVAL_S1_BAND_NAMES,
    EVAL_S2_BAND_NAMES,
    EVAL_TO_HELIOS_S1_BANDS,
    EVAL_TO_HELIOS_S2_BANDS,
)
from .normalize import normalize_bands

torch.multiprocessing.set_sharing_strategy("file_system")

PASTIS_R_DIR = UPath("/weka/dfive-default/presto_eval_sets/pastis_r")


S2_BAND_STATS = {
    "01 - Coastal aerosol": {"mean": 1201.6458740234375, "std": 1254.5341796875},
    "02 - Blue": {"mean": 1201.6458740234375, "std": 1254.5341796875},
    "03 - Green": {"mean": 1398.6396484375, "std": 1200.8133544921875},
    "04 - Red": {"mean": 1452.169921875, "std": 1260.5355224609375},
    "05 - Vegetation Red Edge": {"mean": 1783.147705078125, "std": 1188.0682373046875},
    "06 - Vegetation Red Edge": {"mean": 2698.783935546875, "std": 1163.632080078125},
    "07 - Vegetation Red Edge": {"mean": 3022.353271484375, "std": 1220.4384765625},
    "08 - NIR": {"mean": 3164.72802734375, "std": 1237.6727294921875},
    "08A - Vegetation Red Edge": {"mean": 3270.47412109375, "std": 1232.5126953125},
    "09 - Water vapour": {"mean": 3270.47412109375, "std": 1232.5126953125},
    "10 - SWIR - Cirrus": {"mean": 2392.800537109375, "std": 930.82861328125},
    "11 - SWIR": {"mean": 2392.800537109375, "std": 930.82861328125},
    "12 - SWIR": {"mean": 1632.4835205078125, "std": 829.1475219726562},
}

S1_BAND_STATS = {
    "VV": {"mean": -10.7902, "std": 2.8360},
    "VH": {"mean": -17.3257, "std": 2.8106},
}


class PASTISRDataset(Dataset):
    """PASTIS-R dataset class."""

    def __init__(
        self,
        path_to_splits: Path,
        split: str,
        partition: str,
        norm_stats_from_pretrained: bool = True,
        norm_method: str = "norm_no_clip",
    ):
        """Init PASTIS-R dataset.

        Args:
            path_to_splits: Path where .pt objects returned by process_pastis_r have been saved
            split: Split to use
            partition: Partition to use
            norm_stats_from_pretrained: Whether to use normalization stats from pretrained model
            norm_method: Normalization method to use, only when norm_stats_from_pretrained is False
        """
        assert split in ["train", "val", "valid", "test"]
        if split == "valid":
            split = "val"

        self.s2_means, self.s2_stds = self._get_norm_stats(
            S2_BAND_STATS, EVAL_S2_BAND_NAMES
        )
        self.s1_means, self.s1_stds = self._get_norm_stats(
            S1_BAND_STATS, EVAL_S1_BAND_NAMES
        )
        self.split = split
        self.norm_method = norm_method

        self.norm_stats_from_pretrained = norm_stats_from_pretrained
        # If normalize with pretrained stats, we initialize the normalizer here
        if self.norm_stats_from_pretrained:
            from helios.data.normalize import Normalizer, Strategy

            self.normalizer_computed = Normalizer(Strategy.COMPUTED)

        torch_obj = torch.load(path_to_splits / f"pastis_r_{split}.pt")
        self.s2_images = torch_obj["s2_images"]
        self.s1_images = torch_obj["s1_images"]
        self.labels = torch_obj["targets"]
        self.months = torch_obj["months"]

        if (partition != "default") and (split == "train"):
            with open(path_to_splits / f"{partition}_partition.json") as json_file:
                subset_indices = json.load(json_file)

            self.s2_images = self.s2_images[subset_indices]
            self.s1_images = self.s1_images[subset_indices]
            self.labels = self.labels[subset_indices]
            self.months = self.months[subset_indices]

    @staticmethod
    def _get_norm_stats(
        imputed_band_info: dict[str, dict[str, float]],
        band_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        means = []
        stds = []
        for band_name in band_names:
            assert band_name in imputed_band_info, f"{band_name} not found in band_info"
            means.append(imputed_band_info[band_name]["mean"])  # type: ignore
            stds.append(imputed_band_info[band_name]["std"])  # type: ignore
        return np.array(means), np.array(stds)

    def __len__(self) -> int:
        """Length of the dataset."""
        return self.s2_images.shape[0]

    def __getitem__(self, idx: int) -> tuple[MaskedHeliosSample, torch.Tensor]:
        """Return a single PASTIS data instance."""
        s2_image = self.s2_images[idx]  # (12, 13, 64, 64)
        s2_image = einops.rearrange(s2_image, "t c h w -> h w t c")  # (64, 64, 12, 13)
        s2_image = s2_image[:, :, :, EVAL_TO_HELIOS_S2_BANDS]

        s1_image = self.s1_images[idx]  # (12, 2, 64, 64)
        s1_image = einops.rearrange(s1_image, "t c h w -> h w t c")  # (64, 64, 12, 2)
        s1_image = s1_image[:, :, :, EVAL_TO_HELIOS_S1_BANDS]

        labels = self.labels[idx]  # (64, 64)
        months = self.months[idx]  # (12)

        if not self.norm_stats_from_pretrained:
            s2_image = normalize_bands(
                s2_image.numpy(), self.s2_means, self.s2_stds, self.norm_method
            )
            s1_image = normalize_bands(
                s1_image.numpy(), self.s1_means, self.s1_stds, self.norm_method
            )

        if self.norm_stats_from_pretrained:
            s2_image = self.normalizer_computed.normalize(
                Modality.SENTINEL2_L2A, s2_image
            )
            s1_image = self.normalizer_computed.normalize(Modality.SENTINEL1, s1_image)

        timestamps = []
        # A little hack to get the correct year for the timestamps
        # Pastis months are mostly 2018-09 to 2019-08
        year = 2018
        for month in months:
            if month == 1:
                year = 2019
            timestamps.append(torch.tensor([1, month, year], dtype=torch.long))
        timestamps = torch.stack(timestamps)

        masked_sample = MaskedHeliosSample.from_heliossample(
            HeliosSample(
                sentinel2_l2a=torch.tensor(s2_image).float(),
                sentinel1=torch.tensor(s1_image).float(),
                timestamps=timestamps,
            )
        )

        return masked_sample, labels
