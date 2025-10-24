"""PASTIS-R (S2+S1) dataset class."""

import json
import logging
from pathlib import Path

import einops
import numpy as np
import torch
import torch.multiprocessing
from torch.utils.data import Dataset

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.evals.datasets.constants import (
    EVAL_S1_BAND_NAMES,
    EVAL_S2_BAND_NAMES,
    EVAL_TO_HELIOS_S1_BANDS,
    EVAL_TO_HELIOS_S2_BANDS,
)
from olmoearth_pretrain.evals.datasets.normalize import normalize_bands
from olmoearth_pretrain.evals.datasets.utils import load_min_max_stats
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)

# TODO: Move this into a worker init function and see if this has to do with the eval memory leak on long runs
torch.multiprocessing.set_sharing_strategy("file_system")


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
    "vv": {"mean": -10.7902, "std": 2.8360},
    "vh": {"mean": -17.3257, "std": 2.8106},
}


class PASTISRDataset(Dataset):
    """PASTIS-R dataset class."""

    allowed_modalities = [Modality.SENTINEL1.name, Modality.SENTINEL2_L2A.name]

    def __init__(
        self,
        path_to_splits: Path,
        dir_partition: Path | None = None,
        split: str = "train",
        partition: str = "default",
        norm_stats_from_pretrained: bool = True,
        norm_method: str = "norm_no_clip",
        input_modalities: list[str] = [],
    ):
        """Init PASTIS-R dataset.

        Args:
            path_to_splits: Path where .pt objects returned by process_pastis_r have been saved
            dir_partition: Path to the partition directory, only used if partition is not "default"
            split: Split to use
            partition: Partition to use
            norm_stats_from_pretrained: Whether to use normalization stats from pretrained model
            norm_method: Normalization method to use, only when norm_stats_from_pretrained is False
            input_modalities: List of modalities to use, must be a subset of ["sentinel1", "sentinel2_l2a"]
        """
        assert split in ["train", "valid", "test"]

        assert len(input_modalities) > 0, "input_modalities must be set"
        assert all(
            modality in self.allowed_modalities for modality in input_modalities
        ), f"input_modalities must be a subset of {self.allowed_modalities}"

        self.input_modalities = input_modalities

        # Load min/max stats and merge with band stats
        self.min_max_stats = load_min_max_stats()["pastis"]
        s2_minmax = self.min_max_stats["sentinel2_l2a"]
        s1_minmax = self.min_max_stats["sentinel1"]

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
            for band_name in EVAL_S2_BAND_NAMES
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

        self.s2_means, self.s2_stds, self.s2_mins, self.s2_maxs = self._get_norm_stats(
            merged_s2_stats, EVAL_S2_BAND_NAMES
        )
        self.s1_means, self.s1_stds, self.s1_mins, self.s1_maxs = self._get_norm_stats(
            merged_s1_stats, EVAL_S1_BAND_NAMES
        )
        self.split = split
        self.norm_method = norm_method

        self.norm_stats_from_pretrained = norm_stats_from_pretrained
        # If normalize with pretrained stats, we initialize the normalizer here
        if self.norm_stats_from_pretrained:
            from olmoearth_pretrain.data.normalize import Normalizer, Strategy

            self.normalizer_computed = Normalizer(Strategy.COMPUTED)

        self.s2_images_dir = path_to_splits / f"pastis_r_{split}" / "s2_images"
        self.s1_images_dir = path_to_splits / f"pastis_r_{split}" / "s1_images"
        self.labels = torch.load(path_to_splits / f"pastis_r_{split}" / "targets.pt")
        self.months = torch.load(path_to_splits / f"pastis_r_{split}" / "months.pt")
        if (partition != "default") and (split == "train"):
            assert dir_partition is not None, "dir_partition must be set"
            # PASTIS and PASTIS-R share the same partitions so we just use PASTIS Partitions
            with open(dir_partition / f"{partition}_partition.json") as json_file:
                subset_indices = json.load(json_file)
            self.months = self.months[subset_indices]
            self.labels = self.labels[subset_indices]
            self.indices = subset_indices
        else:
            self.indices = list(range(len(self.months)))

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
        """Return a single PASTIS data instance."""
        image_idx = self.indices[idx]
        s2_image: np.ndarray = torch.load(
            self.s2_images_dir / f"{image_idx}.pt"
        ).numpy()
        s2_image = einops.rearrange(s2_image, "t c h w -> h w t c")  # (64, 64, 12, 13)

        s1_image: np.ndarray = torch.load(
            self.s1_images_dir / f"{image_idx}.pt"
        ).numpy()
        s1_image = einops.rearrange(s1_image, "t c h w -> h w t c")  # (64, 64, 12, 2)

        labels = self.labels[idx]  # (64, 64)
        months = self.months[idx]  # (12)

        # If using norm stats from pretrained we should normalize before we rearrange
        if not self.norm_stats_from_pretrained:
            s2_image = normalize_bands(
                s2_image,
                self.s2_means,
                self.s2_stds,
                self.s2_mins,
                self.s2_maxs,
                self.norm_method,
            )
            s1_image = normalize_bands(
                s1_image,
                self.s1_means,
                self.s1_stds,
                self.s1_mins,
                self.s1_maxs,
                self.norm_method,
            )

        s2_image = s2_image[:, :, :, EVAL_TO_HELIOS_S2_BANDS]
        s1_image = s1_image[:, :, :, EVAL_TO_HELIOS_S1_BANDS]
        if self.norm_stats_from_pretrained:
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

        if Modality.SENTINEL1.name in self.input_modalities:
            sample_dict[Modality.SENTINEL1.name] = torch.from_numpy(s1_image).float()
        if Modality.SENTINEL2_L2A.name in self.input_modalities:
            sample_dict[Modality.SENTINEL2_L2A.name] = torch.from_numpy(
                s2_image
            ).float()

        if not sample_dict:
            raise ValueError(f"No valid modalities found in: {self.input_modalities}")

        masked_sample = MaskedOlmoEarthSample.from_olmoearthsample(
            OlmoEarthSample(**sample_dict)
        )

        return masked_sample, labels.long()
