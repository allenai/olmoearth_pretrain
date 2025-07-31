"""PASTIS-R (S2+S1) dataset class."""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import einops
import numpy as np
import torch
import torch.multiprocessing
from torch.utils.data import Dataset
from upath import UPath

from helios.data.constants import Modality
from helios.data.dataset import HeliosSample
from helios.evals.datasets.constants import (
    EVAL_S1_BAND_NAMES,
    EVAL_S2_BAND_NAMES,
    EVAL_TO_HELIOS_S1_BANDS,
    EVAL_TO_HELIOS_S2_BANDS,
)
from helios.evals.datasets.normalize import normalize_bands
from helios.train.masking import MaskedHeliosSample

logger = logging.getLogger(__name__)

torch.multiprocessing.set_sharing_strategy("file_system")

DATA_DIR = UPath("/weka/dfive-default/helios/evaluation/PASTIS-R")
PASTIS_DIR = UPath("/weka/dfive-default/presto_eval_sets/pastis_r")
PASTIS_DIR_PARTITION = UPath("/weka/dfive-default/presto_eval_sets/pastis")


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


class PASTISRProcessor:
    """Process PASTIS-R dataset into PyTorch objects.

    This class processes the PASTIS-R dataset into PyTorch objects.
    It loads the S2 and S1 images, and the annotations, and splits them into 4 images.
    It also imputes the missing bands in the S2 images.
    """

    def __init__(self, data_dir: str, output_dir: str):
        """Initialize PASTIS-R processor.

        Args:
            data_dir: Path to PASTIS-R dataset
            output_dir: Path to output directory
        """
        self.data_dir = UPath(data_dir)
        self.output_dir = UPath(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.all_months = [
            "201809",
            "201810",
            "201811",
            "201812",
            "201901",
            "201902",
            "201903",
            "201904",
            "201905",
            "201906",
            "201907",
            "201908",
            "201909",
            "201910",
        ]

    def impute(self, img: torch.Tensor) -> torch.Tensor:
        """Impute missing bands in Sentinel-2 images."""
        img = torch.stack(
            [
                img[0, ...],  # fill B1 with B2, IMPUTED!
                img[0, ...],  # fill B2 with B2
                img[1, ...],  # fill B3 with B3
                img[2, ...],  # fill B4 with B4
                img[3, ...],  # fill B5 with B5
                img[4, ...],  # fill B6 with B6
                img[5, ...],  # fill B7 with B7
                img[6, ...],  # fill B8 with B8
                img[7, ...],  # fill B8A with B8A
                img[7, ...],  # fill B9 with B8A, IMPUTED!
                img[8, ...],  # fill B10 with B11, IMPUTED!
                img[8, ...],  # fill B11 with B11
                img[9, ...],  # fill B12 with B12
            ]
        )
        return img

    def aggregate_months(
        self, modality_name: str, images: torch.Tensor, dates: dict[str, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate images into monthly averages."""
        if (
            modality_name != Modality.SENTINEL2_L2A.name
            and modality_name != Modality.SENTINEL1.name
        ):
            raise ValueError(
                f"Unsupported modality: {modality_name} for PASTIS dataset!"
            )

        months_dict = dict[str, list[torch.Tensor]]()
        for m in self.all_months:
            months_dict[m] = []

        for idx, date in dates.items():
            month = str(date)[:6]
            img = torch.tensor(images[int(idx)], dtype=torch.float32)
            # S2 in PASTIS has 10 bands, so imputation is always needed
            if modality_name == Modality.SENTINEL2_L2A.name:
                if img.shape[1] == 10:
                    img = self.impute(img)
                else:
                    raise ValueError(
                        f"Sentinal2 image has {img.shape[0]} bands, expected 10!"
                    )
            months_dict[month].append(img)

        img_list: list[torch.Tensor] = []
        month_list: list[int] = []
        for month in self.all_months:
            if months_dict[month]:
                stacked_imgs = torch.stack(months_dict[month])
                # NOTE: averaging S2 data may not be the best option, given cloudy scenes
                month_avg = stacked_imgs.mean(dim=0)
                if len(img_list) < 12:
                    img_list.append(month_avg)
                    month_list.append(int(month))

        return torch.stack(img_list), torch.tensor(month_list, dtype=torch.long)

    def process_sample(self, sample: dict[str, Any]) -> dict[str, torch.Tensor] | None:
        """Process a single sample from metadata."""
        properties = sample["properties"]
        dates = properties["dates-S2"]
        patch_id = properties["ID_PATCH"]

        s2_path = self.data_dir / f"DATA_S2/S2_{patch_id}.npy"
        s1_path = self.data_dir / f"DATA_S1A/S1A_{patch_id}.npy"
        target_path = self.data_dir / f"ANNOTATIONS/TARGET_{patch_id}.npy"

        try:
            s2_images = np.load(s2_path)
            s1_images = np.load(s1_path)
            targets = np.load(target_path)[0].astype("int64")
        except FileNotFoundError:
            return None  # Skip missing files

        assert len(dates) == s2_images.shape[0], "Mismatch between S2 dates and images"

        # Only extract the first two bands (vv/vh) for S1
        s1_images = s1_images[:, :2, ...]
        s2_images, months = self.aggregate_months(
            Modality.SENTINEL2_L2A.name, s2_images, dates
        )
        s1_images, _ = self.aggregate_months(Modality.SENTINEL1.name, s1_images, dates)

        targets = torch.tensor(targets, dtype=torch.long)
        # PASTIS has 19 classes, the last one is void label, convert it to -1 to ignore
        # https://github.com/VSainteuf/pastis-benchmark
        targets[targets == 19] = -1

        def split_images(images: torch.Tensor) -> torch.Tensor:
            """Split images into 4 quadrants."""
            return torch.stack(
                [
                    images[..., :64, :64],
                    images[..., 64:, :64],
                    images[..., :64, 64:],
                    images[..., 64:, 64:],
                ]
            )

        return {
            "fold": f"fold_{properties['Fold']}",
            "s2_images": split_images(s2_images),
            "s1_images": split_images(s1_images),
            "months": torch.stack([months] * 4),
            "targets": torch.stack(
                [
                    targets[:64, :64],
                    targets[64:, :64],
                    targets[:64, 64:],
                    targets[64:, 64:],
                ]
            ),
        }

    def process(self) -> None:
        """Process the PASTIS-R dataset."""
        with open(self.data_dir / "metadata.geojson") as f:
            meta_data = json.load(f)

        all_data: dict[str, dict[str, list[torch.Tensor]]] = {
            f"fold_{i}": {"s2_images": [], "s1_images": [], "months": [], "targets": []}
            for i in range(1, 6)
        }

        # Count how many samples don't have 12 months of data
        doesnt_have_twelve = 0

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.process_sample, meta_data["features"]))

        for res in results:
            if res:
                fold = res["fold"]
                if res["s2_images"].shape[1] == 12 and res["s1_images"].shape[1] == 12:
                    for key in ["s2_images", "s1_images", "months", "targets"]:
                        all_data[fold][key].append(res[key])
                else:
                    doesnt_have_twelve += 1

        print(f"doesnt_have_twelve: {doesnt_have_twelve}")  # We got 0!

        for fold_idx in range(1, 6):
            fold_key = f"fold_{fold_idx}"
            for key in ["s2_images", "s1_images", "months", "targets"]:
                all_data[fold_key][key] = torch.cat(all_data[fold_key][key], dim=0)

        all_data_splits = {
            "train": {
                key: torch.cat(
                    [
                        all_data["fold_1"][key],
                        all_data["fold_2"][key],
                        all_data["fold_3"][key],
                    ],
                    dim=0,
                )
                for key in ["s2_images", "s1_images", "months", "targets"]
            },
            "valid": {
                key: all_data["fold_4"][key]
                for key in ["s2_images", "s1_images", "months", "targets"]
            },
            "test": {
                key: all_data["fold_5"][key]
                for key in ["s2_images", "s1_images", "months", "targets"]
            },
        }

        for split, data in all_data_splits.items():
            # Save each S1/S2 separately
            split_dir = self.output_dir / f"pastis_r_{split}"
            os.makedirs(split_dir, exist_ok=True)

            torch.save(data["months"], split_dir / "months.pt")
            torch.save(data["targets"], split_dir / "targets.pt")
            print(data["s2_images"].shape)
            print(data["s1_images"].shape)

            s2_dir = split_dir / "s2_images"
            s1_dir = split_dir / "s1_images"
            os.makedirs(s2_dir, exist_ok=True)
            os.makedirs(s1_dir, exist_ok=True)

            for idx in range(data["s2_images"].shape[0]):
                print(data["s2_images"][idx, :, :, :, :].shape)
                torch.save(data["s2_images"][idx].clone(), s2_dir / f"{idx}.pt")

            for idx in range(data["s1_images"].shape[0]):
                print(data["s1_images"][idx, :, :, :, :].shape)
                torch.save(data["s1_images"][idx].clone(), s1_dir / f"{idx}.pt")

        for split in ["train", "valid", "test"]:
            for key in ["s2_images", "s1_images", "months", "targets"]:
                print(f"{split} {key}: {all_data_splits[split][key].shape}")

        for channel_idx in range(13):
            channel_data = all_data_splits["train"]["s2_images"][
                :, :, channel_idx, :, :
            ]
            print(
                f"S2 Channel {channel_idx}: Mean {channel_data.mean().item():.4f}, Std {channel_data.std().item():.4f}"
            )

        for channel_idx in range(2):
            channel_data = all_data_splits["train"]["s1_images"][
                :, :, channel_idx, :, :
            ]
            print(
                f"S1 Channel {channel_idx}: Mean {channel_data.mean().item():.4f}, Std {channel_data.std().item():.4f}"
            )


def process_pastis(
    data_dir: str = DATA_DIR,
    output_dir: str = PASTIS_DIR,
) -> None:
    """Process PASTIS-R dataset."""
    processor = PASTISRProcessor(
        data_dir=data_dir,
        output_dir=output_dir,
    )
    processor.process()


class PASTISRDataset(Dataset):
    """PASTIS-R dataset class."""

    allowed_modalities = [Modality.SENTINEL1.name, Modality.SENTINEL2_L2A.name]

    def __init__(
        self,
        path_to_splits: Path = PASTIS_DIR,
        split: str = "train",
        partition: str = "default",
        norm_stats_from_pretrained: bool = True,
        norm_method: str = "norm_no_clip",
        input_modalities: list[str] = [],
    ):
        """Init PASTIS-R dataset.

        Args:
            path_to_splits: Path where .pt objects returned by process_pastis_r have been saved
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

        # Does not support 12 band L2A data
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

        self.s2_images_dir = path_to_splits / f"pastis_r_{split}" / "s2_images"
        self.s1_images_dir = path_to_splits / f"pastis_r_{split}" / "s1_images"
        self.labels = torch.load(path_to_splits / f"pastis_r_{split}" / "targets.pt")
        self.months = torch.load(path_to_splits / f"pastis_r_{split}" / "months.pt")
        if (partition != "default") and (split == "train"):
            # PASTIS and PASTIS-R share the same partitions so we just use PASTIS Partitions
            with open(
                PASTIS_DIR_PARTITION / f"{partition}_partition.json"
            ) as json_file:
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
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[MaskedHeliosSample, torch.Tensor]:
        """Return a single PASTIS data instance."""
        image_idx = self.indices[idx]
        s2_image = torch.load(self.s2_images_dir / f"{image_idx}.pt")
        s2_image = einops.rearrange(s2_image, "t c h w -> h w t c")  # (64, 64, 12, 13)

        s1_image = torch.load(self.s1_images_dir / f"{image_idx}.pt")
        s1_image = einops.rearrange(s1_image, "t c h w -> h w t c")  # (64, 64, 12, 2)

        labels = self.labels[idx]  # (64, 64)
        months = self.months[idx]  # (12)

        # If using norm stats from pretrained we should normalize before we rearrange
        if not self.norm_stats_from_pretrained:
            s2_image = normalize_bands(
                s2_image.numpy(), self.s2_means, self.s2_stds, self.norm_method
            )
            s1_image = normalize_bands(
                s1_image.numpy(), self.s1_means, self.s1_stds, self.norm_method
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
            sample_dict[Modality.SENTINEL1.name] = torch.tensor(s1_image).float()
        if Modality.SENTINEL2_L2A.name in self.input_modalities:
            sample_dict[Modality.SENTINEL2_L2A.name] = torch.tensor(s2_image).float()

        if not sample_dict:
            raise ValueError(f"No valid modalities found in: {self.input_modalities}")

        masked_sample = MaskedHeliosSample.from_heliossample(
            HeliosSample(**sample_dict)
        )

        return masked_sample, labels.long()
