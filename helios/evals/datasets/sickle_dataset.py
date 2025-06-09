"""SICKLE dataset class."""

import glob
import logging
import os
import random
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import einops
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.multiprocessing
from torch.utils.data import Dataset
from upath import UPath

from helios.data.constants import Modality
from helios.data.dataset import HeliosSample
from helios.evals.datasets.constants import (
    EVAL_L8_BAND_NAMES,
    EVAL_S1_BAND_NAMES,
    EVAL_S2_L2A_BAND_NAMES,
    EVAL_TO_HELIOS_L8_BANDS,
    EVAL_TO_HELIOS_S1_BANDS,
    EVAL_TO_HELIOS_S2_L2A_BANDS,
)
from helios.evals.datasets.normalize import normalize_bands
from helios.train.masking import MaskedHeliosSample

logger = logging.getLogger(__name__)

CSV_PATH = UPath(
    "/weka/dfive-default/helios/evaluation/SICKLE/sickle_dataset_tabular.csv"
)
DATA_DIR = UPath("/weka/dfive-default/helios/evaluation/SICKLE")
SICKLE_DIR = UPath("/weka/dfive-default/presto_eval_sets/sickle")

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


class SICKLEProcessor:
    """Process SICKLE dataset into PyTorch objects."""

    def __init__(self, csv_path: str, data_dir: str, output_dir: str):
        """Initialize SICKLE processor.

        Args:
            csv_path: path to the CSV file.
            data_dir: path to the data directory.
            output_dir: path to the output directory.
        """
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # We resize images and masks to 32*32
        # Here we avoid using Bilinear as it will averaging the neighbors, which is not what we want for masks
        self.resize = A.Resize(height=32, width=32, interpolation=cv2.INTER_NEAREST)

    def impute_l8_bands(self, img: torch.Tensor) -> torch.Tensor:
        """Impute missing bands in L8 images."""
        img = torch.stack(
            [
                img[0, ...],  # fill B1 with B1
                img[1, ...],  # fill B2 with B2
                img[2, ...],  # fill B3 with B3
                img[3, ...],  # fill B4 with B4
                img[4, ...],  # fill B5 with B5
                img[5, ...],  # fill B6 with B6
                img[6, ...],  # fill B7 with B7
                img[6, ...],  # fill B8 with B7, IMPUTED!
                img[6, ...],  # fill B9 with B7, IMPUTED!
                img[7, ...],  # fill B10 with B10
                img[7, ...],  # fill B11 with B10, IMPUTED!
            ]
        )
        return img

    def _read_mask(self, mask_path: str) -> np.ndarray:
        """Read a mask from a path."""
        with rasterio.open(mask_path) as fp:
            mask = fp.read()

        # There're multiple layers in the mask, we only use the first two layers
        # Which is the plot_mask and crop_type_mask
        mask = mask[:2, ...]
        mask[0][mask[0] == 0] = -1
        # Convert crop type mask into binary (Paddy: 0, Non-Paddy: 1)
        # Reference: https://github.com/Depanshu-Sani/SICKLE/blob/main/utils/dataset.py
        mask[1] -= 1
        mask[1][mask[1] > 1] = 1
        # Convert to -1 to ignore
        mask[mask < 0] = -1
        return mask

    def _get_image_date(self, image_path: str) -> str:
        """Get the date of the image.

        Args:
            image_path: Path to the image.

        Returns:
            year-month string.
        """
        if "S2" in image_path:
            # For S2 2018 data?
            if os.path.basename(image_path)[0] == "T":
                image_date = os.path.basename(image_path).split("_")[1][:8]
            else:
                image_date = os.path.basename(image_path).split("_")[0][:8]
        elif "S1" in image_path:
            image_date = os.path.basename(image_path).split("_")[4][:8]
        elif "L8" in image_path:
            image_date = os.path.basename(image_path).split("_")[2][:8]

        return f"{image_date[:4]}-{image_date[4:6]}"  # year-month string

    def _read_image(self, image_path: str) -> torch.Tensor:
        """Read an image from a path."""
        data_file = np.load(image_path)
        if "S2" in image_path:
            bands = S2_BANDS
        elif "S1" in image_path:
            bands = S1_BANDS
        elif "L8" in image_path:
            bands = L8_BANDS
        try:
            all_channels = [
                self.resize(image=data_file[band])["image"] for band in bands
            ]
        except Exception:
            # Not quite sure why we need this, get from the SICKLE code repo
            all_channels = [
                self.resize(image=data_file[band])["image"] for band in bands[:-1]
            ]
            all_channels += [np.zeros((32, 32), dtype=np.float32)]

        data = torch.tensor(np.stack(all_channels, axis=0))  # (C, H, W)
        # Imput missing bands in L8
        if "L8" in image_path:
            data = self.impute_l8_bands(data)

        return data

    def _aggregate_months(
        self, images: list[str], start_date: date, end_date: date
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate images by month.

        Args:
            images: List of image paths.
            start_date: Start date.
            end_date: End date.

        Returns:
            Tuple of aggregated images and dates.
        """
        images = sorted(images)
        # Get all the unique year-month between start_date and end_date
        all_months: list[str] = []
        current_date = start_date
        while current_date <= end_date:
            all_months.append(current_date.strftime("%Y-%m"))
            current_date = current_date.replace(day=1) + timedelta(days=32)
            current_date = current_date.replace(day=1)
        all_months = list(set(all_months))
        all_months.sort()

        months_dict = dict[str, list[torch.Tensor]]()
        for month in all_months:
            months_dict[month] = []

        for image_path in images:
            year_month = self._get_image_date(image_path)
            img = self._read_image(image_path)
            months_dict[year_month].append(img)

        img_list: list[torch.Tensor] = []
        month_list: list[int] = []
        for month in all_months:
            if months_dict[month]:
                stacked_imgs = torch.stack(months_dict[month])
                month_avg = stacked_imgs.mean(dim=0)
                img_list.append(month_avg)
                month_list.append(int(month.replace("-", "")))

        return torch.stack(img_list), torch.tensor(month_list, dtype=torch.long)

    def process_sample(self, sample: dict[str, Any]) -> dict[str, torch.Tensor] | None:
        """Process a single sample from the SICKLE dataset.

        Args:
            sample: A dictionary containing the sample information.

        Returns:
            A dictionary containing the processed sample.
        """
        uid = sample["uid"]
        standard_season = sample["standard_season"]
        year = sample["year"]
        split = sample["split"]

        start_date = date(int(year), MONTH_TO_INT[standard_season.split("-")[0]], 1)
        end_date = date(int(year), MONTH_TO_INT[standard_season.split("-")[1]], 1)
        # Deal with the case where it across the year boundary
        if end_date <= start_date:
            end_date = end_date.replace(year=end_date.year + 1)

        # Get all the S2, S1, and L8 images for the sample
        s2_path = os.path.join(self.data_dir, f"images/S2/npy/{uid}")
        s1_path = os.path.join(self.data_dir, f"images/S1/npy/{uid}")
        l8_path = os.path.join(self.data_dir, f"images/L8/npy/{uid}")
        s2_image_paths = glob.glob(os.path.join(s2_path, "*.npz"))
        s1_image_paths = glob.glob(os.path.join(s1_path, "*.npz"))
        l8_image_paths = glob.glob(os.path.join(l8_path, "*.npz"))

        if (
            len(s2_image_paths) == 0
            or len(s1_image_paths) == 0
            or len(l8_image_paths) == 0
        ):
            return None

        # Aggregate images by month
        s2_images, s2_months = self._aggregate_months(
            s2_image_paths, start_date, end_date
        )
        s1_images, s1_months = self._aggregate_months(
            s1_image_paths, start_date, end_date
        )
        l8_images, l8_months = self._aggregate_months(
            l8_image_paths, start_date, end_date
        )

        # For now, we only use the 10m mask, there're also 3m, 30m masks
        mask_path = os.path.join(self.data_dir, f"masks/10m/{uid}.tif")
        mask = self._read_mask(mask_path)
        plot_mask, crop_type_mask = mask[0], mask[1]

        # Remove plots that are not in this split, only predict for plots in this split
        unmatched_plot_ids = set(np.unique(plot_mask)) - set(self.split_plot_ids[split])
        for unmatched_plot_id in unmatched_plot_ids:
            crop_type_mask[plot_mask == unmatched_plot_id] = -1

        # Resize the mask to 32*32
        crop_type_mask = self.resize(image=crop_type_mask)["image"]
        targets = torch.tensor(crop_type_mask, dtype=torch.long)
        # Here we require at least 5 months of data for each modality
        # Also make sure the first month is the same for all modalities
        print(f"Finish processing {uid}")
        if (
            min(len(s2_months), len(s1_months), len(l8_months)) >= MIN_MONTHS
            and s2_months[0] == s1_months[0] == l8_months[0]
        ):
            # Given the temporal resolution, S2 has the most data
            months = s2_months
            return {
                "split": split,
                "s2_images": s2_images,
                "s1_images": s1_images,
                "l8_images": l8_images,
                "months": months,
                "targets": targets,
            }
        else:
            warnings.warn(
                f"Number of images for S2, S1, and L8 are not the sufficient for {uid}"
            )
            return None

    def process(self) -> None:
        """Process the SICKLE dataset."""
        all_samples = []
        df = pd.read_csv(self.csv_path)
        # Remove test split as there're not ground truth labels for it
        df = df[df["SPLIT"] != "test"]
        for _, row in df.iterrows():
            sample = {
                "uid": row["UNIQUE_ID"],
                "standard_season": row["STANDARD_SEASON"],
                "year": row["YEAR"],
                "split": row["SPLIT"],
            }
            all_samples.append(sample)

        # Get the unique plot_ids per split
        self.split_plot_ids = defaultdict(list)
        for split in ["train", "val"]:
            plot_ids = df[df["SPLIT"] == split]["PLOT_ID"].unique()
            self.split_plot_ids[split] = plot_ids

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.process_sample, all_samples))

        doesnt_have_five = 0

        all_data: dict[str, dict[str, list[torch.Tensor]]] = {
            split: {
                "s2_images": [],
                "s1_images": [],
                "l8_images": [],
                "months": [],
                "targets": [],
            }
            for split in ["train", "val"]
        }
        for res in results:
            if res:
                # Cut all samples to 5 months
                res["s2_images"] = res["s2_images"][:MIN_MONTHS, ...]
                res["s1_images"] = res["s1_images"][:MIN_MONTHS, ...]
                res["l8_images"] = res["l8_images"][:MIN_MONTHS, ...]
                res["months"] = res["months"][:MIN_MONTHS]

                for key in [
                    "s2_images",
                    "s1_images",
                    "l8_images",
                    "months",
                    "targets",
                ]:
                    all_data[res["split"]][key].append(res[key])
            else:
                doesnt_have_five += 1

        print(f"doesnt_have_five: {doesnt_have_five}")

        all_data_stacked = {
            split: {
                key: torch.stack(all_data[split][key], dim=0)
                for key in ["s2_images", "s1_images", "l8_images", "months", "targets"]
            }
            for split in ["train", "val"]
        }

        all_data_splits = {
            "train": {
                key: all_data_stacked["train"][key]
                for key in ["s2_images", "s1_images", "l8_images", "months", "targets"]
            },
            "val": {
                key: all_data_stacked["val"][key]
                for key in ["s2_images", "s1_images", "l8_images", "months", "targets"]
            },
        }

        for split, data in all_data_splits.items():
            split_dir = os.path.join(self.output_dir, f"sickle_{split}")
            os.makedirs(split_dir, exist_ok=True)

            torch.save(data["months"], os.path.join(split_dir, "months.pt"))
            torch.save(data["targets"], os.path.join(split_dir, "targets.pt"))

            s2_dir = os.path.join(split_dir, "s2_images")
            s1_dir = os.path.join(split_dir, "s1_images")
            l8_dir = os.path.join(split_dir, "l8_images")
            os.makedirs(s2_dir, exist_ok=True)
            os.makedirs(s1_dir, exist_ok=True)
            os.makedirs(l8_dir, exist_ok=True)

            for idx in range(data["s2_images"].shape[0]):
                print(data["s2_images"][idx, :, :, :, :].shape)
                torch.save(
                    data["s2_images"][idx].clone(), os.path.join(s2_dir, f"{idx}.pt")
                )

            for idx in range(data["s1_images"].shape[0]):
                print(data["s1_images"][idx, :, :, :, :].shape)
                torch.save(
                    data["s1_images"][idx].clone(), os.path.join(s1_dir, f"{idx}.pt")
                )

            for idx in range(data["l8_images"].shape[0]):
                print(data["l8_images"][idx, :, :, :, :].shape)
                torch.save(
                    data["l8_images"][idx].clone(), os.path.join(l8_dir, f"{idx}.pt")
                )

        for split in ["train", "val"]:
            for key in ["s2_images", "s1_images", "l8_images", "months", "targets"]:
                print(f"{split} {key}: {all_data_splits[split][key].shape}")

        for channel_idx, band_name in enumerate(S2_BANDS):
            channel_data = all_data_splits["train"]["s2_images"][
                :, :, channel_idx, :, :
            ]
            print(
                f"S2 {band_name}: Mean {channel_data.mean().item():.4f}, Std {channel_data.std().item():.4f}"
            )

        for channel_idx, band_name in enumerate(S1_BANDS):
            channel_data = all_data_splits["train"]["s1_images"][
                :, :, channel_idx, :, :
            ]
            print(
                f"S1 {band_name}: Mean {channel_data.mean().item():.4f}, Std {channel_data.std().item():.4f}"
            )

        for channel_idx, band_name in enumerate(L8_BANDS_IMPUTED):
            channel_data = all_data_splits["train"]["l8_images"][
                :, :, channel_idx, :, :
            ]
            print(
                f"L8 {band_name}: Mean {channel_data.mean().item():.4f}, Std {channel_data.std().item():.4f}"
            )


def process_sickle(
    csv_path: str = CSV_PATH,
    data_dir: str = DATA_DIR,
    output_dir: str = SICKLE_DIR,
) -> None:
    """Process SICKLE dataset."""
    processor = SICKLEProcessor(
        csv_path=csv_path,
        data_dir=data_dir,
        output_dir=output_dir,
    )
    processor.process()


class SICKLEDataset(Dataset):
    """SICKLE dataset class."""

    allowed_modalities = [
        Modality.LANDSAT.name,
        Modality.SENTINEL1.name,
        Modality.SENTINEL2_L2A.name,
    ]

    def __init__(
        self,
        path_to_splits: Path = SICKLE_DIR,
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

        self.s2_means, self.s2_stds = self._get_norm_stats(
            S2_BAND_STATS, EVAL_S2_L2A_BAND_NAMES
        )
        self.s1_means, self.s1_stds = self._get_norm_stats(
            S1_BAND_STATS, EVAL_S1_BAND_NAMES
        )
        self.l8_means, self.l8_stds = self._get_norm_stats(
            L8_BAND_STATS, EVAL_L8_BAND_NAMES
        )

        self.norm_method = norm_method

        self.norm_stats_from_pretrained = norm_stats_from_pretrained
        # If normalize with pretrained stats, we initialize the normalizer here
        if self.norm_stats_from_pretrained:
            from helios.data.normalize import Normalizer, Strategy

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
                l8_image.numpy(), self.l8_means, self.l8_stds, self.norm_method
            )
            s2_image = normalize_bands(
                s2_image.numpy(), self.s2_means, self.s2_stds, self.norm_method
            )
            s1_image = normalize_bands(
                s1_image.numpy(), self.s1_means, self.s1_stds, self.norm_method
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

        masked_sample = MaskedHeliosSample.from_heliossample(
            HeliosSample(**sample_dict)
        )

        return masked_sample, labels.long()
