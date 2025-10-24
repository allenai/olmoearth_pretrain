"""SICKLE processor."""

import argparse
import glob
import logging
import os
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from typing import Any

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import rasterio
import torch
from upath import UPath

from olmoearth_pretrain.evals.datasets.sickle_dataset import (
    L8_BANDS,
    L8_BANDS_IMPUTED,
    S1_BANDS,
    S2_BANDS,
)

logger = logging.getLogger(__name__)

CSV_PATH = UPath(
    "/weka/dfive-default/helios/evaluation/SICKLE/sickle_dataset_tabular.csv"
)
DATA_DIR = UPath("/weka/dfive-default/helios/evaluation/SICKLE")

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
    csv_path: str,
    data_dir: str,
    output_dir: str,
) -> None:
    """Process SICKLE dataset."""
    processor = SICKLEProcessor(
        csv_path=csv_path,
        data_dir=data_dir,
        output_dir=output_dir,
    )
    processor.process()


def main() -> None:
    """Main function to process SICKLE dataset."""
    parser = argparse.ArgumentParser(
        description="Process SICKLE dataset into PyTorch objects."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=str(CSV_PATH),
        help="Path to the SICKLE tabular CSV file.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DATA_DIR),
        help="Path to the raw SICKLE image data directory.",
    )
    parser.add_argument(
        "--output_dir", type=str, help="Output directory to save processed dataset."
    )
    args = parser.parse_args()

    process_sickle(
        csv_path=args.csv_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
