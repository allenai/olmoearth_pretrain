"""Script to process the pastis dataset."""

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
from upath import UPath

from olmoearth_pretrain.data.constants import Modality

logger = logging.getLogger(__name__)


class PASTISRProcessor:
    """Process PASTIS-R dataset into PyTorch objects.

    This class processes the PASTIS-R dataset into PyTorch objects.
    It loads the S2 and S1 images, and the annotations, and splits them into 4 images.
    It also imputes the missing bands in the S2 images.
    """

    def __init__(self, data_dir: str, output_dir: str, resize_to_64: bool = True):
        """Initialize PASTIS-R processor.

        Args:
            data_dir: Path to PASTIS-R dataset
            output_dir: Path to output directory
            resize_to_64: Whether or not to resize the pastis dataset into 64x64 tiles
                from 128x128 tiles
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
        self.resize_to_64 = resize_to_64

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
                if img.shape[0] == 10:
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

        if self.resize_to_64:
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
        else:
            return {
                "fold": f"fold_{properties['Fold']}",
                "s2_images": s2_images.unsqueeze(0),
                "s1_images": s1_images.unsqueeze(0),
                "months": months.unsqueeze(0),
                "targets": targets.unsqueeze(0),
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


def process_pastis(data_dir: str, output_dir: str) -> None:
    """Process PASTIS-R dataset."""
    processor = PASTISRProcessor(
        data_dir=data_dir,
        output_dir=output_dir,
    )
    processor.process()


def process_pastis_orig_size(
    data_dir: str,
    output_dir: str,
) -> None:
    """Process PASTIS-R dataset."""
    processor = PASTISRProcessor(
        data_dir=data_dir, output_dir=output_dir, resize_to_64=False
    )
    processor.process()


def main() -> None:
    """Main function to process PASTIS-R dataset."""
    parser = argparse.ArgumentParser(
        description="Process PASTIS-R dataset into PyTorch objects."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to the raw PASTIS-R dataset directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory to save processed dataset.",
    )
    parser.add_argument(
        "--orig_size",
        action="store_true",
        help="If set, do not resize to 64x64 (use original size).",
    )
    args = parser.parse_args()

    if args.orig_size:
        process_pastis_orig_size(data_dir=args.data_dir, output_dir=args.output_dir)
    else:
        process_pastis(data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
