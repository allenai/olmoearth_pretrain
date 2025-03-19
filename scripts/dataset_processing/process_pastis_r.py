"""Process PASTIS-R dataset into PyTorch objects.

This script processes the PASTIS-R dataset into PyTorch objects.
It loads the S2 and S1 images, and the annotations, and splits them into 4 images.
It also imputes the missing bands in the S2 images.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
from upath import UPath

# Paths
PASTIS_PATH = UPath("/weka/dfive-default/helios/evaluation/PASTIS-R")
OUTPUT_DIR = UPath("/weka/dfive-default/presto_eval_sets/pastis_r")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# All months
ALL_MONTHS = [
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


def impute(img: torch.Tensor) -> torch.Tensor:
    """Impute missing bands in Sentinel-2 images."""
    # img is shape (10, 128, 128)
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
    )  # (13, 128, 128)
    return img


def aggregate_months(
    images: torch.Tensor, dates: dict[str, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate images into monthly averages."""
    months_dict: dict[str, list[torch.Tensor]]
    months_dict = {m: [] for m in ALL_MONTHS}

    for idx, date in dates.items():
        month = str(date)[:6]
        img = torch.tensor(images[int(idx)], dtype=torch.float32)
        # Impute for Sentinel-2
        if images.shape[1] == 10:
            img = impute(img)
        months_dict[month].append(img)

    img_list: list[torch.Tensor] = []
    month_list: list[int] = []
    for month in ALL_MONTHS:
        if months_dict[month]:
            stacked_imgs = torch.stack(months_dict[month])  # (N, channels, 128, 128)
            month_avg = stacked_imgs.mean(dim=0)  # (channels, 128, 128)
            if len(img_list) < 12:
                img_list.append(month_avg)
                month_list.append(int(month[4:]))

    return torch.stack(img_list), torch.tensor(month_list, dtype=torch.long)


def process_sample(sample: dict[str, Any]) -> dict[str, torch.Tensor] | None:
    """Process a single sample from metadata."""
    properties = sample["properties"]
    dates = properties["dates-S2"]
    patch_id = properties["ID_PATCH"]

    # Load S2 and S1 images, only load ascending orbit for S1
    s2_path = PASTIS_PATH / f"DATA_S2/S2_{patch_id}.npy"
    s1_path = PASTIS_PATH / f"DATA_S1A/S1A_{patch_id}.npy"
    target_path = PASTIS_PATH / f"ANNOTATIONS/TARGET_{patch_id}.npy"

    try:
        s2_images = np.load(s2_path)
        s1_images = np.load(s1_path)
        targets = np.load(target_path)[0].astype("int64")
    except FileNotFoundError:
        return None  # Skip missing files

    assert len(dates) == s2_images.shape[0], "Mismatch between S2 dates and images"

    # Keep only VV and VH bands from S1, the third band is VV/VH ratio
    s1_images = s1_images[:, :2, ...]

    s2_images, months = aggregate_months(s2_images, dates)
    s1_images, _ = aggregate_months(s1_images, dates)

    targets = torch.tensor(targets, dtype=torch.long)
    targets[targets == 19] = -1  # Convert void class to -1

    # Split into 4 quadrants
    def split_images(images: torch.Tensor) -> torch.Tensor:
        """Split images into 4 quadrants."""
        return torch.stack(
            [
                images[..., :64, :64],
                images[..., 64:, :64],
                images[..., :64, 64:],
                images[..., 64:, 64:],
            ]
        )  # (4, 12, channels, 64, 64)

    return {
        "fold": f'fold_{properties["Fold"]}',
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


# Load metadata
with open(PASTIS_PATH / "metadata.geojson") as f:
    meta_data = json.load(f)

# Process samples in parallel
all_data: dict[str, dict[str, list[torch.Tensor]]] = {}
for i in range(1, 6):
    all_data[f"fold_{i}"] = {
        "s2_images": [],
        "s1_images": [],
        "months": [],
        "targets": [],
    }

doesnt_have_twelve = 0

with ThreadPoolExecutor() as executor:
    results = list(executor.map(process_sample, meta_data["features"]))

for res in results:
    if res:
        fold = res["fold"]
        if res["s2_images"].shape[1] == 12 and res["s1_images"].shape[1] == 12:
            all_data[fold]["s2_images"].append(res["s2_images"])
            all_data[fold]["s1_images"].append(res["s1_images"])
            all_data[fold]["months"].append(res["months"])
            all_data[fold]["targets"].append(res["targets"])
        else:
            doesnt_have_twelve += 1

print(f"doesnt_have_twelve: {doesnt_have_twelve}")

# Concatenate tensors
for fold_idx in range(1, 6):
    fold_key = f"fold_{fold_idx}"
    for key in ["s2_images", "s1_images", "months", "targets"]:
        all_data[fold_key][key] = torch.cat(all_data[fold_key][key], dim=0)

# Split into train/valid/test sets
all_data_splits = {
    "train": {
        key: torch.cat(
            [all_data["fold_1"][key], all_data["fold_2"][key], all_data["fold_3"][key]],
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

# Save data
for split, data in all_data_splits.items():
    torch.save(data, OUTPUT_DIR / f"pastis_r_{split}.pt")

# Print stats
for split in ["train", "valid", "test"]:
    for key in ["s2_images", "s1_images", "months", "targets"]:
        print(f"{split} {key}: {all_data_splits[split][key].shape}")

# Compute mean and std for normalization
for channel_idx in range(13):
    channel_data = all_data_splits["train"]["s2_images"][:, :, channel_idx, :, :]
    print(
        f"S2 Channel {channel_idx}: Mean {channel_data.mean().item():.4f}, Std {channel_data.std().item():.4f}"
    )

for channel_idx in range(2):
    channel_data = all_data_splits["train"]["s1_images"][:, :, channel_idx, :, :]
    print(
        f"S1 Channel {channel_idx}: Mean {channel_data.mean().item():.4f}, Std {channel_data.std().item():.4f}"
    )
