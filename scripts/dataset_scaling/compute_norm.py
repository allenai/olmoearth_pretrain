"""Compute the normalization stats for a given dataset."""

import argparse
import json
import logging
import random
from typing import Any

from tqdm import tqdm
from upath import UPath

from helios.data.constants import IMAGE_TILE_SIZE, Modality
from helios.data.dataset import GetItemArgs, HeliosDataset, HeliosDatasetConfig
from helios.data.utils import update_streaming_stats

logger = logging.getLogger(__name__)


def compute_normalization_values(
    dataset: HeliosDataset,
    estimate_from: int | None = None,
) -> dict[str, Any]:
    """Compute the normalization values for the dataset in a streaming manner.

    Args:
        dataset: The dataset to compute the normalization values for.
        estimate_from: The number of samples to estimate the normalization values from.

    Returns:
        dict: A dictionary containing the normalization values for the dataset.
    """
    dataset_len = len(dataset)
    if estimate_from is not None:
        indices_to_sample = random.sample(list(range(dataset_len)), k=estimate_from)
    else:
        indices_to_sample = list(range(dataset_len))

    norm_dict: dict[str, Any] = {}

    for i in tqdm(indices_to_sample):
        get_item_args = GetItemArgs(idx=i, patch_size=1, sampled_hw_p=IMAGE_TILE_SIZE)
        _, sample = dataset[get_item_args]
        for modality in sample.modalities:
            # Shall we compute the norm stats for worldcover?
            if modality == "timestamps" or modality == "latlon":
                continue
            modality_data = sample.as_dict(ignore_nones=True)[modality]
            modality_spec = Modality.get(modality)
            modality_bands = modality_spec.band_order
            if modality_data is None:
                continue
            if modality not in norm_dict:
                norm_dict[modality] = {}
                for band in modality_bands:
                    norm_dict[modality][band] = {
                        "mean": 0.0,
                        "var": 0.0,
                        "std": 0.0,
                        "count": 0,
                    }
            # Compute the normalization stats for the modality
            for idx, band in enumerate(modality_bands):
                modality_band_data = modality_data[:, :, :, idx]  # (H, W, T, C)
                current_stats = norm_dict[modality][band]
                new_count, new_mean, new_var = update_streaming_stats(
                    current_stats["count"],
                    current_stats["mean"],
                    current_stats["var"],
                    modality_band_data,
                )
                # Update the normalization stats
                norm_dict[modality][band]["count"] = new_count
                norm_dict[modality][band]["mean"] = new_mean
                norm_dict[modality][band]["var"] = new_var

    # Compute the standard deviation
    for modality in norm_dict:
        for band in norm_dict[modality]:
            norm_dict[modality][band]["std"] = (
                norm_dict[modality][band]["var"] / norm_dict[modality][band]["count"]
            ) ** 0.5

    norm_dict["total_n"] = dataset_len
    norm_dict["sampled_n"] = len(indices_to_sample)
    norm_dict["tile_path"] = dataset.tile_path

    return norm_dict


args = argparse.ArgumentParser()

args.add_argument("--tile_path", type=str, required=True)
args.add_argument("--supported_modalities", type=str, required=True)
args.add_argument("--estimate_from", type=int, required=False, default=None)
args.add_argument("--output_path", type=str, required=True)
args_dict = args.parse_args().__dict__  # type: ignore

logger.info(
    f"Computing normalization stats for {args_dict['tile_path']} with modalities {args_dict['supported_modalities']}"
)


def parse_supported_modalities(supported_modalities: str) -> list[str]:
    """Parse the supported modalities from a string."""
    return supported_modalities.split(",")


# Use the config to build the dataset
dataset_config = HeliosDatasetConfig(
    tile_path=UPath(args_dict["tile_path"]),
    supported_modality_names=parse_supported_modalities(
        args_dict["supported_modalities"]
    ),
    normalize=False,
)
dataset = dataset_config.build()

norm_dict = compute_normalization_values(
    dataset=dataset,
    estimate_from=args_dict["estimate_from"],
)
logger.info(f"Normalization stats: {norm_dict}")

with open(args_dict["output_path"], "w") as f:
    json.dump(norm_dict, f)


# Example usage:
# 20250304 run:
# python3 compute_norm.py --tile_path "/weka/dfive-default/helios/dataset/presto" --supported_modalities "sentinel2_l2a,sentinel1,worldcover" --output_path "/weka/dfive-default/yawenz/helios/data/norm_configs/computed_20250304.json"
