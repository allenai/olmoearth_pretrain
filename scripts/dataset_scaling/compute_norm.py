"""Compute the normalization stats for a given dataset."""

import argparse
import json
import logging

from upath import UPath

from helios.data.dataset import HeliosDatasetConfig

logger = logging.getLogger(__name__)

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

norm_dict = dataset.compute_normalization_values(
    estimate_from=args_dict["estimate_from"]
)
logger.info(f"Normalization stats: {norm_dict}")

with open(args_dict["output_path"], "w") as f:
    json.dump(norm_dict, f)


# Example usage:
# 20250304 run:
# python3 compute_norm.py --tile_path "/weka/dfive-default/helios/dataset/presto" --supported_modalities "sentinel2_l2a,sentinel1,worldcover" --output_path "/weka/dfive-default/yawenz/helios/data/norm_configs/computed_20250304.json"
