"""Compute the normalization stats for a given dataset.

Example usage:
    python3 olmoearth_pretrain/internal/compute_h5_norm.py --h5py_dir /path/to/h5pydir  --supported_modalities "era5_10,landsat,naip_10,sentinel1,sentinel2_l2a,srtm,worldcover" --estimate_from 100 --output_path /weka/dfive-default/yawenz/helios/helios/data/norm_configs/computed_20250722.json
"""

import argparse
import json
import logging
import random
from typing import Any

from olmo_core.utils import prepare_cli_environment
from tqdm import tqdm

from olmoearth_pretrain.data.constants import (
    IMAGE_TILE_SIZE,
    MISSING_VALUE,
    Modality,
)
from olmoearth_pretrain.data.dataset import (
    GetItemArgs,
    OlmoEarthDataset,
    OlmoEarthDatasetConfig,
)
from olmoearth_pretrain.data.utils import update_streaming_stats

logger = logging.getLogger(__name__)

# Per-(modality, band) sentinel values that encode "not measured" rather than a
# measurement, and so must be excluded from the stats. glo30 aspect uses -1 for
# flat pixels, which have no compass bearing; including them drags the mean
# toward north and inflates the std.
BAND_STAT_EXCLUDED_VALUES: dict[tuple[str, str], float] = {
    ("glo30", "aspect"): -1.0,
}

# Sampling a subset of a large h5 dir will hit some samples that carry none of
# the requested modalities; those raise instead of loading. Tolerate them, but
# fail loudly if they stop being a small minority.
DEFAULT_MAX_SKIP_FRACTION = 0.05


def compute_normalization_values(
    dataset: OlmoEarthDataset,
    estimate_from: int | None = None,
    max_skip_fraction: float = DEFAULT_MAX_SKIP_FRACTION,
) -> dict[str, Any]:
    """Compute the normalization values for the dataset in a streaming manner.

    Args:
        dataset: The dataset to compute the normalization values for.
        estimate_from: The number of samples to estimate the normalization values from.
        max_skip_fraction: Abort if more than this fraction of the sampled
            indices fail to load. Guards against silently computing stats from a
            handful of samples when something is systemically wrong.

    Returns:
        dict: A dictionary containing the normalization values for the dataset.
    """
    dataset_len = len(dataset)
    if estimate_from is not None:
        indices_to_sample = random.sample(list(range(dataset_len)), k=estimate_from)
    else:
        indices_to_sample = list(range(dataset_len))
    norm_dict: dict[str, Any] = {}
    skipped = 0
    for i in tqdm(indices_to_sample):
        get_item_args = GetItemArgs(idx=i, patch_size=1, sampled_hw_p=IMAGE_TILE_SIZE)
        try:
            _, sample = dataset[get_item_args]
        except Exception:
            # Most commonly: the sample has none of the requested modalities, so
            # there is no spatial modality to read H/W/T from.
            logger.warning(f"Skipping sample {i}; failed to load.", exc_info=True)
            skipped += 1
            continue
        for modality in sample.modalities:
            # Shall we compute the norm stats for worldcover?
            if modality == "latlon":
                continue
            modality_data = sample.as_dict()[modality]
            modality_spec = Modality.get(modality)
            modality_bands = modality_spec.band_order
            if modality_data is None:
                continue
            # To avoid the case where we include missing values in the stats
            if (modality_data == MISSING_VALUE).any():
                logger.info(
                    f"Skipping sample {i} because modality {modality} contains missing values."
                )
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
                modality_band_data = modality_data[..., idx]
                excluded = BAND_STAT_EXCLUDED_VALUES.get((modality, band))
                if excluded is not None:
                    modality_band_data = modality_band_data[
                        modality_band_data != excluded
                    ]
                    if modality_band_data.size == 0:
                        # Every pixel was the sentinel; nothing to accumulate.
                        continue
                current_stats = norm_dict[modality][band]
                new_count, new_mean, new_var = update_streaming_stats(
                    current_stats["count"],
                    current_stats["mean"],
                    current_stats["var"],
                    modality_band_data,
                )
                # Update the normalization stats
                norm_dict[modality][band]["count"] = int(new_count)
                norm_dict[modality][band]["mean"] = float(new_mean)
                norm_dict[modality][band]["var"] = float(new_var)

    used = len(indices_to_sample) - skipped
    if skipped > max_skip_fraction * len(indices_to_sample):
        raise RuntimeError(
            f"{skipped}/{len(indices_to_sample)} sampled indices failed to load, "
            f"above the {max_skip_fraction:.0%} tolerance. Refusing to emit stats "
            "computed from the remainder; check the dataset and modality list."
        )
    if skipped:
        logger.warning(
            f"Skipped {skipped}/{len(indices_to_sample)} unloadable samples."
        )

    # Compute the standard deviation
    for modality in norm_dict:
        for band in norm_dict[modality]:
            count = norm_dict[modality][band]["count"]
            norm_dict[modality][band]["std"] = (
                (norm_dict[modality][band]["var"] / count) ** 0.5 if count else 0.0
            )

    norm_dict["total_n"] = dataset_len
    norm_dict["sampled_n"] = used
    norm_dict["skipped_n"] = skipped
    path = dataset.h5py_dir or dataset.tile_path
    norm_dict["tile_path"] = str(path)

    return norm_dict


if __name__ == "__main__":
    prepare_cli_environment()
    args = argparse.ArgumentParser()
    args.add_argument("--h5py_dir", type=str, required=True)
    args.add_argument("--supported_modalities", type=str, required=True)
    args.add_argument("--estimate_from", type=int, required=False, default=None)
    args.add_argument("--output_path", type=str, required=True)
    args_dict = args.parse_args().__dict__  # type: ignore

    logger.info(
        f"Computing normalization stats with modalities {args_dict['supported_modalities']}"
    )

    def parse_supported_modalities(supported_modalities: str) -> list[str]:
        """Parse the supported modalities from a string."""
        return supported_modalities.split(",")

    # FOr some reason landsat and naip were missi g from every sample
    supported_modalities = parse_supported_modalities(args_dict["supported_modalities"])
    logger.info(f"Supported modalities: {supported_modalities}")
    # Use the config to build the dataset
    dataset_config = OlmoEarthDatasetConfig(
        h5py_dir=args_dict["h5py_dir"],
        training_modalities=supported_modalities,
        normalize=False,
    )
    dataset = dataset_config.build()
    dataset.prepare()
    logger.info(f"Dataset: {dataset.normalize}")

    norm_dict = compute_normalization_values(
        dataset=dataset,
        estimate_from=args_dict["estimate_from"],
    )
    logger.info(f"Normalization stats: {norm_dict}")

    with open(args_dict["output_path"], "w") as f:
        json.dump(norm_dict, f)
