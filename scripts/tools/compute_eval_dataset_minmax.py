#!/usr/bin/env python3
"""Compute min/max values per band for each modality across all non-geobench eval datasets.

This script reads all training split data from non-geobench datasets (those NOT prefixed
with m_ or m-) and computes the min and max value for each band present in each modality.
The results are saved to a JSON config file.

Non-geobench datasets include:
- mados (Sentinel-2 L2A)
- sen1floods11 (Sentinel-1)
- pastis (Sentinel-2 L2A, Sentinel-1)
- pastis128 (Sentinel-2 L2A, Sentinel-1)
- breizhcrops (Sentinel-2 L2A)
"""

import argparse
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm
from upath import UPath

from helios.evals.datasets import (
    BREIZHCROPS_DIR,
    FLOODS_DIR,
    MADOS_DIR,
    PASTIS_DIR,
    PASTIS_DIR_ORIG,
)
from helios.evals.datasets.breizhcrops import LEVEL, BreizhCrops
from helios.evals.datasets.constants import (
    EVAL_S1_BAND_NAMES,
    EVAL_S2_BAND_NAMES,
)


def _process_image_file(file_path, modality_type):
    """Helper function to process a single image file and return min/max values."""
    image = torch.load(file_path).numpy()  # Shape: (T, C, H, W)
    image = rearrange(image, "t c h w -> h w t c")
    # check the last band

    num_bands = image.shape[-1]
    mins = [float("inf")] * num_bands
    maxs = [float("-inf")] * num_bands

    for band_idx in range(num_bands):
        band_data = image[:, :, :, band_idx]
        valid_data = band_data[~np.isnan(band_data) & ~np.isinf(band_data)]
        if len(valid_data) > 0:
            mins[band_idx] = float(valid_data.min())
            maxs[band_idx] = float(valid_data.max())
        else:
            raise ValueError(f"Band {band_idx} has no valid data points")

    return mins, maxs


def compute_mados_stats(
    path_to_splits: UPath, max_samples: int | None = None, num_workers: int = 1
) -> dict:
    """Compute min/max stats for MADOS dataset."""
    print("Computing stats for MADOS (Sentinel-2 L2A)...")

    # Load training data
    torch_obj = torch.load(path_to_splits / "MADOS_train.pt")
    images = torch_obj["images"]  # Shape: (N, 80, 80, 13)

    # Limit samples if requested
    if max_samples is not None:
        images = images[:max_samples]
        print(f"  Limited to {max_samples} samples (smoke test mode)")

    # Compute min/max per band
    images_np = images.numpy()
    min_vals = np.nanmin(images_np, axis=(0, 1, 2))  # Shape: (13,)
    max_vals = np.nanmax(images_np, axis=(0, 1, 2))  # Shape: (13,)

    stats = {
        "sentinel2_l2a": {
            band_name: {"min": float(min_vals[i]), "max": float(max_vals[i])}
            for i, band_name in enumerate(EVAL_S2_BAND_NAMES)
        }
    }

    print(f"  Processed {images.shape[0]} samples")
    return stats


def compute_sen1floods11_stats(
    path_to_splits: UPath, max_samples: int | None = None, num_workers: int = 1
) -> dict:
    """Compute min/max stats for Sen1Floods11 dataset."""
    print("Computing stats for Sen1Floods11 (Sentinel-1)...")

    # Load training data
    torch_obj = torch.load(path_to_splits / "flood_train_data.pt")
    s1 = torch_obj["s1"]  # Shape: (N, 2, 64, 64)

    # Limit samples if requested
    if max_samples is not None:
        s1 = s1[:max_samples]
        print(f"  Limited to {max_samples} samples (smoke test mode)")

    # Rearrange to (N, 64, 64, 2)

    s1 = rearrange(s1, "n c h w -> n h w c")

    # Remove NaN values
    s1_np = s1.numpy()
    min_vals = []
    max_vals = []

    for band_idx in range(2):
        band_data = s1_np[:, :, :, band_idx]
        # Filter out NaN and inf values
        valid_data = band_data[~np.isnan(band_data) & ~np.isinf(band_data)]
        min_vals.append(float(valid_data.min()))
        max_vals.append(float(valid_data.max()))

    stats = {
        "sentinel1": {
            band_name: {"min": min_vals[i], "max": max_vals[i]}
            for i, band_name in enumerate(EVAL_S1_BAND_NAMES)
        }
    }

    print(f"  Processed {s1.shape[0]} samples")
    return stats


def compute_pastis_stats(
    path_to_splits: UPath,
    dataset_name: str,
    max_samples: int | None = None,
    num_workers: int = 1,
) -> dict:
    """Compute min/max stats for PASTIS or PASTIS128 dataset."""
    print(f"Computing stats for {dataset_name} (Sentinel-2 L2A, Sentinel-1)...")

    split_dir = "pastis_r_train"
    s2_dir = path_to_splits / split_dir / "s2_images"
    s1_dir = path_to_splits / split_dir / "s1_images"

    # Get all image files
    s2_files = sorted(s2_dir.glob("*.pt"))
    s1_files = sorted(s1_dir.glob("*.pt"))

    # Limit samples if requested
    if max_samples is not None:
        s2_files = s2_files[:max_samples]
        s1_files = s1_files[:max_samples]
        print(f"  Limited to {max_samples} samples (smoke test mode)")

    # Initialize min/max trackers
    s2_mins = [float("inf")] * len(EVAL_S2_BAND_NAMES)
    s2_maxs = [float("-inf")] * len(EVAL_S2_BAND_NAMES)
    s1_mins = [float("inf")] * len(EVAL_S1_BAND_NAMES)
    s1_maxs = [float("-inf")] * len(EVAL_S1_BAND_NAMES)

    # Process S2 images in parallel
    print(
        f"  Processing {len(s2_files)} Sentinel-2 images with {num_workers} workers..."
    )
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_process_image_file, f, "s2"): f for f in s2_files
            }
            for future in tqdm(as_completed(futures), total=len(s2_files), desc="S2"):
                mins, maxs = future.result()
                for band_idx in range(len(mins)):
                    s2_mins[band_idx] = min(s2_mins[band_idx], mins[band_idx])
                    s2_maxs[band_idx] = max(s2_maxs[band_idx], maxs[band_idx])
    else:
        for s2_file in tqdm(s2_files, desc="S2"):
            mins, maxs = _process_image_file(s2_file, "s2")
            for band_idx in range(len(mins)):
                s2_mins[band_idx] = min(s2_mins[band_idx], mins[band_idx])
                s2_maxs[band_idx] = max(s2_maxs[band_idx], maxs[band_idx])

    # Process S1 images in parallel
    print(
        f"  Processing {len(s1_files)} Sentinel-1 images with {num_workers} workers..."
    )
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_process_image_file, f, "s1"): f for f in s1_files
            }
            for future in tqdm(as_completed(futures), total=len(s1_files), desc="S1"):
                mins, maxs = future.result()
                for band_idx in range(len(mins)):
                    s1_mins[band_idx] = min(s1_mins[band_idx], mins[band_idx])
                    s1_maxs[band_idx] = max(s1_maxs[band_idx], maxs[band_idx])

    stats = {
        "sentinel2_l2a": {
            band_name: {"min": s2_mins[i], "max": s2_maxs[i]}
            for i, band_name in enumerate(EVAL_S2_BAND_NAMES)
        },
        "sentinel1": {
            band_name: {"min": s1_mins[i], "max": s1_maxs[i]}
            for i, band_name in enumerate(EVAL_S1_BAND_NAMES)
        },
    }

    return stats


def compute_breizhcrops_stats(
    path_to_splits: UPath, max_samples: int | None = None, num_workers: int = 1
) -> dict:
    """Compute min/max stats for BreizhCrops dataset."""
    print("Computing stats for BreizhCrops (Sentinel-2 L2A)...")

    # Load training regions (frh01 and frh02)
    regions = ["frh01", "frh02"]

    # Initialize min/max trackers
    mins = [float("inf")] * len(EVAL_S2_BAND_NAMES)
    maxs = [float("-inf")] * len(EVAL_S2_BAND_NAMES)

    total_samples = 0
    for region in regions:
        print(f"  Processing region {region}...")
        ds = BreizhCrops(
            root=path_to_splits,
            region=region,
            preload_ram=False,
            level=LEVEL,
        )

        # Limit number of samples per region if requested
        num_to_process = len(ds)
        if max_samples is not None:
            remaining = max_samples - total_samples
            if remaining <= 0:
                break
            num_to_process = min(len(ds), remaining)
            print(
                f"  Limited to {num_to_process} samples for this region (smoke test mode)"
            )

        for idx in tqdm(range(num_to_process), desc=f"  {region}"):
            x, _, _ = ds[idx]  # x shape: (T, C)

            # First 13 bands are S2 bands
            s2_data = x[:, :13]

            # Update min/max per band
            for band_idx in range(13):
                band_data = s2_data[:, band_idx]
                valid_data = band_data[~np.isnan(band_data) & ~np.isinf(band_data)]
                if len(valid_data) > 0:
                    mins[band_idx] = min(mins[band_idx], float(valid_data.min()))
                    maxs[band_idx] = max(maxs[band_idx], float(valid_data.max()))

            total_samples += 1

    stats = {
        "sentinel2_l2a": {
            band_name: {"min": mins[i], "max": maxs[i]}
            for i, band_name in enumerate(EVAL_S2_BAND_NAMES)
        }
    }

    print(f"  Processed {total_samples} samples")
    return stats


def main():
    """Main function to compute min/max stats for all non-geobench datasets."""
    parser = argparse.ArgumentParser(
        description="Compute min/max values per modality for non-geobench eval datasets"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process per dataset (for smoke testing). Default: process all samples.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers for processing. Default: 1 (no parallelization)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path. Default: helios/evals/datasets/minmax_stats.json",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Computing min/max values per modality for non-geobench eval datasets")
    if args.max_samples is not None:
        print(f"SMOKE TEST MODE: Processing max {args.max_samples} samples per dataset")
    print(f"Using {args.num_workers} worker(s) for parallel processing")
    print("=" * 80)

    all_stats = {}

    # MADOS
    try:
        all_stats["mados"] = compute_mados_stats(
            MADOS_DIR, args.max_samples, args.num_workers
        )
    except Exception as e:
        print(f"ERROR processing MADOS: {e}")
        traceback.print_exc()

    # Sen1Floods11
    try:
        all_stats["sen1floods11"] = compute_sen1floods11_stats(
            FLOODS_DIR, args.max_samples, args.num_workers
        )
    except Exception as e:
        print(f"ERROR processing Sen1Floods11: {e}")
        traceback.print_exc()

    # PASTIS
    try:
        all_stats["pastis"] = compute_pastis_stats(
            PASTIS_DIR, "PASTIS", args.max_samples, args.num_workers
        )
    except Exception as e:
        print(f"ERROR processing PASTIS: {e}")
        traceback.print_exc()

    # PASTIS128
    try:
        all_stats["pastis128"] = compute_pastis_stats(
            PASTIS_DIR_ORIG, "PASTIS128", args.max_samples, args.num_workers
        )
    except Exception as e:
        print(f"ERROR processing PASTIS128: {e}")
        traceback.print_exc()

    # BreizhCrops
    try:
        all_stats["breizhcrops"] = compute_breizhcrops_stats(
            BREIZHCROPS_DIR, args.max_samples, args.num_workers
        )
    except Exception as e:
        print(f"ERROR processing BreizhCrops: {e}")
        traceback.print_exc()

    project_root = Path(__file__).parent.parent
    # Save to JSON
    if args.output is not None:
        output_file = Path(args.output)
    else:
        output_file = (
            project_root / "helios" / "evals" / "datasets" / "minmax_stats.json"
        )

    print("\n" + "=" * 80)
    print(f"Saving results to {output_file}")
    print("=" * 80)

    with open(output_file, "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\nSuccessfully computed stats for {len(all_stats)} datasets")
    print(f"Results saved to: {output_file}")
    if args.max_samples is not None:
        print(
            f"\nNote: This was a smoke test run with max {args.max_samples} samples per dataset."
        )
        print("Run without --max-samples to process all data.")


if __name__ == "__main__":
    main()
