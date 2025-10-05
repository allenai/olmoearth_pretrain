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
- sickle (Sentinel-2 L2A, Sentinel-1, Landsat-8)
- cropharvest (Sentinel-2 L2A, Sentinel-1, SRTM)
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
    CROPHARVEST_DIR,
    FLOODS_DIR,
    MADOS_DIR,
    PASTIS_DIR,
    PASTIS_DIR_ORIG,
    SICKLE_DIR,
)
from helios.evals.datasets.breizhcrops import LEVEL, BreizhCrops
from helios.evals.datasets.constants import (
    EVAL_S1_BAND_NAMES,
    EVAL_S2_BAND_NAMES,
    EVAL_SRTM_BAND_NAMES,
)
from helios.evals.datasets.cropharvest import (
    S2_EVAL_BANDS_BEFORE_IMPUTATION as CROPHARVEST_S2_EVAL_BANDS_BEFORE_IMPUTATION,
)
from helios.evals.datasets.cropharvest import _get_eval_datasets
from helios.evals.datasets.cropharvest_package.datasets import CropHarvest
from helios.evals.datasets.sickle_dataset import (
    L8_BAND_STATS as SICKLE_L8_BAND_STATS,
)
from helios.evals.datasets.sickle_dataset import (
    S1_BAND_STATS as SICKLE_S1_BAND_STATS,
)
from helios.evals.datasets.sickle_dataset import (
    S2_BAND_STATS as SICKLE_S2_BAND_STATS,
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


def compute_sickle_stats(
    path_to_splits: UPath, max_samples: int | None = None, num_workers: int = 1
) -> dict:
    """Compute min/max stats for SICKLE dataset."""
    print("Computing stats for SICKLE (Sentinel-2 L2A, Sentinel-1, Landsat-8)...")

    split_dir = "sickle_train"
    s2_dir = path_to_splits / split_dir / "s2_images"
    s1_dir = path_to_splits / split_dir / "s1_images"
    l8_dir = path_to_splits / split_dir / "l8_images"

    # Get all image files
    s2_files = sorted(s2_dir.glob("*.pt"))
    s1_files = sorted(s1_dir.glob("*.pt"))
    l8_files = sorted(l8_dir.glob("*.pt"))

    # Limit samples if requested
    if max_samples is not None:
        s2_files = s2_files[:max_samples]
        s1_files = s1_files[:max_samples]
        l8_files = l8_files[:max_samples]
        print(f"Limited to {max_samples} samples (smoke test mode)")

    # Initialize min/max trackers
    s2_mins = [float("inf")] * len(SICKLE_S2_BAND_STATS.keys())
    s2_maxs = [float("-inf")] * len(SICKLE_S2_BAND_STATS.keys())
    s1_mins = [float("inf")] * len(SICKLE_S1_BAND_STATS.keys())
    s1_maxs = [float("-inf")] * len(SICKLE_S1_BAND_STATS.keys())
    l8_mins = [float("inf")] * len(SICKLE_L8_BAND_STATS.keys())
    l8_maxs = [float("-inf")] * len(SICKLE_L8_BAND_STATS.keys())

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
    else:
        for s1_file in tqdm(s1_files, desc="S1"):
            mins, maxs = _process_image_file(s1_file, "s1")
            for band_idx in range(len(mins)):
                s1_mins[band_idx] = min(s1_mins[band_idx], mins[band_idx])
                s1_maxs[band_idx] = max(s1_maxs[band_idx], maxs[band_idx])

    # Process L8 images in parallel
    print(
        f"  Processing {len(l8_files)} Landsat-8 images with {num_workers} workers..."
    )
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_process_image_file, f, "l8"): f for f in l8_files
            }
            for future in tqdm(as_completed(futures), total=len(l8_files), desc="L8"):
                mins, maxs = future.result()
                for band_idx in range(len(mins)):
                    l8_mins[band_idx] = min(l8_mins[band_idx], mins[band_idx])
                    l8_maxs[band_idx] = max(l8_maxs[band_idx], maxs[band_idx])
    else:
        for l8_file in tqdm(l8_files, desc="L8"):
            mins, maxs = _process_image_file(l8_file, "l8")
            for band_idx in range(len(mins)):
                l8_mins[band_idx] = min(l8_mins[band_idx], mins[band_idx])
                l8_maxs[band_idx] = max(l8_maxs[band_idx], maxs[band_idx])

    stats = {
        "sentinel2_l2a": {
            band_name: {"min": s2_mins[i], "max": s2_maxs[i]}
            for i, band_name in enumerate(SICKLE_S2_BAND_STATS.keys())
        },
        "sentinel1": {
            band_name: {"min": s1_mins[i], "max": s1_maxs[i]}
            for i, band_name in enumerate(SICKLE_S1_BAND_STATS.keys())
        },
        "landsat": {
            band_name: {"min": l8_mins[i], "max": l8_maxs[i]}
            for i, band_name in enumerate(SICKLE_L8_BAND_STATS.keys())
        },
    }

    return stats


def compute_cropharvest_stats(
    cropharvest_dir: UPath, max_samples: int | None = None, num_workers: int = 1
) -> dict:
    """Compute min/max stats for CropHarvest dataset."""
    print("Computing stats for CropHarvest (Sentinel-2 L2A, Sentinel-1, SRTM)...")

    # Download if needed
    CropHarvest(cropharvest_dir, download=True)

    # Get all evaluation datasets
    evaluation_datasets = _get_eval_datasets(cropharvest_dir)

    # Initialize min/max trackers for all modalities
    # CropHarvest has 18 bands total: 9 S2, 2 S1, 1 SRTM, plus some extras
    all_mins = [float("inf")] * 18
    all_maxs = [float("-inf")] * 18

    total_samples = 0
    for dataset in evaluation_datasets:
        print(f"  Processing {dataset.id}...")
        try:
            array, _, _ = dataset.as_array()  # Shape: (N, T, C)

            # Limit samples if requested
            num_to_process = array.shape[0]
            if max_samples is not None:
                remaining = max_samples - total_samples
                if remaining <= 0:
                    print("    Skipping (already reached max_samples limit)")
                    continue
                num_to_process = min(array.shape[0], remaining)
                print(f"    Limited to {num_to_process} samples (smoke test mode)")

            # Update min/max across all samples and timesteps
            for sample_idx in range(num_to_process):
                for band_idx in range(array.shape[2]):
                    band_data = array[sample_idx, :, band_idx]
                    valid_data = band_data[~np.isnan(band_data) & ~np.isinf(band_data)]
                    if len(valid_data) > 0:
                        all_mins[band_idx] = min(
                            all_mins[band_idx], float(valid_data.min())
                        )
                        all_maxs[band_idx] = max(
                            all_maxs[band_idx], float(valid_data.max())
                        )

            total_samples += array.shape[0]
        except Exception as e:
            print(f"    Warning: Failed to process {dataset.id}: {e}")
            continue

    # Map to actual band names based on CropHarvest structure
    # First 9 are S2 bands (without some bands like coastal aerosol, cirrus)
    # Then 2 S1 bands
    # Then 1 SRTM band
    stats = {
        "sentinel1": {
            band_name: {"min": all_mins[i], "max": all_maxs[i]}
            for i, band_name in enumerate(EVAL_S1_BAND_NAMES)
        },
        "sentinel2_l2a": {
            band_name: {
                "min": all_mins[len(EVAL_S1_BAND_NAMES) + i],
                "max": all_maxs[len(EVAL_S1_BAND_NAMES) + i],
            }
            for i, band_name in enumerate(CROPHARVEST_S2_EVAL_BANDS_BEFORE_IMPUTATION)
        },
        "srtm": {
            band_name: {
                "min": all_mins[
                    len(EVAL_S1_BAND_NAMES)
                    + len(CROPHARVEST_S2_EVAL_BANDS_BEFORE_IMPUTATION)
                    + i
                ],
                "max": all_maxs[
                    len(EVAL_S1_BAND_NAMES)
                    + len(CROPHARVEST_S2_EVAL_BANDS_BEFORE_IMPUTATION)
                    + i
                ],
            }
            for i, band_name in enumerate(EVAL_SRTM_BAND_NAMES)
        },
    }

    print(f"  Processed {total_samples} samples across all countries")
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

    # SICKLE
    try:
        all_stats["sickle"] = compute_sickle_stats(
            SICKLE_DIR, args.max_samples, args.num_workers
        )
    except Exception as e:
        print(f"ERROR processing SICKLE: {e}")
        traceback.print_exc()

    # CropHarvest
    try:
        all_stats["cropharvest"] = compute_cropharvest_stats(
            CROPHARVEST_DIR, args.max_samples, args.num_workers
        )
    except Exception as e:
        print(f"ERROR processing CropHarvest: {e}")
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
