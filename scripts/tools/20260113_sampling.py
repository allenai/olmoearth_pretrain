"""Sampling experiments based on worldcover dominant category."""

import argparse
import csv
import random
from collections import defaultdict

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from tqdm import tqdm
from upath import UPath

from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality

# Worldcover has 12 classes (0-11)
NUM_WORLDCOVER_CLASSES = 12

# Worldcover category names
WORLDCOVER_CATEGORY_NAMES = {
    0: "missing",
    1: "tree_cover",
    2: "shrubland",
    3: "grassland",
    4: "cropland",
    5: "built_up",
    6: "bare_sparse_vegetable",
    7: "snow_and_ice",
    8: "permanent_water_bodies",
    9: "herbaceous_wetland",
    10: "moss_and_lichen",
    11: "mangroves",
}


def process_worldcover(wc: np.ndarray) -> np.ndarray:
    """Process WC data."""
    wc[wc == 95] = 110
    wc = wc / 10  # now we should be to classes
    # keep missing values
    wc[wc == MISSING_VALUE / 10] = MISSING_VALUE
    return wc


def read_worldcover_from_h5(h5_file_path: UPath) -> np.ndarray | None:
    """Read worldcover data from h5 file."""
    with h5_file_path.open("rb") as f:
        with h5py.File(f, "r") as h5file:
            if Modality.WORLDCOVER.name not in h5file:
                return None
            wc = h5file[Modality.WORLDCOVER.name][()]
            wc = process_worldcover(wc)
            return wc
    return None


def compute_category_percentages(wc: np.ndarray) -> dict[int, float]:
    """Compute percentage of pixels per category.

    Args:
        wc: Worldcover array with shape (H, W, T, C) or similar

    Returns:
        Dictionary mapping category (0-11) to percentage of pixels
    """
    # Flatten the array to get all pixels
    wc_flat = wc.flatten()

    # Filter out missing values
    wc_flat = wc_flat[wc_flat != MISSING_VALUE]

    if len(wc_flat) == 0:
        return {}

    # Count pixels per category
    unique, counts = np.unique(wc_flat, return_counts=True)
    total_pixels = len(wc_flat)

    # Create dictionary with percentages
    percentages = {}
    for category, count in zip(unique, counts):
        category_int = int(category)
        if 0 <= category_int < NUM_WORLDCOVER_CLASSES:
            percentages[category_int] = (count / total_pixels) * 100.0

    return percentages


def get_dominant_category(percentages: dict[int, float]) -> int | None:
    """Get the category with the highest percentage.

    Args:
        percentages: Dictionary mapping category to percentage

    Returns:
        The dominant category, or None if no valid categories
    """
    if not percentages:
        return None
    return max(percentages.items(), key=lambda x: x[1])[0]


def sample_by_dominant_category(
    files_by_category: dict[int, list[UPath]], samples_per_category: int
) -> list[UPath]:
    """Sample equal number of files for each dominant category.

    Args:
        files_by_category: Dictionary mapping category to list of file paths
        samples_per_category: Number of samples to take per category

    Returns:
        List of sampled file paths
    """
    sampled_files = []
    for category, files in files_by_category.items():
        if len(files) >= samples_per_category:
            sampled = random.sample(files, samples_per_category)
        else:
            # If we don't have enough files, take all of them
            sampled = files
        sampled_files.extend(sampled)
        category_name = WORLDCOVER_CATEGORY_NAMES.get(category, f"unknown_{category}")
        print(
            f"Category {category} ({category_name}): sampled {len(sampled)} files (out of {len(files)} available)"
        )

    return sampled_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample h5 files based on worldcover dominant category"
    )
    parser.add_argument(
        "--h5_path",
        type=str,
        required=True,
        help="Path to directory containing h5 files",
    )
    parser.add_argument(
        "--samples_per_category",
        type=int,
        default=1000,
        help="Number of samples to take per dominant category (default: 1000)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="sampling_by_category.csv",
        help="Output CSV file path (default: sampling_by_category.csv)",
    )
    parser.add_argument(
        "--stats_csv",
        type=str,
        default="category_stats.csv",
        help="Output CSV file for category statistics (default: category_stats.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=5000,
        help="Maximum number of files to process (default: 5000)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    path_to_h5s = UPath(args.h5_path)
    h5s_to_process = list(path_to_h5s.glob("*.h5"))
    print(f"Found {len(h5s_to_process)} h5 files")

    # Limit to first max_files
    if len(h5s_to_process) > args.max_files:
        h5s_to_process = h5s_to_process[: args.max_files]
        print(f"Limiting to first {args.max_files} files")

    # Step 1: Process all files and compute category percentages
    files_by_category: dict[int, list[UPath]] = defaultdict(list)
    category_stats = []

    print(
        f"\nStep 1: Computing category percentages for {len(h5s_to_process)} files..."
    )
    for h5_file in tqdm(h5s_to_process):
        if h5_file.name == "normalizing_dict.h5":
            continue

        try:
            wc = read_worldcover_from_h5(h5_file)
            if wc is None:
                continue

            percentages = compute_category_percentages(wc)
            dominant_category = get_dominant_category(percentages)

            if dominant_category is not None:
                files_by_category[dominant_category].append(h5_file)

                # Store statistics
                stats = {
                    "filename": h5_file.name,
                    "dominant_category": dominant_category,
                    "dominant_category_name": WORLDCOVER_CATEGORY_NAMES.get(
                        dominant_category, f"unknown_{dominant_category}"
                    ),
                }
                # Add percentage for each category
                for cat in range(NUM_WORLDCOVER_CLASSES):
                    cat_name = WORLDCOVER_CATEGORY_NAMES.get(cat, f"unknown_{cat}")
                    stats[f"{cat_name}_percent"] = percentages.get(cat, 0.0)
                category_stats.append(stats)
        except Exception as e:
            print(f"Error processing {h5_file.name}: {e}")

    print("\nFound files with dominant categories:")
    total_files = sum(len(files) for files in files_by_category.values())
    for category in sorted(files_by_category.keys()):
        category_name = WORLDCOVER_CATEGORY_NAMES.get(category, f"unknown_{category}")
        count = len(files_by_category[category])
        percentage = (count / total_files * 100.0) if total_files > 0 else 0.0
        print(
            f"  Category {category} ({category_name}): {count} files ({percentage:.2f}%)"
        )

    # Compute average percentage of each category across all files
    if category_stats:
        print("\nAverage percentage of each category across all files:")
        avg_percentages = {}
        for cat in range(NUM_WORLDCOVER_CLASSES):
            cat_name = WORLDCOVER_CATEGORY_NAMES.get(cat, f"unknown_{cat}")
            cat_key = f"{cat_name}_percent"
            if cat_key in category_stats[0]:
                avg_percentages[cat] = np.mean(
                    [stats[cat_key] for stats in category_stats]
                )

        for cat in sorted(avg_percentages.keys()):
            cat_name = WORLDCOVER_CATEGORY_NAMES.get(cat, f"unknown_{cat}")
            print(f"  Category {cat} ({cat_name}): {avg_percentages[cat]:.2f}%")

    # Step 2: Sample equal numbers per category
    print(f"\nStep 2: Sampling {args.samples_per_category} files per category...")
    sampled_files = sample_by_dominant_category(
        files_by_category, args.samples_per_category
    )

    print(f"\nTotal sampled files: {len(sampled_files)}")

    # Save sampled files
    output_path = UPath(args.output_csv)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename"])
        writer.writeheader()
        for file_path in sampled_files:
            writer.writerow({"filename": file_path.name})

    print(f"Saved sampled files to {output_path}")

    # Save category statistics
    if category_stats:
        stats_path = UPath(args.stats_csv)
        fieldnames = ["filename", "dominant_category", "dominant_category_name"] + [
            f"{WORLDCOVER_CATEGORY_NAMES.get(cat, f'unknown_{cat}')}_percent"
            for cat in range(NUM_WORLDCOVER_CLASSES)
        ]
        with open(stats_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(category_stats)

        print(f"Saved category statistics to {stats_path}")
