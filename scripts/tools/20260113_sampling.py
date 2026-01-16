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


def sample_by_weighted_strategy(
    files_by_category: dict[int, list[UPath]],
    total_samples: int,
    category_weights: dict[int, float],
    others_weight: float,
) -> list[UPath]:
    """Sample files according to a weighted strategy.

    Args:
        files_by_category: Dictionary mapping category to list of file paths
        total_samples: Total number of samples to generate
        category_weights: Dict mapping specific categories to their target weights (0-1)
        others_weight: Weight for all other categories combined (0-1)

    Returns:
        List of sampled file paths
    """
    sampled_files = []
    specified_categories = set(category_weights.keys())
    other_categories = [
        c for c in files_by_category.keys() if c not in specified_categories
    ]

    # Sample from specified categories
    for category, weight in category_weights.items():
        target_count = int(total_samples * weight)
        files = files_by_category.get(category, [])
        if len(files) >= target_count:
            sampled = random.sample(files, target_count)
        else:
            sampled = files
            print(
                f"  Warning: Only {len(files)} available for category {category}, needed {target_count}"
            )
        sampled_files.extend(sampled)
        category_name = WORLDCOVER_CATEGORY_NAMES.get(category, f"unknown_{category}")
        print(
            f"  {category_name}: sampled {len(sampled)} files (target: {target_count})"
        )

    # Sample from "others" using native distribution
    if others_weight > 0 and other_categories:
        others_total = int(total_samples * others_weight)
        # Compute native distribution among other categories
        other_counts = {c: len(files_by_category[c]) for c in other_categories}
        total_other_files = sum(other_counts.values())

        if total_other_files > 0:
            for category in other_categories:
                native_proportion = other_counts[category] / total_other_files
                target_count = int(others_total * native_proportion)
                files = files_by_category[category]
                if len(files) >= target_count:
                    sampled = random.sample(files, target_count)
                else:
                    sampled = files
                sampled_files.extend(sampled)
                category_name = WORLDCOVER_CATEGORY_NAMES.get(
                    category, f"unknown_{category}"
                )
                print(
                    f"  {category_name} (other): sampled {len(sampled)} files (target: {target_count})"
                )

    return sampled_files


def sample_random(
    files_by_category: dict[int, list[UPath]],
    total_samples: int,
) -> list[UPath]:
    """Random sample maintaining native distribution.

    Args:
        files_by_category: Dictionary mapping category to list of file paths
        total_samples: Total number of samples to generate

    Returns:
        List of sampled file paths
    """
    all_files = []
    for files in files_by_category.values():
        all_files.extend(files)

    if len(all_files) >= total_samples:
        sampled = random.sample(all_files, total_samples)
    else:
        sampled = all_files
        print(
            f"  Warning: Only {len(all_files)} files available, needed {total_samples}"
        )

    print(f"  Random sampling: {len(sampled)} files")
    return sampled


# Sampling strategies for 10K samples
SAMPLING_STRATEGIES = {
    "strategy1": {
        # 40% tree cover, 40% cropland, 20% others
        "category_weights": {1: 0.40, 4: 0.40},
        "others_weight": 0.20,
        "description": "40% tree_cover, 40% cropland, 20% others",
    },
    "strategy2": {
        # 30% tree cover, 30% cropland, 30% built-up, 10% others
        "category_weights": {1: 0.30, 4: 0.30, 5: 0.30},
        "others_weight": 0.10,
        "description": "30% tree_cover, 30% cropland, 30% built_up, 10% others",
    },
    "strategy3": {
        # 22% tree cover, 22% cropland, 22% grassland, 22% built-up, 8% others
        "category_weights": {1: 0.22, 4: 0.22, 3: 0.22, 5: 0.22},
        "others_weight": 0.12,  # 100 - 88 = 12% for others (rounding)
        "description": "22% tree_cover, 22% cropland, 22% grassland, 22% built_up, 12% others",
    },
    "strategy4": {
        # 50% cropland, 20% tree cover, 20% built-up, 10% others
        "category_weights": {4: 0.50, 1: 0.20, 5: 0.20},
        "others_weight": 0.10,
        "description": "50% cropland, 20% tree_cover, 20% built_up, 10% others",
    },
    "strategy5": {
        # Random sampling (native distribution)
        "category_weights": {},
        "others_weight": 0.0,
        "random": True,
        "description": "Random sampling (native distribution)",
    },
    "strategy6": {
        # 50% water, 20% tree cover, 10% cropland, 10% built-up, 10% others
        "category_weights": {8: 0.50, 1: 0.20, 4: 0.10, 5: 0.10},
        "others_weight": 0.10,
        "description": "50% permanent_water_bodies, 20% tree_cover, 10% cropland, 10% built_up, 10% others",
    },
}


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
        "--total_samples",
        type=int,
        default=10000,
        help="Total number of samples for weighted strategies",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for strategy npy files",
    )
    parser.add_argument(
        "--stats_csv",
        type=str,
        default="category_stats.csv",
        help="Output CSV file for category statistics",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=-1,
        help="Maximum number of files to process (set to -1 for no limit)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    path_to_h5s = UPath(args.h5_path)
    stats_path = UPath(args.stats_csv)

    # Step 1: Load existing stats or compute from scratch
    files_by_category: dict[int, list[UPath]] = defaultdict(list)
    category_stats = []

    if stats_path.exists():
        print(f"\nStep 1: Loading existing stats from {stats_path}...")
        with open(stats_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["filename"]
                dominant_category = int(row["dominant_category"])
                h5_file = path_to_h5s / filename
                files_by_category[dominant_category].append(h5_file)
                # Convert percentage strings to floats for np.mean later
                for cat in range(NUM_WORLDCOVER_CLASSES):
                    cat_name = WORLDCOVER_CATEGORY_NAMES.get(cat, f"unknown_{cat}")
                    cat_key = f"{cat_name}_percent"
                    if cat_key in row:
                        row[cat_key] = float(row[cat_key])
                category_stats.append(row)
        print(f"Loaded stats for {len(category_stats)} files")
    else:
        h5s_to_process = list(path_to_h5s.glob("*.h5"))
        print(f"Found {len(h5s_to_process)} h5 files")

        # Limit to first max_files (unless -1 for no limit)
        if args.max_files > 0 and len(h5s_to_process) > args.max_files:
            h5s_to_process = h5s_to_process[: args.max_files]
            print(f"Limiting to first {args.max_files} files")

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

    # Save category statistics only if we computed them fresh
    if category_stats and not stats_path.exists():
        fieldnames = ["filename", "dominant_category", "dominant_category_name"] + [
            f"{WORLDCOVER_CATEGORY_NAMES.get(cat, f'unknown_{cat}')}_percent"
            for cat in range(NUM_WORLDCOVER_CLASSES)
        ]
        with open(stats_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(category_stats)

        print(f"Saved category statistics to {stats_path}")

    # Helper function to extract indices and save
    def extract_and_save_indices(
        sampled_files: list[UPath], output_path: UPath
    ) -> None:
        """Extract indices and save to npy file."""
        indices = []
        for file_path in sampled_files:
            # Extract index from filename like "sample_123.h5" -> 123
            filename = file_path.stem  # "sample_123"
            idx = int(filename.split("_")[1])
            indices.append(idx)

        indices_array = np.array(indices, dtype=np.int64)
        np.save(output_path, indices_array)
        print(f"Saved {len(indices_array)} indices to {output_path}")

    # Step 2: Run all 5 sampling strategies
    print(f"\n{'=' * 60}")
    print(
        f"Running all 5 sampling strategies with {args.total_samples} total samples each"
    )
    print(f"{'=' * 60}")

    output_dir = UPath(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for strategy_name, strategy_config in SAMPLING_STRATEGIES.items():
        print(f"\n--- {strategy_name}: {strategy_config['description']} ---")

        # Reset seed for each strategy for reproducibility
        random.seed(args.seed)
        np.random.seed(args.seed)

        if strategy_config.get("random", False):
            sampled_files = sample_random(files_by_category, args.total_samples)
        else:
            sampled_files = sample_by_weighted_strategy(
                files_by_category,
                args.total_samples,
                strategy_config["category_weights"],
                strategy_config["others_weight"],
            )

        print(f"Total sampled: {len(sampled_files)} files")

        output_path = output_dir / f"{strategy_name}_10k.npy"
        extract_and_save_indices(sampled_files, output_path)
