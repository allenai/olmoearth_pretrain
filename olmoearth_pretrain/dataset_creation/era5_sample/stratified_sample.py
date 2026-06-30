"""Stratified sampling from the candidate grid to produce the final selection.

Applies power allocation (n_b ~ count_b^alpha) across occupied
Köppen × elevation × latitude bins to draw 250k primary samples, then
marks 25% as overlap-pair seeds and generates a second point within the
same ERA5-Land 0.1° cell (offset ~5km for future S2 diversity).

Output: selected.parquet in the metadata directory.

Usage:
    python -m olmoearth_pretrain.dataset_creation.era5_sample.stratified_sample \
        --metadata-dir /weka/dfive-default/helios/dataset/era5enc_pretrain/metadata \
        --target-count 250000 \
        --overlap-fraction 0.25 \
        --seed 42
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ERA5_RESOLUTION_DEG = 0.1
OFFSET_KM = 5.0  # Spatial offset for overlap secondary windows


def _power_allocation(
    bin_counts: np.ndarray,
    target_total: int,
    alpha: float = 0.5,
    floor: int = 5,
) -> np.ndarray:
    """Compute per-bin sample counts using power allocation.

    n_b = max(floor, round(target_total * count_b^alpha / sum(count_b^alpha)))
    Then scale to exactly hit target_total.
    """
    powered = bin_counts.astype(float) ** alpha
    raw_alloc = powered / powered.sum() * target_total
    alloc = np.maximum(np.round(raw_alloc).astype(int), floor)

    # Cap each bin at its actual count
    alloc = np.minimum(alloc, bin_counts)

    # Scale to hit target exactly (iterative adjustment)
    for _ in range(100):
        diff = target_total - alloc.sum()
        if diff == 0:
            break
        if diff > 0:
            # Add to largest under-allocated bins
            headroom = bin_counts - alloc
            eligible = np.where(headroom > 0)[0]
            if len(eligible) == 0:
                break
            # Distribute proportionally to powered weight
            weights = powered[eligible]
            add = np.round(weights / weights.sum() * diff).astype(int)
            add = np.minimum(add, headroom[eligible])
            alloc[eligible] += add
        else:
            # Remove from largest over-floor bins
            removable = alloc - floor
            eligible = np.where(removable > 0)[0]
            if len(eligible) == 0:
                break
            weights = powered[eligible]
            remove = np.round(weights / weights.sum() * abs(diff)).astype(int)
            remove = np.minimum(remove, removable[eligible])
            alloc[eligible] -= remove

    return alloc


def _offset_within_cell(
    lon: float, lat: float, offset_km: float, rng: np.random.Generator
) -> tuple[float, float]:
    """Generate a point offset ~offset_km from (lon, lat), clamped to same ERA5 cell.

    The offset direction is random. The result is guaranteed to remain within
    the same 0.1° cell as the original point.
    """
    # ERA5-Land cell boundaries
    cell_lon_min = math.floor(lon / ERA5_RESOLUTION_DEG) * ERA5_RESOLUTION_DEG
    cell_lon_max = cell_lon_min + ERA5_RESOLUTION_DEG
    cell_lat_min = math.floor(lat / ERA5_RESOLUTION_DEG) * ERA5_RESOLUTION_DEG
    cell_lat_max = cell_lat_min + ERA5_RESOLUTION_DEG

    # Convert offset_km to approximate degrees
    lat_rad = math.radians(lat)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(lat_rad)

    # Random direction
    angle = rng.uniform(0, 2 * math.pi)
    d_lat = (offset_km * math.sin(angle)) / km_per_deg_lat
    d_lon = (offset_km * math.cos(angle)) / km_per_deg_lon if km_per_deg_lon > 1 else 0

    new_lon = lon + d_lon
    new_lat = lat + d_lat

    # Clamp to cell with small inset to avoid landing exactly on boundary
    inset = ERA5_RESOLUTION_DEG * 0.01
    new_lon = max(cell_lon_min + inset, min(new_lon, cell_lon_max - inset))
    new_lat = max(cell_lat_min + inset, min(new_lat, cell_lat_max - inset))

    return new_lon, new_lat


def stratified_sample(
    metadata_dir: str | Path,
    target_count: int = 250_000,
    overlap_fraction: float = 0.25,
    alpha: float = 0.5,
    floor: int = 5,
    seed: int = 42,
) -> Path:
    """Draw stratified sample and generate overlap pairs.

    Args:
        metadata_dir: Directory containing candidates.parquet.
        target_count: Number of primary samples to draw.
        overlap_fraction: Fraction of primaries that get a second (overlap) window.
        alpha: Power allocation exponent (0.5 = square root).
        floor: Minimum samples per occupied bin.
        seed: Random seed for reproducibility.

    Returns:
        Path to the output selected.parquet.
    """
    metadata_dir = Path(metadata_dir)
    output_path = metadata_dir / "selected.parquet"

    if output_path.exists():
        logger.info("selected.parquet already exists at %s, skipping", output_path)
        return output_path

    candidates_path = metadata_dir / "candidates.parquet"
    if not candidates_path.exists():
        raise FileNotFoundError(
            f"candidates.parquet not found at {candidates_path}. "
            "Run build_candidate_grid.py first."
        )

    logger.info("Loading candidates from %s", candidates_path)
    df = pd.read_parquet(candidates_path)
    logger.info("Loaded %d candidates", len(df))

    rng = np.random.default_rng(seed)

    # Create bin key
    df["bin_key"] = (
        df["koppen_code"].astype(str)
        + "_"
        + df["elev_band"].astype(str)
        + "_"
        + df["lat_band"].astype(str)
    )

    # Compute per-bin counts
    bin_counts_series = df["bin_key"].value_counts()
    bin_keys = bin_counts_series.index.values
    bin_counts = bin_counts_series.values

    logger.info(
        "Found %d occupied bins (of %d possible)",
        len(bin_keys),
        30 * 5 * 6,
    )

    # Compute allocation
    actual_target = min(target_count, len(df))
    alloc = _power_allocation(bin_counts, actual_target, alpha=alpha, floor=floor)
    total_allocated = alloc.sum()
    logger.info(
        "Power allocation: target=%d, allocated=%d across %d bins",
        actual_target,
        total_allocated,
        len(bin_keys),
    )

    # Sample within each bin
    selected_indices: list[int] = []
    for bin_key, n_samples in zip(bin_keys, alloc):
        bin_mask = df["bin_key"] == bin_key
        bin_indices = df.index[bin_mask].values
        n_samples = min(n_samples, len(bin_indices))
        if n_samples > 0:
            chosen = rng.choice(bin_indices, size=n_samples, replace=False)
            selected_indices.extend(chosen)

    selected = df.loc[selected_indices].copy().reset_index(drop=True)
    logger.info("Selected %d primary samples", len(selected))

    # Assign roles and pair IDs
    selected["role"] = "primary"
    selected["pair_id"] = -1

    # Mark overlap-pair seeds (25% of primaries)
    n_overlap = int(len(selected) * overlap_fraction)
    overlap_mask = np.zeros(len(selected), dtype=bool)
    overlap_indices = rng.choice(len(selected), size=n_overlap, replace=False)
    overlap_mask[overlap_indices] = True

    # Update roles for overlap primaries
    selected.loc[overlap_mask, "role"] = "overlap_primary"

    # Generate pair IDs
    pair_counter = 0
    pair_ids = selected["pair_id"].values.copy()
    for idx in np.where(overlap_mask)[0]:
        pair_ids[idx] = pair_counter
        pair_counter += 1
    selected["pair_id"] = pair_ids

    # Generate overlap secondary points
    secondary_rows: list[dict] = []
    for idx in np.where(overlap_mask)[0]:
        row = selected.iloc[idx]
        new_lon, new_lat = _offset_within_cell(row["lon"], row["lat"], OFFSET_KM, rng)
        secondary_rows.append(
            {
                "lon": new_lon,
                "lat": new_lat,
                "era5_cell_id": row["era5_cell_id"],
                "koppen_code": row["koppen_code"],
                "koppen_class": row["koppen_class"],
                "elevation_m": row["elevation_m"],
                "elev_band": row["elev_band"],
                "elev_band_label": row["elev_band_label"],
                "lat_band": row["lat_band"],
                "lat_band_label": row["lat_band_label"],
                "bin_key": row["bin_key"],
                "role": "overlap_secondary",
                "pair_id": row["pair_id"],
            }
        )

    secondary_df = pd.DataFrame(secondary_rows)
    result = pd.concat([selected, secondary_df], ignore_index=True)
    logger.info(
        "Final selection: %d primary + %d overlap_primary + %d overlap_secondary = %d total",
        (result["role"] == "primary").sum(),
        (result["role"] == "overlap_primary").sum(),
        (result["role"] == "overlap_secondary").sum(),
        len(result),
    )

    result.to_parquet(output_path, index=False)
    logger.info("Wrote selected.parquet to %s", output_path)

    # Print coverage summary
    _print_coverage_summary(result)

    return output_path


def _print_coverage_summary(df: pd.DataFrame) -> None:
    """Print stratification coverage statistics."""
    primaries = df[df["role"].isin(["primary", "overlap_primary"])]

    logger.info("--- Coverage Summary (primary windows) ---")
    logger.info("  Total primary windows: %d", len(primaries))

    # Köppen coverage
    koppen_counts = primaries["koppen_class"].value_counts()
    logger.info("  Köppen classes covered: %d / 30", koppen_counts.nunique())
    logger.info("  Top 5 by count:\n%s", koppen_counts.head().to_string())
    logger.info("  Bottom 5 by count:\n%s", koppen_counts.tail().to_string())

    # Elevation
    elev_counts = primaries["elev_band_label"].value_counts()
    logger.info("  Elevation distribution:\n%s", elev_counts.to_string())

    # Latitude
    lat_counts = primaries["lat_band_label"].value_counts()
    logger.info("  Latitude distribution:\n%s", lat_counts.to_string())


def main() -> None:
    """Run the stratified sampling CLI."""
    parser = argparse.ArgumentParser(
        description="Stratified sampling from ERA5 candidate grid."
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        required=True,
        help="Directory with candidates.parquet and output location.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=250_000,
        help="Number of primary samples to draw (default: 250000).",
    )
    parser.add_argument(
        "--overlap-fraction",
        type=float,
        default=0.25,
        help="Fraction of primaries that get an overlap pair (default: 0.25).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Power allocation exponent (default: 0.5 = square root).",
    )
    parser.add_argument(
        "--floor",
        type=int,
        default=5,
        help="Minimum samples per occupied bin (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()

    stratified_sample(
        metadata_dir=args.metadata_dir,
        target_count=args.target_count,
        overlap_fraction=args.overlap_fraction,
        alpha=args.alpha,
        floor=args.floor,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
