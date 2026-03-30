"""Generate filter_idx_file .npy arrays for axis-isolated ablation experiments.

All experiments start from a quality-gated "clean" pool, then split along
one axis at a time into high/low subsets of equal size. This isolates the
causal effect of each data trait on downstream eval performance.

Usage:
    python scripts/tools/generate_ablation_filters.py \
        --scores v0_osm_sampling_scores.parquet \
        --output-dir ablation_filters \
        --samples-per-split 50000

    # Custom quality gate thresholds
    python scripts/tools/generate_ablation_filters.py \
        --scores v0_osm_sampling_scores.parquet \
        --output-dir ablation_filters \
        --max-bright-frac 0.10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class AxisDef:
    """Definition of an ablation axis."""

    name: str
    description: str
    hypothesis: str
    feature: str
    low_label: str
    high_label: str
    # How to split: "quantile" splits by feature quantile,
    # "categorical" uses custom logic
    split_mode: str = "quantile"
    # For quantile mode: what quantile range defines low/high
    low_quantile_max: float = 0.33
    high_quantile_min: float = 0.67
    # For quantile mode: ascending=True means low values → "low" group
    ascending: bool = True


# Threshold justifications are based on the v0_osm_sampling_scores distribution
# analysis of 1.14M samples. See v0_osm_sampling_scores_analysis.pdf.
#
# Quality gate: bright_frac < 0.15 catches p95=0.115, removing the worst 5-8%
# of cloudy samples. dark_frac < 0.15 mirrors this for shadows (p95=0.121).
# constant_frac < 0.05 catches p99=0.109, removing sensor fill / dead strips.
# valid_timesteps >= 8 is loose — p5=11, so this only removes truly broken
# samples (<1% of corpus). Together these pass 92.3%.
#
# Per-axis p33/p67 splits were chosen because the distributions are smooth
# and unimodal — terciles give clean separation with large pools (~347k each).

AXES = [
    AxisDef(
        name="spatial_complexity",
        description="Spatial texture and boundary density",
        hypothesis="Spatially complex scenes (edges, mixed surfaces) produce richer "
        "gradients and improve segmentation tasks (PASTIS, MADOS)",
        feature="spatial_autocorr",
        low_label="complex",
        high_label="smooth",
        # Distribution: mean=0.902, std=0.068. p33=0.880, p67=0.939.
        # The distribution is left-skewed with a long tail toward 0.5-0.8
        # (complex scenes). p33/p67 split gives: complex pool mean=0.825,
        # smooth pool mean=0.972 — a 0.15 gap on a 0-1 scale.
        # low autocorr = complex, high autocorr = smooth (ascending=True)
        low_quantile_max=0.33,
        high_quantile_min=0.67,
    ),
    AxisDef(
        name="temporal_dynamics",
        description="Strength of seasonal/temporal change",
        hypothesis="Strong temporal signal teaches the model phenological reasoning, "
        "improving multi-temporal tasks (PASTIS) and change detection",
        feature="sentinel2_l2a_seasonal_amplitude",
        low_label="static",
        high_label="dynamic",
        # Distribution: mean=3285, std=2426. Highly right-skewed — the bottom
        # third (p33=1443) are near-static scenes (desert, ocean, evergreen
        # forest), top third (p67=4257) have dramatic seasonal swings. This
        # gives a 7x difference in amplitude between groups.
        low_quantile_max=0.33,
        high_quantile_min=0.67,
    ),
    AxisDef(
        name="land_cover_diversity",
        description="Number and balance of land cover classes per tile",
        hypothesis="Tiles with diverse land cover force the model to learn "
        "discriminative features at class boundaries, improving classification broadly",
        feature="lc_entropy",
        low_label="homogeneous",
        high_label="diverse",
        # Distribution: mean=0.862, std=0.380, max=2.101. Roughly normal but
        # with a spike near 0 (homogeneous tiles). p33=0.724 corresponds to
        # ~2-3 classes with one dominant; p67=1.075 corresponds to 4+ classes
        # with reasonable balance. 3x entropy gap between groups.
        low_quantile_max=0.33,
        high_quantile_min=0.67,
    ),
    AxisDef(
        name="geographic_regime",
        description="Latitude band / climate zone diversity",
        hypothesis="Training on diverse geographic regimes improves generalization "
        "to unseen regions and biomes in downstream evals",
        feature="abs_lat",
        low_label="tropical_subtropical",
        high_label="temperate_boreal",
        # Distribution: mean=38.5°, std=13.8°. The corpus is heavily temperate
        # (63% at 35-55°). p33=35.0° cleanly separates tropical+subtropical
        # (mean=22.5°) from temperate+boreal (mean=52.1°). The 30° gap spans
        # fundamentally different biomes, solar angles, and seasonal patterns.
        low_quantile_max=0.33,
        high_quantile_min=0.67,
    ),
    AxisDef(
        name="spectral_richness",
        description="Pixel value diversity and dynamic range",
        hypothesis="Spectrally rich scenes (diverse reflectance values) teach the "
        "model finer spectral discrimination than spectrally flat scenes",
        feature="sentinel2_l2a_entropy",
        low_label="flat_spectrum",
        high_label="rich_spectrum",
        # Distribution: mean=3.811, std=0.261, tight and near-normal. The gap
        # between p33=3.696 and p67=3.924 is only 0.23 nats — modest but this
        # is a log-scale measure so it represents ~26% more unique spectral
        # configurations. Low entropy = dominated by one surface reflectance;
        # high entropy = rich mix of materials.
        low_quantile_max=0.33,
        high_quantile_min=0.67,
    ),
    AxisDef(
        name="infrastructure_content",
        description="Presence of human-built structures and infrastructure",
        hypothesis="The dataset has almost no dense urban content (0% dense urban) — "
        "enriching for infrastructure-adjacent tiles improves classification of "
        "built environments and teaches the model human land use patterns",
        feature="osm_feature_count",
        low_label="rural",
        high_label="infrastructure_rich",
        # Distribution: mean=3.0, std=1.8, integer-valued 0-13. p33=2 (1-2 OSM
        # layers active = just roads), p67=4 (4+ active layers = buildings,
        # roads, parking, etc). The infrastructure-rich pool has 3.5x more
        # active OSM feature types — meaningfully more complex human landscape.
        low_quantile_max=0.33,
        high_quantile_min=0.67,
    ),
]


def apply_quality_gate(
    df: pd.DataFrame,
    max_bright_frac: float = 0.15,
    max_dark_frac: float = 0.15,
    max_constant_frac: float = 0.05,
    min_valid_timesteps: int = 8,
) -> pd.DataFrame:
    """Apply quality gate to get the clean pool."""
    mask = (
        (df["sentinel2_l2a_bright_frac"] < max_bright_frac)
        & (df["sentinel2_l2a_dark_frac"] < max_dark_frac)
        & (df["sentinel2_l2a_constant_frac"] < max_constant_frac)
        & (df["sentinel2_l2a_valid_timesteps"] >= min_valid_timesteps)
    )
    return df[mask]


def generate_axis_splits(
    clean: pd.DataFrame,
    axis: AxisDef,
    samples_per_split: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate high/low index arrays for one axis.

    Returns (low_indices, high_indices, metadata_dict).
    """
    feat = clean[axis.feature]

    if axis.split_mode == "quantile":
        low_thresh = feat.quantile(axis.low_quantile_max)
        high_thresh = feat.quantile(axis.high_quantile_min)

        if axis.ascending:
            low_pool = clean[feat <= low_thresh]
            high_pool = clean[feat >= high_thresh]
        else:
            # Inverted: low feature value = "high" group
            low_pool = clean[feat >= high_thresh]
            high_pool = clean[feat <= low_thresh]

        meta = {
            "low_threshold": float(low_thresh),
            "high_threshold": float(high_thresh),
        }
    else:
        raise ValueError(f"Unknown split_mode: {axis.split_mode}")

    # Sample equal counts from each pool
    n = min(samples_per_split, len(low_pool), len(high_pool))
    if n < 1000:
        logger.warning(
            f"Axis '{axis.name}': only {n} samples available per split "
            f"(low_pool={len(low_pool)}, high_pool={len(high_pool)})"
        )

    low_sampled = low_pool.sample(n=n, random_state=42)
    high_sampled = high_pool.sample(n=n, random_state=42)

    low_indices = low_sampled["sample_index"].astype(int).values
    high_indices = high_sampled["sample_index"].astype(int).values

    meta.update(
        {
            "axis": axis.name,
            "feature": axis.feature,
            "description": axis.description,
            "hypothesis": axis.hypothesis,
            "samples_per_split": int(n),
            "low_label": axis.low_label,
            "high_label": axis.high_label,
            "low_pool_size": len(low_pool),
            "high_pool_size": len(high_pool),
            "low_feature_mean": float(low_sampled[axis.feature].mean()),
            "high_feature_mean": float(high_sampled[axis.feature].mean()),
            "low_feature_std": float(low_sampled[axis.feature].std()),
            "high_feature_std": float(high_sampled[axis.feature].std()),
        }
    )

    return low_indices, high_indices, meta


def main() -> None:
    """Generate ablation filter files."""
    parser = argparse.ArgumentParser(
        description="Generate axis-isolated ablation filters"
    )
    parser.add_argument("--scores", required=True, help="Path to scored parquet")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for .npy files"
    )
    parser.add_argument(
        "--samples-per-split",
        type=int,
        default=50_000,
        help="Number of samples per high/low split",
    )
    parser.add_argument("--max-bright-frac", type=float, default=0.15)
    parser.add_argument("--max-dark-frac", type=float, default=0.15)
    parser.add_argument("--max-constant-frac", type=float, default=0.05)
    parser.add_argument("--min-valid-timesteps", type=int, default=8)
    args = parser.parse_args()

    logger.info(f"Loading scores from {args.scores}")
    df = pd.read_parquet(args.scores)
    logger.info(f"Loaded {len(df):,} samples")
    os.makedirs(args.output_dir, exist_ok=True)

    n = args.samples_per_split

    # ---- Standalone filters: quality gate + random baseline ----
    clean = apply_quality_gate(
        df,
        max_bright_frac=args.max_bright_frac,
        max_dark_frac=args.max_dark_frac,
        max_constant_frac=args.max_constant_frac,
        min_valid_timesteps=args.min_valid_timesteps,
    )
    logger.info(
        f"Quality gate: {len(clean):,} / {len(df):,} samples pass "
        f"({len(clean) / len(df):.1%})"
    )

    # Quality gate filter — sampled to same size as axis splits for fair comparison
    clean_sampled = clean.sample(n=min(n, len(clean)), random_state=42)
    clean_path = os.path.join(args.output_dir, "quality_gate.npy")
    np.save(clean_path, clean_sampled["sample_index"].astype(int).values)
    logger.info(f"Saved quality_gate ({len(clean_sampled):,} samples) -> {clean_path}")

    # Random baseline — same size, no filtering at all
    random_sampled = df.sample(n=min(n, len(df)), random_state=42)
    random_path = os.path.join(args.output_dir, "random_baseline.npy")
    np.save(random_path, random_sampled["sample_index"].astype(int).values)
    logger.info(
        f"Saved random_baseline ({len(random_sampled):,} samples) -> {random_path}"
    )

    # ---- Axis splits: from the FULL dataset, no quality gate ----
    all_meta: dict = {
        "total_dataset_size": len(df),
        "quality_gate_size": len(clean),
        "samples_per_split": n,
    }
    axis_summaries = []

    for axis in AXES:
        logger.info(f"\nAxis: {axis.name} ({axis.feature})")
        low_idx, high_idx, meta = generate_axis_splits(df, axis, n)

        low_path = os.path.join(args.output_dir, f"{axis.name}_{axis.low_label}.npy")
        high_path = os.path.join(args.output_dir, f"{axis.name}_{axis.high_label}.npy")
        np.save(low_path, low_idx)
        np.save(high_path, high_idx)

        logger.info(
            f"  {axis.low_label}: {len(low_idx):,} samples "
            f"(feature mean={meta['low_feature_mean']:.3f})"
        )
        logger.info(
            f"  {axis.high_label}: {len(high_idx):,} samples "
            f"(feature mean={meta['high_feature_mean']:.3f})"
        )

        all_meta[axis.name] = meta
        axis_summaries.append(meta)

    # Save experiment manifest
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(all_meta, f, indent=2)
    logger.info(f"\nManifest saved -> {manifest_path}")

    # Print summary table
    n_experiments = (
        len(AXES) * 2 + 2
    )  # axes × high/low + quality_gate + random_baseline
    print("\n" + "=" * 90)
    print("  ABLATION EXPERIMENT PLAN")
    print("=" * 90)
    print(f"\nDataset: {len(df):,} samples")
    print(f"Quality gate passes: {len(clean):,} ({len(clean) / len(df):.1%})")
    print(f"Samples per split: {n:,}")

    print(f"\n{'Experiment':<50s} {'N':>8s}  Description")
    print("-" * 90)
    print(f"{'random_baseline':<50s} {n:>8,}  Unfiltered random sample (control)")
    print(
        f"{'quality_gate':<50s} {len(clean_sampled):>8,}  "
        f"Quality-gated sample (bright<{args.max_bright_frac}, dark<{args.max_dark_frac}, "
        f"const<{args.max_constant_frac}, ts>={args.min_valid_timesteps})"
    )
    for m in axis_summaries:
        for side in ["low", "high"]:
            label = m[f"{side}_label"]
            name = f"{m['axis']}_{label}"
            feat_mean = m[f"{side}_feature_mean"]
            print(
                f"{name:<50s} {m['samples_per_split']:>8,}  {m['feature']} mean={feat_mean:.3f}"
            )

    print(f"\nTotal experiments: {n_experiments}")
    print(f"Filter files: {args.output_dir}/")
    print()

    print("HYPOTHESES:")
    print("-" * 90)
    print(
        "  random_baseline: Control — no filtering. All other experiments should beat this."
    )
    print(
        "  quality_gate: Removing cloudy/shadowed/degenerate samples improves training "
        "efficiency without changing the data distribution."
    )
    for m in axis_summaries:
        print(f"  {m['axis']}: {m['hypothesis']}")
    print()

    print("SUGGESTED EXPERIMENT COMMANDS (base model, dry_run first):")
    print("-" * 90)

    all_names = ["random_baseline", "quality_gate"]
    for axis in AXES:
        all_names.append(f"{axis.name}_{axis.low_label}")
        all_names.append(f"{axis.name}_{axis.high_label}")

    for name in all_names:
        npy_path = os.path.join(args.output_dir, f"{name}.npy")
        run_name = f"data_ablation_{name}"
        print(
            f"  # {run_name}\n"
            f"  python scripts/official/base.py dry_run {run_name} local "
            f'--dataset.filter_idx_file="{npy_path}"\n'
        )


if __name__ == "__main__":
    main()
