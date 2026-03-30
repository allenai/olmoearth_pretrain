"""Analyze and visualize sample scores produced by score_samples.py.

Generates an HTML report with:
- Summary statistics for all features
- Distribution histograms for key features
- Correlation heatmap between features
- Geographic scatter of samples colored by traits
- Land cover and modality composition breakdowns
- Feature pair scatter plots

Usage:
    python scripts/tools/analyze_scores.py test_1k_scores.parquet
    python scripts/tools/analyze_scores.py test_1k_scores.parquet --output report.html
    python scripts/tools/analyze_scores.py test_1k_scores.parquet --top-k 30
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class PngSaver:
    """Drop-in replacement for PdfPages that saves each figure as a numbered PNG."""

    def __init__(self, output_dir: str) -> None:
        """Initialize PNG saver with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._idx = 0

    def __enter__(self) -> PngSaver:
        """Enter context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager."""

    def savefig(self, fig: plt.Figure, **kwargs: Any) -> None:
        """Save figure as PNG."""
        path = os.path.join(self.output_dir, f"page_{self._idx:02d}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        self._idx += 1


# Features that are most interesting to visualize distributions for
KEY_FEATURES = [
    "lc_entropy",
    "lc_edge_density",
    "sentinel2_l2a_entropy",
    "sentinel2_l2a_temporal_std",
    "sentinel2_l2a_seasonal_amplitude",
    "ndvi_mean",
    "ndvi_seasonal_amplitude",
    "elevation_mean",
    "terrain_ruggedness",
    "population_mean",
    "overall_missing_frac",
    "completeness_ratio",
    "num_modalities",
    "modality_type_diversity",
    "sentinel2_l2a_bright_frac",
    "sentinel2_l2a_constant_frac",
    "spatial_autocorr",
    "edge_density",
    "s2_bimodality",
    "frac_pixels_changed",
    "change_concentration",
    "ndwi_mean",
    "ndbi_mean",
    "canopy_height_mean",
    "osm_feature_count",
    "temperature_mean",
    "abs_lat",
    "sentinel2_l2a_dynamic_range",
]

LAND_COVER_COLS = [
    c
    for c in [
        "lc_frac_tree_cover",
        "lc_frac_shrubland",
        "lc_frac_grassland",
        "lc_frac_cropland",
        "lc_frac_built_up",
        "lc_frac_bare_sparse",
        "lc_frac_snow_ice",
        "lc_frac_water",
        "lc_frac_herbaceous_wetland",
        "lc_frac_mangroves",
        "lc_frac_moss_lichen",
    ]
]


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics for all numeric columns."""
    numeric = df.select_dtypes(include="number")
    stats = numeric.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    stats["non_zero_frac"] = (numeric != 0).mean()
    stats["nunique"] = numeric.nunique()
    return stats.sort_index()


def plot_distributions(df: pd.DataFrame, features: list[str], pdf: PdfPages) -> None:
    """Plot histograms for key features, 4x4 per page."""
    present = [f for f in features if f in df.columns]
    for page_start in range(0, len(present), 16):
        batch = present[page_start : page_start + 16]
        n = len(batch)
        cols = 4
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 3 * rows))
        axes = np.array(axes).flatten()

        for i, feat in enumerate(batch):
            ax = axes[i]
            vals = df[feat].dropna()
            if len(vals) > 100_000:
                vals = vals.sample(n=100_000, random_state=42)
            ax.hist(vals, bins=50, color="steelblue", edgecolor="none", alpha=0.8)
            ax.set_title(feat, fontsize=9)
            ax.tick_params(labelsize=7)
            # Add mean/median lines
            ax.axvline(
                vals.mean(),
                color="red",
                linestyle="--",
                linewidth=0.8,
                label=f"mean={vals.mean():.2f}",
            )
            ax.axvline(
                vals.median(),
                color="orange",
                linestyle="-",
                linewidth=0.8,
                label=f"med={vals.median():.2f}",
            )
            ax.legend(fontsize=6)

        for i in range(n, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle("Feature Distributions", fontsize=14, y=1.01)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def plot_correlation_heatmap(
    df: pd.DataFrame, features: list[str], pdf: PdfPages
) -> None:
    """Plot correlation heatmap for key features."""
    present = [f for f in features if f in df.columns]
    if len(present) < 3:
        return

    corr = df[present].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(present)))
    ax.set_yticks(range(len(present)))
    ax.set_xticklabels(present, rotation=90, fontsize=6)
    ax.set_yticklabels(present, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature Correlation Matrix", fontsize=14)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


MAX_SCATTER_POINTS = 50_000


def _subsample(df: pd.DataFrame, max_n: int = MAX_SCATTER_POINTS) -> pd.DataFrame:
    """Subsample a dataframe for plotting if too large."""
    if len(df) <= max_n:
        return df
    return df.sample(n=max_n, random_state=42)


def plot_geographic(df: pd.DataFrame, pdf: PdfPages) -> None:
    """Scatter plots of sample locations colored by key traits."""
    if "lat" not in df.columns or "lon" not in df.columns:
        return

    color_features = [
        "lc_entropy",
        "ndvi_mean",
        "elevation_mean",
        "overall_missing_frac",
        "sentinel2_l2a_entropy",
        "population_mean",
    ]
    present = [f for f in color_features if f in df.columns]
    sub = _subsample(df)

    for feat in present:
        fig, ax = plt.subplots(figsize=(14, 7))
        vals = sub[feat].fillna(0)
        sc = ax.scatter(
            sub["lon"],
            sub["lat"],
            c=vals,
            s=2,
            alpha=0.5,
            cmap="viridis",
            edgecolors="none",
            rasterized=True,
        )
        fig.colorbar(sc, ax=ax, shrink=0.7, label=feat)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        n_label = f" (showing {len(sub):,}/{len(df):,})" if len(sub) < len(df) else ""
        ax.set_title(f"Geographic Distribution — {feat}{n_label}", fontsize=12)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def plot_land_cover_composition(df: pd.DataFrame, pdf: PdfPages) -> None:
    """Stacked bar / pie of average land cover fractions."""
    present = [c for c in LAND_COVER_COLS if c in df.columns]
    if not present:
        return

    means = df[present].mean()
    labels = [c.replace("lc_frac_", "").replace("_", " ") for c in present]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    nonzero = means[means > 0.001]
    nonzero_labels = [labels[i] for i, v in enumerate(means) if v > 0.001]
    ax1.pie(
        nonzero, labels=nonzero_labels, autopct="%1.1f%%", textprops={"fontsize": 8}
    )
    ax1.set_title("Mean Land Cover Composition")

    # Bar chart
    ax2.barh(labels, means, color="steelblue")
    ax2.set_xlabel("Mean Fraction")
    ax2.set_title("Land Cover Fractions")
    ax2.tick_params(labelsize=8)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_modality_presence(df: pd.DataFrame, pdf: PdfPages) -> None:
    """Bar chart of modality availability across the dataset."""
    has_cols = sorted(
        [c for c in df.columns if c.startswith("has_") and df[c].mean() > 0]
    )
    if not has_cols:
        return

    means = df[has_cols].mean().sort_values(ascending=True)
    labels = [c.replace("has_", "") for c in means.index]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.3)))
    ax.barh(labels, means.values, color="steelblue")
    ax.set_xlabel("Fraction of Samples Present")
    ax.set_title("Modality Availability")
    ax.set_xlim(0, 1.05)
    for i, v in enumerate(means.values):
        ax.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_quality_overview(df: pd.DataFrame, pdf: PdfPages) -> None:
    """Overview of data quality metrics across modalities."""
    quality_cols = sorted(
        [
            c
            for c in df.columns
            if any(
                c.endswith(s)
                for s in [
                    "_missing_pixel_frac",
                    "_constant_frac",
                    "_bright_frac",
                    "_dark_frac",
                ]
            )
        ]
    )
    if not quality_cols:
        return

    means = df[quality_cols].mean().sort_values(ascending=False)
    # Only show non-trivial ones
    means = means[means > 0.001]
    if means.empty:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(means) * 0.35)))
    colors = []
    for c in means.index:
        if "missing" in c:
            colors.append("gray")
        elif "constant" in c:
            colors.append("orange")
        elif "bright" in c:
            colors.append("red")
        else:
            colors.append("navy")
    ax.barh(range(len(means)), means.values, color=colors)
    ax.set_yticks(range(len(means)))
    ax.set_yticklabels(means.index, fontsize=7)
    ax.set_xlabel("Mean Fraction")
    ax.set_title("Data Quality Issues (higher = worse)")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_pairs(df: pd.DataFrame, pdf: PdfPages) -> None:
    """Scatter plots of interesting feature pairs."""
    pairs = [
        ("lc_entropy", "sentinel2_l2a_entropy"),
        ("ndvi_mean", "elevation_mean"),
        ("lc_entropy", "edge_density"),
        ("abs_lat", "ndvi_seasonal_amplitude"),
        ("overall_missing_frac", "sentinel2_l2a_entropy"),
        ("population_mean", "lc_frac_built_up"),
        ("elevation_mean", "temperature_mean"),
        ("sentinel2_l2a_temporal_std", "frac_pixels_changed"),
    ]
    valid_pairs = [(a, b) for a, b in pairs if a in df.columns and b in df.columns]
    if not valid_pairs:
        return

    sub = _subsample(df)

    cols = 4
    rows = (len(valid_pairs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = np.array(axes).flatten()

    for i, (xa, ya) in enumerate(valid_pairs):
        ax = axes[i]
        ax.scatter(
            sub[xa],
            sub[ya],
            s=2,
            alpha=0.3,
            color="steelblue",
            edgecolors="none",
            rasterized=True,
        )
        ax.set_xlabel(xa, fontsize=7)
        ax.set_ylabel(ya, fontsize=7)
        ax.tick_params(labelsize=6)

    for i in range(len(valid_pairs), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Feature Pair Scatter Plots", fontsize=14, y=1.01)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_highly_correlated(df: pd.DataFrame, pdf: PdfPages, top_k: int = 30) -> None:
    """Find and display the most correlated feature pairs."""
    numeric = df.select_dtypes(include="number")
    # Drop near-constant columns
    numeric = numeric.loc[:, numeric.std() > 1e-10]
    corr = numeric.corr()

    # Extract upper triangle pairs
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    top = pairs[:top_k]

    fig, ax = plt.subplots(figsize=(12, max(4, top_k * 0.3)))
    labels = [f"{a} ↔ {b}" for a, b, _ in top]
    values = [c for _, _, c in top]
    colors = ["red" if v > 0 else "blue" for v in values]
    ax.barh(range(len(top)), values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Pearson Correlation")
    ax.set_title(f"Top {top_k} Most Correlated Feature Pairs")
    ax.axvline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate PDF analysis report from sample scores."""
    parser = argparse.ArgumentParser(description="Analyze sample scores")
    parser.add_argument("input", help="Path to scores parquet file")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: input_analysis.pdf or input_analysis_pngs/)",
    )
    parser.add_argument(
        "--top-k", type=int, default=30, help="Number of top correlations to show"
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Save as numbered PNGs in a directory instead of PDF",
    )
    args = parser.parse_args()

    input_path = args.input
    if args.png:
        output_path = args.output or input_path.rsplit(".", 1)[0] + "_analysis_pngs"
    else:
        output_path = args.output or input_path.rsplit(".", 1)[0] + "_analysis.pdf"

    logger.info(f"Loading {input_path}")
    df = (
        pd.read_parquet(input_path)
        if input_path.endswith(".parquet")
        else pd.read_csv(input_path)
    )
    logger.info(f"Loaded {df.shape[0]} samples, {df.shape[1]} columns")

    # Print summary stats to stdout
    stats = summary_table(df)
    print("\n" + "=" * 80)
    print(f"  DATASET SUMMARY  ({df.shape[0]} samples, {df.shape[1]} features)")
    print("=" * 80)
    print(stats.to_string())
    print()

    # Save summary CSV alongside
    stats_path = output_path.rsplit(".", 1)[0] + "_stats.csv"
    stats.to_csv(stats_path)
    logger.info(f"Summary stats saved to {stats_path}")

    # Generate report
    logger.info(f"Generating report -> {output_path}")
    saver_cls = PngSaver if args.png else PdfPages
    with saver_cls(output_path) as saver:
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.text(
            0.5,
            0.5,
            f"Sample Score Analysis\n{df.shape[0]} samples × {df.shape[1]} features\n{Path(input_path).name}",
            ha="center",
            va="center",
            fontsize=16,
            transform=ax.transAxes,
        )
        ax.axis("off")
        saver.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        plot_distributions(df, KEY_FEATURES, saver)
        plot_correlation_heatmap(df, KEY_FEATURES, saver)
        plot_highly_correlated(df, saver, top_k=args.top_k)
        plot_geographic(df, saver)
        plot_land_cover_composition(df, saver)
        plot_modality_presence(df, saver)
        plot_quality_overview(df, saver)
        plot_scatter_pairs(df, saver)

    logger.info(f"Done. Report: {output_path}")


if __name__ == "__main__":
    main()
