"""Visualize geographic distribution of 150k samples per selection strategy.

Reads the selection parquet and produces a 2x3 grid of globe scatter plots,
one subplot per strategy, for side-by-side comparison.

Usage:
  python visualize_strategy_map.py \
    --parquet /path/to/selection_top250000.parquet \
    --output  strategy_map.png \
    --darkmode
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STRATEGIES = [
    "in_top_combined",
    "in_top_solo_novelty",
    "in_top_solo_xglobal_bridge",
    "in_top_solo_sparse_infill",
    "in_top_solo_xlocal_bridge",
    "in_top_solo_prototypes",
]

NICE_NAMES = {
    "in_top_combined": "Combined",
    "in_top_solo_novelty": "Novelty",
    "in_top_solo_xglobal_bridge": "X-Global Bridge",
    "in_top_solo_sparse_infill": "Sparse Infill",
    "in_top_solo_xlocal_bridge": "X-Local Bridge",
    "in_top_solo_prototypes": "Prototypes",
}

STRATEGY_COLORS = {
    "in_top_combined": "#e74c3c",
    "in_top_solo_novelty": "#3498db",
    "in_top_solo_xglobal_bridge": "#2ecc71",
    "in_top_solo_sparse_infill": "#f39c12",
    "in_top_solo_xlocal_bridge": "#9b59b6",
    "in_top_solo_prototypes": "#1abc9c",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for strategy-map visualization."""
    p = argparse.ArgumentParser(
        description="Compare geographic spread of selection strategies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--parquet",
        default="/weka/dfive-default/rslearn-eai/datasets/globe_land_grid/"
        "s50ix24_embeddings/_scores/selection_top250000.parquet",
        help="Path to the selection parquet file",
    )
    p.add_argument(
        "--output",
        default="strategy_map.png",
        help="Output image path",
    )
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--point-size", type=float, default=0.15)
    p.add_argument("--alpha", type=float, default=0.4)
    p.add_argument("--darkmode", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Render a geographic scatter plot comparing selection strategies."""
    args = parse_args(argv)

    print(f"[load] Reading {args.parquet} ...")
    df = pd.read_parquet(args.parquet)
    print(f"[load] {len(df):,} total rows")

    for s in STRATEGIES:
        n = df[s].sum()
        print(f"  {NICE_NAMES[s]:>20s}: {n:,} samples selected")

    bg = "#111111" if args.darkmode else "#ffffff"
    fg = "#eeeeee" if args.darkmode else "#222222"
    coast_color = "#444444" if args.darkmode else "#aaaaaa"

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        use_cartopy = True
    except ImportError:
        use_cartopy = False
        print("[warn] cartopy not installed — falling back to plain axes")

    nrows, ncols = 2, 3
    fig_kw = dict(figsize=(24, 12), facecolor=bg)

    if use_cartopy:
        fig, axes = plt.subplots(
            nrows,
            ncols,
            subplot_kw={"projection": ccrs.Robinson()},
            **fig_kw,
        )
    else:
        fig, axes = plt.subplots(nrows, ncols, **fig_kw)

    axes_flat = axes.flatten()
    rng = np.random.RandomState(args.seed)

    for ax, strategy in zip(axes_flat, STRATEGIES):
        ax.set_facecolor(bg)

        mask = df[strategy] == 1
        sub = df.loc[mask, ["lat", "lon"]].copy()
        n_pts = len(sub)

        shuffle = rng.permutation(n_pts)
        lats = sub["lat"].values[shuffle]
        lons = sub["lon"].values[shuffle]

        if use_cartopy:
            ax.set_global()
            ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor=coast_color)
            ax.add_feature(
                cfeature.BORDERS, linewidth=0.2, edgecolor=coast_color, alpha=0.5
            )
            ax.add_feature(
                cfeature.LAND,
                facecolor="#1a1a1a" if args.darkmode else "#f0f0f0",
                zorder=0,
            )
            ax.add_feature(
                cfeature.OCEAN,
                facecolor="#0a0a0a" if args.darkmode else "#ddeeff",
                zorder=0,
            )
            transform = ccrs.PlateCarree()
        else:
            transform = ax.transData

        ax.scatter(
            lons,
            lats,
            c=STRATEGY_COLORS[strategy],
            s=args.point_size,
            alpha=args.alpha,
            edgecolors="none",
            rasterized=True,
            transform=transform,
        )

        title = f"{NICE_NAMES[strategy]}  ({n_pts:,} pts)"
        ax.set_title(title, color=fg, fontsize=13, fontweight="bold", pad=8)

        if not use_cartopy:
            ax.set_xlabel("Longitude", color=fg, fontsize=8)
            ax.set_ylabel("Latitude", color=fg, fontsize=8)
            ax.tick_params(colors=fg, labelsize=6)
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_aspect("equal")

    fig.suptitle(
        "Selection Strategy — Geographic Distribution (150k each)",
        color=fg,
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.output, dpi=args.dpi, facecolor=bg, bbox_inches="tight")
    print(f"[save] {args.output}")
    plt.close(fig)
    print("[done]")


if __name__ == "__main__":
    main()
