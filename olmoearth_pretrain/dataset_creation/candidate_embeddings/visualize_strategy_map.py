"""Visualize geographic distribution of top-scoring samples per strategy.

Two modes:

1. **All strategies, one suffix** (default):
   Produces a 2x3 grid comparing combined + 5 solo strategies for a single
   score suffix.

     python olmoearth_pretrain/dataset_creation/candidate_embeddings/visualize_strategy_map.py --suffix normalized_score --top 150000 --output map_normscores.png

2. **One strategy, all flavors** (``--strategy``):
   Produces a grid comparing every available score suffix for a single
   strategy, auto-detected from the parquet columns.

     python olmoearth_pretrain/dataset_creation/candidate_embeddings/visualize_strategy_map.py --strategy novelty --top 150000 --darkmode --output map_novelty.png
"""

from __future__ import annotations

import argparse
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _schema import STRATEGIES as _BASE_STRATEGIES

STRATEGIES = ["combined", *_BASE_STRATEGIES]

NICE_NAMES = {
    "combined": "Combined",
    "novelty": "Novelty",
    "xglobal_bridge": "X-Global Bridge",
    "sparse_infill": "Sparse Infill",
    "xlocal_bridge": "X-Local Bridge",
    "prototypes": "Prototypes",
}

STRATEGY_COLORS = {
    "combined": "#e74c3c",
    "novelty": "#3498db",
    "xglobal_bridge": "#2ecc71",
    "sparse_infill": "#f39c12",
    "xlocal_bridge": "#9b59b6",
    "prototypes": "#1abc9c",
}


def score_column(strategy: str, suffix: str) -> str:
    """Return the parquet column name for a given strategy and score suffix."""
    if strategy == "combined":
        return "combined_score"
    return f"{strategy}_{suffix}"


def detect_suffixes(df: pd.DataFrame, strategy: str) -> list[str]:
    """Find all score-column suffixes present for *strategy* in *df*.

    For ``combined`` returns ``["score"]`` (the single ``combined_score`` col).
    For other strategies, matches ``{strategy}_*`` and strips the prefix.
    """
    if strategy == "combined":
        return ["score"] if "combined_score" in df.columns else []
    prefix = f"{strategy}_"
    suffixes = [c[len(prefix) :] for c in sorted(df.columns) if c.startswith(prefix)]
    suffixes = [s for s in suffixes if s not in ("weight", "raw_score")]
    return suffixes


def _suffix_nice_name(suffix: str) -> str:
    """Human-friendly label for a score suffix."""
    return suffix.replace("_", " ").title()


def _grid_shape(n: int) -> tuple[int, int]:
    """Return (nrows, ncols) for *n* subplots, preferring wider layouts."""
    if n <= 3:
        return 1, n
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def _make_figure(
    nrows: int,
    ncols: int,
    use_cartopy: bool,
    bg: str,
    ccrs_mod: Any | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Create a figure with the right projection and return (fig, flat axes)."""
    fig_kw = dict(figsize=(8 * ncols, 6 * nrows), facecolor=bg)

    if use_cartopy:
        assert ccrs_mod is not None, "cartopy.crs required when use_cartopy=True"
        fig, axes = plt.subplots(
            nrows,
            ncols,
            subplot_kw={"projection": ccrs_mod.Robinson()},
            **fig_kw,
        )
    else:
        fig, axes = plt.subplots(nrows, ncols, **fig_kw)

    axes_flat = np.atleast_1d(axes).flatten().tolist()
    return fig, axes_flat


def _plot_one(
    ax: plt.Axes,
    df: pd.DataFrame,
    col: str,
    top_k: int,
    *,
    color: str,
    label: str,
    point_size: float,
    alpha: float,
    darkmode: bool,
    use_cartopy: bool,
    rng: np.random.RandomState,
) -> int:
    """Plot top-K samples by *col* on *ax*. Returns the point count."""
    bg = "#111111" if darkmode else "#ffffff"
    coast_color = "#444444" if darkmode else "#aaaaaa"
    fg = "#eeeeee" if darkmode else "#222222"
    ax.set_facecolor(bg)

    k = min(top_k, len(df))
    sub = df.nlargest(k, col)[["lat", "lon"]].copy()
    n_pts = len(sub)

    shuffle = rng.permutation(n_pts)
    lats = sub["lat"].values[shuffle]
    lons = sub["lon"].values[shuffle]

    if use_cartopy:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor=coast_color)
        ax.add_feature(
            cfeature.BORDERS, linewidth=0.2, edgecolor=coast_color, alpha=0.5
        )
        ax.add_feature(
            cfeature.LAND,
            facecolor="#1a1a1a" if darkmode else "#f0f0f0",
            zorder=0,
        )
        ax.add_feature(
            cfeature.OCEAN,
            facecolor="#0a0a0a" if darkmode else "#ddeeff",
            zorder=0,
        )
        transform = ccrs.PlateCarree()
    else:
        transform = ax.transData

    ax.scatter(
        lons,
        lats,
        c=color,
        s=point_size,
        alpha=alpha,
        edgecolors="none",
        rasterized=True,
        transform=transform,
    )

    ax.set_title(
        f"{label}  ({n_pts:,} pts)", color=fg, fontsize=13, fontweight="bold", pad=8
    )

    if not use_cartopy:
        ax.set_xlabel("Longitude", color=fg, fontsize=8)
        ax.set_ylabel("Latitude", color=fg, fontsize=8)
        ax.tick_params(colors=fg, labelsize=6)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_aspect("equal")

    return n_pts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for strategy-map visualization."""
    p = argparse.ArgumentParser(
        description="Compare geographic spread of selection strategies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--parquet",
        default="/weka/dfive-default/rslearn-eai/datasets/globe_land_grid/"
        "s50ix24_embeddings/_scores/combined_acquisition_scores.parquet",
        help="Path to the combined acquisition-scores parquet file",
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--suffix",
        default=None,
        help="Score column suffix (all-strategies mode). "
        "e.g. normalized_score, diverse_score_p90, diverse_score_p95",
    )
    mode.add_argument(
        "--strategy",
        default=None,
        choices=STRATEGIES,
        help="Show a single strategy across all available score flavors",
    )

    p.add_argument(
        "--top",
        type=int,
        default=150_000,
        help="Number of top-scoring samples to display per subplot",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output image path (auto-generated if omitted)",
    )
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--point-size", type=float, default=0.15)
    p.add_argument("--alpha", type=float, default=0.4)
    p.add_argument("--darkmode", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Render geographic scatter plots comparing strategies or flavors."""
    args = parse_args(argv)

    # Default to all-strategies mode with normalized_score if neither flag set
    if args.strategy is None and args.suffix is None:
        args.suffix = "normalized_score"

    print(f"[load] Reading {args.parquet} ...")
    df = pd.read_parquet(args.parquet)
    print(f"[load] {len(df):,} total rows")

    bg = "#111111" if args.darkmode else "#ffffff"
    fg = "#eeeeee" if args.darkmode else "#222222"

    try:
        import cartopy.crs as ccrs

        use_cartopy = True
        ccrs_mod = ccrs
    except ImportError:
        use_cartopy = False
        ccrs_mod = None
        print("[warn] cartopy not installed — falling back to plain axes")

    rng = np.random.RandomState(args.seed)

    # ── Single-strategy mode: one strategy, N flavor subplots ────────────
    if args.strategy is not None:
        suffixes = detect_suffixes(df, args.strategy)
        if not suffixes:
            raise ValueError(
                f"No score columns found for strategy '{args.strategy}'. "
                f"Available columns: {sorted(df.columns.tolist())}"
            )
        print(f"[conf] strategy={args.strategy}  top={args.top:,}  flavors={suffixes}")
        if args.output is None:
            args.output = f"strategy_map_{args.strategy}_flavors.png"

        nrows, ncols = _grid_shape(len(suffixes))
        fig, axes_flat = _make_figure(nrows, ncols, use_cartopy, bg, ccrs_mod)
        color = STRATEGY_COLORS[args.strategy]
        nice = NICE_NAMES[args.strategy]

        for ax, suffix in zip(axes_flat, suffixes):
            col = score_column(args.strategy, suffix)
            label = f"{nice} — {_suffix_nice_name(suffix)}"
            n = _plot_one(
                ax,
                df,
                col,
                args.top,
                color=color,
                label=label,
                point_size=args.point_size,
                alpha=args.alpha,
                darkmode=args.darkmode,
                use_cartopy=use_cartopy,
                rng=rng,
            )
            print(f"  {label:>40s}: top {n:,} by {col}")

        for ax in axes_flat[len(suffixes) :]:
            ax.set_visible(False)

        fig.suptitle(
            f"{nice} — Top {args.top:,} across score flavors",
            color=fg,
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

    # ── All-strategies mode: 6 strategies, one suffix ────────────────────
    else:
        print(f"[conf] suffix={args.suffix}  top={args.top:,}")
        for s in STRATEGIES:
            col = score_column(s, args.suffix)
            if col not in df.columns:
                raise KeyError(
                    f"Column '{col}' not found in parquet. "
                    f"Available columns: {sorted(df.columns.tolist())}"
                )

        if args.output is None:
            args.output = f"strategy_map_{args.suffix}.png"

        nrows, ncols = 2, 3
        fig, axes_flat = _make_figure(nrows, ncols, use_cartopy, bg, ccrs_mod)

        for ax, strategy in zip(axes_flat, STRATEGIES):
            col = score_column(strategy, args.suffix)
            label = NICE_NAMES[strategy]
            n = _plot_one(
                ax,
                df,
                col,
                args.top,
                color=STRATEGY_COLORS[strategy],
                label=label,
                point_size=args.point_size,
                alpha=args.alpha,
                darkmode=args.darkmode,
                use_cartopy=use_cartopy,
                rng=rng,
            )
            print(f"  {label:>20s}: top {n:,} by {col}")

        suffix_label = _suffix_nice_name(args.suffix)
        fig.suptitle(
            f"Top {args.top:,} by {suffix_label} — Geographic Distribution",
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
