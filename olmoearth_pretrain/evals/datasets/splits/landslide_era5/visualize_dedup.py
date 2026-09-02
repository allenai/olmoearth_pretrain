"""Visualize the deduped landslide window distribution.

Produces a 3x3 figure:
  Row 1-2 (merged): World map of ~6k deduped windows colored by positive/negative.
  Row 3, col 1: Histogram of counts per location (country).
  Row 3, col 2: Histogram of counts per event year.
  Row 3, col 3: Histogram of counts per event type.

Usage:
    python visualize_dedup.py [--parquet PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PQ = SCRIPT_DIR / "deduped_windows.parquet"
DEFAULT_OUT = SCRIPT_DIR / "dedup_overview.png"

POINT_COLOR = "#d62728"

# Cap the bar axes so a few huge categories don't crush the small ones.
# Bars exceeding the cap are clipped, but their true total is still labeled.
BAR_CAP = 750

# Per-source-group colors for the stacked bar plots. Order controls stack order.
GROUP_COLORS = {
    "sen12_landslides": "#5a9bd5",
    "glc": "#ed7d31",
    "icimod": "#70ad47",
}
GROUP_FALLBACK_COLOR = "#999999"


def _groups_in(df: pd.DataFrame) -> list[str]:
    """Groups present in the data, ordered by the GROUP_COLORS palette first."""
    present = set(df["group"].unique())
    ordered = [g for g in GROUP_COLORS if g in present]
    ordered += sorted(g for g in present if g not in GROUP_COLORS)
    return ordered


def _stacked_crosstab(
    df: pd.DataFrame, category: str, groups: list[str]
) -> pd.DataFrame:
    """Counts of (category value x group), columns ordered as ``groups``."""
    ct = pd.crosstab(df[category].fillna("unknown"), df["group"])
    for g in groups:
        if g not in ct.columns:
            ct[g] = 0
    return ct[groups]


def _stacked_barh(
    ax: plt.Axes,
    ct: pd.DataFrame,
    groups: list[str],
    title: str,
    cap: int = BAR_CAP,
) -> None:
    """Horizontal stacked bars (one segment per group), with total labels.

    The x-axis is capped at ``cap``; bars beyond it are clipped but their true
    total is annotated (right-aligned inside the bar) so they stay readable.
    """
    y = np.arange(len(ct))
    left = np.zeros(len(ct))
    for g in groups:
        vals = ct[g].to_numpy()
        ax.barh(
            y,
            vals,
            left=left,
            color=GROUP_COLORS.get(g, GROUP_FALLBACK_COLOR),
            edgecolor="white",
            linewidth=0.5,
            label=g,
        )
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(ct.index, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(f"Window count (axis capped at {cap:,})")
    ax.set_title(title, fontsize=13, fontweight="bold")
    if cap:
        ax.set_xlim(0, cap)
    totals = ct.sum(axis=1).to_numpy()
    for yi, total in zip(y, totals):
        if cap and total > cap:
            ax.text(
                cap * 0.98,
                yi,
                f"{int(total):,}",
                va="center",
                ha="right",
                fontsize=9,
                color="white",
                fontweight="bold",
            )
        else:
            ax.text(
                total + cap * 0.01,
                yi,
                f"{int(total):,}",
                va="center",
                ha="left",
                fontsize=9,
            )


def _stacked_barv(
    ax: plt.Axes,
    ct: pd.DataFrame,
    groups: list[str],
    title: str,
    cap: int = BAR_CAP,
) -> None:
    """Vertical stacked bars (one segment per group), with total labels.

    The y-axis is capped at ``cap``; bars beyond it are clipped but their true
    total is annotated (inside the top of the bar) so they stay readable.
    """
    x = np.arange(len(ct))
    bottom = np.zeros(len(ct))
    for g in groups:
        vals = ct[g].to_numpy()
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=GROUP_COLORS.get(g, GROUP_FALLBACK_COLOR),
            edgecolor="white",
            linewidth=0.5,
            label=g,
        )
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in ct.index], fontsize=10)
    ax.set_xlabel("Event year")
    ax.set_ylabel(f"Window count (axis capped at {cap:,})")
    ax.set_title(title, fontsize=13, fontweight="bold")
    if cap:
        ax.set_ylim(0, cap)
    totals = ct.sum(axis=1).to_numpy()
    for xi, total in zip(x, totals):
        if cap and total > cap:
            ax.text(
                xi,
                cap * 0.98,
                f"{int(total):,}",
                ha="center",
                va="top",
                fontsize=9,
                color="white",
                fontweight="bold",
            )
        else:
            ax.text(
                xi,
                total + cap * 0.01,
                f"{int(total):,}",
                ha="center",
                va="bottom",
                fontsize=9,
            )


def main() -> None:
    """CLI entry point for plotting the deduped window distribution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=str, default=str(DEFAULT_PQ))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)

    fig = plt.figure(figsize=(20, 14), dpi=150)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.30)

    # --- Rows 1-2: map spanning all 6 cells -----------------------------------
    ax_map = fig.add_subplot(gs[0:2, :], projection=ccrs.Robinson())
    ax_map.set_global()
    ax_map.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none")
    ax_map.add_feature(cfeature.OCEAN, facecolor="#e6f2ff")
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.4, color="#888888")
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.3, color="#bbbbbb")

    # Deduplicate to unique locations (pos/neg twins share the same lat/lon).
    df_unique = df.drop_duplicates(subset=["latitude", "longitude"])
    ax_map.scatter(
        df_unique["longitude"],
        df_unique["latitude"],
        c=POINT_COLOR,
        s=8,
        alpha=0.6,
        transform=ccrs.PlateCarree(),
        zorder=2,
        edgecolors="none",
    )

    n_pos = (df["window_type"] == "positive").sum()
    n_neg = (df["window_type"] == "negative").sum()
    ax_map.set_title(
        f"Deduped Landslide Windows — {len(df):,} total  "
        f"({n_pos:,} pos / {n_neg:,} neg)  •  "
        f"{len(df_unique):,} unique locations",
        fontsize=14,
        fontweight="bold",
    )

    groups = _groups_in(df)

    # --- Row 3, col 1: location counts (stacked by source) ---------------------
    ax_loc = fig.add_subplot(gs[2, 0])
    loc_totals = df["location"].fillna("unknown").value_counts()
    top_n = 12
    keep = list(loc_totals.head(top_n).index)
    loc_ct = _stacked_crosstab(df, "location", groups)
    loc_ct = loc_ct.reindex(keep)
    if len(loc_totals) > top_n:
        rest = _stacked_crosstab(
            df[~df["location"].fillna("unknown").isin(keep)], "location", groups
        ).sum(axis=0)
        loc_ct.loc["other"] = rest
    _stacked_barh(ax_loc, loc_ct, groups, "By location")

    # --- Row 3, col 2: event year histogram (stacked by source) ----------------
    ax_year = fig.add_subplot(gs[2, 1])
    year_ct = _stacked_crosstab(df, "event_year", groups).sort_index()
    _stacked_barv(ax_year, year_ct, groups, "By event year")

    # --- Row 3, col 3: event type histogram (stacked by source) ----------------
    ax_type = fig.add_subplot(gs[2, 2])
    type_ct = _stacked_crosstab(df, "event_type", groups)
    type_ct = type_ct.loc[type_ct.sum(axis=1).sort_values(ascending=False).index]
    _stacked_barh(ax_type, type_ct, groups, "By event type", cap=500)

    # Shared legend for the source-group coloring.
    handles, labels = ax_loc.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Source group",
        loc="lower center",
        ncol=len(groups),
        fontsize=10,
        title_fontsize=11,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    print(f"Saved to {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
