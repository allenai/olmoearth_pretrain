"""Smoke-check visualization of ERA5 climate-stratified sampling windows.

Reads rslearn windows directly from the dataset, extracts their center
coordinates (from UTM projection + bounds) and stratification metadata
(from window options), then produces a 3-row x 3-col figure:

  Row 1-2 (spanning 2x3): Global scatter map of window centers.
  Row 3, col 1: Histogram of windows per Köppen-Geiger class (30 classes).
  Row 3, col 2: Histogram of windows per elevation band.
  Row 3, col 3: Histogram of windows per latitude band.

Usage:
    python -m olmoearth_pretrain.dataset_creation.era5_sample.visualize_windows \
        --ds-path /weka/dfive-default/helios/dataset/era5enc_pretrain/rslearn_dataset \
        --output /weka/dfive-default/helios/dataset/era5enc_pretrain/metadata/sampling_smokecheck.png \
        --workers 16
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
from rslearn.dataset import Dataset, Window
from upath import UPath

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# All 30 Köppen-Geiger classes in canonical order
KOPPEN_ORDER = [
    "Af",
    "Am",
    "Aw",
    "BWh",
    "BWk",
    "BSh",
    "BSk",
    "Csa",
    "Csb",
    "Csc",
    "Cwa",
    "Cwb",
    "Cwc",
    "Cfa",
    "Cfb",
    "Cfc",
    "Dsa",
    "Dsb",
    "Dsc",
    "Dsd",
    "Dwa",
    "Dwb",
    "Dwc",
    "Dwd",
    "Dfa",
    "Dfb",
    "Dfc",
    "Dfd",
    "ET",
    "EF",
]

ELEV_ORDER = ["<250", "250-750", "750-1500", "1500-3000", ">3000"]
LAT_ORDER = ["[-90,-60)", "[-60,-30)", "[-30,0)", "[0,30)", "[30,60)", "[60,90]"]


def _window_center_lonlat(window: Window) -> tuple[float, float]:
    """Compute the lon/lat center of a window from its UTM projection and bounds."""
    cx_px = (window.bounds[0] + window.bounds[2]) / 2.0
    cy_px = (window.bounds[1] + window.bounds[3]) / 2.0

    cx_m = cx_px * window.projection.x_resolution
    cy_m = cy_px * window.projection.y_resolution

    transformer = Transformer.from_crs(
        window.projection.crs, "EPSG:4326", always_xy=True
    )
    lon, lat = transformer.transform(cx_m, cy_m)
    return lon, lat


def visualize_windows(
    ds_path: str,
    output_path: str,
    group: str | None = None,
    workers: int = 16,
    max_windows: int | None = None,
) -> None:
    """Generate the smoke-check visualization figure.

    Args:
        ds_path: Path to the rslearn dataset.
        output_path: Path to save the output PNG.
        group: Optional window group filter.
        workers: Number of workers for loading windows.
        max_windows: Optional cap on number of windows to process.
    """
    logger.info("Loading windows from %s", ds_path)
    dataset = Dataset(UPath(ds_path))

    groups = [group] if group else None
    windows = dataset.load_windows(groups=groups)
    logger.info("Loaded %d windows", len(windows))

    if max_windows and len(windows) > max_windows:
        rng = np.random.default_rng(0)
        indices = rng.choice(len(windows), size=max_windows, replace=False)
        windows = [windows[i] for i in indices]
        logger.info("Subsampled to %d windows", len(windows))

    # Extract data from windows
    lons: list[float] = []
    lats: list[float] = []
    koppen_classes: list[str] = []
    elev_bands: list[str] = []
    lat_bands: list[str] = []
    roles: list[str] = []

    for i, w in enumerate(windows):
        if i % 10000 == 0 and i > 0:
            logger.info("Processing window %d / %d", i, len(windows))

        try:
            lon, lat = _window_center_lonlat(w)
        except Exception as e:
            logger.debug("Skipping window %d: failed to compute center (%s)", i, e)
            continue

        lons.append(lon)
        lats.append(lat)
        koppen_classes.append(w.options.get("koppen_class", "unknown"))
        elev_bands.append(w.options.get("elev_band_label", "unknown"))
        lat_bands.append(w.options.get("lat_band_label", "unknown"))
        roles.append(w.options.get("role", "primary"))

    logger.info("Extracted coordinates for %d windows", len(lons))

    lons_arr = np.array(lons)
    lats_arr = np.array(lats)

    # --- Build figure ---
    fig = plt.figure(figsize=(18, 14))

    # Use GridSpec: rows 0-1 span the map (2 rows x 3 cols), row 2 has 3 histograms
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Map subplot spanning rows 0-1, cols 0-2
    ax_map = fig.add_subplot(gs[0:2, :])

    # Color by role
    colors = []
    for r in roles:
        if r == "overlap_secondary":
            colors.append("#e74c3c")
        elif r == "overlap_primary":
            colors.append("#f39c12")
        else:
            colors.append("#3498db")

    ax_map.scatter(
        lons_arr,
        lats_arr,
        s=0.3,
        alpha=0.4,
        c=colors,
        rasterized=True,
    )
    ax_map.set_xlim(-180, 180)
    ax_map.set_ylim(-90, 90)
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.set_title(
        f"ERA5 Climate-Stratified Windows (n={len(lons):,})",
        fontsize=14,
        fontweight="bold",
    )
    ax_map.set_aspect("equal")
    ax_map.grid(True, alpha=0.3)

    # Add simple coastline reference lines
    ax_map.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_map.axvline(0, color="gray", linewidth=0.5, linestyle="--")

    # Legend for roles
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#3498db",
            markersize=8,
            label="Primary",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#f39c12",
            markersize=8,
            label="Overlap primary",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#e74c3c",
            markersize=8,
            label="Overlap secondary",
        ),
    ]
    ax_map.legend(handles=legend_elements, loc="lower left", fontsize=9)

    # --- Histograms ---
    # 1. Köppen-Geiger classes
    ax_koppen = fig.add_subplot(gs[2, 0])
    koppen_counts = {k: 0 for k in KOPPEN_ORDER}
    for kc in koppen_classes:
        if kc in koppen_counts:
            koppen_counts[kc] += 1
    bars = [koppen_counts[k] for k in KOPPEN_ORDER]
    ax_koppen.bar(range(len(KOPPEN_ORDER)), bars, color="#2ecc71", edgecolor="none")
    ax_koppen.set_xticks(range(len(KOPPEN_ORDER)))
    ax_koppen.set_xticklabels(KOPPEN_ORDER, rotation=90, fontsize=7)
    ax_koppen.set_xlabel("Köppen-Geiger Class")
    ax_koppen.set_ylabel("Window Count")
    ax_koppen.set_title("Distribution by Climate Class")

    # 2. Elevation bands
    ax_elev = fig.add_subplot(gs[2, 1])
    elev_counts = {e: 0 for e in ELEV_ORDER}
    for eb in elev_bands:
        if eb in elev_counts:
            elev_counts[eb] += 1
    bars_e = [elev_counts[e] for e in ELEV_ORDER]
    ax_elev.bar(range(len(ELEV_ORDER)), bars_e, color="#9b59b6", edgecolor="none")
    ax_elev.set_xticks(range(len(ELEV_ORDER)))
    ax_elev.set_xticklabels(ELEV_ORDER, fontsize=9)
    ax_elev.set_xlabel("Elevation Band (m)")
    ax_elev.set_ylabel("Window Count")
    ax_elev.set_title("Distribution by Elevation")

    # 3. Latitude bands
    ax_lat = fig.add_subplot(gs[2, 2])
    lat_counts = {lb: 0 for lb in LAT_ORDER}
    for lb in lat_bands:
        if lb in lat_counts:
            lat_counts[lb] += 1
    bars_l = [lat_counts[lb] for lb in LAT_ORDER]
    ax_lat.bar(range(len(LAT_ORDER)), bars_l, color="#e67e22", edgecolor="none")
    ax_lat.set_xticks(range(len(LAT_ORDER)))
    ax_lat.set_xticklabels(LAT_ORDER, fontsize=9)
    ax_lat.set_xlabel("Latitude Band")
    ax_lat.set_ylabel("Window Count")
    ax_lat.set_title("Distribution by Latitude")

    # Save
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved visualization to %s", output_p)


def main() -> None:
    """Run the window visualization smoke-check CLI."""
    parser = argparse.ArgumentParser(
        description="Smoke-check visualization of ERA5 sampling windows."
    )
    parser.add_argument(
        "--ds-path",
        type=str,
        required=True,
        help="Path to the rslearn dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/weka/dfive-default/helios/dataset/era5enc_pretrain/metadata/sampling_smokecheck.png",
        help="Output path for the PNG figure.",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Optional window group filter.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of workers for loading windows.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional maximum number of windows to process (for quick checks).",
    )
    args = parser.parse_args()

    visualize_windows(
        ds_path=args.ds_path,
        output_path=args.output,
        group=args.group,
        workers=args.workers,
        max_windows=args.max_windows,
    )


if __name__ == "__main__":
    main()
