"""Plot embedding cluster assignments on a geographic map.

Reads an index CSV (with lat/lon per sample) and an integer label .npy
file, then produces a scatter plot of points colored by cluster.

Usage example:
  python scripts/embeddings/visualize_map.py \
    --index-csv  /path/to/embeds/index.csv \
    --labels-npy /path/to/embeds/_cluster/labels_kmeans_int.npy \
    --output-dir /path/to/output \
    --output-name cluster_map.png \
    --subsample  200000 \
    --darkmode
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for geographic cluster-map plotting."""
    p = argparse.ArgumentParser(
        description="Plot cluster labels on a lat/lon map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--index-csv",
        required=True,
        help="CSV with columns: sample_idx, shard, row, lat, lon",
    )
    p.add_argument(
        "--labels-npy", required=True, help="Integer cluster labels .npy  (N,)"
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write the output image. "
        "Defaults to the directory containing --labels-npy.",
    )
    p.add_argument(
        "--output-name",
        default="cluster_map.png",
        help="Filename for the output image (written inside --output-dir)",
    )
    p.add_argument(
        "--subsample",
        type=int,
        default=0,
        help="Plot at most this many points (0 = all). "
        "Random subset, preserving cluster proportions.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument(
        "--point-size", type=float, default=0.3, help="Marker size for scatter points"
    )
    p.add_argument("--alpha", type=float, default=0.5, help="Marker opacity")
    p.add_argument("--darkmode", action="store_true")
    p.add_argument(
        "--lat-bounds",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Denormalize lat from [0,1] to [MIN, MAX]. "
        "Leave unset if lat is already in degrees.",
    )
    p.add_argument(
        "--lon-bounds",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Denormalize lon from [0,1] to [MIN, MAX]. "
        "Leave unset if lon is already in degrees.",
    )
    p.add_argument(
        "--figsize", type=float, nargs=2, default=[16, 10], metavar=("W", "H")
    )
    p.add_argument("--no-legend", action="store_true", help="Hide the legend")
    p.add_argument("--no-title", action="store_true", help="Hide the title")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Render and save the geographic scatter plot for cluster labels."""
    args = parse_args(argv)

    # -- Resolve output path ------------------------------------------------
    output_dir = args.output_dir or os.path.dirname(args.labels_npy)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.output_name)

    # -- Load data ----------------------------------------------------------
    print(f"[load] Reading {args.index_csv} ...")
    df = pd.read_csv(args.index_csv)
    lat = df["lat"].values.astype(np.float32)
    lon = df["lon"].values.astype(np.float32)

    labels = np.load(args.labels_npy)
    assert labels.shape[0] == lat.shape[0], (
        f"Label count ({labels.shape[0]}) != row count ({lat.shape[0]})"
    )
    print(f"[load] {lat.shape[0]:,} points, {len(np.unique(labels))} unique labels")

    # -- Denormalize if requested ------------------------------------------
    if args.lat_bounds is not None:
        lo, hi = args.lat_bounds
        lat = lat * (hi - lo) + lo
    if args.lon_bounds is not None:
        lo, hi = args.lon_bounds
        lon = lon * (hi - lo) + lo

    # -- Subsample ---------------------------------------------------------
    if 0 < args.subsample < lat.shape[0]:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(lat.shape[0], size=args.subsample, replace=False)
        idx.sort()
        lat, lon, labels = lat[idx], lon[idx], labels[idx]
        print(f"[subsample] Kept {lat.shape[0]:,} points")

    # -- Color palette -----------------------------------------------------
    unique_labels = np.unique(labels)
    has_noise = -1 in unique_labels
    real_labels = sorted(lab for lab in unique_labels if lab != -1)
    n_real = len(real_labels)

    try:
        import glasbey

        palette = glasbey.create_palette(
            palette_size=max(n_real, 2),
            lightness_bounds=(40, 80),
            chroma_bounds=(50, 100),
        )
        cmap_dict = {lab: palette[i] for i, lab in enumerate(real_labels)}
    except ImportError:
        base_cmap = plt.cm.get_cmap("tab20", max(n_real, 2))
        cmap_dict = {lab: base_cmap(i) for i, lab in enumerate(real_labels)}

    if has_noise:
        cmap_dict[-1] = (0.5, 0.5, 0.5, 0.15)

    colors = np.array([cmap_dict[label_id] for label_id in labels])

    # -- Plot --------------------------------------------------------------
    bg = "#111111" if args.darkmode else "#ffffff"
    fg = "#eeeeee" if args.darkmode else "#222222"

    fig, ax = plt.subplots(figsize=args.figsize, facecolor=bg)
    ax.set_facecolor(bg)

    # Try adding coastlines via cartopy
    use_cartopy = False
    lat_is_degrees = args.lat_bounds is not None or (
        lat.min() < -0.5 or lat.max() > 1.5
    )
    lon_is_degrees = args.lon_bounds is not None or (
        lon.min() < -0.5 or lon.max() > 1.5
    )

    if lat_is_degrees and lon_is_degrees:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature

            ax.remove()
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(), facecolor=bg)
            coast_color = "#444444" if args.darkmode else "#cccccc"
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor=coast_color)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor=coast_color)
            use_cartopy = True
        except ImportError:
            pass

    shuffle = np.random.RandomState(args.seed).permutation(len(lat))
    ax.scatter(
        lon[shuffle],
        lat[shuffle],
        c=colors[shuffle],
        s=args.point_size,
        alpha=args.alpha,
        edgecolors="none",
        rasterized=True,
        transform=ccrs.PlateCarree() if use_cartopy else ax.transData,
    )

    # Legend
    if not args.no_legend:
        handles = []
        for lab in sorted(unique_labels):
            name = "Noise" if lab == -1 else f"Cluster {lab}"
            h = plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=cmap_dict[lab],
                markersize=6,
                label=name,
            )
            handles.append(h)
        ax.legend(
            handles=handles,
            loc="lower left",
            fontsize=7,
            framealpha=0.6,
            ncol=max(1, n_real // 10),
            facecolor=bg,
            edgecolor=fg,
            labelcolor=fg,
        )

    xlabel = "Longitude" if lon_is_degrees else "lon (normalized)"
    ylabel = "Latitude" if lat_is_degrees else "lat (normalized)"
    ax.set_xlabel(xlabel, color=fg, fontsize=9)
    ax.set_ylabel(ylabel, color=fg, fontsize=9)
    ax.tick_params(colors=fg, labelsize=7)

    if not args.no_title:
        method = os.path.basename(args.labels_npy).replace("_int.npy", "")
        ax.set_title(
            f"Cluster map — {method}  ({lat.shape[0]:,} points)",
            color=fg,
            fontsize=12,
            pad=10,
        )

    plt.tight_layout()

    fig.savefig(output_path, dpi=args.dpi, facecolor=bg, bbox_inches="tight")
    print(f"[save] {output_path}")

    plt.close(fig)
    print("[done]")


if __name__ == "__main__":
    main()
