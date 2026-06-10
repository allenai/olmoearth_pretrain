"""Plot and summarize geographies for generated Presto OSM split candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRESTO_H5PY_DIR = Path(
    "/weka/dfive-default/helios/dataset/presto/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_"
    "worldcereal_worldcover_wri_canopy_height_map/469728"
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("olmoearth_pretrain/evals/datasets/splits/presto_osm_balanced"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/presto_osm_balanced_split_geographies"),
    )
    parser.add_argument("--h5py-dir", type=Path, default=PRESTO_H5PY_DIR)
    parser.add_argument("--bin-size-deg", type=float, default=5.0)
    return parser.parse_args()


def load_latlons(h5py_dir: Path) -> np.ndarray:
    """Load Presto lat/lon distribution."""
    return np.load(h5py_dir / "latlon_distribution.npy")


def variant_dirs(splits_dir: Path) -> list[Path]:
    """Return generated variant directories."""
    return sorted(path for path in splits_dir.iterdir() if path.is_dir())


def split_csvs(variant_dir: Path) -> list[Path]:
    """Return train/valid/test CSVs if present."""
    paths = []
    for split in ("train", "valid", "test"):
        path = variant_dir / f"{split}.csv"
        if path.exists():
            paths.append(path)
    return paths


def add_geography(df: pd.DataFrame, latlons: np.ndarray, bin_size_deg: float) -> pd.DataFrame:
    """Add lat/lon and coarse geographic bins to selected samples."""
    out = df.copy()
    out["lat"] = latlons[out["sample_index"].to_numpy(dtype=np.int64), 0]
    out["lon"] = latlons[out["sample_index"].to_numpy(dtype=np.int64), 1]
    out["lat_bin"] = np.floor(out["lat"] / bin_size_deg).astype(int)
    out["lon_bin"] = np.floor(out["lon"] / bin_size_deg).astype(int)
    out["geo_bin"] = out["lat_bin"].astype(str) + "," + out["lon_bin"].astype(str)
    return out


def summarize_geography(df: pd.DataFrame) -> dict:
    """Summarize geographic spread for a split."""
    if df.empty:
        return {
            "samples": 0,
            "unique_geo_bins": 0,
            "lat_min": None,
            "lat_max": None,
            "lon_min": None,
            "lon_max": None,
            "top_geo_bins": [],
        }
    top_bins = (
        df["geo_bin"]
        .value_counts()
        .head(10)
        .rename_axis("geo_bin")
        .reset_index(name="samples")
        .to_dict("records")
    )
    return {
        "samples": int(len(df)),
        "unique_geo_bins": int(df["geo_bin"].nunique()),
        "lat_min": float(df["lat"].min()),
        "lat_max": float(df["lat"].max()),
        "lon_min": float(df["lon"].min()),
        "lon_max": float(df["lon"].max()),
        "top_geo_bins": top_bins,
    }


def plot_variant(variant: str, split_to_df: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Plot split sample geographies."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = {"train": "tab:blue", "valid": "tab:orange", "test": "tab:green"}
    for split, df in split_to_df.items():
        if df.empty:
            continue
        ax.scatter(
            df["lon"],
            df["lat"],
            s=4,
            alpha=0.45,
            label=f"{split} ({len(df)})",
            color=colors[split],
        )
    ax.set_title(f"{variant} geographic distribution")
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.legend(markerscale=3)
    ax.grid(True, linewidth=0.2, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_geo_bin_hist(variant: str, all_df: pd.DataFrame, output_path: Path) -> None:
    """Plot samples per geographic bin."""
    counts = all_df["geo_bin"].value_counts().head(40)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(np.arange(len(counts)), counts.to_numpy())
    ax.set_title(f"{variant} top 5-degree geographic bins")
    ax.set_xlabel("Geo bin rank")
    ax.set_ylabel("Samples")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    """Generate geography artifacts."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latlons = load_latlons(args.h5py_dir)
    manifest: dict[str, dict] = {}

    for variant_dir in variant_dirs(args.splits_dir):
        variant = variant_dir.name
        variant_out = args.output_dir / variant
        variant_out.mkdir(parents=True, exist_ok=True)
        split_to_df = {}
        manifest[variant] = {"splits": {}}
        for path in split_csvs(variant_dir):
            split = path.stem
            df = pd.read_csv(path)
            df = add_geography(df, latlons, args.bin_size_deg)
            geo_path = variant_out / f"{split}_with_geography.csv"
            df.to_csv(geo_path, index=False)
            split_to_df[split] = df
            manifest[variant]["splits"][split] = {
                **summarize_geography(df),
                "path": str(geo_path),
            }

        all_df = pd.concat(split_to_df.values(), ignore_index=True)
        manifest[variant]["all"] = summarize_geography(all_df)
        map_path = variant_out / "geography_scatter.png"
        hist_path = variant_out / "geo_bin_histogram.png"
        plot_variant(variant, split_to_df, map_path)
        plot_geo_bin_hist(variant, all_df, hist_path)
        manifest[variant]["plots"] = {
            "scatter": str(map_path),
            "geo_bin_histogram": str(hist_path),
        }

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
