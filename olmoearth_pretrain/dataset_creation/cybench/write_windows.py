"""Stage 3: merge partials into per-region series and write rslearn windows.

Step A (``series``): for every spatial chunk, concatenate its time-chunk
partials from ``aggregate_era5`` into one ``(T_all, 14)`` float32 series per
region (``NODATA`` where nothing was fetched or no valid cell existed) and
save it under ``<agg>/series_<variant>/<ridx>.npy``. Regions straddling a
chunk boundary are merged from their partial sums / weight totals.

Step B (``windows``): for every CY-Bench label ``(crop, cc, adm_id,
harvest_year)`` take the 448 days ending at the crop-calendar end of season
and write a materialized rslearn window:

* group ``<crop>``, name ``<crop>_<cc>_<adm_id>_<year>``;
* WGS84 projection at 0.1 deg/px, bounds = the ERA5 cell containing the
  region's representative point (the layer is 1x1, so only the time range and
  tags carry information);
* raster layer ``era5_daily``: ``(14, 448, 1, 1)`` float32 ``NumpyRasterFormat``
  with per-day timestamps, nodata ``-9999``;
* vector layer ``label``: one point feature with ``yield`` (t/ha) and metadata;
* window options / tags: ``crop, country_code, adm_id, harvest_year, split``.

The dataset's ``config.json`` is written by this script (layers have no
data_source; everything is pre-materialized).

Usage::

    python -m olmoearth_pretrain.dataset_creation.cybench.write_windows series --agg-dir ... --workers 8
    python -m olmoearth_pretrain.dataset_creation.cybench.write_windows windows --agg-dir ... \
        --labels-root ... --ds-path /weka/.../cybench/rslearn_dataset --workers 16
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import re
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    DEFAULT_LABELS_ROOT,
    DEFAULT_ROOT,
    ERA5_LAYER,
    ERA5L_BANDS,
    LABEL_LAYER,
    NODATA,
    WINDOW_DAYS,
    list_crop_countries,
    load_calendar,
    load_labels,
    safe_name,
    split_for_year,
    window_range,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PARTIAL_RE = re.compile(r"t(\d+)_y(\d+)_x(\d+)\.npz$")
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def dataset_config() -> dict[str, Any]:
    """Rslearn ``config.json`` for the output dataset (pre-materialized layers)."""
    return {
        "layers": {
            ERA5_LAYER: {
                "type": "raster",
                "band_sets": [
                    {
                        "bands": list(ERA5L_BANDS),
                        "dtype": "float32",
                        "nodata_vals": [NODATA] * len(ERA5L_BANDS),
                        "format": {
                            "class_path": "rslearn.utils.raster_format.NumpyRasterFormat"
                        },
                    }
                ],
            },
            LABEL_LAYER: {"type": "vector"},
        }
    }


# ---------------------------------------------------------------------------
# Step A: series
# ---------------------------------------------------------------------------
def _series_for_spatial_chunk(args: tuple) -> tuple[str, int, int]:
    """Worker: assemble series for regions fully inside one spatial chunk.

    Edge regions' partial sums are accumulated and saved to an ``edge_*.npz``
    for the main process to merge. Returns (chunk, n_inside_written, n_edge).
    """
    key, files, agg_dir, variant, n_days = args
    dates_all = np.load(Path(agg_dir) / "dates.npy")
    series_dir = Path(agg_dir) / f"series_{variant}"
    series_dir.mkdir(parents=True, exist_ok=True)
    inside: dict[int, np.ndarray] = {}
    edge_sums: dict[int, np.ndarray] = {}
    edge_wsum: dict[int, np.ndarray] = {}
    c = len(ERA5L_BANDS)
    for f in files:
        with np.load(f) as z:
            pos = np.searchsorted(dates_all, z["dates"])
            ok = (pos < n_days) & (dates_all[np.minimum(pos, n_days - 1)] == z["dates"])
            if not ok.any():
                continue
            pos = pos[ok]
            mean = z[f"mean_{variant}"][:, ok, :]
            for k, r in enumerate(z["ridx_in"]):
                arr = inside.get(int(r))
                if arr is None:
                    arr = np.full((n_days, c), NODATA, dtype=np.float32)
                    inside[int(r)] = arr
                arr[pos] = mean[k]
            sums = z[f"sums_{variant}"][:, ok, :]
            wsum = z[f"wsum_{variant}"][:, ok, :]
            for k, r in enumerate(z["ridx_edge"]):
                r = int(r)
                if r not in edge_sums:
                    edge_sums[r] = np.zeros((n_days, c), dtype=np.float64)
                    edge_wsum[r] = np.zeros((n_days, c), dtype=np.float64)
                edge_sums[r][pos] += sums[k]
                edge_wsum[r][pos] += wsum[k]
    for r, arr in inside.items():
        np.save(series_dir / f"{r}.npy", arr)
    if edge_sums:
        ridx = np.array(sorted(edge_sums))
        np.savez(
            Path(agg_dir) / f"edge_{variant}_{key}.npz",
            ridx=ridx,
            sums=np.stack([edge_sums[r] for r in ridx]).astype(np.float32),
            wsum=np.stack([edge_wsum[r] for r in ridx]).astype(np.float32),
        )
    return key, len(inside), len(edge_sums)


def build_series(agg_dir: Path, variant: str, workers: int) -> None:
    """Assemble per-region series from all partials (Step A)."""
    dates_all = np.load(agg_dir / "dates.npy")
    n_days = len(dates_all)
    by_chunk: dict[str, list[Path]] = defaultdict(list)
    for f in sorted((agg_dir / "partials").glob("*.npz")):
        m = _PARTIAL_RE.search(f.name)
        if not m:
            continue
        by_chunk[f"y{m.group(2)}_x{m.group(3)}"].append(f)
    logger.info(
        "%d spatial chunks, %d partials",
        len(by_chunk),
        sum(map(len, by_chunk.values())),
    )
    jobs = [
        (k, [str(f) for f in fs], str(agg_dir), variant, n_days)
        for k, fs in sorted(by_chunk.items())
    ]
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for key, n_in, n_edge in pool.imap_unordered(_series_for_spatial_chunk, jobs):
            logger.info(
                "chunk %s: %d inside regions written, %d edge regions",
                key,
                n_in,
                n_edge,
            )

    # Merge edge regions across spatial chunks.
    sums: dict[int, np.ndarray] = {}
    wsum: dict[int, np.ndarray] = {}
    for f in agg_dir.glob(f"edge_{variant}_*.npz"):
        with np.load(f) as z:
            for k, r in enumerate(z["ridx"]):
                r = int(r)
                if r not in sums:
                    sums[r] = np.zeros_like(z["sums"][k], dtype=np.float64)
                    wsum[r] = np.zeros_like(z["wsum"][k], dtype=np.float64)
                sums[r] += z["sums"][k]
                wsum[r] += z["wsum"][k]
    series_dir = agg_dir / f"series_{variant}"
    for r in sums:
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(wsum[r] > 0, sums[r] / wsum[r], NODATA).astype(np.float32)
        np.save(series_dir / f"{r}.npy", mean)
    logger.info(
        "merged %d edge regions; %d series total",
        len(sums),
        len(list(series_dir.glob("*.npy"))),
    )


# ---------------------------------------------------------------------------
# Step B: windows
# ---------------------------------------------------------------------------
def _write_group(args: tuple) -> dict[str, Any]:
    """Worker: write every window for one (crop, cc)."""
    crop, cc, agg_dir, labels_root, ds_path, variant, min_year, max_year, fresh = args
    from rasterio.crs import CRS
    from rslearn.dataset import Dataset, Window
    from rslearn.utils.feature import Feature
    from rslearn.utils.geometry import WGS84_PROJECTION, Projection, STGeometry
    from rslearn.utils.raster_array import RasterArray, RasterMetadata
    from rslearn.utils.raster_format import NumpyRasterFormat
    from rslearn.utils.vector_format import GeojsonCoordinateMode, GeojsonVectorFormat
    from shapely.geometry import Point
    from upath import UPath

    agg_dir = Path(agg_dir)
    dates_all = np.load(agg_dir / "dates.npy")
    regions = pd.read_parquet(agg_dir / "regions.parquet")
    regions = regions[(regions["crop"] == crop) & (regions["cc"] == cc)]
    ridx_by_adm = {row.adm_id: (idx, row) for idx, row in regions.iterrows()}
    labels = load_labels(Path(labels_root), crop, cc)
    cal = load_calendar(Path(labels_root), crop, cc).set_index("adm_id")
    if min_year is not None:
        labels = labels[labels["year"] >= min_year]
    if max_year is not None:
        labels = labels[labels["year"] <= max_year]

    dataset = Dataset(UPath(ds_path))
    raster_format = NumpyRasterFormat()
    vector_format = GeojsonVectorFormat(coordinate_mode=GeojsonCoordinateMode.WGS84)
    proj = Projection(CRS.from_epsg(4326), 0.1, -0.1)
    series_cache: dict[int, np.ndarray] = {}
    counts: dict[str, int] = defaultdict(int)
    yields_by_split: dict[str, list[float]] = defaultdict(list)

    # ``yield`` is a Python keyword, so itertuples would rename the column.
    labels = labels.rename(columns={"yield": "y"})
    for row in labels.itertuples(index=False):
        counts["labels"] += 1
        if row.adm_id not in cal.index:
            counts["skip_no_calendar"] += 1
            continue
        if row.adm_id not in ridx_by_adm:
            counts["skip_no_region"] += 1
            continue
        ridx, reg = ridx_by_adm[row.adm_id]
        start, end = window_range(int(row.year), int(cal.loc[row.adm_id, "eos"]))
        s_day = (start - EPOCH).days
        s_pos = int(np.searchsorted(dates_all, s_day))
        if (
            s_pos >= len(dates_all)
            or dates_all[s_pos] != s_day
            or s_pos + WINDOW_DAYS > len(dates_all)
        ):
            counts["skip_outside_range"] += 1
            continue
        if ridx not in series_cache:
            f = agg_dir / f"series_{variant}" / f"{ridx}.npy"
            if not f.exists():
                counts["skip_no_series"] += 1
                continue
            series_cache[ridx] = np.load(f)
        block = series_cache[ridx][s_pos : s_pos + WINDOW_DAYS]  # (448, C)
        missing_days = (block == NODATA).all(axis=1)
        if missing_days.any():
            counts["skip_missing_days"] += 1
            continue

        name = safe_name(f"{crop}_{cc}_{row.adm_id}_{row.year}")
        split = split_for_year(int(row.year))
        lon, lat = float(reg.centroid_lon), float(reg.centroid_lat)
        col = int(np.floor(lon / 0.1))
        r0 = int(np.floor(lat / -0.1))
        bounds = (col, r0, col + 1, r0 + 1)
        window = Window(
            storage=dataset.storage,
            group=crop,
            name=name,
            projection=proj,
            bounds=bounds,
            time_range=(start, end),
            options={
                "crop": crop,
                "country_code": cc,
                "adm_id": row.adm_id,
                "harvest_year": str(int(row.year)),
                "split": split,
                "yield": float(row.y),
                "eos_doy": int(cal.loc[row.adm_id, "eos"]),
                "sos_doy": int(cal.loc[row.adm_id, "sos"]),
                "weights": variant,
            },
        )
        if (
            not fresh
            and window.is_layer_completed(ERA5_LAYER)
            and window.is_layer_completed(LABEL_LAYER)
        ):
            counts["exists"] += 1
            yields_by_split[split].append(window.options["yield"])
            continue
        window.save()

        array = np.ascontiguousarray(block.T)[:, :, None, None].astype(
            np.float32
        )  # (C, T, 1, 1)
        timestamps = [
            (start + timedelta(days=k), start + timedelta(days=k + 1))
            for k in range(WINDOW_DAYS)
        ]
        raster_format.encode_raster(
            window.get_raster_dir(ERA5_LAYER, list(ERA5L_BANDS)),
            proj,
            bounds,
            RasterArray(
                array=array,
                timestamps=timestamps,
                metadata=RasterMetadata(nodata_value=NODATA),
            ),
        )
        window.mark_layer_completed(ERA5_LAYER)

        feat = Feature(
            STGeometry(WGS84_PROJECTION, Point(lon, lat), (start, end)),
            {k: v for k, v in window.options.items()},
        )
        vector_format.encode_vector(window.get_layer_dir(LABEL_LAYER), [feat])
        window.mark_layer_completed(LABEL_LAYER)
        counts["written"] += 1
        yields_by_split[split].append(window.options["yield"])

    return dict(
        crop=crop,
        cc=cc,
        counts=dict(counts),
        yields_by_split={k: v for k, v in yields_by_split.items()},
    )


def write_windows(
    agg_dir: Path,
    labels_root: Path,
    ds_path: Path,
    variant: str,
    workers: int,
    min_year: int | None,
    max_year: int | None,
    fresh: bool,
    only: list[str] | None,
) -> None:
    """Write every label's window (Step B) and summarize."""
    ds_path.mkdir(parents=True, exist_ok=True)
    cfg_path = ds_path / "config.json"
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps(dataset_config(), indent=2))
    pairs = list_crop_countries(labels_root)
    if only:
        pairs = [(c, cc) for c, cc in pairs if f"{c}_{cc}" in only or cc in only]
    jobs = [
        (
            crop,
            cc,
            str(agg_dir),
            str(labels_root),
            str(ds_path),
            variant,
            min_year,
            max_year,
            fresh,
        )
        for crop, cc in pairs
    ]
    rows = []
    yields: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for res in pool.imap_unordered(_write_group, jobs):
            logger.info("%s/%s: %s", res["crop"], res["cc"], res["counts"])
            rows.append(dict(crop=res["crop"], cc=res["cc"], **res["counts"]))
            for split, ys in res["yields_by_split"].items():
                yields[res["crop"]][split].extend(ys)
    meta_dir = ds_path.parent / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows).fillna(0)
    summary.to_csv(meta_dir / f"windows_summary_{variant}.csv", index=False)
    logger.info("\n%s", summary.to_string(index=False))
    stats = {
        crop: {
            split: dict(n=len(ys), mean=float(np.mean(ys)), std=float(np.std(ys)))
            for split, ys in by_split.items()
        }
        for crop, by_split in yields.items()
    }
    (meta_dir / f"label_stats_{variant}.json").write_text(json.dumps(stats, indent=2))
    logger.info(
        "label stats (fill target_mean/target_std in direct_registry.json from 'train'):\n%s",
        json.dumps(stats, indent=2),
    )


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("step", choices=["series", "windows", "all"])
    ap.add_argument("--agg-dir", default=str(DEFAULT_ROOT / "agg"))
    ap.add_argument("--labels-root", default=str(DEFAULT_LABELS_ROOT))
    ap.add_argument("--ds-path", default=str(DEFAULT_ROOT / "rslearn_dataset"))
    ap.add_argument("--variant", choices=["crop", "area"], default="crop")
    ap.add_argument(
        "--min-year", type=int, default=2000, help="CY-Bench's MIN_INPUT_YEAR"
    )
    ap.add_argument("--max-year", type=int, default=None)
    ap.add_argument(
        "--only", nargs="*", default=None, help="restrict to 'crop_CC' or 'CC' groups"
    )
    ap.add_argument(
        "--fresh", action="store_true", help="rewrite windows even if completed"
    )
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    if args.step in ("series", "all"):
        build_series(Path(args.agg_dir), args.variant, args.workers)
    if args.step in ("windows", "all"):
        write_windows(
            Path(args.agg_dir),
            Path(args.labels_root),
            Path(args.ds_path),
            args.variant,
            args.workers,
            args.min_year,
            args.max_year,
            args.fresh,
            args.only,
        )


if __name__ == "__main__":
    main()
