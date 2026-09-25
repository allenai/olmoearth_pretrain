"""Stage 3: merge partials into per-region series and write rslearn windows.

``series``: concatenate each spatial chunk's time partials from
``aggregate_era5`` into one ``(T_all, 14)`` series per region under
``<agg>/series_<variant>/<ridx>.npy`` (``NODATA`` where nothing was fetched or
no cell was valid); regions on a chunk boundary are merged from their partial
sums.

``windows``: for every CY-Bench label ``(crop, cc, adm_id, harvest_year)`` slice
the 448 days ending at the crop-calendar end of season and write a
materialized rslearn window: group ``<crop>``, WGS84 0.1 deg/px bounds on the
ERA5 cell holding the region's representative point, raster layer
``era5_daily`` ``(14, 448, 1, 1)`` with per-day timestamps, vector layer
``label`` with ``yield`` (t/ha), and tags ``crop, country_code, adm_id,
harvest_year, split``. The dataset ``config.json`` is written here too (layers
have no data_source; everything is pre-materialized).

Usage::

    python -m olmoearth_pretrain.dataset_creation.cybench.write_windows all \
        --agg-dir .../agg --labels-root .../cybench-data --ds-path .../rslearn_dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    split_for_year,
    window_name,
    window_range,
)

if TYPE_CHECKING:
    from rslearn.dataset import Window

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PARTIAL_RE = re.compile(r"t(\d+)_y(\d+)_x(\d+)\.npz$")
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
N_BANDS = len(ERA5L_BANDS)


def dataset_config() -> dict[str, Any]:
    """Rslearn ``config.json`` for the output dataset (pre-materialized layers)."""
    band_set = {
        "bands": list(ERA5L_BANDS),
        "dtype": "float32",
        "nodata_vals": [NODATA] * N_BANDS,
        "format": {"class_path": "rslearn.utils.raster_format.NumpyRasterFormat"},
    }
    return {
        "layers": {
            ERA5_LAYER: {"type": "raster", "band_sets": [band_set]},
            LABEL_LAYER: {"type": "vector"},
        }
    }


# --- series -----------------------------------------------------------------
def build_series(agg_dir: Path, variant: str, workers: int) -> None:
    """Turn the per-chunk partials into one ``(T_all, 14)`` series per region."""
    partials_by_chunk: dict[str, list[str]] = defaultdict(list)
    for f in sorted((agg_dir / "partials").glob("*.npz")):
        if m := PARTIAL_RE.search(f.name):
            partials_by_chunk[f"y{m.group(2)}_x{m.group(3)}"].append(str(f))
    logger.info(
        "%d spatial chunks, %d partials",
        len(partials_by_chunk),
        sum(map(len, partials_by_chunk.values())),
    )

    jobs = [
        (key, files, str(agg_dir), variant)
        for key, files in sorted(partials_by_chunk.items())
    ]
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for key, n_inside, n_edge in pool.imap_unordered(series_for_chunk, jobs):
            logger.info(
                "chunk %s: %d inside regions written, %d edge regions",
                key,
                n_inside,
                n_edge,
            )
    n_edge = merge_edge_regions(agg_dir, variant)
    logger.info(
        "merged %d edge regions; %d series total",
        n_edge,
        len(list((agg_dir / f"series_{variant}").glob("*.npy"))),
    )


def series_for_chunk(job: tuple[str, list[str], str, str]) -> tuple[str, int, int]:
    """Worker: write final series for regions inside one spatial chunk; stash edge-region sums."""
    key, files, agg_dir, variant = job
    agg = Path(agg_dir)
    dates_all = np.load(agg / "dates.npy")
    series_dir = agg / f"series_{variant}"
    series_dir.mkdir(parents=True, exist_ok=True)

    means: dict[int, np.ndarray] = {}
    edge_sums: dict[int, np.ndarray] = defaultdict(
        lambda: np.zeros((len(dates_all), N_BANDS))
    )
    edge_wsum: dict[int, np.ndarray] = defaultdict(
        lambda: np.zeros((len(dates_all), N_BANDS))
    )
    for f in files:
        with np.load(f) as z:
            pos, keep = _positions(dates_all, z["dates"])
            if not keep.any():
                continue
            for k, r in enumerate(z["ridx_in"]):
                means.setdefault(
                    int(r), np.full((len(dates_all), N_BANDS), NODATA, np.float32)
                )[pos] = z[f"mean_{variant}"][k, keep]
            for k, r in enumerate(z["ridx_edge"]):
                edge_sums[int(r)][pos] += z[f"sums_{variant}"][k, keep]
                edge_wsum[int(r)][pos] += z[f"wsum_{variant}"][k, keep]

    for r, arr in means.items():
        np.save(series_dir / f"{r}.npy", arr)
    if edge_sums:
        ridx = np.array(sorted(edge_sums))
        np.savez(
            agg / f"edge_{variant}_{key}.npz",
            ridx=ridx,
            sums=np.stack([edge_sums[r] for r in ridx]).astype(np.float32),
            wsum=np.stack([edge_wsum[r] for r in ridx]).astype(np.float32),
        )
    return key, len(means), len(edge_sums)


def merge_edge_regions(agg_dir: Path, variant: str) -> int:
    """Sum the stashed partial sums of boundary regions across chunks and write their series."""
    sums: dict[int, np.ndarray] = {}
    wsum: dict[int, np.ndarray] = {}
    for f in agg_dir.glob(f"edge_{variant}_*.npz"):
        with np.load(f) as z:
            for k, r in enumerate(z["ridx"]):
                sums[int(r)] = sums.get(int(r), 0) + z["sums"][k].astype(np.float64)
                wsum[int(r)] = wsum.get(int(r), 0) + z["wsum"][k].astype(np.float64)
    for r in sums:
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(wsum[r] > 0, sums[r] / wsum[r], NODATA).astype(np.float32)
        np.save(agg_dir / f"series_{variant}" / f"{r}.npy", mean)
    return len(sums)


def _positions(
    dates_all: np.ndarray, dates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Indices of ``dates`` in the global axis, and which of them are actually on it."""
    pos = np.searchsorted(dates_all, dates)
    keep = (pos < len(dates_all)) & (
        dates_all[np.minimum(pos, len(dates_all) - 1)] == dates
    )
    return pos[keep], keep


# --- windows ----------------------------------------------------------------
def write_windows(agg_dir: Path, labels_root: Path, ds_path: Path, variant: str, workers: int,
                  min_year: int | None, max_year: int | None, fresh: bool, only: list[str] | None) -> None:  # fmt: skip
    """Write one window per label for every (crop, country); summarize counts and yield stats."""
    ds_path.mkdir(parents=True, exist_ok=True)
    if not (ds_path / "config.json").exists():
        (ds_path / "config.json").write_text(json.dumps(dataset_config(), indent=2))

    groups = list_crop_countries(labels_root)
    if only:
        groups = [
            (crop, cc) for crop, cc in groups if f"{crop}_{cc}" in only or cc in only
        ]
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
        for crop, cc in groups
    ]

    rows: list[dict[str, Any]] = []
    yields: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for res in pool.imap_unordered(write_group, jobs):
            logger.info("%s/%s: %s", res["crop"], res["cc"], res["counts"])
            rows.append(dict(crop=res["crop"], cc=res["cc"], **res["counts"]))
            for split, ys in res["yields"].items():
                yields[res["crop"]][split].extend(ys)

    meta_dir = ds_path.parent / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows).fillna(0)
    summary.to_csv(meta_dir / f"windows_summary_{variant}.csv", index=False)
    stats = {
        crop: {
            split: dict(n=len(ys), mean=float(np.mean(ys)), std=float(np.std(ys)))
            for split, ys in by_split.items()
        }
        for crop, by_split in yields.items()
    }
    (meta_dir / f"label_stats_{variant}.json").write_text(json.dumps(stats, indent=2))
    logger.info(
        "\n%s\nlabel stats (registry target_mean/std come from 'train'):\n%s",
        summary.to_string(index=False),
        json.dumps(stats, indent=2),
    )


@dataclass
class GroupInputs:
    """Everything one (crop, cc) worker needs to slice its windows."""

    crop: str
    cc: str
    labels: pd.DataFrame  # adm_id, year, y
    calendar: pd.DataFrame  # indexed by adm_id: sos, eos
    regions: dict[str, tuple[int, Any]]  # adm_id -> (row index in regions.parquet, row)
    dates_all: np.ndarray
    series_dir: Path
    _series_cache: dict[int, np.ndarray] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        crop: str,
        cc: str,
        agg_dir: Path,
        labels_root: Path,
        variant: str,
        min_year: int | None,
        max_year: int | None,
    ) -> GroupInputs:
        """Read labels, calendar and region table for one (crop, cc)."""
        labels = load_labels(labels_root, crop, cc).rename(
            columns={"yield": "y"}
        )  # 'yield' is a keyword
        if min_year is not None:
            labels = labels[labels["year"] >= min_year]
        if max_year is not None:
            labels = labels[labels["year"] <= max_year]
        regions = pd.read_parquet(agg_dir / "regions.parquet")
        regions = regions[(regions["crop"] == crop) & (regions["cc"] == cc)]
        return cls(
            crop=crop, cc=cc, labels=labels,
            calendar=load_calendar(labels_root, crop, cc).set_index("adm_id"),
            regions={row.adm_id: (idx, row) for idx, row in regions.iterrows()},
            dates_all=np.load(agg_dir / "dates.npy"),
            series_dir=agg_dir / f"series_{variant}",
        )  # fmt: skip

    def series(self, ridx: int) -> np.ndarray | None:
        """The region's ``(T_all, 14)`` series, or None if it was never aggregated."""
        if ridx not in self._series_cache:
            f = self.series_dir / f"{ridx}.npy"
            if not f.exists():
                return None
            self._series_cache[ridx] = np.load(f)
        return self._series_cache[ridx]

    def window_block(
        self, adm_id: str, year: int
    ) -> tuple[np.ndarray, datetime, datetime] | str:
        """The label's ``(448, 14)`` block and its time range, or the reason it must be skipped."""
        if adm_id not in self.calendar.index:
            return "skip_no_calendar"
        if adm_id not in self.regions:
            return "skip_no_region"
        start, end = window_range(year, int(self.calendar.loc[adm_id, "eos"]))
        first = int(np.searchsorted(self.dates_all, (start - EPOCH).days))
        if (
            first + WINDOW_DAYS > len(self.dates_all)
            or self.dates_all[first] != (start - EPOCH).days
        ):
            return "skip_outside_range"
        series = self.series(self.regions[adm_id][0])
        if series is None:
            return "skip_no_series"
        block = series[first : first + WINDOW_DAYS]
        if (block == NODATA).all(axis=1).any():
            return "skip_missing_days"
        return block, start, end


class WindowWriter:
    """Writes pre-materialized rslearn windows (ERA5 raster + yield label)."""

    def __init__(self, ds_path: str) -> None:
        """Open the dataset and prepare the raster/vector formats and projection."""
        from rasterio.crs import CRS
        from rslearn.dataset import Dataset
        from rslearn.utils.geometry import Projection
        from rslearn.utils.raster_format import NumpyRasterFormat
        from rslearn.utils.vector_format import (
            GeojsonCoordinateMode,
            GeojsonVectorFormat,
        )
        from upath import UPath

        self.dataset = Dataset(UPath(ds_path))
        self.projection = Projection(CRS.from_epsg(4326), 0.1, -0.1)
        self.raster_format = NumpyRasterFormat()
        self.vector_format = GeojsonVectorFormat(
            coordinate_mode=GeojsonCoordinateMode.WGS84
        )

    def window(
        self,
        group: str,
        name: str,
        lon: float,
        lat: float,
        start: datetime,
        end: datetime,
        options: dict,
    ) -> Window:
        """An (unsaved) window on the 0.1 deg cell containing (lon, lat)."""
        from rslearn.dataset import Window

        col, row = int(np.floor(lon / 0.1)), int(np.floor(lat / -0.1))
        return Window(storage=self.dataset.storage, group=group, name=name, projection=self.projection,
                      bounds=(col, row, col + 1, row + 1), time_range=(start, end), options=options)  # fmt: skip

    @staticmethod
    def is_complete(window: Window) -> bool:
        """Both layers already materialized (resume support)."""
        return window.is_layer_completed(ERA5_LAYER) and window.is_layer_completed(
            LABEL_LAYER
        )

    def write(self, window: Window, block: np.ndarray, lon: float, lat: float) -> None:
        """Save the window with its ``(448, 14)`` block as a 1x1 raster and a point label feature."""
        from rslearn.utils.feature import Feature
        from rslearn.utils.geometry import WGS84_PROJECTION, STGeometry
        from rslearn.utils.raster_array import RasterArray, RasterMetadata
        from shapely.geometry import Point

        window.save()
        start, end = window.time_range
        array = np.ascontiguousarray(block.T)[:, :, None, None].astype(
            np.float32
        )  # (C, T, 1, 1)
        timestamps = [
            (start + timedelta(days=k), start + timedelta(days=k + 1))
            for k in range(WINDOW_DAYS)
        ]
        raster = RasterArray(
            array=array,
            timestamps=timestamps,
            metadata=RasterMetadata(nodata_value=NODATA),
        )
        self.raster_format.encode_raster(
            window.get_raster_dir(ERA5_LAYER, list(ERA5L_BANDS)),
            self.projection,
            window.bounds,
            raster,
        )
        window.mark_layer_completed(ERA5_LAYER)

        feature = Feature(
            STGeometry(WGS84_PROJECTION, Point(lon, lat), (start, end)),
            dict(window.options),
        )
        self.vector_format.encode_vector(window.get_layer_dir(LABEL_LAYER), [feature])
        window.mark_layer_completed(LABEL_LAYER)


def write_group(job: tuple) -> dict[str, Any]:
    """Worker: write every window for one (crop, cc)."""
    crop, cc, agg_dir, labels_root, ds_path, variant, min_year, max_year, fresh = job
    inputs = GroupInputs.load(
        crop, cc, Path(agg_dir), Path(labels_root), variant, min_year, max_year
    )
    writer = WindowWriter(ds_path)
    counts: Counter[str] = Counter()
    yields: dict[str, list[float]] = defaultdict(list)

    for label in inputs.labels.itertuples(index=False):
        counts["labels"] += 1
        result = inputs.window_block(label.adm_id, int(label.year))
        if isinstance(result, str):
            counts[result] += 1
            continue
        block, start, end = result
        region = inputs.regions[label.adm_id][1]
        split = split_for_year(int(label.year))
        options = dict(
            crop=crop, country_code=cc, adm_id=label.adm_id, harvest_year=str(int(label.year)), split=split,
            **{"yield": float(label.y)},
            eos_doy=int(inputs.calendar.loc[label.adm_id, "eos"]), sos_doy=int(inputs.calendar.loc[label.adm_id, "sos"]),
            weights=variant,
        )  # fmt: skip
        window = writer.window(crop, window_name(crop, cc, label.adm_id, int(label.year)),
                               region.centroid_lon, region.centroid_lat, start, end, options)  # fmt: skip
        if not fresh and writer.is_complete(window):
            counts["exists"] += 1
        else:
            writer.write(window, block, region.centroid_lon, region.centroid_lat)
            counts["written"] += 1
        yields[split].append(float(label.y))
    return dict(crop=crop, cc=cc, counts=dict(counts), yields=dict(yields))


# --- CLI --------------------------------------------------------------------
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
        write_windows(Path(args.agg_dir), Path(args.labels_root), Path(args.ds_path), args.variant, args.workers,
                      args.min_year, args.max_year, args.fresh, args.only)  # fmt: skip


if __name__ == "__main__":
    main()
