"""Export PANGAEA-bench datasets as rslearn datasets that keep the original imagery.

Unlike pastis_rslearn_export.py, which re-materializes Sentinel imagery from
Planetary Computer, these exports write the exact pixels PANGAEA feeds every
model as passthrough layers, so inputs, labels and splits match the benchmark.
Band names are mapped to the OlmoEarth band_order names; bands a dataset lacks
are zeroed after normalization by the eval loader (see
RslearnToOlmoEarthDataset._init_band_scatter).

Every window is WINDOW_PX x WINDOW_PX. Source tiles are cut on their native
grid; the last row/column of windows is shifted flush with the tile edge, and
the pixels it shares with its neighbour are labelled IGNORE_LABEL, so each
source pixel is scored exactly once and no window holds padded imagery.

    python -m olmoearth_pretrain.evals.datasets.pangaea_rslearn_export \
        ai4smallfarms --src <pangaea root_path> --ds_path <out rslearn root>
"""

import argparse
import ast
import calendar
import json
import logging
import shutil
import zipfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import fiona
import numpy as np
import rasterio
from rasterio.crs import CRS
from rslearn.dataset import Dataset as RslearnDataset
from rslearn.dataset import Window
from rslearn.utils.geometry import Projection
from rslearn.utils.raster_array import RasterArray, RasterMetadata
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parents[3] / "data/rslearn_dataset_configs"

WINDOW_PX = 128
IGNORE_LABEL = 255


def window_offsets(size: int, window: int) -> list[tuple[int, int]]:
    """(start, first_new_pixel) along one axis, covering [0, size) exactly once.

    The last window is shifted to end at ``size``; ``first_new_pixel`` is the
    offset inside it where pixels not covered by the previous window begin.
    """
    if size < window:
        raise ValueError(f"tile side {size} is smaller than the window {window}")
    starts = list(range(0, size - window + 1, window))
    out = [(s, 0) for s in starts]
    covered = starts[-1] + window
    if covered < size:
        out.append((size - window, covered - (size - window)))
    return out


def write_raster(
    window: Window,
    layer: str,
    bands: list[str],
    chw: np.ndarray,
    time_range: tuple[datetime, datetime] | None,
    nodata: float | None = None,
) -> None:
    """Write one completed raster layer covering the whole window."""
    GeotiffRasterFormat().encode_raster(
        window.get_raster_dir(layer, bands),
        window.projection,
        window.bounds,
        RasterArray(
            chw_array=chw,
            time_range=time_range,
            metadata=RasterMetadata(nodata_value=nodata),
        ),
    )
    window.mark_layer_completed(layer)


def init_dataset(ds_path: UPath, config_name: str) -> RslearnDataset:
    """Create the dataset root and install its config.json."""
    ds_path.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CONFIG_DIR / config_name, ds_path / "config.json")
    return RslearnDataset(ds_path)


# --- AI4SmallFarms --------------------------------------------------------------
# Sentinel-2 (B2, B3, B4, B8) monthly tiles over Vietnam and Cambodia, ~9.3-9.9 m
# per tile, with a binary crop-field mask stored as 0/255. PANGAEA uses the
# sentinel-2-asia release and renames validate -> val.

AI4SF_GROUP = "ai4smallfarms"
AI4SF_BANDS = ["B02", "B03", "B04", "B08"]
AI4SF_SPLITS = {"train": "train", "validate": "val", "test": "test"}


def ai4sf_tile_dates(src: Path) -> dict[str, tuple[datetime, datetime]]:
    """Tile stem (e.g. '0_vietnam') -> its acquisition month, from tiles_asia.gpkg."""
    out = {}
    with fiona.open(src / "sentinel-2-asia/tiles_asia.gpkg") as tiles:
        for t in tiles:
            p = t["properties"]
            year, month = int(p["year"]), int(p["month"])
            last = calendar.monthrange(year, month)[1]
            out[f"{p['id']}_{p['country']}"] = (
                datetime(year, month, 1, tzinfo=UTC),
                datetime(year, month, last, 23, 59, 59, tzinfo=UTC),
            )
    return out


def ai4sf_windows(src: Path) -> Iterator[tuple[str, str, Path]]:
    """(split tag, tile stem, image path) for every tile."""
    for folder, split in AI4SF_SPLITS.items():
        for image in sorted((src / f"sentinel-2-asia/{folder}/images").glob("*.tif")):
            yield split, image.stem, image


def export_ai4sf_tile(
    dataset: RslearnDataset,
    split: str,
    stem: str,
    image_path: Path,
    time_range: tuple[datetime, datetime],
) -> int:
    """Cut one tile into windows and write its imagery and label layers."""
    mask_path = image_path.parent.parent / "masks" / image_path.name
    with rasterio.open(image_path) as img, rasterio.open(mask_path) as msk:
        if (
            not img.transform.almost_equals(msk.transform, 1e-6)
            or img.shape != msk.shape
        ):
            raise ValueError(f"{stem}: image and mask grids differ")
        if img.transform.b != 0 or img.transform.d != 0:
            raise ValueError(f"{stem}: rotated geotransform")
        image = img.read().astype(np.float32)
        mask = msk.read(1)
        tf, crs = img.transform, img.crs
    if set(np.unique(mask)) - {0, 255}:
        raise ValueError(f"{stem}: unexpected mask values {np.unique(mask)}")
    # PANGAEA zeroes NaNs and maps the mask 0/255 -> 0/1.
    image = np.nan_to_num(image, nan=0.0)
    label = (mask // 255).astype(np.uint8)

    projection = Projection(crs, tf.a, tf.e)
    col0, row0 = round(tf.c / tf.a), round(tf.f / tf.e)
    h, w = label.shape
    n = 0
    for r, r_new in window_offsets(h, WINDOW_PX):
        for c, c_new in window_offsets(w, WINDOW_PX):
            window = Window(
                storage=dataset.storage,
                group=AI4SF_GROUP,
                name=f"{stem}_r{r}_c{c}",
                projection=projection,
                bounds=(
                    col0 + c,
                    row0 + r,
                    col0 + c + WINDOW_PX,
                    row0 + r + WINDOW_PX,
                ),
                time_range=time_range,
                options={"split": split, "tile": stem},
            )
            window.save()
            sl = np.s_[r : r + WINDOW_PX, c : c + WINDOW_PX]
            win_label = label[sl].copy()
            win_label[:r_new, :] = IGNORE_LABEL
            win_label[:, :c_new] = IGNORE_LABEL
            write_raster(
                window,
                "sentinel2_l2a",
                AI4SF_BANDS,
                image[(slice(None), *sl)],
                time_range,
            )
            write_raster(window, "label", ["label"], win_label[None], None)
            n += 1
    return n


def export_ai4smallfarms(src: Path, ds_path: UPath, workers: int) -> None:
    """Export AI4SmallFarms (sentinel-2-asia) to an rslearn dataset."""
    dataset = init_dataset(ds_path, "config_pangaea_ai4smallfarms.json")
    dates = ai4sf_tile_dates(src)
    tiles = list(ai4sf_windows(src))
    missing = sorted({stem for _, stem, _ in tiles} - set(dates))
    if missing:
        raise ValueError(f"tiles without a date in tiles_asia.gpkg: {missing}")

    def run(t: tuple[str, str, Path]) -> tuple[str, int]:
        split, stem, image = t
        return split, export_ai4sf_tile(dataset, split, stem, image, dates[stem])

    counts: dict[str, list[int]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for split, n in ex.map(run, tiles):
            counts.setdefault(split, []).append(n)
    logger.info(
        json.dumps({s: {"tiles": len(v), "windows": sum(v)} for s, v in counts.items()})
    )


# --- CropTypeMappingSouthSudan --------------------------------------------------
# SustainBench's South Sudan crop-type split: 64x64 Sentinel-2 / Sentinel-1
# series from 2017 stored as (C, H, W, T) npz arrays with no geolocation.
# PANGAEA flips the time axis and right-pads to 6 timesteps with F.pad; every
# sample has more than 6, so the negative pad CROPS, keeping the last 6 array
# entries. We store exactly those, in their original array order, each as its
# own layer dated from the sample's json. Labels 1-4 (sorghum, maize, rice,
# groundnut) become 0-3; everything else is ignored. S1 ships in dB but rslearn
# datasets store linear power (the eval loader applies convert_to_db), so it is
# written as 10 ** (dB / 10).

CTM_GROUP = "croptypemapping"
CTM_TIMESTEPS = 6
CTM_PX = 64
CTM_S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
CTM_S1_BANDS = ["vv", "vh"]
CTM_SPLITS = {0: "train", 1: "val", 2: "test"}
# No geolocation ships with the npz files, so windows are laid out on a
# synthetic 10 m grid; options["synthetic_geometry"] marks them so no
# embedding product is ever materialized from these coordinates.
CTM_CRS = "EPSG:32636"
CTM_GRID_COLS = 32


def ctm_dates(root: Path, sensor: str, sample_id: str) -> list[datetime]:
    """Acquisition dates for one sample, in array order."""
    raw = json.loads(
        (root / sensor / f"{sensor}_southsudan_{sample_id}.json").read_text()
    )
    dates = raw["dates"]
    if isinstance(dates, str):
        dates = ast.literal_eval(dates)
    return [datetime.fromisoformat(d).replace(tzinfo=UTC) for d in dates]


def export_ctm_sample(
    dataset: RslearnDataset, root: Path, index: int, sample_id: str, split: str
) -> bool:
    """Write one sample's window; False if its npz is unreadable."""
    try:
        arrays = np.load(root / f"npy/southsudan_{sample_id}.npz")
        s2, s1 = arrays["s2"], 10 ** (arrays["s1"][:2] / 10)
    except zipfile.BadZipFile:
        logger.warning(f"{sample_id}: unreadable npz, skipped")
        return False
    truth = np.load(root / f"truth/southsudan_{sample_id}.npz")["truth"]

    series = {}
    for layer, sensor, arr, bands in (
        ("sentinel2_l2a", "s2", s2, CTM_S2_BANDS),
        ("sentinel1", "s1", s1, CTM_S1_BANDS),
    ):
        dates = ctm_dates(root, sensor, sample_id)
        if arr.shape != (len(bands), CTM_PX, CTM_PX, len(dates)):
            raise ValueError(
                f"{sample_id} {sensor}: shape {arr.shape} vs {len(dates)} dates"
            )
        if len(dates) < CTM_TIMESTEPS:
            raise ValueError(f"{sample_id} {sensor}: only {len(dates)} timesteps")
        series[layer] = (arr[..., -CTM_TIMESTEPS:], dates[-CTM_TIMESTEPS:], bands)

    label = np.full(truth.shape, IGNORE_LABEL, dtype=np.uint8)
    for src_class in range(1, 5):
        label[truth == src_class] = src_class - 1

    all_dates = [d for _, ds, _ in series.values() for d in ds]
    row, col = divmod(index, CTM_GRID_COLS)
    window = Window(
        storage=dataset.storage,
        group=CTM_GROUP,
        name=sample_id,
        projection=Projection(CRS.from_string(CTM_CRS), 10, -10),
        bounds=(
            col * CTM_PX,
            row * CTM_PX,
            (col + 1) * CTM_PX,
            (row + 1) * CTM_PX,
        ),
        time_range=(min(all_dates), max(all_dates) + timedelta(days=1)),
        options={"split": split, "synthetic_geometry": True},
    )
    window.save()
    for layer, (arr, dates, bands) in series.items():
        for t, date in enumerate(dates):
            write_raster(
                window,
                f"{layer}_t{t}",
                bands,
                arr[..., t].astype(np.float32),
                (date, date + timedelta(days=1)),
            )
    write_raster(window, "label", ["label"], label[None], None)
    return True


def export_croptypemapping(src: Path, ds_path: UPath, workers: int) -> None:
    """Export CropTypeMappingSouthSudan to an rslearn dataset."""
    root = src / "southsudan"
    dataset = init_dataset(ds_path, "config_pangaea_croptypemapping.json")
    rows = [
        line.split(",")
        for line in (root / "list_eval_partition.csv").read_text().splitlines()[1:]
    ]
    samples = [(i, sid, CTM_SPLITS[int(part)]) for i, (_, sid, part) in enumerate(rows)]

    def run(s: tuple[int, str, str]) -> tuple[str, bool]:
        return s[2], export_ctm_sample(dataset, root, *s)

    counts: dict[str, dict[str, int]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for split, ok in ex.map(run, samples):
            c = counts.setdefault(split, {"windows": 0, "skipped": 0})
            c["windows" if ok else "skipped"] += 1
    logger.info(json.dumps(counts))


EXPORTERS = {
    "ai4smallfarms": export_ai4smallfarms,
    "croptypemapping": export_croptypemapping,
}


def main() -> None:
    """Run one dataset's export."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", choices=sorted(EXPORTERS))
    parser.add_argument("--src", required=True, help="PANGAEA root_path for it")
    parser.add_argument("--ds_path", required=True, help="Output rslearn root")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if UPath(args.ds_path).exists():
        raise FileExistsError(f"{args.ds_path} exists; pick a new path")
    EXPORTERS[args.dataset](Path(args.src), UPath(args.ds_path), args.workers)


if __name__ == "__main__":
    main()
