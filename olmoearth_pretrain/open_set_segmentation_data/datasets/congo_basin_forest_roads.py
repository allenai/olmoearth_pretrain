"""Congo Basin forest roads -> open-set-segmentation forest-road masks.

Source: "Forest roads (Congo Basin)" (Zenodo record 13739812, doi
10.5281/zenodo.13739812; Slagter et al. 2024, Remote Sensing of Environment,
doi 10.1016/j.rse.2024.114380). Road development in Congo Basin forests is monitored
from 2019 onward by applying a deep-learning model to 10 m Sentinel-1 + Sentinel-2
imagery, producing automated monthly road detections. This release covers 2019-2023
(46,311 km of roads). The data is 355,995 LineString segments (World Mollweide,
ESRI:54009, metres) with attributes:
  NetworkID (connected-network id), SegLenM (segment length m), NetLenM (network
  length m), Month + Year (segment OPENING month/year), MonthNum (months since the
  2019-01 monitoring start).

  https://zenodo.org/records/13739812  (file forestroads_afr_2019-01_2023-12.zip)

Recipe (spec S4 "lines"): rasterize the road centerlines into a thin dilated mask so
they are visible at 10 m/pixel. Single foreground class:
  0 = forest road (a mapped forest-road segment; dilated to ~20-30 m so it registers
      at 10 m).
This is a **positive-only** dataset (spec S5): non-road pixels are left as
nodata/ignore (255) -- we do NOT fabricate a background class or negative tiles; the
assembly step supplies negatives from other datasets.

Suitability at 10 m: the source product is *itself* derived from 10 m S1/S2 imagery
with a deep-learning detector, i.e. these roads are exactly the linear disturbance
features resolvable at 10 m in Sentinel imagery (logging / selective-logging access,
a forest-degradation proxy). A centerline dilated to ~2-3 px is a meaningful 10 m
label. ACCEPTED.

Presence vs change (spec S5): each segment carries an OPENING month/year, which could
support a change-label framing. But a road is a **persistent** feature once built, so
(per the task instruction and spec S5's persistent-post-change-state clause) we treat
this as a presence/state segmentation with ``change_time=null`` and a static 1-year
window. The window is anchored on the **latest** opening year among the segments in a
tile, so imagery in that year is guaranteed to post-date construction of every road in
the mask (roads opened earlier persist). All anchor years fall in the manifest range
[2019, 2024].

Tiling: the road network is partitioned onto a fixed 640 m grid in the source
(Mollweide) CRS; each occupied cell becomes one 64x64 (640 m) UTM 10 m tile centered on
the cell center, into which every road segment overlapping the cell is rasterized
(clipped to the tile). 94,284 cells are occupied; we sample MAX_SAMPLES (25,000) cells
(deterministic seeded subsample) to honor the 25k per-dataset cap. Tiles whose
rasterized road mask has < MIN_ROAD_PIXELS pixels are dropped.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.congo_basin_forest_roads
"""

import argparse
import math
import multiprocessing
import random
import zipfile
from collections import Counter, defaultdict
from typing import Any

import fiona
import shapely
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered
from shapely.geometry import shape

from olmoearth_pretrain.open_set_segmentation_data import (
    download,
    io,
    manifest,
    sampling,
)
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)

SLUG = "congo_basin_forest_roads"
NAME = "Congo Basin Forest Roads"
ZENODO_RECORD = "13739812"
ZIP_NAME = "forestroads_afr_2019-01_2023-12.zip"
SHP_NAME = "forestroads_afr_2019-01_2023-12.shp"

# Source CRS is World Mollweide (ESRI:54009, metres). Projection res (1,1) => geometry
# coords are treated as pixel==metre, matching the geom_to_pixels convention (as the
# TermPicks EPSG:3413 script does).
SRC_CRS = CRS.from_string("ESRI:54009")
SRC_PROJ = Projection(SRC_CRS, 1, 1)

TILE = 64  # 640 m tiles.
CELL_M = TILE * io.RESOLUTION  # 640 m grid cells in the source CRS.
DILATE_RADIUS_PX = 1.0  # buffer the centerline by ~1 px -> ~2-3 px (20-30 m) wide.
MIN_ROAD_PIXELS = 3  # drop tiles whose road mask is a trivial sliver / empty.

# Sentinel-era manifest range; opening years present in the source are 2019-2023.
YEAR_MIN = 2019
YEAR_MAX = 2024

CID_ROAD = 0
CLASSES = [
    {
        "id": CID_ROAD,
        "name": "forest_road",
        "description": (
            "A mapped Congo Basin forest-road segment (Slagter et al. 2024): a linear "
            "logging / selective-logging access road automatically detected from 10 m "
            "Sentinel-1 + Sentinel-2 imagery with a deep-learning model, a "
            "forest-degradation proxy. Rasterized from the LineString and dilated to "
            "~20-30 m (2-3 px) so it is visible at 10 m/pixel. Non-road pixels are "
            "nodata (255): this is a positive-only mask."
        ),
    }
]

_TO_WGS84 = None  # lazily-built pyproj transformer (per process).


def _lonlat(x: float, y: float) -> tuple[float, float]:
    """ESRI:54009 (x, y) metres -> (lon, lat) degrees."""
    global _TO_WGS84
    if _TO_WGS84 is None:
        from pyproj import Transformer

        _TO_WGS84 = Transformer.from_crs("ESRI:54009", 4326, always_xy=True)
    lon, lat = _TO_WGS84.transform(x, y)
    return lon, lat


def _download_and_extract() -> None:
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    if not (raw / SHP_NAME).exists():
        print("downloading forest-roads archive from Zenodo ...")
        download.download_http(
            f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{ZIP_NAME}/content",
            raw / ZIP_NAME,
        )
        with zipfile.ZipFile((raw / ZIP_NAME).path) as z:
            z.extractall(raw.path)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Forest roads (Congo Basin) -- Slagter et al. 2024, Remote Sensing of "
            f"Environment. Zenodo record {ZENODO_RECORD} "
            "(doi 10.5281/zenodo.13739812).\n"
            f"https://zenodo.org/records/{ZENODO_RECORD}  (file {ZIP_NAME})\n"
            "Deep-learning-mapped forest-road LineStrings from 10 m Sentinel-1+2 "
            "imagery, 2019-2023, CRS ESRI:54009 (World Mollweide). Attributes: "
            "NetworkID, SegLenM, NetLenM, Month, Year (segment opening month/year), "
            "MonthNum. License CC-BY-4.0.\n"
        )


def build_cells() -> dict[tuple[int, int], dict[str, Any]]:
    """Partition road segments onto a 640 m grid in the source (Mollweide) CRS.

    Returns cell (ix, iy) -> {"wkbs": [...], "max_year": int, "min_year": int}. A
    segment is added to every grid cell its bounding box overlaps (segments are short --
    median ~42 m -- so this is 1-4 cells for almost all; the rare long segment's extra
    cells rasterize to empty and get dropped). Road pixels outside the tile clip away at
    rasterization, so bbox-overlap membership is a safe superset of tile membership.
    """
    shp = (io.raw_dir(SLUG) / SHP_NAME).path
    cells: dict[tuple[int, int], dict[str, Any]] = defaultdict(
        lambda: {"wkbs": [], "max_year": 0, "min_year": 9999}
    )
    with fiona.open(shp) as src:
        for feat in src:
            p = feat["properties"]
            year = p.get("Year")
            if year is None:
                continue
            year = int(year)
            if year < YEAR_MIN or year > YEAR_MAX:
                continue
            geom = shape(feat["geometry"])
            if geom.is_empty or geom.length == 0:
                continue
            wkb = shapely.to_wkb(geom)
            minx, miny, maxx, maxy = geom.bounds
            ix0, ix1 = math.floor(minx / CELL_M), math.floor(maxx / CELL_M)
            iy0, iy1 = math.floor(miny / CELL_M), math.floor(maxy / CELL_M)
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    c = cells[(ix, iy)]
                    c["wkbs"].append(wkb)
                    if year > c["max_year"]:
                        c["max_year"] = year
                    if year < c["min_year"]:
                        c["min_year"] = year
    return cells


def _write_one(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"

    ix, iy = rec["cell"]
    # Cell center in the source CRS -> lon/lat -> local UTM tile centered there.
    cx = (ix + 0.5) * CELL_M
    cy = (iy + 0.5) * CELL_M
    lon, lat = _lonlat(cx, cy)
    proj = io.utm_projection_for_lonlat(lon, lat)
    _, col, row = io.lonlat_to_utm_pixel(lon, lat, proj)
    bounds = io.centered_bounds(col, row, TILE, TILE)

    shapes: list[tuple[Any, int]] = []
    for wkb in rec["wkbs"]:
        geom = shapely.from_wkb(wkb)
        try:
            pix = geom_to_pixels(geom, SRC_PROJ, proj)
        except Exception:
            continue
        if pix.is_empty:
            continue
        dil = pix.buffer(DILATE_RADIUS_PX)
        if dil.is_empty:
            continue
        shapes.append((dil, CID_ROAD))
    if not shapes:
        return "empty"

    arr = rasterize_shapes(
        shapes, bounds, fill=io.CLASS_NODATA, dtype="uint8", all_touched=True
    )[0]
    if int((arr == CID_ROAD).sum()) < MIN_ROAD_PIXELS:
        return "empty"

    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        change_time=None,
        source_id=rec["source_id"],
        classes_present=[CID_ROAD],
    )
    return "written"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--probe", action="store_true", help="scan/report only, no writes"
    )
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    _download_and_extract()

    print("partitioning road segments onto 640 m grid ...")
    cells = build_cells()
    print(f"  {len(cells)} occupied cells")

    io.check_disk()

    # One tile per cell; anchor the 1-year window on the cell's latest opening year
    # (imagery then post-dates construction of every road in the mask).
    records: list[dict[str, Any]] = []
    for (ix, iy), c in cells.items():
        records.append(
            {
                "cell": (ix, iy),
                "wkbs": c["wkbs"],
                "year": int(c["max_year"]),
                "min_year": int(c["min_year"]),
                "source_id": f"cell_{ix}_{iy}/opened_{c['min_year']}_{c['max_year']}",
            }
        )

    # Deterministic seeded subsample down to the 25k hard cap.
    records.sort(key=lambda r: r["cell"])
    rng = random.Random(42)
    rng.shuffle(records)
    if len(records) > sampling.MAX_SAMPLES_PER_DATASET:
        records = records[: sampling.MAX_SAMPLES_PER_DATASET]
        print(f"capped to {sampling.MAX_SAMPLES_PER_DATASET} cells")
    records.sort(key=lambda r: r["cell"])  # stable id assignment
    for i, r in enumerate(records):
        r["sample_id"] = f"{i:06d}"

    year_hist = Counter(r["year"] for r in records)
    print("anchor-year distribution:", dict(sorted(year_hist.items())))

    if args.probe:
        print("probe only; exiting before writes")
        return

    results: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in star_imap_unordered(p, _write_one, [dict(rec=r) for r in records]):
            results[res] += 1
    print("write results:", dict(results))

    io.check_disk()

    # Recompute the anchor-year histogram from the tiles actually on disk (some selected
    # cells rasterize to < MIN_ROAD_PIXELS and are dropped), so metadata is accurate and
    # stable across idempotent re-runs.
    import json as _json

    written_year_hist: Counter = Counter()
    for jp in io.locations_dir(SLUG).glob("*.json"):
        with jp.open() as _f:
            _tr = _json.load(_f)["time_range"]
        written_year_hist[int(_tr[0][:4])] += 1
    num_samples = sum(written_year_hist.values())

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Zenodo / RSE",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.org/10.5281/zenodo.13739812",
                "have_locally": False,
                "annotation_method": (
                    "deep-learning detection on 10 m Sentinel-1+2 imagery + manual "
                    "training (Slagter et al. 2024, RSE)"
                ),
                "citation": (
                    "Slagter B., Fesenmyer K., Hethcoat M., Belair E., Ellis P., "
                    "Kleinschroth F., Pena-Claros M., Herold M., Reiche J. (2024). "
                    "Monitoring road development in Congo Basin forests with multi-sensor "
                    "satellite imagery and deep learning. Remote Sensing of Environment. "
                    "doi:10.1016/j.rse.2024.114380"
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": num_samples,
            "anchor_year_counts": {
                str(k): v for k, v in sorted(written_year_hist.items())
            },
            "notes": (
                "Positive-only forest-road segmentation. Congo Basin forest-road "
                "LineStrings (ESRI:54009 World Mollweide, deep-learning-mapped from 10 m "
                "Sentinel-1+2, Slagter et al. 2024) rasterized (buffered ~1 px -> "
                "~20-30 m wide, all_touched) into 64x64 UTM 10 m tiles; class 0 = "
                "forest_road, non-road = nodata (255). The road network is partitioned "
                "onto a 640 m grid in the source CRS; each occupied cell (94,284 total) "
                "is one tile with every overlapping segment rasterized (clipped); tiles "
                f"with < {MIN_ROAD_PIXELS} road px are dropped. Sampled "
                f"{sampling.MAX_SAMPLES_PER_DATASET} cells (seeded random) to honor the "
                "25k cap. Roads carry an opening month/year but are treated as a "
                "persistent presence/state label (change_time=null): a 1-year window is "
                "anchored on the latest opening year among a tile's segments so imagery "
                "post-dates construction of every mapped road. Opening years span "
                "2019-2023 (within manifest range [2019,2024]). NetworkID/SegLenM/"
                "NetLenM/MonthNum attributes exist in the source but are collapsed to a "
                "single road class per the task spec. Caveat: source is a derived "
                "deep-learning product (not in-situ reference), so mislabeled/omitted "
                "roads are possible; narrow single-track roads are near the 10 m "
                "resolution limit but were detectable enough to be mapped by the source."
            ),
        },
    )

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=num_samples
    )
    print(
        f"done: {num_samples} samples; "
        f"anchor-year={dict(sorted(written_year_hist.items()))}"
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
