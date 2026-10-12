"""Process JRC Tropical Moist Forest (TMF) into open-set-segmentation label patches.

Source: EC JRC Tropical Moist Forest product (Vancutsem et al. 2021, Science Advances;
https://forobs.jrc.ec.europa.eu/TMF/data). Pan-tropical 30 m derived-product map from
Landsat, distributed as 10x10 degree tiles in EPSG:4326. We use the **AnnualChange**
collection, whose per-year raster encodes the forest state of each pixel for that year:

    1 = Undisturbed tropical moist forest
    2 = Degraded tropical moist forest
    3 = Deforested land
    4 = Tropical moist forest regrowth
    5 = Permanent or seasonal water
    6 = Other land cover
    0 = no data / outside the tropical belt

These six classes map exactly to the manifest classes. We treat a single year's state as
a per-pixel **classification** label (task_type=classification) with a ~1-year time range
anchored on the chosen year (no per-event change_time -- the annual state map is a clean
per-year classification, so we keep it simple as the spec directs).

This is a HUGE global product, so per the spec we do BOUNDED-TILE sampling: we download a
handful of representative tiles across the three tropical-forest basins (Amazon, Congo,
SE Asia) for ONE year, then draw spatially-homogeneous <=64x64 windows (dominant-class
majority) and balance to <=1000 tiles/class. 30 m source windows are reprojected to a
local UTM projection at 10 m with nearest resampling (categorical labels).

Tile download URL (discovered from the TMF download portal SPA):
    https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py
        ?type=tile&dataset=AnnualChange_<year>&lat=<latLabel>&lon=<lonLabel>
where <latLabel>_<lonLabel> is the 10x10-deg tile id (e.g. N0_E20, S10_W60).

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.jrc_tropical_moist_forest_tmf
"""

import argparse
import multiprocessing
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import tqdm
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds

from olmoearth_pretrain.open_set_segmentation_data import download, io

SLUG = "jrc_tropical_moist_forest_tmf"

# One representative year within the manifest range 2016-2025. AnnualChange 2020 is a
# recent, fully-observed Landsat-era year; the annual state map is a per-year label.
YEAR = 2020

PER_CLASS = 1000
BLOCK = 22  # native (30 m) block ~= 660 m ~= a 64 px @ 10 m UTM tile footprint
DOMINANCE_FLOOR = (
    0.5  # a candidate window must be >=50% one class (majority = its label)
)
MAX_NODATA_FRAC = 0.2  # reject windows that are mostly outside the product
TILE = 64
BASE_URL = "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py"

# Representative 10x10-deg tiles across the three tropical moist-forest basins. Each tile
# id is "<latLabel>_<lonLabel>" with the lat/lon being the tile's NW corner label used by
# the JRC download portal. ~85 MB per tile per year.
TILES = {
    "S10_W60": "Amazon - S Brazil / Rondonia (heavy deforestation, degradation, regrowth)",
    "S10_W70": "Amazon - W Brazil / Peru / Bolivia",
    "N0_E20": "Congo Basin - DR Congo",
    "N0_E10": "Congo Basin - Gabon / Cameroon",
    "N0_E110": "SE Asia - Borneo (Kalimantan)",
    "N0_E100": "SE Asia - Sumatra / Malay Peninsula",
}

# Manifest class order -> id. Source AnnualChange value = id + 1. Descriptions from the
# TMF product definition (Vancutsem et al. 2021 / TMF Data Users Guide).
CLASSES = [
    (
        "undisturbed forest",
        "Undisturbed tropical moist forest: closed evergreen / semi-evergreen forest with no "
        "disturbance detected over the Landsat observation record.",
    ),
    (
        "degraded forest",
        "Degraded tropical moist forest: forest that underwent temporary canopy disturbance "
        "(selective logging, fire, blow-down or other short-duration events) but remained forest.",
    ),
    (
        "deforested",
        "Deforested land: former tropical moist forest cleared and converted to other land cover "
        "(cropland, pasture, plantations, built-up or bare ground).",
    ),
    (
        "regrowth",
        "Tropical moist forest regrowth: vegetation regrowth / secondary forest on land that was "
        "previously deforested.",
    ),
    ("water", "Permanent or seasonal water bodies."),
    (
        "other",
        "Other land cover: land that was not tropical moist forest over the observation period "
        "(non-TMF vegetation, savanna, bare soil, built-up outside deforestation).",
    ),
]
# source value (1..6) -> class id (0..5); source 0 -> nodata
SRC_TO_ID = {v: v - 1 for v in range(1, 7)}


def tile_url(tile: str, year: int) -> str:
    lat_label, lon_label = tile.split("_")
    return f"{BASE_URL}?type=tile&dataset=AnnualChange_{year}&lat={lat_label}&lon={lon_label}"


def raw_tile_path(tile: str):
    return io.raw_dir(SLUG) / f"JRC_TMF_AnnualChange_{YEAR}_{tile}.tif"


def download_tiles() -> None:
    """Download the chosen AnnualChange tiles for YEAR (idempotent, disk-guarded)."""
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    for tile in TILES:
        io.check_disk()  # tiles are large; re-check before each
        dst = raw_tile_path(tile)
        if dst.exists():
            print(f"  [skip] {dst.name} already present")
            continue
        url = tile_url(tile, YEAR)
        print(f"  downloading {tile} -> {dst.name}")
        download.download_http(url, dst)


def _scan_tile(tile: str) -> list[dict[str, Any]]:
    """Find homogeneous BLOCKxBLOCK native windows in one tile; return candidate records.

    A block qualifies if a single class is >= DOMINANCE_FLOOR of the block and the nodata
    fraction is <= MAX_NODATA_FRAC. The block's label is its dominant class.
    """
    path = str(raw_tile_path(tile))
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        st = ds.transform
    h, w = arr.shape
    nby, nbx = h // BLOCK, w // BLOCK
    a = arr[: nby * BLOCK, : nbx * BLOCK].reshape(nby, BLOCK, nbx, BLOCK)
    denom = float(BLOCK * BLOCK)
    best_frac = np.zeros((nby, nbx), np.float32)
    best_src = np.zeros((nby, nbx), np.uint8)  # source class value (1..6)
    for v in range(1, 7):
        cnt = (a == v).sum(axis=(1, 3)).astype(np.float32) / denom
        m = cnt > best_frac
        best_frac[m] = cnt[m]
        best_src[m] = v
    nod = (a == 0).sum(axis=(1, 3)).astype(np.float32) / denom
    qual = (best_frac >= DOMINANCE_FLOOR) & (nod <= MAX_NODATA_FRAC) & (best_src > 0)
    brs, bcs = np.nonzero(qual)
    # center native pixel of each qualifying block -> lon/lat via source transform
    cx = bcs * BLOCK + BLOCK / 2.0
    cy = brs * BLOCK + BLOCK / 2.0
    lons = st.c + cx * st.a
    lats = st.f + cy * st.e
    recs = []
    for br, bc, lon, lat in zip(
        brs.tolist(), bcs.tolist(), lons.tolist(), lats.tolist()
    ):
        src_v = int(best_src[br, bc])
        recs.append(
            {
                "tile": tile,
                "lon": float(lon),
                "lat": float(lat),
                "label": SRC_TO_ID[src_v],  # class id 0..5
                "frac": float(best_frac[br, bc]),
                "source_id": f"{tile}_r{br}_c{bc}",
            }
        )
    return recs


def _write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    lon, lat = rec["lon"], rec["lat"]
    dst_proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    dst_transform = get_transform_from_projection_and_bounds(dst_proj, bounds)

    # Geographic bbox of the UTM tile so we can window the source read.
    xs = [bounds[0] * io.RESOLUTION, bounds[2] * io.RESOLUTION]
    ys = [bounds[1] * -io.RESOLUTION, bounds[3] * -io.RESOLUTION]
    left, right = min(xs), max(xs)
    bottom, top = min(ys), max(ys)
    l2, b2, r2, t2 = transform_bounds(
        dst_proj.crs, "EPSG:4326", left, bottom, right, top
    )
    pad = 0.003  # ~330 m margin so the tile is fully covered before nearest-resampling

    with rasterio.open(str(raw_tile_path(rec["tile"]))) as ds:
        win = from_bounds(l2 - pad, b2 - pad, r2 + pad, t2 + pad, ds.transform)
        src = ds.read(1, window=win, boundless=True, fill_value=0)
        win_transform = ds.window_transform(win)
        src_crs = ds.crs

    src_state = np.zeros((TILE, TILE), np.uint8)
    reproject(
        source=src,
        destination=src_state,
        src_transform=win_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_proj.crs,
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0,
    )
    # Remap source values 1..6 -> class ids 0..5; 0 (nodata/outside) -> CLASS_NODATA.
    out = np.full((TILE, TILE), io.CLASS_NODATA, np.uint8)
    for v, cid in SRC_TO_ID.items():
        out[src_state == v] = cid

    io.write_label_geotiff(
        SLUG, sample_id, out, dst_proj, bounds, nodata=io.CLASS_NODATA
    )
    present = sorted(int(x) for x in np.unique(out) if x != io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        dst_proj,
        bounds,
        io.year_range(YEAR),
        source_id=rec["source_id"],
        classes_present=present,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()

    print("Downloading TMF AnnualChange tiles...")
    download_tiles()
    io.check_disk()

    print("Scanning tiles for homogeneous windows...")
    with multiprocessing.Pool(min(len(TILES), 8)) as p:
        all_recs: list[dict[str, Any]] = []
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan_tile, [dict(tile=t) for t in TILES]),
            total=len(TILES),
        ):
            all_recs.extend(recs)
    print(f"scanned {len(all_recs)} candidate homogeneous windows")

    # Balance to <=PER_CLASS per class (seeded random subsample per class).
    from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

    selected = balance_by_class(all_recs, "label", per_class=PER_CLASS)
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} windows (<= {PER_CLASS}/class)")

    io.check_disk()
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            pass

    counts = Counter(r["label"] for r in selected)
    class_counts = {name: counts.get(i, 0) for i, (name, _d) in enumerate(CLASSES)}
    print("class counts:", class_counts)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "JRC Tropical Moist Forest (TMF)",
            "task_type": "classification",
            "source": "EC JRC",
            "license": "free with attribution (no limitations on use)",
            "provenance": {
                "url": "https://forobs.jrc.ec.europa.eu/TMF/data",
                "have_locally": False,
                "annotation_method": "derived-product (JRC TMF AnnualChange, Landsat)",
                "citation": "Vancutsem et al. 2021, Science Advances, doi:10.1126/sciadv.abe1603",
                "product": "AnnualChange",
                "year": YEAR,
                "tiles": TILES,
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": class_counts,
            "notes": (
                "Bounded-tile sampling of the pan-tropical JRC TMF AnnualChange product: "
                f"{len(TILES)} representative 10x10-deg tiles across the Amazon, Congo and "
                f"SE-Asia basins, year {YEAR}. Homogeneous (>=50% dominant-class) 64x64 "
                "windows reprojected from native 30 m EPSG:4326 to local UTM at 10 m with "
                "nearest resampling. Per-year forest-state used as a classification label "
                "with a 1-year time range (no per-event change_time)."
            ),
        },
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
