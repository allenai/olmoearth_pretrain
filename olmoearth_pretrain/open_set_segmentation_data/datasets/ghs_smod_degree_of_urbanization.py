"""Process GHS-SMOD (Degree of Urbanization) into open-set-segmentation label patches.

Source: EC JRC / GHSL GHS-SMOD R2023A "Settlement Model" grid, which encodes the UN
Degree of Urbanisation (DEGURBA) level-2 rural-urban classification per grid cell:

    10 = Water grid cell
    11 = Very low density rural grid cell
    12 = Low density rural grid cell
    13 = Rural cluster grid cell
    21 = Suburban / peri-urban grid cell
    22 = Semi-dense urban cluster grid cell
    23 = Dense urban cluster grid cell
    30 = Urban centre grid cell
   -200 = no data

The product is distributed globally as a single-band raster in Mollweide (ESRI:54009) at
**1000 m** native resolution (GHS_SMOD_E<epoch>_GLOBE_R2023A_54009_1000). We treat the
settlement class of a cell as a per-pixel **classification** label with a 1-year time
range anchored on the chosen epoch.

This is a GLOBAL derived-product map, so per the spec we do BOUNDED-TILE sampling: we
download only the single (small, ~29 MB) global 1 km file for ONE epoch, then draw a
class-balanced set of grid cells globally (<=1000 cells/class), and around each selected
cell cut a 64x64 label tile in a local UTM projection at **10 m**, reprojected from the
1 km Mollweide source with **nearest** resampling (categorical labels). Because a 64x64
@10 m tile (640 m) is smaller than one native 1 km cell, each tile is essentially the
homogeneous settlement class at that location -- this heavy 1 km -> 10 m upsampling is
intentional and documented (the DEGURBA class is defined on the 1 km grid).

Manifest classes (7) collapse the 8 source codes by merging very-low (11) + low (12)
density rural into a single "very-low/low-density rural" class.

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.ghs_smod_degree_of_urbanization
"""

import argparse
import multiprocessing
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import tqdm
from rasterio.warp import Resampling, reproject, transform, transform_bounds
from rasterio.windows import from_bounds
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds

from olmoearth_pretrain.open_set_segmentation_data import download, io

SLUG = "ghs_smod_degree_of_urbanization"

# One representative epoch within the manifest range 2016-2025. E2020 is a recent,
# Sentinel-era GHSL epoch; the settlement model is a per-epoch label.
YEAR = 2020

PER_CLASS = 1000
TILE = 64
SRC_CRS = "ESRI:54009"  # World Mollweide
SRC_NODATA = -200

BASE_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_SMOD_GLOBE_R2023A/"
    "GHS_SMOD_E{year}_GLOBE_R2023A_54009_1000/V1-0/"
    "GHS_SMOD_E{year}_GLOBE_R2023A_54009_1000_V1_0.zip"
)

# Manifest class order -> (name, description, [source codes]). id = position in list.
# Descriptions follow the GHSL DEGURBA level-2 settlement-model definitions.
CLASSES: list[tuple[str, str, list[int]]] = [
    (
        "water",
        "Water grid cell: 1 km grid cell where the majority of the surface is permanent "
        "water according to the GHSL land/water mask.",
        [10],
    ),
    (
        "very-low/low-density rural",
        "Rural grid cells outside any settlement cluster: very-low-density rural (mostly "
        "uninhabited / sparsely built) and low-density rural cells that belong to neither an "
        "urban cluster nor a rural cluster.",
        [11, 12],
    ),
    (
        "rural cluster",
        "Rural cluster grid cell: part of a contiguous cluster of cells with total population "
        ">=500 and density >=300 inh/km2 that does not reach urban-cluster size (village scale).",
        [13],
    ),
    (
        "suburban",
        "Suburban / peri-urban grid cell: part of an urban cluster (>=5,000 population, "
        ">=300 inh/km2) that is not classified as a dense or semi-dense urban cluster cell.",
        [21],
    ),
    (
        "semi-dense urban cluster",
        "Semi-dense urban cluster grid cell: urban-cluster cell located at least 3 km away "
        "from the nearest urban centre.",
        [22],
    ),
    (
        "dense urban cluster",
        "Dense urban cluster grid cell: urban-cluster cell adjacent to / near an urban centre "
        "with higher density, not meeting the urban-centre thresholds.",
        [23],
    ),
    (
        "urban centre",
        "Urban centre grid cell: part of a contiguous high-density cluster with density "
        ">=1,500 inh/km2 (or built-up >=50%) and total cluster population >=50,000.",
        [30],
    ),
]
# source code -> class id
SRC_TO_ID: dict[int, int] = {}
for _cid, (_n, _d, _codes) in enumerate(CLASSES):
    for _code in _codes:
        SRC_TO_ID[_code] = _cid


def raw_path():
    return io.raw_dir(SLUG) / f"GHS_SMOD_E{YEAR}_GLOBE_R2023A_54009_1000_V1_0.tif"


def download_source() -> None:
    """Download + unzip the global 1 km GHS-SMOD file for YEAR (idempotent, disk-guarded)."""
    io.check_disk()
    tif = raw_path()
    if tif.exists():
        print(f"  [skip] {tif.name} already present")
        return
    import zipfile

    zip_dst = io.raw_dir(SLUG) / f"GHS_SMOD_E{YEAR}_R2023A_54009_1000.zip"
    url = BASE_URL.format(year=YEAR)
    print(f"  downloading {url}")
    download.download_http(url, zip_dst)
    print("  unzipping")
    with zipfile.ZipFile(zip_dst.path) as zf:
        zf.extractall(io.raw_dir(SLUG).path)


# ---- worker: opened once per process via initializer ----
_SRC = None


def _init_worker() -> None:
    global _SRC
    _SRC = rasterio.open(str(raw_path()))


def _write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    lon, lat = rec["lon"], rec["lat"]
    dst_proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    dst_transform = get_transform_from_projection_and_bounds(dst_proj, bounds)

    # Geographic extent of the UTM tile (metres) -> window in the Mollweide source.
    xs = [bounds[0] * io.RESOLUTION, bounds[2] * io.RESOLUTION]
    ys = [bounds[1] * -io.RESOLUTION, bounds[3] * -io.RESOLUTION]
    left, right = min(xs), max(xs)
    bottom, top = min(ys), max(ys)
    l2, b2, r2, t2 = transform_bounds(dst_proj.crs, SRC_CRS, left, bottom, right, top)
    pad = 2000.0  # ~2 native cells of margin so the tile is fully covered

    ds = _SRC
    win = from_bounds(l2 - pad, b2 - pad, r2 + pad, t2 + pad, ds.transform)
    src = ds.read(1, window=win, boundless=True, fill_value=SRC_NODATA)
    win_transform = ds.window_transform(win)

    dst_arr = np.full((TILE, TILE), SRC_NODATA, dtype=np.int16)
    reproject(
        source=src,
        destination=dst_arr,
        src_transform=win_transform,
        src_crs=ds.crs,
        dst_transform=dst_transform,
        dst_crs=dst_proj.crs,
        resampling=Resampling.nearest,
        src_nodata=SRC_NODATA,
        dst_nodata=SRC_NODATA,
    )
    out = np.full((TILE, TILE), io.CLASS_NODATA, np.uint8)
    for v, cid in SRC_TO_ID.items():
        out[dst_arr == v] = cid

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


def _sample_candidates() -> list[dict[str, Any]]:
    """Draw up to PER_CLASS grid-cell centres per class, globally, from the source raster."""
    with rasterio.open(str(raw_path())) as ds:
        a = ds.read(1)
        st = ds.transform
        width = ds.width
    rng = np.random.default_rng(42)
    recs: list[dict[str, Any]] = []
    for cid, (name, _desc, codes) in enumerate(CLASSES):
        if len(codes) == 1:
            idx = np.flatnonzero(a == codes[0])
        else:
            idx = np.flatnonzero(np.isin(a, codes))
        n_total = len(idx)
        if n_total > PER_CLASS:
            idx = rng.choice(idx, PER_CLASS, replace=False)
        rows = (idx // width).astype(np.int64)
        cols = (idx % width).astype(np.int64)
        # cell-centre coords in Mollweide
        mx = st.c + st.a * (cols + 0.5)
        my = st.f + st.e * (rows + 0.5)
        lons, lats = transform(SRC_CRS, "EPSG:4326", mx.tolist(), my.tolist())
        for r, c, lon, lat in zip(rows.tolist(), cols.tolist(), lons, lats):
            recs.append(
                {
                    "lon": float(lon),
                    "lat": float(lat),
                    "label": cid,
                    "source_id": f"r{r}_c{c}",
                }
            )
        print(
            f"  class {cid} ({name}): {n_total} cells -> {min(n_total, PER_CLASS)} sampled"
        )
    return recs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest_write_start()

    print("Downloading GHS-SMOD global 1 km file...")
    download_source()
    io.check_disk()

    print("Sampling class-balanced grid cells...")
    selected = _sample_candidates()
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} cells (<= {PER_CLASS}/class)")

    io.check_disk()
    with multiprocessing.Pool(args.workers, initializer=_init_worker) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            pass

    counts = Counter(r["label"] for r in selected)
    class_counts = {name: counts.get(i, 0) for i, (name, _d, _c) in enumerate(CLASSES)}
    print("class counts:", class_counts)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "GHS-SMOD (Degree of Urbanization)",
            "task_type": "classification",
            "source": "EC JRC / GHSL",
            "license": "open + attribution (CC BY 4.0)",
            "provenance": {
                "url": "https://human-settlement.emergency.copernicus.eu/ghs_smod2023.php",
                "have_locally": False,
                "annotation_method": "authoritative/model (GHSL DEGURBA settlement model)",
                "product": "GHS_SMOD_R2023A_54009_1000",
                "epoch": YEAR,
                "native_resolution_m": 1000,
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc, _c) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": class_counts,
            "notes": (
                "Bounded-tile sampling of the GLOBAL JRC GHSL GHS-SMOD R2023A product "
                f"(epoch {YEAR}). The single global 1 km Mollweide (ESRI:54009) file was "
                "downloaded; up to 1000 grid cells per class were sampled globally and "
                "class-balanced. Around each cell a 64x64 tile in local UTM at 10 m was cut "
                "and reprojected from 1 km with NEAREST resampling (categorical). Source "
                "codes 11 and 12 (very-low + low density rural) were merged into one class. "
                "NOTE the heavy 1 km -> 10 m upsampling: a 64x64 @10 m tile (640 m) is "
                "smaller than one native cell, so each tile is essentially the homogeneous "
                "DEGURBA class at that location (the class is defined on the 1 km grid)."
            ),
        },
    )
    from olmoearth_pretrain.open_set_segmentation_data import manifest

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


def manifest_write_start() -> None:
    from olmoearth_pretrain.open_set_segmentation_data import manifest

    manifest.write_registry_entry(SLUG, "in_progress")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
