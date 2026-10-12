"""Process Copernicus HRL Small Woody Features (SWF) into open-set-segmentation labels.

Source: Copernicus Land Monitoring Service, High Resolution Layer "Small Woody Features"
(EEA / Copernicus Land). Pan-European (EEA38 + UK) 5 m raster derived by photo-
interpretation of 2.5-5 m VHR imagery, marking hedgerows, tree rows, and small
woods/patches that are too small/narrow to be captured by the standard forest layers.
Reference year 2021 (also available: 2015, 2018).

The CLMS download portal is login-gated, but the products are published open-access as
public ArcGIS ImageServer layers on the EEA DiscoMap server (no credential needed):
    https://image.discomap.eea.europa.eu/arcgis/rest/services/GioLandPublic/
        HRL_SmallWoodyFeatures_2021_005m/ImageServer/exportImage
We pull raw 5 m pixel blocks via exportImage (EPSG:3035 / LAEA Europe, U8 thematic).

**5 m raster legend (2021):** 0 = Non-SWF area, 1 = SWF area  (binary presence mask).
(2018 uses 0=non-woody, 1=woody, 254=unclassified, 255=outside.) The manifest lists two
classes -- "linear woody features (hedgerows)" and "patchy woody features" -- but that
linear/patchy split lives only in the SWF *vector* product; the public 5 m raster is a
single woody-presence class. So this becomes a **2-class dense segmentation**:
    0 = non_woody (background; a REAL observed class, not a fabricated negative)
    1 = small_woody_feature (hedgerow / tree row / small wood)

**10 m observability (spec S4 VHR-native guidance).** The label is native 5 m. We
reproject to a local UTM grid at 10 m with **mode** resampling (categorical). A 10 m
pixel aggregates ~4 native 5 m pixels; mode keeps woody only where it occupies the
majority of the 10 m pixel -- i.e. it retains resolvable woody cover (multi-row
hedgerows, small woods, dense field boundaries) and coarsens away single-row sub-pixel
hedgerows. This is the intended behaviour: the fine sub-pixel features are recorded as
lost in the summary rather than fabricated at 10 m.

Large European product -> **bounded-tile sampling** (spec S5): we download 5 m blocks
over a spread of representative hedgerow/small-woody landscapes across Europe, scan them
for 64x64 (@10 m = 640 m) windows that contain woody cover, and balance to <=1000
tiles/class. We do NOT attempt full-continent coverage.

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.copernicus_hrl_small_woody_features
"""

import argparse
import multiprocessing
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import tqdm
from pyproj import Transformer
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds

from olmoearth_pretrain.open_set_segmentation_data import download, io
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    select_tiles_per_class,
)

SLUG = "copernicus_hrl_small_woody_features"

# Reference year: 2021 is the most recent SWF vintage and sits at the end of the manifest
# range (2016-2021), well inside the Sentinel era. Its 5 m raster is a clean binary mask.
YEAR = 2021
SERVICE = "HRL_SmallWoodyFeatures_2021_005m"
BASE_URL = (
    "https://image.discomap.eea.europa.eu/arcgis/rest/services/GioLandPublic/"
    f"{SERVICE}/ImageServer/exportImage"
)
SRC_CRS = "EPSG:3035"
NATIVE_RES = 5  # m

PER_CLASS = 1000
TILE = 64  # output tile: 64 px @ 10 m = 640 m
NATIVE_WIN = 128  # 128 px @ 5 m = 640 m (same ground footprint as one output tile)
BLOCK_PX = 2560  # 2560 px @ 5 m = 12.8 km block; 20x20 = 400 windows/block
WOODY_FLOOR = (
    0.02  # a window must be >= 2% woody (>=~328 native px) to count as containing SWF
)
MAX_NODATA_FRAC = (
    0.2  # reject windows that are mostly unclassified / outside the product
)

# Source pixel value -> output class id. 1 -> woody(1), 0 -> background(0), else -> nodata.
WOODY_VAL = 1
BG_VAL = 0

CLASSES = [
    (
        "non_woody",
        "Non-woody area: land observed by the Copernicus HRL SWF product as NOT belonging to "
        "a small woody feature (open land, cropland, grassland, built-up, water, or large "
        "forest that is captured by the forest layers rather than the SWF layer). A real "
        "observed background class, not a fabricated negative.",
    ),
    (
        "small_woody_feature",
        "Small woody feature: hedgerow, tree row, small wood/grove or small patchy woody "
        "vegetation, typically < ~0.5 ha or a narrow linear woody strip, mapped by photo-"
        "interpretation of 2.5-5 m VHR imagery. The manifest's linear-vs-patchy split is a "
        "vector-only attribute and is not present in the public 5 m raster, so both are merged "
        "into this single class. At 10 m the 5 m mask is mode-resampled, so only woody cover "
        "occupying the majority of a 10 m pixel is kept; single-row sub-pixel hedgerows are "
        "coarsened away.",
    ),
]

# Representative European landscapes with notable small-woody-feature / hedgerow density,
# spread across the EEA38+UK product. (name, lon, lat) of block center.
REGIONS = [
    ("brittany_fr", -2.8, 48.1),  # Breton bocage (dense hedgerow network)
    ("vendee_fr", -1.4, 46.7),  # W France bocage
    ("normandy_fr", 0.2, 48.9),  # Normandy bocage
    ("devon_uk", -3.9, 50.7),  # SW England hedgerows
    ("ireland_midlands", -7.6, 53.3),  # Irish field boundaries
    ("lower_saxony_de", 9.0, 52.7),  # N Germany Knicks/hedgebanks
    ("netherlands", 5.8, 52.0),  # NL wooded banks
    ("denmark", 9.4, 56.0),  # Danish hedgerows / shelterbelts
    ("galicia_es", -8.0, 42.8),  # NW Spain bocage
    ("po_valley_it", 10.0, 45.4),  # N Italy field trees / rows
    ("austria", 15.4, 47.2),  # SE Austria woody patches
    ("poland", 19.0, 52.2),  # C Poland mid-field woodlots / rows
]


def _region_bbox_3035(lon: float, lat: float) -> tuple[float, float, float, float]:
    """LAEA (EPSG:3035) bbox of a BLOCK_PX x BLOCK_PX @5 m block centered at lon/lat."""
    t = Transformer.from_crs("EPSG:4326", SRC_CRS, always_xy=True)
    cx, cy = t.transform(lon, lat)
    half = BLOCK_PX * NATIVE_RES / 2.0
    # snap origin to the 5 m grid so the returned raster aligns cleanly
    xmin = round((cx - half) / NATIVE_RES) * NATIVE_RES
    ymin = round((cy - half) / NATIVE_RES) * NATIVE_RES
    return (xmin, ymin, xmin + BLOCK_PX * NATIVE_RES, ymin + BLOCK_PX * NATIVE_RES)


def raw_block_path(region: str):
    return io.raw_dir(SLUG) / f"SWF_{YEAR}_{region}.tif"


def block_url(bbox: tuple[float, float, float, float]) -> str:
    xmin, ymin, xmax, ymax = bbox
    return (
        f"{BASE_URL}?bbox={xmin},{ymin},{xmax},{ymax}&bboxSR=3035&imageSR=3035"
        f"&size={BLOCK_PX},{BLOCK_PX}&format=tiff&pixelType=U8"
        f"&interpolation=RSP_NearestNeighbor&f=image"
    )


def download_blocks() -> None:
    """Download one 5 m SWF block per region (idempotent, disk-guarded)."""
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    for region, lon, lat in REGIONS:
        io.check_disk()
        dst = raw_block_path(region)
        if dst.exists():
            print(f"  [skip] {dst.name} present")
            continue
        url = block_url(_region_bbox_3035(lon, lat))
        print(f"  downloading {region} -> {dst.name}")
        download.download_http(url, dst)


def _scan_block(region: str) -> list[dict[str, Any]]:
    """Find NATIVE_WIN x NATIVE_WIN windows containing woody cover in one block."""
    path = str(raw_block_path(region))
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        st = ds.transform
    h, w = arr.shape
    nby, nbx = h // NATIVE_WIN, w // NATIVE_WIN
    a = arr[: nby * NATIVE_WIN, : nbx * NATIVE_WIN].reshape(
        nby, NATIVE_WIN, nbx, NATIVE_WIN
    )
    denom = float(NATIVE_WIN * NATIVE_WIN)
    woody = (a == WOODY_VAL).sum(axis=(1, 3)).astype(np.float32) / denom
    nod = (~np.isin(a, [WOODY_VAL, BG_VAL])).sum(axis=(1, 3)).astype(np.float32) / denom
    qual = (woody >= WOODY_FLOOR) & (nod <= MAX_NODATA_FRAC)
    brs, bcs = np.nonzero(qual)
    cx = bcs * NATIVE_WIN + NATIVE_WIN / 2.0
    cy = brs * NATIVE_WIN + NATIVE_WIN / 2.0
    xs = st.c + cx * st.a
    ys = st.f + cy * st.e
    tr = Transformer.from_crs(SRC_CRS, "EPSG:4326", always_xy=True)
    recs = []
    for br, bc, x3035, y3035 in zip(
        brs.tolist(), bcs.tolist(), xs.tolist(), ys.tolist()
    ):
        lon, lat = tr.transform(x3035, y3035)
        recs.append(
            {
                "region": region,
                "lon": float(lon),
                "lat": float(lat),
                "woody_frac": float(woody[br, bc]),
                "classes_present": [
                    BG_VAL,
                    WOODY_VAL,
                ],  # both classes present in the tile
                "source_id": f"{region}_r{br}_c{bc}",
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

    # Geographic bbox of the UTM tile so we can window-read the source block.
    xs = [bounds[0] * io.RESOLUTION, bounds[2] * io.RESOLUTION]
    ys = [bounds[1] * -io.RESOLUTION, bounds[3] * -io.RESOLUTION]
    left, right = min(xs), max(xs)
    bottom, top = min(ys), max(ys)
    l2, b2, r2, t2 = transform_bounds(dst_proj.crs, SRC_CRS, left, bottom, right, top)
    pad = 100.0  # metres of margin so the tile is fully covered before mode-resampling

    with rasterio.open(str(raw_block_path(rec["region"]))) as ds:
        win = from_bounds(l2 - pad, b2 - pad, r2 + pad, t2 + pad, ds.transform)
        src = ds.read(1, window=win, boundless=True, fill_value=255)
        win_transform = ds.window_transform(win)
        src_crs = ds.crs

    # Reproject 5 m -> UTM 10 m with MODE (categorical): keeps woody only where it is the
    # majority of the coarser 10 m pixel; sub-pixel single-row hedgerows are coarsened out.
    resampled = np.full((TILE, TILE), 255, np.uint8)
    reproject(
        source=src,
        destination=resampled,
        src_transform=win_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_proj.crs,
        resampling=Resampling.mode,
        src_nodata=255,
        dst_nodata=255,
    )
    out = np.full((TILE, TILE), io.CLASS_NODATA, np.uint8)
    out[resampled == BG_VAL] = 0
    out[resampled == WOODY_VAL] = 1

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

    print("Downloading SWF 5 m blocks...")
    download_blocks()
    io.check_disk()

    print("Scanning blocks for woody windows...")
    with multiprocessing.Pool(min(len(REGIONS), 12)) as p:
        all_recs: list[dict[str, Any]] = []
        for recs in tqdm.tqdm(
            star_imap_unordered(
                p, _scan_block, [dict(region=r) for r, _lo, _la in REGIONS]
            ),
            total=len(REGIONS),
        ):
            all_recs.extend(recs)
    print(f"scanned {len(all_recs)} candidate woody windows")

    # Tiles-per-class balanced: both classes present in every tile, so this caps at
    # ~PER_CLASS tiles (woody=bg=PER_CLASS) spread across regions.
    selected = select_tiles_per_class(all_recs, "classes_present", per_class=PER_CLASS)
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} windows")

    io.check_disk()
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            pass

    region_counts = Counter(r["region"] for r in selected)
    print("per-region counts:", dict(region_counts))

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "Copernicus HRL Small Woody Features",
            "task_type": "classification",
            "source": "EEA / Copernicus Land Monitoring Service",
            "license": "open (Copernicus data policy, free use with attribution)",
            "provenance": {
                "url": "https://land.copernicus.eu/en/products/high-resolution-layer-small-woody-features",
                "access": (
                    "public EEA DiscoMap ArcGIS ImageServer exportImage (no credential): "
                    f"GioLandPublic/{SERVICE}"
                ),
                "have_locally": False,
                "annotation_method": "photo-interpretation of 2.5-5 m VHR imagery",
                "reference_year": YEAR,
                "native_resolution_m": NATIVE_RES,
                "native_crs": SRC_CRS,
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "per_region_counts": dict(region_counts),
            "notes": (
                "Bounded-tile sampling of the pan-European Copernicus HRL Small Woody "
                f"Features 5 m raster (reference year {YEAR}) over {len(REGIONS)} "
                "representative hedgerow/small-woody landscapes across the EEA38+UK. Native "
                "5 m binary presence mask (0=non-woody, 1=SWF) reprojected from EPSG:3035 to "
                "local UTM at 10 m with MODE resampling; woody kept only where it is the "
                "majority of a 10 m pixel, coarsening away sub-pixel single-row hedgerows. "
                "Manifest linear/patchy split is vector-only (absent from the public raster) "
                "so both are merged into one small_woody_feature class. 1-year time range "
                f"anchored on {YEAR}."
            ),
        },
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
