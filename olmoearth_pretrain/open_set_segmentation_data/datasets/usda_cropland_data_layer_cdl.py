"""Process USDA Cropland Data Layer (CDL) into open-set-segmentation label patches.

Source: USDA NASS Cropland Data Layer -- an annual 30 m crop-specific land-cover raster
for the conterminous US (CONUS) with ~130 active categories, produced by a decision-tree
classifier trained on FSA farm-program ground truth plus NASA/USGS imagery
(https://www.nass.usda.gov/Research_and_Science/Cropland/). It is a derived-product MAP
(annotation_method: derived-product), but the major crop classes are high-accuracy, so per
the spec (§4 dense_raster; §5 large derived-product) we do BOUNDED-TILE sampling over
representative agricultural regions and prefer windows where a class has a confident
presence (>= PRESENT_FRAC of the tile).

Access (frugal -- no national download): the full-CONUS CDL is only distributed as
~2 GB/year national archives, but we only need bounded regional windows. We therefore use
the NASS CroplandCROS / CropScape web service ``GetCDLFile`` endpoint, which clips the CDL
to an arbitrary EPSG:5070 (CONUS Albers) bounding box and returns a small GeoTIFF (a 45 km
region is ~2 MB). We fetch one clip per (region, year) -- a few dozen MB total -- rather
than pulling any national raster (spec §5: "download only enough of the product to draw
the target count from representative regions").

Regions: ~16 boxes chosen to span the major US crop geographies and thereby cover the CDL
class taxonomy (Corn Belt corn/soy; Great Plains wheat/sorghum; Northern Plains
canola/sunflower/dry beans/sugarbeets/spring wheat; Mississippi/Louisiana rice/cotton/
sugarcane; California Central & Coastal Valleys almonds/pistachios/grapes/citrus/tomatoes/
vegetables; Texas High Plains cotton/sorghum; Southeast peanuts/cotton/tobacco; Pacific
Northwest & Idaho potatoes/apples/sugarbeets; Great Lakes alfalfa/cherries/blueberries/
cranberries; Florida citrus/sugarcane). Years 2021 and 2024 (early + late in the manifest
2016-2024 range) give temporal diversity; each annual label gets a 1-year time range
anchored on its CDL year.

Class scheme: CDL codes are remapped to a compact uint8 id space built from the observed
frequency of codes across the sampled windows (ids 0..N-1 in descending frequency), keeping
the top 254 (spec §5 uint8 254-class cap; CDL has ~130 active codes so nothing is dropped in
practice). CDL code 0 (Background/out-of-CONUS) and 81 (Clouds/No Data) are treated as
nodata (255) and never become classes. Native 30 m EPSG:5070 windows are reprojected to a
local UTM projection at 10 m with NEAREST resampling (categorical labels).

Selection is tiles-per-class balanced (rarest class first) up to 1000 tiles/class, capped at
the per-dataset 25,000-sample limit (spec §5); with ~130 classes the 25k cap lowers the
effective per-class target to ~190.

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.usda_cropland_data_layer_cdl
"""

import argparse
import multiprocessing
import time
import urllib.request
from collections import Counter
from typing import Any
from xml.etree import ElementTree

import numpy as np
import rasterio
import tqdm
from pyproj import Transformer
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds

from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    select_tiles_per_class,
)

SLUG = "usda_cropland_data_layer_cdl"

YEARS = [2021, 2024]

TILE = 64
BLOCK = 21  # native 30 m block ~= 630 m ~= a 64 px @ 10 m UTM tile footprint
PER_CLASS = 1000  # tiles-per-class target (25k total cap lowers effective per-class)
PRESENT_FRAC = 0.10  # a code counts as "present" if it covers >= 10% of a block
MAX_CLASSES = 254  # uint8 cap (ids 0..253; 255 = nodata)
REGION_HALF_M = 22500  # half-width of each region box (45 km square)

CROPSCAPE_URL = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile"

# CDL codes that are NOT usable classes -> mapped to nodata (255), never counted.
NODATA_CODES = {0, 81}  # 0 = Background/out-of-CONUS, 81 = Clouds/No Data

# Representative CONUS agricultural regions: (key, description, center_lon, center_lat).
REGIONS: list[tuple[str, str, float, float]] = [
    ("iowa_cornbelt", "Iowa Corn Belt - corn, soybeans", -93.6, 42.0),
    ("illinois", "Central Illinois - corn, soybeans", -89.0, 40.0),
    ("kansas_wheat", "Central Kansas - winter wheat, sorghum, corn", -98.3, 38.3),
    (
        "north_dakota",
        "North Dakota - spring wheat, canola, sunflower, dry beans, barley",
        -99.0,
        47.6,
    ),
    (
        "red_river_valley",
        "Red River Valley MN/ND - sugarbeets, potatoes, spring wheat, soybeans",
        -96.8,
        47.9,
    ),
    (
        "mississippi_delta",
        "Arkansas/Mississippi Delta - rice, soybeans, cotton",
        -91.2,
        34.6,
    ),
    ("louisiana", "South Louisiana - rice, sugarcane, soybeans", -92.2, 30.4),
    (
        "ca_central_valley_n",
        "N California Central Valley - rice, tomatoes, almonds, walnuts, grapes",
        -121.6,
        38.7,
    ),
    (
        "ca_central_valley_s",
        "S California Central Valley - almonds, pistachios, grapes, citrus, cotton",
        -119.6,
        36.3,
    ),
    (
        "ca_coast",
        "California Central Coast - grapes, strawberries, lettuce/vegetables",
        -120.6,
        35.0,
    ),
    (
        "texas_high_plains",
        "Texas High Plains - cotton, sorghum, corn, winter wheat",
        -101.9,
        34.2,
    ),
    ("georgia", "South Georgia - peanuts, cotton", -83.6, 31.5),
    (
        "north_carolina",
        "E North Carolina - tobacco, cotton, peanuts, soybeans",
        -78.0,
        35.4,
    ),
    (
        "wa_columbia_basin",
        "Washington Columbia Basin - potatoes, apples, wheat, corn",
        -119.3,
        46.6,
    ),
    (
        "idaho_snake",
        "Idaho Snake River Plain - potatoes, sugarbeets, barley, alfalfa",
        -113.8,
        42.9,
    ),
    (
        "great_lakes",
        "Michigan/Wisconsin - corn, alfalfa, cherries, blueberries, cranberries, sugarbeets",
        -85.5,
        43.6,
    ),
    ("florida", "S Florida - citrus/oranges, sugarcane, vegetables", -81.2, 26.9),
]

# Official USDA NASS CDL category legend (public domain). code -> category name.
CDL_LEGEND: dict[int, str] = {
    1: "Corn",
    2: "Cotton",
    3: "Rice",
    4: "Sorghum",
    5: "Soybeans",
    6: "Sunflower",
    10: "Peanuts",
    11: "Tobacco",
    12: "Sweet Corn",
    13: "Pop or Orn Corn",
    14: "Mint",
    21: "Barley",
    22: "Durum Wheat",
    23: "Spring Wheat",
    24: "Winter Wheat",
    25: "Other Small Grains",
    26: "Dbl Crop WinWht/Soybeans",
    27: "Rye",
    28: "Oats",
    29: "Millet",
    30: "Speltz",
    31: "Canola",
    32: "Flaxseed",
    33: "Safflower",
    34: "Rape Seed",
    35: "Mustard",
    36: "Alfalfa",
    37: "Other Hay/Non Alfalfa",
    38: "Camelina",
    39: "Buckwheat",
    41: "Sugarbeets",
    42: "Dry Beans",
    43: "Potatoes",
    44: "Other Crops",
    45: "Sugarcane",
    46: "Sweet Potatoes",
    47: "Misc Vegs & Fruits",
    48: "Watermelons",
    49: "Onions",
    50: "Cucumbers",
    51: "Chick Peas",
    52: "Lentils",
    53: "Peas",
    54: "Tomatoes",
    55: "Caneberries",
    56: "Hops",
    57: "Herbs",
    58: "Clover/Wildflowers",
    59: "Sod/Grass Seed",
    60: "Switchgrass",
    61: "Fallow/Idle Cropland",
    63: "Forest",
    64: "Shrubland",
    65: "Barren",
    66: "Cherries",
    67: "Peaches",
    68: "Apples",
    69: "Grapes",
    70: "Christmas Trees",
    71: "Other Tree Crops",
    72: "Citrus",
    74: "Pecans",
    75: "Almonds",
    76: "Walnuts",
    77: "Pears",
    81: "Clouds/No Data",
    82: "Developed",
    83: "Water",
    87: "Wetlands",
    88: "Nonag/Undefined",
    92: "Aquaculture",
    111: "Open Water",
    112: "Perennial Ice/Snow",
    121: "Developed/Open Space",
    122: "Developed/Low Intensity",
    123: "Developed/Med Intensity",
    124: "Developed/High Intensity",
    131: "Barren",
    141: "Deciduous Forest",
    142: "Evergreen Forest",
    143: "Mixed Forest",
    152: "Shrubland",
    176: "Grassland/Pasture",
    190: "Woody Wetlands",
    195: "Herbaceous Wetlands",
    204: "Pistachios",
    205: "Triticale",
    206: "Carrots",
    207: "Asparagus",
    208: "Garlic",
    209: "Cantaloupes",
    210: "Prunes",
    211: "Olives",
    212: "Oranges",
    213: "Honeydew Melons",
    214: "Broccoli",
    215: "Avocados",
    216: "Peppers",
    217: "Pomegranates",
    218: "Nectarines",
    219: "Greens",
    220: "Plums",
    221: "Strawberries",
    222: "Squash",
    223: "Apricots",
    224: "Vetch",
    225: "Dbl Crop WinWht/Corn",
    226: "Dbl Crop Oats/Corn",
    227: "Lettuce",
    228: "Dbl Crop Triticale/Corn",
    229: "Pumpkins",
    230: "Dbl Crop Lettuce/Durum Wht",
    231: "Dbl Crop Lettuce/Cantaloupe",
    232: "Dbl Crop Lettuce/Cotton",
    233: "Dbl Crop Lettuce/Barley",
    234: "Dbl Crop Durum Wht/Sorghum",
    235: "Dbl Crop Barley/Sorghum",
    236: "Dbl Crop WinWht/Sorghum",
    237: "Dbl Crop Barley/Corn",
    238: "Dbl Crop WinWht/Cotton",
    239: "Dbl Crop Soybeans/Cotton",
    240: "Dbl Crop Soybeans/Oats",
    241: "Dbl Crop Corn/Soybeans",
    242: "Blueberries",
    243: "Cabbage",
    244: "Cauliflower",
    245: "Celery",
    246: "Radishes",
    247: "Turnips",
    248: "Eggplants",
    249: "Gourds",
    250: "Cranberries",
    254: "Dbl Crop Barley/Soybeans",
}

# EPSG:5070 (CONUS Albers) is the CDL native CRS.
_TO_5070 = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
_TO_4326 = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)


def clip_path(region_key: str, year: int):
    return io.raw_dir(SLUG) / "clips" / f"cdl_{year}_{region_key}.tif"


def region_bbox_5070(center_lon: float, center_lat: float) -> tuple[int, int, int, int]:
    """EPSG:5070 (x1,y1,x2,y2) box of REGION_HALF_M around a lon/lat center."""
    x, y = _TO_5070.transform(center_lon, center_lat)
    h = REGION_HALF_M
    return (int(x - h), int(y - h), int(x + h), int(y + h))


def _download(url: str, dst) -> Any:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / (dst.name + ".tmp")
    with urllib.request.urlopen(url, timeout=300) as r, tmp.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dst)
    return dst


def download_all_clips() -> list[tuple[str, int]]:
    """Fetch every region-year clip. Returns the list of (region_key, year) available."""
    (io.raw_dir(SLUG) / "clips").mkdir(parents=True, exist_ok=True)
    available: list[tuple[str, int]] = []
    for year in YEARS:
        for region_key, _desc, lon, lat in REGIONS:
            io.check_disk()
            dst = clip_path(region_key, year)
            if dst.exists():
                available.append((region_key, year))
                continue
            x1, y1, x2, y2 = region_bbox_5070(lon, lat)
            query = f"{CROPSCAPE_URL}?year={year}&bbox={x1},{y1},{x2},{y2}"
            ok = False
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(query, timeout=180) as r:
                        xml = r.read().decode()
                    node = ElementTree.fromstring(xml).find(".//returnURL")
                    if node is None or not node.text:
                        raise RuntimeError(f"no returnURL: {xml[:200]}")
                    _download(node.text.strip(), dst)
                    ok = True
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"  [retry {attempt}] {region_key} {year}: {e}")
                    time.sleep(5 * (attempt + 1))
            if ok:
                print(f"  fetched {dst.name}")
                available.append((region_key, year))
            else:
                print(f"  [FAILED] {region_key} {year} - skipping this region-year")
    return available


def _scan_clip(region_key: str, year: int) -> list[dict[str, Any]]:
    """Scan non-overlapping BLOCKxBLOCK native windows of a clip; return candidate records.

    Each record lists the raw CDL codes present (>= PRESENT_FRAC of the block, excluding
    nodata codes) plus the block-center lon/lat and a source id.
    """
    path = str(clip_path(region_key, year))
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        st = ds.transform
    h, w = arr.shape
    nby, nbx = h // BLOCK, w // BLOCK
    if nby == 0 or nbx == 0:
        return []
    a = arr[: nby * BLOCK, : nbx * BLOCK]
    denom = float(BLOCK * BLOCK)
    recs: list[dict[str, Any]] = []
    for br in range(nby):
        for bc in range(nbx):
            block = a[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            codes, counts = np.unique(block, return_counts=True)
            present = [
                int(code)
                for code, cnt in zip(codes.tolist(), counts.tolist())
                if code not in NODATA_CODES and (cnt / denom) >= PRESENT_FRAC
            ]
            if not present:
                continue
            cx = st.c + (bc * BLOCK + BLOCK / 2.0) * st.a
            cy = st.f + (br * BLOCK + BLOCK / 2.0) * st.e
            lon, lat = _TO_4326.transform(cx, cy)
            recs.append(
                {
                    "region": region_key,
                    "year": year,
                    "lon": float(lon),
                    "lat": float(lat),
                    "codes": present,
                    "source_id": f"{region_key}_{year}_r{br}_c{bc}",
                }
            )
    return recs


def _build_lut(code_to_id: dict[int, int]) -> np.ndarray:
    """256-entry LUT: raw CDL code -> compact id, unmapped -> CLASS_NODATA."""
    lut = np.full(256, io.CLASS_NODATA, dtype=np.uint8)
    for code, cid in code_to_id.items():
        lut[code] = cid
    return lut


def _write_one(rec: dict[str, Any], code_to_id: dict[int, int]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    lon, lat = rec["lon"], rec["lat"]
    dst_proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    dst_transform = get_transform_from_projection_and_bounds(dst_proj, bounds)

    xs = [bounds[0] * io.RESOLUTION, bounds[2] * io.RESOLUTION]
    ys = [bounds[1] * -io.RESOLUTION, bounds[3] * -io.RESOLUTION]
    left, right = min(xs), max(xs)
    bottom, top = min(ys), max(ys)
    # Source is EPSG:5070 (CONUS Albers).
    l2, b2, r2, t2 = transform_bounds(
        dst_proj.crs, "EPSG:5070", left, bottom, right, top
    )
    pad = 60.0  # ~2 native px margin so the tile is fully covered before nearest-resampling

    with rasterio.open(str(clip_path(rec["region"], rec["year"]))) as ds:
        win = from_bounds(l2 - pad, b2 - pad, r2 + pad, t2 + pad, ds.transform)
        src = ds.read(1, window=win, boundless=True, fill_value=0)
        win_transform = ds.window_transform(win)
        src_crs = ds.crs

    dst = np.zeros((TILE, TILE), np.uint8)
    reproject(
        source=src,
        destination=dst,
        src_transform=win_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_proj.crs,
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0,
    )
    lut = _build_lut(code_to_id)
    out = lut[dst]  # remap raw codes -> compact ids; unmapped/background -> 255

    io.write_label_geotiff(
        SLUG, sample_id, out, dst_proj, bounds, nodata=io.CLASS_NODATA
    )
    present = sorted(int(x) for x in np.unique(out) if x != io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        dst_proj,
        bounds,
        io.year_range(rec["year"]),
        source_id=rec["source_id"],
        classes_present=present,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()

    print("Fetching CDL regional clips via CropScape (frugal, no national download)...")
    available = download_all_clips()
    io.check_disk()
    if not available:
        raise RuntimeError("no CDL clips fetched")

    print(f"Scanning {len(available)} region-year clips for candidate windows...")
    with multiprocessing.Pool(min(len(available), 16)) as p:
        all_recs: list[dict[str, Any]] = []
        for recs in tqdm.tqdm(
            star_imap_unordered(
                p, _scan_clip, [dict(region_key=r, year=y) for (r, y) in available]
            ),
            total=len(available),
        ):
            all_recs.extend(recs)
    print(f"scanned {len(all_recs)} candidate windows")

    # Build the compact class scheme from observed code frequency (windows containing code),
    # keep top MAX_CLASSES (uint8 cap). CDL has ~130 active codes so nothing drops in practice.
    code_freq: Counter = Counter()
    for r in all_recs:
        for c in set(r["codes"]):
            code_freq[c] += 1
    ordered = [c for c, _ in code_freq.most_common()]
    dropped = ordered[MAX_CLASSES:]
    kept = ordered[:MAX_CLASSES]
    code_to_id = {code: i for i, code in enumerate(kept)}
    print(f"{len(kept)} classes kept (dropped {len(dropped)} beyond {MAX_CLASSES}-cap)")

    # Map each record's raw codes -> compact ids; drop records with nothing left.
    records: list[dict[str, Any]] = []
    for r in all_recs:
        ids = sorted({code_to_id[c] for c in r["codes"] if c in code_to_id})
        if not ids:
            continue
        r["present_ids"] = ids
        records.append(r)

    selected = select_tiles_per_class(
        records, classes_key="present_ids", per_class=PER_CLASS
    )
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(
        f"selected {len(selected)} windows (tiles-per-class, <= {PER_CLASS}/class, 25k cap)"
    )

    io.check_disk()
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(
                p, _write_one, [dict(rec=r, code_to_id=code_to_id) for r in selected]
            ),
            total=len(selected),
        ):
            pass

    # Report tiles-per-class counts (a tile counts toward every class present in it).
    id_to_code = {i: code for code, i in code_to_id.items()}
    class_counts: dict[str, int] = {}
    for r in selected:
        for cid in r["present_ids"]:
            name = CDL_LEGEND.get(id_to_code[cid], f"code_{id_to_code[cid]}")
            class_counts[name] = class_counts.get(name, 0) + 1

    classes_meta = [
        {
            "id": cid,
            "name": CDL_LEGEND.get(id_to_code[cid], f"code_{id_to_code[cid]}"),
            "cdl_code": id_to_code[cid],
            "description": None,
        }
        for cid in range(len(code_to_id))
    ]

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "USDA Cropland Data Layer (CDL)",
            "task_type": "classification",
            "source": "USDA NASS",
            "license": "public domain",
            "provenance": {
                "url": "https://www.nass.usda.gov/Research_and_Science/Cropland/",
                "have_locally": False,
                "annotation_method": "derived-product (trained on FSA ground truth)",
                "access": "CropScape GetCDLFile web service (regional EPSG:5070 clips)",
                "years": YEARS,
                "regions": {k: d for k, d, _lon, _lat in REGIONS},
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes_meta,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": class_counts,
            "notes": (
                "Bounded-tile sampling of the USDA NASS CDL derived-product map. Regional "
                f"clips ({len(REGIONS)} representative CONUS ag regions x {len(YEARS)} years) "
                "were pulled via the CropScape GetCDLFile web service in EPSG:5070; no "
                "national raster was downloaded. Non-overlapping ~64px-footprint (630 m) "
                "windows were scanned; a CDL code counts as present in a window when it "
                f"covers >= {int(PRESENT_FRAC * 100)}% of the block (high-confidence "
                "presence). CDL code 0 (Background) and 81 (Clouds/No Data) are mapped to "
                "nodata (255). Raw CDL codes were remapped to a compact uint8 id space by "
                f"descending frequency, keeping the top {MAX_CLASSES} (uint8 cap); each "
                "class's original CDL code is recorded in classes[].cdl_code. Windows were "
                "selected tiles-per-class (rarest class first) up to 1000/class, capped at "
                "the 25,000-sample per-dataset limit. Native 30 m EPSG:5070 reprojected to "
                "local UTM at 10 m with nearest resampling. Each sample's time range is the "
                "1-year window of its CDL year."
            ),
        },
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
