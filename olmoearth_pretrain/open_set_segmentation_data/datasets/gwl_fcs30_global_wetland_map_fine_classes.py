"""Process GWL_FCS30 Global Wetland Map (fine classes) into open-set-segmentation labels.

Source: Zhang et al. 2023, ESSD "GWL_FCS30: a global 30 m wetland map with a fine
classification system using multi-sourced and time-series remote sensing imagery in 2020"
(doi:10.5281/zenodo.7340516; https://essd.copernicus.org/articles/15/265/2023/). The map
was generated on GEE by fusing pre-existing wetland products with time-series Landsat /
Sentinel observations for the 2020 epoch. Distributed on Zenodo as 12 longitude-band ZIPs
(~2.4 GB total) of 5x5-deg GeoTIFF tiles (EPSG:4326, ~0.00027 deg ~= 30 m/px, uint8).

Raster value legend (verified by reading tiles + the GEE-community catalog / paper):
    0   = non-wetland / background        -> nodata (255)
    180 = permanent water
    181 = swamp
    182 = marsh
    183 = flooded flat
    184 = saline (inland saline wetland)
    185 = mangrove forest
    186 = salt marsh
    187 = tidal flat

This is a wetland-only product defined inside a wetland mask (0 = everything else), so per
spec §2 non-wetland is set to nodata=255 rather than inventing a background class.

Output class ids (uint8), in ascending source-value order:
    0 permanent water   (180)   4 saline        (184)
    1 swamp             (181)   5 mangrove      (185)
    2 marsh             (182)   6 salt marsh    (186)
    3 flooded flat      (183)   7 tidal flat    (187)

task_type=classification, dense_raster. 2020 is a static per-year state, so change_time is
null and the time range is a 1-year window on 2020 (§5). (The manifest's [2016, 2022] is
the product's temporal-validity envelope; the map itself is the 2020 epoch.)

GLOBAL derived-product => BOUNDED-TILE sampling (spec §5). We download the 12 band ZIPs
(only ~2.4 GB total -- well under any bulk-download concern -- so we take the full set for
flexibility rather than a partial pull), then extract only a curated set of 5x5-deg tiles
covering wetland-rich regions on every continent and every one of the 8 classes. Each tile
is scanned on its native 30 m grid for spatially-homogeneous ~640 m windows where a single
wetland class dominates the wetland pixels (>=80% of wetland pixels are one class) and the
window carries a meaningful amount of that wetland (>=15% of the block) -- §4 guidance to
prefer homogeneous/high-confidence windows for derived-product maps. Windows are balanced
tiles-per-class (<=1000/class) and reprojected from native EPSG:4326 to a local UTM
projection at 10 m with NEAREST resampling (categorical labels). Non-wetland stays nodata.

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.gwl_fcs30_global_wetland_map_fine_classes
"""

import argparse
import math
import multiprocessing
import zipfile
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import tqdm
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds

from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "gwl_fcs30_global_wetland_map_fine_classes"
ZENODO_RECORD = "7340516"

YEAR = 2020
PER_CLASS = 1000
TILE = 64  # output UTM tile is 64x64 @ 10 m (= 640 m)
BLOCK = 22  # native ~30 m block ~= 660 m ~= one 64 px @ 10 m UTM tile
MIN_WETLAND = 0.15  # candidate block must be >=15% wetland (meaningful signal)
DOM_OF_WETLAND = (
    0.8  # ... and one class must be >=80% of the wetland pixels (homogeneous)
)
PAD_DEG = 0.004  # ~440 m geographic pad so the reprojected UTM tile is fully covered

# source raster value -> output class id ; source 0 -> nodata (non-wetland)
SRC_TO_ID = {180: 0, 181: 1, 182: 2, 183: 3, 184: 4, 185: 5, 186: 6, 187: 7}
WET_VALUES = tuple(SRC_TO_ID.keys())

# Output classes (id order) with definitions from Zhang et al. 2023 (ESSD 15, 265).
CLASSES = [
    (
        "permanent water",
        "Inland open freshwater bodies that hold water year-round: lakes, reservoirs, ponds "
        "and wide rivers. GWL_FCS30 inland sub-category (value 180).",
    ),
    (
        "swamp",
        "Forested / woody wetlands -- wetlands dominated by trees and shrubs (e.g. peat "
        "swamp forest, floodplain forest). GWL_FCS30 inland sub-category (value 181).",
    ),
    (
        "marsh",
        "Herbaceous freshwater wetlands dominated by emergent grasses, sedges and reeds on "
        "waterlogged mineral or organic soils. GWL_FCS30 inland sub-category (value 182).",
    ),
    (
        "flooded flat",
        "Seasonally / periodically inundated flats and floodplains that are bare or sparsely "
        "vegetated during the flooded period. GWL_FCS30 inland sub-category (value 183).",
    ),
    (
        "saline",
        "Inland saline wetlands -- salt lakes, salt pans/playas and saline marshes with high "
        "salinity substrate (e.g. altiplano salars, Central-Asian salt lakes). GWL_FCS30 "
        "inland sub-category (value 184).",
    ),
    (
        "mangrove",
        "Coastal intertidal forests of salt-tolerant mangrove trees along tropical/subtropical "
        "shorelines and estuaries. GWL_FCS30 coastal tidal sub-category (value 185).",
    ),
    (
        "salt marsh",
        "Coastal intertidal herbaceous marshes (halophytic grasses/succulents) in the upper "
        "tidal zone of temperate and subtropical coasts. GWL_FCS30 coastal tidal sub-category "
        "(value 186).",
    ),
    (
        "tidal flat",
        "Unvegetated coastal intertidal mud/sand flats exposed at low tide (e.g. Yellow Sea, "
        "Wadden Sea, Amazon coast). GWL_FCS30 coastal tidal sub-category (value 187).",
    ),
]

# Wetland-rich (lon, lat) sampling regions across continents, chosen to collectively cover
# all 8 classes. Each maps to the 5x5-deg tile that contains it (deduplicated at runtime);
# a single tile typically supplies several classes, so the class assignment is by the
# dominant wetland class found in each homogeneous window rather than by the region tag.
REGIONS: dict[str, tuple[float, float]] = {
    # --- mangrove (coastal tropics) ---
    "sundarbans": (89.0, 22.0),
    "sumatra_east": (104.0, -1.0),
    "kalimantan_coast": (109.0, -1.0),
    "papua_south": (138.0, -8.0),
    "amazon_coast_gf": (-51.0, 3.0),
    "niger_delta": (6.0, 5.0),
    "florida_mangrove": (-81.0, 25.0),
    "australia_nt": (132.0, -12.0),
    "mozambique_coast": (36.5, -18.0),
    "philippines": (122.0, 11.0),
    # --- salt marsh (temperate coasts) ---
    "georgia_coast_us": (-81.0, 32.0),
    "chesapeake": (-76.0, 38.0),
    "louisiana_marsh": (-91.0, 29.0),
    "wadden_sea": (8.0, 53.0),
    "uk_wash": (0.0, 53.0),
    "jiangsu_coast": (120.5, 33.0),
    "patagonia_coast": (-65.0, -43.0),
    "san_francisco_bay": (-122.0, 38.0),
    # --- tidal flat (macrotidal coasts) ---
    "yellow_sea_korea": (126.0, 37.0),
    "bohai_china": (118.5, 39.0),
    "nw_australia": (122.0, -18.0),
    "amazon_coast_para": (-49.0, 0.0),
    "wash_uk2": (-3.0, 54.0),
    # --- swamp (forested wetlands) ---
    "congo_cuvette": (18.0, 0.0),
    "congo_east2": (20.0, 1.0),
    "amazon_swamp": (-63.0, -3.0),
    "kalimantan_peat": (113.0, -2.0),
    "sumatra_peat": (103.0, -1.0),
    "atchafalaya": (-91.5, 30.5),
    "pantanal_swamp": (-57.0, -18.0),
    # --- marsh (inland herbaceous) ---
    "everglades": (-81.0, 26.0),
    "prairie_potholes": (-99.0, 47.0),
    "west_siberia": (75.0, 61.0),
    "ob_river": (70.0, 60.0),
    "sudd_south_sudan": (30.0, 8.0),
    "poyang_dongting": (116.0, 29.0),
    "camargue": (4.5, 43.5),
    "biebrza_poland": (22.5, 53.5),
    # --- flooded flat (floodplains / seasonal) ---
    "amazon_floodplain": (-64.0, -3.5),
    "ganges_delta": (89.0, 24.0),
    "mekong_floodplain": (105.0, 12.0),
    "okavango": (23.0, -19.0),
    "niger_inland_delta": (-4.0, 15.0),
    "brahmaputra": (92.0, 26.0),
    # --- saline (salt lakes / pans) ---
    "kazakhstan_salt": (60.0, 46.0),
    "caspian_depression": (52.0, 47.0),
    "altiplano_salars": (-67.0, -21.0),
    "australia_salt": (137.0, -29.0),
    "australia_salt2": (122.0, -30.0),
    "chott_tunisia": (8.0, 34.0),
    "iran_playa": (54.0, 32.0),
    "great_salt_lake": (-112.0, 41.0),
    "etosha": (16.0, -19.0),
    "qinghai": (99.0, 37.0),
    # --- permanent water (lakes / reservoirs) ---
    "great_lakes": (-84.0, 45.0),
    "lake_victoria": (33.0, -1.0),
    "finland_lakes": (26.0, 62.0),
    "canada_shield": (-95.0, 55.0),
    "amazon_water": (-60.0, -3.0),
}


def tile_stem_for(lon: float, lat: float) -> str:
    """5x5-deg tile file stem containing (lon, lat). Tiles are named by (left_lon, top_lat)."""
    left = int(math.floor(lon / 5.0) * 5)
    top = int(math.ceil(lat / 5.0) * 5)
    lon_part = f"E{left}" if left >= 0 else f"W{-left}"
    lat_part = f"N{top}" if top >= 0 else f"S{-top}"
    return f"GWL_FCS30_2020_{lon_part}{lat_part}"


def tiles_dir():
    return io.raw_dir(SLUG) / "tiles"


def tile_path(stem: str):
    return tiles_dir() / f"{stem}.tif"


def _zip_paths() -> list:
    d = io.raw_dir(SLUG)
    return sorted(d.glob("GWL_FCS30_2020_*.zip"))


def build_tile_index() -> dict[str, str]:
    """Map tile stem -> zip filename by reading each band ZIP's namelist (no extraction)."""
    index: dict[str, str] = {}
    for zp in _zip_paths():
        with zipfile.ZipFile(str(zp)) as zf:
            for name in zf.namelist():
                if name.endswith(".tif"):
                    stem = name.split("/")[-1][:-4]
                    index[stem] = zp.name
    return index


def extract_tile(stem: str, zip_name: str) -> None:
    """Extract one 5x5-deg tile tif from its band ZIP into tiles/ (idempotent, atomic)."""
    dst = tile_path(stem)
    if dst.exists():
        return
    zp = io.raw_dir(SLUG) / zip_name
    with zipfile.ZipFile(str(zp)) as zf:
        member = next(n for n in zf.namelist() if n.split("/")[-1] == f"{stem}.tif")
        data = zf.read(member)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / (dst.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
    tmp.rename(dst)


def _scan_tile(stem: str) -> list[dict[str, Any]]:
    """Find homogeneous, single-class BLOCKxBLOCK wetland windows in one 5x5-deg tile."""
    path = str(tile_path(stem))
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        st = ds.transform
    h, w = arr.shape
    nby, nbx = h // BLOCK, w // BLOCK
    if nby == 0 or nbx == 0:
        return []
    a = arr[: nby * BLOCK, : nbx * BLOCK].reshape(nby, BLOCK, nbx, BLOCK)
    denom = float(BLOCK * BLOCK)
    wet = np.zeros((nby, nbx), np.float32)
    best = np.zeros((nby, nbx), np.float32)
    best_src = np.zeros((nby, nbx), np.uint8)
    for v in WET_VALUES:
        cnt = (a == v).sum(axis=(1, 3)).astype(np.float32)
        wet += cnt
        m = cnt > best
        best[m] = cnt[m]
        best_src[m] = v
    wet_frac = wet / denom
    dom_of_wet = np.divide(best, np.maximum(wet, 1.0))
    qual = (wet_frac >= MIN_WETLAND) & (dom_of_wet >= DOM_OF_WETLAND) & (best_src > 0)
    brs, bcs = np.nonzero(qual)
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
                "stem": stem,
                "lon": float(lon),
                "lat": float(lat),
                "label": SRC_TO_ID[src_v],
                "source_id": f"{stem}_r{br}_c{bc}",
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

    with rasterio.open(str(tile_path(rec["stem"]))) as ds:
        win = from_bounds(
            l2 - PAD_DEG, b2 - PAD_DEG, r2 + PAD_DEG, t2 + PAD_DEG, ds.transform
        )
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
        change_time=None,
        source_id=rec["source_id"],
        classes_present=present,
    )


def _write_source_txt(index: dict[str, str], used_tiles: list[str]) -> None:
    d = io.raw_dir(SLUG)
    d.mkdir(parents=True, exist_ok=True)
    (d / "SOURCE.txt").write_text(
        "GWL_FCS30: global 30 m wetland map with a fine classification system (2020).\n"
        "Zhang et al. 2023, ESSD 15, 265. doi:10.5281/zenodo.7340516  License: CC BY.\n"
        f"Zenodo record {ZENODO_RECORD}: 12 longitude-band ZIPs of 5x5-deg GeoTIFF tiles\n"
        "(EPSG:4326, ~30 m, uint8), ~2.4 GB total, downloaded in full to this dir.\n"
        f"{len(used_tiles)} curated wetland-rich tiles extracted under tiles/ for\n"
        "bounded-tile sampling; the full global mosaic is not tiled/scanned.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()

    from olmoearth_pretrain.open_set_segmentation_data import download

    print(f"Downloading {ZENODO_RECORD} band ZIPs (~2.4 GB)...")
    download.download_zenodo(ZENODO_RECORD, io.raw_dir(SLUG))
    io.check_disk()

    index = build_tile_index()
    print(f"tile index has {len(index)} tiles across the band ZIPs")

    # Resolve curated regions -> tiles (dedup); keep only tiles that exist.
    stems: dict[str, None] = {}
    missing = []
    for name, (lon, lat) in REGIONS.items():
        stem = tile_stem_for(lon, lat)
        if stem in index:
            stems[stem] = None
        else:
            missing.append((name, stem))
    used_tiles = sorted(stems)
    print(
        f"{len(used_tiles)} unique tiles to scan; {len(missing)} regions missing a tile"
    )
    if missing:
        print("  missing:", missing)

    _write_source_txt(index, used_tiles)

    print("Extracting curated tiles...")
    with multiprocessing.Pool(min(len(used_tiles), 16)) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(
                p, extract_tile, [dict(stem=s, zip_name=index[s]) for s in used_tiles]
            ),
            total=len(used_tiles),
        ):
            pass
    io.check_disk()

    print("Scanning tiles for homogeneous single-class wetland windows...")
    with multiprocessing.Pool(min(len(used_tiles), 16)) as p:
        all_recs: list[dict[str, Any]] = []
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan_tile, [dict(stem=s) for s in used_tiles]),
            total=len(used_tiles),
        ):
            all_recs.extend(recs)
    print(f"scanned {len(all_recs)} candidate homogeneous windows")
    print("candidate class counts:", dict(Counter(r["label"] for r in all_recs)))

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
            "name": "GWL_FCS30 Global Wetland Map (fine classes)",
            "task_type": "classification",
            "source": "Zenodo / ESSD",
            "license": "CC-BY",
            "provenance": {
                "url": "https://doi.org/10.5281/zenodo.7340516",
                "have_locally": False,
                "annotation_method": "derived-product (GWL_FCS30, 30 m, EPSG:4326, 2020)",
                "citation": (
                    "Zhang, X., Liu, L., Zhao, T., et al. (2023): GWL_FCS30: a global 30 m "
                    "wetland map with a fine classification system using multi-sourced and "
                    "time-series remote sensing imagery in 2020. Earth Syst. Sci. Data 15, "
                    "265-293. doi:10.5194/essd-15-265-2023 / doi:10.5281/zenodo.7340516"
                ),
                "year": YEAR,
                "source_value_legend": {
                    "0": "non-wetland / background (nodata)",
                    "180": "permanent water",
                    "181": "swamp",
                    "182": "marsh",
                    "183": "flooded flat",
                    "184": "saline (inland saline wetland)",
                    "185": "mangrove forest",
                    "186": "salt marsh",
                    "187": "tidal flat",
                },
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
                "Bounded-tile sampling of the global GWL_FCS30 2020 30 m fine-class wetland "
                f"product. The full 12-band ZIP set (~2.4 GB) is downloaded, then {len(used_tiles)} "
                "curated 5x5-deg tiles over wetland-rich regions on all continents are "
                "extracted and scanned on their native 30 m grid for homogeneous ~640 m "
                "windows (>=15% wetland, >=80% of the wetland pixels a single class). Windows "
                "are balanced tiles-per-class (<=1000/class) and reprojected from native "
                "EPSG:4326 to local UTM at 10 m with nearest resampling. Non-wetland (source "
                "0) is nodata=255 (no fabricated background). Static 2020 label: "
                "change_time=null, 1-year time range on 2020. Coastal classes (mangrove/salt "
                "marsh/tidal flat) and inland saline are naturally rarer/patchier than water "
                "and marsh; per spec §5 all classes are kept even where sparse."
            ),
        },
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
