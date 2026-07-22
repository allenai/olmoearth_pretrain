"""Process the Antarctic Ice-Shelf Surface Damage dataset into open-set-seg labels.

Source: "Surface Damage Dataset for Antarctic Ice Shelves 1999-2024" (Tang, Bamber, Li,
Qiao, 2026). Zenodo record 20425952 (concept DOI 10.5281/zenodo.20425951), license
CC-BY-4.0. One 118 MB zip: annual 30 m surface-damage maps over nine ice shelves (Amery,
Brunt, Crosson, Dotson, Holmes, Larsen B, Pine Island, Thwaites, Totten), 1999-2024,
EPSG:3031, produced by a deep-learning segmentation model on Landsat 7/8/9 optical imagery
(Landsat-7 SLC-off gaps in-painted with a diffusion model, DiffGF).

label_type: dense_raster ; task_type: classification (BINARY damage segmentation).

**Class scheme (verified against the rasters + README).** Although the manifest lists three
feature types (crevasses, rifts, heavily fractured areas), the *raster* does not label them
separately -- those are the kinds of features that are collectively mapped as a single
"surface damage" class. Each yearly folder holds two products:
  * Type 1  ``*_damage_map.tif``  (effective ice-shelf extent):  0 = no damage,
    1 = damage, 255 = NoData (outside the effective ice-shelf extent).   <-- USED
  * Type 2  ``*_damage.tif``      (full ROI, no extent mask, "requires additional manual
    checking"): 0 = no damage, 255 = damage.                              <-- NOT used
We use Type 1 only: it carries a proper NoData mask, so class 0 is genuine *undamaged
ice-shelf surface* (a spatially-meaningful within-tile negative), not ocean/rock. Output:
  id 0 = background (undamaged ice-shelf surface)
  id 1 = surface_damage (crevasses / rifts / heavily fractured areas, combined)
  255  = nodata (outside effective ice-shelf extent). The source->output code map is the
         identity {0:0, 1:1}; source 255 stays nodata.

Only the nine main shelves are used; Amery also ships an ``Amery_front`` subregion that
spatially overlaps Amery's main extent, so it is EXCLUDED to avoid duplicate tiles (170
main-shelf Type-1 maps, matching the README's headline count).

Time / change: each annual map is a persistent-state class map for one year -> a 1-year
window on that year (spec section 5, annual labels), change_time=null (surface damage is a
persistent structural feature, not a dated change event). Only Sentinel-era years (>= 2016)
are kept; pre-2016 maps are dropped (spec section 8 pre-2016 rule).

Tiling: source is 30 m EPSG:3031. We scan each annual raster in native-pixel blocks
(~640 m = a 64 px @ 10 m UTM tile), keep blocks that contain damage, balance tiles per
class (spec section 5, tiles-per-class balanced, <=1000/class), and reproject each selected
block footprint to a local UTM projection at 10 m / 64x64 with NEAREST resampling
(categorical 30 m -> 10 m, spec section 4 dense_raster).

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.antarctic_ice_shelf_surface_damage_crevasses_rifts
Inspect the raw rasters:
  python3 -m ...antarctic_ice_shelf_surface_damage_crevasses_rifts --inspect
"""

import argparse
import multiprocessing
import re
import urllib.request
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

from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    MAX_SAMPLES_PER_DATASET,
    balance_tiles_by_class,
)

SLUG = "antarctic_ice_shelf_surface_damage_crevasses_rifts"
NAME = "Antarctic Ice-Shelf Surface Damage (crevasses/rifts)"
ZENODO_RECORD = "20425952"
ZENODO_DOI = "https://doi.org/10.5281/zenodo.20425951"
ZENODO_FILE = "Multi_decadal_Antarctic_ice_shelf_surface_damage_1999_2024.zip"
# Zenodo fingerprints generic User-Agents (returns HTTP 403); a real browser UA works.
BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"

SRC_CRS_EPSG = 3031  # Antarctic Polar Stereographic (product's native CRS)
NATIVE_RES_M = 30.0
MIN_YEAR = 2016  # Sentinel era; drop pre-2016 maps (spec section 8)
MAX_YEAR = 2024

TILE = io.MAX_TILE  # 64 px @ 10 m = 640 m output tile
BLOCK = int(
    round(TILE * io.RESOLUTION / NATIVE_RES_M)
)  # native px per output tile (~21)
PER_CLASS = 1000
SEED = 42
PAD_M = 300.0  # geographic pad (metres in 3031) so the reprojected UTM tile is fully covered

# Source (Type-1) pixel code -> output class id. Identity for {0,1}; 255 stays nodata.
SRC_TO_ID = {0: 0, 1: 1}
CID_BACKGROUND = 0
CID_DAMAGE = 1
CLASSES = [
    {
        "id": CID_BACKGROUND,
        "name": "background",
        "description": "Undamaged ice-shelf surface within the effective ice-shelf extent "
        "(no crevasses, rifts or heavy fracturing detected). A genuine within-tile negative "
        "surrounding the damage, not fabricated; distinct from outside-shelf NoData (255).",
    },
    {
        "id": CID_DAMAGE,
        "name": "surface_damage",
        "description": "Surface damage on the ice shelf: crevasses, rifts and heavily "
        "fractured areas, mapped collectively as a single damage class by a deep-learning "
        "segmentation model on Landsat 7/8/9 optical imagery (Tang et al. 2026). Indicates "
        "reduced structural integrity / buttressing of the ice shelf.",
    },
]

# ---------------------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------------------


def _download_and_extract() -> None:
    """Download the single Zenodo zip (browser UA) and extract it (idempotent)."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    extracted = raw / "extracted"
    if extracted.exists() and any(extracted.rglob("*_damage_map.tif")):
        return
    zip_path = raw / ZENODO_FILE
    if not zip_path.exists():
        url = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{ZENODO_FILE}/content"
        print(f"downloading {ZENODO_FILE} ...", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA})
        tmp = raw / (ZENODO_FILE + ".tmp")
        with urllib.request.urlopen(req, timeout=600) as r, tmp.open("wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        tmp.rename(zip_path)
    print("extracting ...", flush=True)
    import zipfile

    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path.path) as zf:
        zf.extractall(extracted.path)


def _write_source_txt() -> None:
    d = io.raw_dir(SLUG)
    d.mkdir(parents=True, exist_ok=True)
    (d / "SOURCE.txt").write_text(
        "Surface Damage Dataset for Antarctic Ice Shelves 1999-2024 "
        "(Tang, Bamber, Li, Qiao, 2026).\n"
        f"Zenodo record {ZENODO_RECORD} (concept DOI {ZENODO_DOI}), license CC-BY-4.0.\n"
        f"Single file: {ZENODO_FILE} (118 MB). Downloaded with a browser User-Agent "
        "(Zenodo returns HTTP 403 to generic UAs).\n"
        "Used: Type-1 '*_damage_map.tif' (effective ice-shelf extent; 0=no damage, "
        "1=damage, 255=NoData), nine main shelves only (Amery_front excluded as it overlaps "
        "Amery). Binary damage segmentation. Only labels are used; pretraining supplies "
        "its own imagery.\n"
    )


# ---------------------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------------------


def discover_rasters() -> list[dict[str, Any]]:
    """Return [{path, shelf, year}] for Type-1 main-shelf maps in the in-range years."""
    raw = io.raw_dir(SLUG)
    recs: list[dict[str, Any]] = []
    for p in sorted(raw.rglob("*_damage_map.tif")):
        parts = p.path.split("/")
        # .../<Shelf>/<YYYY>/<file>_damage_map.tif  -> shelf = parts[-3], year = parts[-2]
        shelf = parts[-3]
        if shelf.endswith("_front"):
            continue  # overlaps the main shelf extent; skip to avoid duplicate tiles
        m = re.search(r"(19|20)\d{2}", parts[-2])
        if not m:
            continue
        year = int(m.group(0))
        if year < MIN_YEAR or year > MAX_YEAR:
            continue
        recs.append({"path": p.path, "shelf": shelf, "year": year})
    return recs


# ---------------------------------------------------------------------------------------
# Scan phase: find damage-containing blocks in each annual raster
# ---------------------------------------------------------------------------------------

_T3031_TO_WGS84 = None


def _to_wgs84() -> Transformer:
    global _T3031_TO_WGS84
    if _T3031_TO_WGS84 is None:
        _T3031_TO_WGS84 = Transformer.from_crs(SRC_CRS_EPSG, 4326, always_xy=True)
    return _T3031_TO_WGS84


def scan_raster(path: str, shelf: str, year: int) -> list[dict[str, Any]]:
    """Scan one annual raster; one record per block that CONTAINS damage."""
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        st = ds.transform
    h, w = arr.shape
    nby, nbx = h // BLOCK, w // BLOCK
    if nby == 0 or nbx == 0:
        return []
    a = arr[: nby * BLOCK, : nbx * BLOCK].reshape(nby, BLOCK, nbx, BLOCK)
    has_damage = (a == 1).any(axis=(1, 3))
    has_bg = (a == 0).any(axis=(1, 3))
    brs, bcs = np.nonzero(has_damage)
    tf = _to_wgs84()
    recs: list[dict[str, Any]] = []
    for br, bc in zip(brs.tolist(), bcs.tolist()):
        cx = bc * BLOCK + BLOCK / 2.0
        cy = br * BLOCK + BLOCK / 2.0
        x3031 = st.c + cx * st.a + cy * st.b
        y3031 = st.f + cx * st.d + cy * st.e
        lon, lat = tf.transform(x3031, y3031)
        classes_present = [CID_DAMAGE]
        if bool(has_bg[br, bc]):
            classes_present.append(CID_BACKGROUND)
        recs.append(
            {
                "path": path,
                "shelf": shelf,
                "year": year,
                "lon": float(lon),
                "lat": float(lat),
                "classes_present": sorted(classes_present),
                "source_id": f"{shelf}_{year}_r{br}_c{bc}",
            }
        )
    return recs


# ---------------------------------------------------------------------------------------
# Write phase: reproject each selected block footprint to a local UTM 10 m 64x64 tile
# ---------------------------------------------------------------------------------------


def write_tile(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return "skip"
    lon, lat = rec["lon"], rec["lat"]
    dst_proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    dst_transform = get_transform_from_projection_and_bounds(dst_proj, bounds)

    # UTM tile extent in metres -> transform to source CRS (EPSG:3031) for a windowed read.
    left = bounds[0] * io.RESOLUTION
    right = bounds[2] * io.RESOLUTION
    top = bounds[1] * -io.RESOLUTION
    bottom = bounds[3] * -io.RESOLUTION
    lo, bo, ro, to = (
        min(left, right),
        min(bottom, top),
        max(left, right),
        max(bottom, top),
    )
    l2, b2, r2, t2 = transform_bounds(
        dst_proj.crs, f"EPSG:{SRC_CRS_EPSG}", lo, bo, ro, to
    )

    with rasterio.open(rec["path"]) as ds:
        win = from_bounds(l2 - PAD_M, b2 - PAD_M, r2 + PAD_M, t2 + PAD_M, ds.transform)
        # fill_value=255 (nodata) so out-of-shelf/out-of-raster padding never fakes background.
        src = ds.read(1, window=win, boundless=True, fill_value=255)
        win_transform = ds.window_transform(win)
        src_crs = ds.crs

    dst = np.full((TILE, TILE), 255, np.uint8)
    reproject(
        source=src,
        destination=dst,
        src_transform=win_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_proj.crs,
        resampling=Resampling.nearest,
        src_nodata=255,
        dst_nodata=255,
    )
    out = np.full((TILE, TILE), io.CLASS_NODATA, np.uint8)
    for code, cid in SRC_TO_ID.items():
        out[dst == code] = cid

    if not np.any(out == CID_DAMAGE):
        return "empty"  # damage fell outside the reprojected footprint (rare)

    io.write_label_geotiff(
        SLUG, sample_id, out, dst_proj, bounds, nodata=io.CLASS_NODATA
    )
    present = sorted(int(v) for v in np.unique(out) if v != io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        dst_proj,
        bounds,
        io.year_range(rec["year"]),
        change_time=None,
        source_id=rec["source_id"],
        classes_present=present,
    )
    return "written"


# ---------------------------------------------------------------------------------------
# Inspect helper
# ---------------------------------------------------------------------------------------


def inspect() -> None:
    raw = io.raw_dir(SLUG)
    maps = sorted(raw.rglob("*_damage_map.tif"))
    main = [p for p in maps if not p.path.split("/")[-3].endswith("_front")]
    print(f"raw dir: {raw}")
    print(f"Type-1 damage_map tifs: {len(maps)} total, {len(main)} main-shelf")

    def yr(p):
        return int(re.search(r"(19|20)\d{2}", p.path.split("/")[-2]).group(0))

    years = Counter(yr(p) for p in main)
    print("main-shelf year histogram:", dict(sorted(years.items())))
    in_range = [p for p in main if MIN_YEAR <= yr(p) <= MAX_YEAR]
    print(f"in-range [{MIN_YEAR},{MAX_YEAR}] main-shelf maps: {len(in_range)}")
    for p in main[:6]:
        with rasterio.open(p.path) as ds:
            arr = ds.read(1)
        v, c = np.unique(arr, return_counts=True)
        print(
            f"  {'/'.join(p.path.split('/')[-3:])}: crs={ds.crs} res={ds.res} "
            f"nodata={ds.nodata} vals={dict(zip(v.tolist(), c.tolist()))}"
        )


# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    io.check_disk()
    _download_and_extract()
    _write_source_txt()
    io.check_disk()

    if args.inspect:
        inspect()
        return

    rasters = discover_rasters()
    print(
        f"discovered {len(rasters)} Type-1 main-shelf rasters in [{MIN_YEAR},{MAX_YEAR}]",
        flush=True,
    )
    if not rasters:
        raise RuntimeError("no in-range rasters found; run --inspect to debug")
    print(
        "per-shelf raster counts:",
        dict(sorted(Counter(r["shelf"] for r in rasters).items())),
        flush=True,
    )

    # --- scan phase (parallel over rasters) ---
    records: list[dict[str, Any]] = []
    with multiprocessing.Pool(min(args.workers, len(rasters))) as p:
        for recs in tqdm.tqdm(
            star_imap_unordered(
                p,
                scan_raster,
                [
                    dict(path=r["path"], shelf=r["shelf"], year=r["year"])
                    for r in rasters
                ],
            ),
            total=len(rasters),
        ):
            records.extend(recs)
    print(f"scanned {len(records)} damage-containing candidate blocks", flush=True)
    cand_class = Counter()
    for r in records:
        for c in r["classes_present"]:
            cand_class[c] += 1
    print("candidate class block counts:", dict(sorted(cand_class.items())), flush=True)

    # --- select: tiles-per-class balanced ---
    selected = balance_tiles_by_class(
        records,
        "classes_present",
        per_class=PER_CLASS,
        seed=SEED,
        total_cap=MAX_SAMPLES_PER_DATASET,
    )
    selected.sort(key=lambda r: (r["shelf"], r["year"], r["lon"], r["lat"]))
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} tiles (<= {PER_CLASS}/class)", flush=True)

    io.check_disk()

    # --- write phase (parallel) ---
    results: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, write_tile, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            results[res] += 1
    print("write results:", dict(results), flush=True)

    # Count only tiles that were actually written (a few candidate blocks reproject to a
    # footprint whose damage pixels fall just outside the 640 m tile -> "empty", skipped).
    written_recs = [
        r
        for r in selected
        if (io.locations_dir(SLUG) / f"{r['sample_id']}.tif").exists()
    ]
    class_tile_counts: Counter = Counter()
    shelf_counts: Counter = Counter()
    year_counts: Counter = Counter()
    for r in written_recs:
        for c in r["classes_present"]:
            class_tile_counts[c] += 1
        shelf_counts[r["shelf"]] += 1
        year_counts[r["year"]] += 1
    id_to_name = {c["id"]: c["name"] for c in CLASSES}
    print(f"actually-written tiles: {len(written_recs)}", flush=True)

    io.check_disk()
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Zenodo / ESSD",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": ZENODO_DOI,
                "zenodo_record": ZENODO_RECORD,
                "have_locally": False,
                "annotation_method": "deep-learning segmentation on Landsat 7/8/9 optical "
                "imagery (Landsat-7 SLC-off restored with the DiffGF diffusion model)",
                "citation": "Tang, L., Bamber, J. L., Li, T., Qiao, G. (2026): A "
                "multi-decadal dataset of surface damage on Antarctic ice shelves "
                "(1999-2024).",
                "native_crs": f"EPSG:{SRC_CRS_EPSG}",
                "native_resolution_m": NATIVE_RES_M,
                "product_used": "Type-1 '*_damage_map.tif' (effective ice-shelf extent)",
                "source_value_legend": {
                    "0": "no damage (background)",
                    "1": "damage",
                    "255": "NoData (outside effective ice-shelf extent)",
                },
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(written_recs),
            "class_tile_counts": {
                id_to_name.get(k, str(k)): v
                for k, v in sorted(class_tile_counts.items())
            },
            "shelf_counts": dict(sorted(shelf_counts.items())),
            "year_counts": dict(sorted(year_counts.items())),
            "sampling": {
                "per_class": PER_CLASS,
                "tile_size_px": TILE,
                "native_block_px": BLOCK,
                "total_cap": MAX_SAMPLES_PER_DATASET,
                "min_year": MIN_YEAR,
                "max_year": MAX_YEAR,
                "candidate_blocks": len(records),
            },
            "time_range_rule": "annual persistent-state map -> 1-year window on the map year; "
            "change_time=null (persistent structural feature, not a dated change event)",
            "notes": (
                "Binary Antarctic ice-shelf surface-damage segmentation (crevasses / rifts / "
                "heavily fractured areas mapped collectively as one 'surface_damage' class). "
                "Source: 30 m Landsat-derived DL segmentation maps over nine ice shelves, "
                "1999-2024; Type-1 '*_damage_map.tif' (effective ice-shelf extent) used so "
                "class 0 is genuine undamaged ice (255=outside-shelf nodata). Manifest listed "
                "three feature types, but the raster labels damage as a single binary class -- "
                "the types are not separately encoded, so we emit background + surface_damage. "
                "Only Sentinel-era years (>=2016) kept; each annual map -> 1-year window on "
                "its year, change_time=null. Nine main shelves only (Amery_front excluded: it "
                "overlaps Amery). Candidate 64x64 (640 m) blocks that contain damage are "
                "reprojected from native 30 m EPSG:3031 to local UTM at 10 m with NEAREST "
                "resampling; tiles-per-class balanced (<=1000/class). Multiple years per shelf "
                "are eligible for temporal diversity."
            ),
        },
    )
    print(
        f"done: {len(written_recs)} tiles; class tile counts: "
        f"{ {id_to_name.get(k, k): v for k, v in sorted(class_tile_counts.items())} }"
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
