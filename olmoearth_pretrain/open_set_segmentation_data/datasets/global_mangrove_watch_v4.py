"""Process Global Mangrove Watch v4 (2020 baseline) into open-set segmentation label patches.

Source: Global Mangrove Watch (GMW), UNEP-WCMC / JAXA / Aberystwyth University / soloEO.
Product: "Global Mangrove Watch: Annual Mangrove Extent" v4.0.19, the **2020 10 m
Sentinel-2 baseline** (Zenodo record 12756047, DOI 10.5281/zenodo.12756047, CC-BY-4.0).
Over 30,000 ML models trained on 5M+ reference points classify mangrove vs non-mangrove
from Copernicus Sentinel-2 imagery at 10 m. Distributed as 1647 one-degree GeoTIFF tiles
(gmw_mng_2020_v4019_gtiff.zip, ~180 MB), each 10000x10000 uint8 in EPSG:4326 at 0.0001 deg
(~10 m), value 1 = mangrove, 0 = non-mangrove (file nodata is set to 0). Tiles exist only
where mangroves occur (coastal tropics/subtropics), so the tile set already delimits
representative mangrove regions.

Data reality vs manifest: the manifest lists classes [mangrove, non-mangrove, gain, loss].
gain/loss are the GMW *change* products (baseline-to-year comparisons resolved only to
annual/multi-year epochs), which the task spec rejects as change labels (change date not
resolvable to ~1-2 months). We therefore produce the **mangrove EXTENT** product only, a
2-class dense_raster: id 0 = mangrove, id 1 = non-mangrove. The 2020 Sentinel-2 baseline is
near-static, so we assign a static 1-year time range anchored on 2020 (matching the mapped
year). gain/loss are intentionally not encoded (documented in the summary).

Sampling (global derived-product -> BOUNDED, per spec 5/4): only the single 2020 baseline
zip is downloaded (not the full 1990-2024 series). Windows are drawn across the global set
of mangrove tiles with a per-tile cap for geographic diversity, then class-balanced to
<=1000 windows per class. Following the tidal-flats template:
  * mangrove windows (native block with mangrove fraction >= MANGROVE_MIN_FRAC) are the
    rare/valuable positives and carry BOTH classes (mangrove + surrounding non-mangrove);
  * "non-mangrove" windows have no mangrove but are within NEG_NEIGHBORHOOD blocks of a
    mangrove block (genuine coastal context, not open ocean/inland) and carry only class 1.
Native 0.0001 deg EPSG:4326 windows are reprojected to a local UTM projection at 10 m with
**nearest** resampling (categorical). task_type=classification.

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_mangrove_watch_v4
"""

import argparse
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

SLUG = "global_mangrove_watch_v4"

ZENODO_RECORD = "12756047"
GTIFF_ZIP_NAME = "gmw_mng_2020_v4019_gtiff.zip"
GTIFF_ZIP = io.raw_dir(SLUG) / GTIFF_ZIP_NAME

YEAR = 2020  # the mapped baseline year (10 m Sentinel-2)

PER_CLASS = 1000
TILE = 64  # output tile size (px @ 10 m UTM)
BLOCK = 64  # native 0.0001 deg block ~= 0.0064 deg ~= a 64 px @ 10 m UTM footprint
MANGROVE_MIN_FRAC = 0.10  # a "mangrove" window: >= 10% of the block is mangrove
PER_TILE_CAP = 10  # cap candidates per class per tile for geographic diversity
NEG_NEIGHBORHOOD = (
    3  # a non-mangrove window must be within this many blocks of a mangrove block
)

# GMW extent raster is binary: source 1 -> mangrove (id 0); source 0 -> non-mangrove (id 1).
SRC_TO_ID = {1: 0, 0: 1}
CLASSES = [
    (
        "mangrove",
        "Mangrove forest present in 2020 as mapped by Global Mangrove Watch v4 from Copernicus "
        "Sentinel-2 at 10 m (over 30,000 ML models trained on 5M+ photointerpreted reference "
        "points). Mangroves are salt-tolerant intertidal forests/shrublands of tropical and "
        "subtropical coastlines. (GMW source value 1.)",
    ),
    (
        "non-mangrove",
        "No mangrove present: all other cover within the mangrove analysis area (open water, "
        "tidal flats, terrestrial vegetation, built-up, bare ground). (GMW source value 0.)",
    ),
]


def download() -> None:
    """Download only the 2020 baseline GeoTIFF zip (bounded; ~180 MB, not the full series)."""
    from olmoearth_pretrain.open_set_segmentation_data import download as dl

    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    if GTIFF_ZIP.exists():
        print(f"  [skip] {GTIFF_ZIP.name} present")
        return
    print(f"  downloading {GTIFF_ZIP_NAME} from Zenodo record {ZENODO_RECORD}")
    dl.download_zenodo(ZENODO_RECORD, io.raw_dir(SLUG), filenames=[GTIFF_ZIP_NAME])


def list_tiles() -> list[str]:
    """Return zip member names of the mangrove-extent GeoTIFF tiles."""
    with zipfile.ZipFile(GTIFF_ZIP.path) as zf:
        return sorted(m for m in zf.namelist() if m.lower().endswith(".tif"))


def _vsipath(member: str) -> str:
    return f"/vsizip/{GTIFF_ZIP.path}/{member}"


def _scan_tile(member: str) -> list[dict[str, Any]]:
    """Scan one 1-degree tile for candidate BLOCK-sized windows.

    Returns records for mangrove windows (>= MANGROVE_MIN_FRAC mangrove) and non-mangrove
    windows (no mangrove but within NEG_NEIGHBORHOOD blocks of a mangrove block), capped
    per tile for geographic diversity.
    """
    rng = np.random.default_rng(abs(hash(member)) % (2**32))
    with rasterio.open(_vsipath(member)) as ds:
        H, W = ds.height, ds.width
        st = ds.transform
        a = ds.read(1)
    nby, nbx = H // BLOCK, W // BLOCK
    if nby == 0 or nbx == 0:
        return []
    a = (a[: nby * BLOCK, : nbx * BLOCK] == 1).astype(np.float32)
    mng_frac = a.reshape(nby, BLOCK, nbx, BLOCK).sum(axis=(1, 3)) / float(BLOCK * BLOCK)

    is_mng = mng_frac >= MANGROVE_MIN_FRAC
    if not is_mng.any():
        return []
    from scipy.ndimage import binary_dilation

    k = 2 * NEG_NEIGHBORHOOD + 1
    near_mng = binary_dilation(is_mng, structure=np.ones((k, k), bool))
    is_neg = near_mng & (mng_frac == 0.0)

    def sample(mask, label):
        brs, bcs = np.nonzero(mask)
        if len(brs) == 0:
            return []
        idx = np.arange(len(brs))
        if len(idx) > PER_TILE_CAP:
            idx = rng.choice(idx, PER_TILE_CAP, replace=False)
        recs = []
        for i in idx:
            br, bc = int(brs[i]), int(bcs[i])
            cx = bc * BLOCK + BLOCK / 2.0
            cy = br * BLOCK + BLOCK / 2.0
            lon = st.c + cx * st.a
            lat = st.f + cy * st.e
            recs.append(
                {
                    "member": member,
                    "lon": float(lon),
                    "lat": float(lat),
                    "label": label,
                    "mng_frac": float(mng_frac[br, bc]),
                    "source_id": f"{member.split('/')[-1].replace('.tif', '')}_r{br}_c{bc}",
                }
            )
        return recs

    return sample(is_mng, 0) + sample(is_neg, 1)


def _write_one(rec: dict[str, Any]) -> None:
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
    l2, b2, r2, t2 = transform_bounds(
        dst_proj.crs, "EPSG:4326", left, bottom, right, top
    )
    pad = 0.01  # deg margin so the tile is fully covered before nearest-resampling

    with rasterio.open(_vsipath(rec["member"])) as ds:
        win = from_bounds(l2 - pad, b2 - pad, r2 + pad, t2 + pad, ds.transform)
        src = ds.read(1, window=win, boundless=True, fill_value=0)
        win_transform = ds.window_transform(win)
        src_crs = ds.crs

    dst = np.zeros((TILE, TILE), np.uint8)
    reproject(
        source=src.astype(np.uint8),
        destination=dst,
        src_transform=win_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_proj.crs,
        resampling=Resampling.nearest,
    )
    out = np.full((TILE, TILE), io.CLASS_NODATA, np.uint8)
    for src_v, cid in SRC_TO_ID.items():
        out[dst == src_v] = cid

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
    parser.add_argument("--scan-workers", type=int, default=32)
    args = parser.parse_args()

    io.check_disk()
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    with (io.raw_dir(SLUG) / "SOURCE.txt").open("w") as f:
        f.write(
            "Global Mangrove Watch: Annual Mangrove Extent v4.0.19 (UNEP-WCMC / JAXA / "
            "Aberystwyth Univ / soloEO)\n"
            "2020 10 m Sentinel-2 mangrove-extent baseline; 1647 one-degree GeoTIFF tiles "
            "(EPSG:4326, 0.0001 deg, uint8; 1=mangrove, 0=non-mangrove).\n"
            f"Zenodo record {ZENODO_RECORD} (DOI 10.5281/zenodo.12756047), CC-BY-4.0.\n"
            f"File downloaded: {GTIFF_ZIP_NAME}\n"
        )

    print("Ensuring 2020 baseline download...")
    download()
    io.check_disk()

    tiles = list_tiles()
    print(f"{len(tiles)} mangrove-extent tiles in zip")

    print("Scanning tiles for candidate windows...")
    with multiprocessing.Pool(args.scan_workers) as p:
        all_recs: list[dict[str, Any]] = []
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan_tile, [dict(member=m) for m in tiles]),
            total=len(tiles),
        ):
            all_recs.extend(recs)
    print(f"  {len(all_recs)} candidate windows")
    print("  raw class-candidate counts:", Counter(r["label"] for r in all_recs))
    tiles_with = len({r["member"] for r in all_recs})
    print(f"  from {tiles_with} tiles")

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
    n_tiles_selected = len({r["member"] for r in selected})
    print("selected class counts (primary window label):", class_counts)
    print("distinct source tiles in selection:", n_tiles_selected)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "Global Mangrove Watch v4",
            "task_type": "classification",
            "source": "UNEP-WCMC / JAXA (Global Mangrove Watch)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://www.globalmangrovewatch.org/",
                "have_locally": False,
                "annotation_method": "derived-product (Sentinel-2 ML classification) with photointerpreted validation",
                "citation": "Bunting et al. 2018/2022; GMW Annual Mangrove Extent v4.0.19",
                "product": "GMW Annual Mangrove Extent v4.0.19, 2020 baseline (10 m Sentinel-2)",
                "download": f"Zenodo record {ZENODO_RECORD} (DOI 10.5281/zenodo.12756047), file {GTIFF_ZIP_NAME}",
                "year": YEAR,
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
                "EXTENT product only (2-class: mangrove / non-mangrove). The manifest's "
                "gain/loss classes are GMW change products resolved only to annual/multi-year "
                "epochs; per the change-timing rule (event must be datable to ~1-2 months) they "
                "are NOT encoded here. Bounded sampling of a global derived-product: only the "
                "single 2020 10 m Sentinel-2 baseline zip (~180 MB) was downloaded, not the full "
                "1990-2024 series. 64x64 windows reprojected from native 0.0001 deg EPSG:4326 to "
                "local UTM at 10 m with nearest resampling. Tiles-per-class balanced across the "
                "global set of mangrove tiles (per-tile cap for diversity): mangrove windows "
                "(>=10% mangrove) carry both classes; non-mangrove windows are coastal negatives "
                "within 3 blocks of a mangrove block (not open ocean/inland). Static 1-year time "
                "range anchored on 2020 (the mapped baseline year); task=classification."
            ),
        },
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
