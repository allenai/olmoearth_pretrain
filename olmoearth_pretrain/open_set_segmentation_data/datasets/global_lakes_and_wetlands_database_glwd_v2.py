"""Process the Global Lakes and Wetlands Database (GLWD) v2 into open-set-segmentation labels.

Source: Lehner, B., Anand, M., Fluet-Chouinard, E., et al. (2025): "Mapping the world's
inland surface waters: an upgrade to the Global Lakes and Wetlands Database (GLWD v2)".
Earth Syst. Sci. Data 17, 2277-2329, doi:10.5194/essd-17-2277-2025. Data on figshare
(doi:10.6084/m9.figshare.28519994, CC-BY-4.0), hosted by HydroSHEDS / WWF. The product is
a global (excl. Antarctica) 15 arc-second (~500 m at the equator) raster in EPSG:4326
mapping 33 lake/river/wetland types (plus dryland), derived by fusing many input products
for the ~1990-2020 period. We use the "combined_classes" GeoTIFF distribution
(GLWD_v2_0_combined_classes_tif.zip, ~925 MB) and, within it, the dominant-class raster
GLWD_v2_0_main_class.tif (uint8, value 0 = inland pixel without wetland, 255 = nodata,
1..33 = dominant wetland class within the pixel).

Class legend (GLWD_ID -> class name); output class id = GLWD_ID - 1 (so ids 0..32):
    1  Freshwater lake                        18 Palustrine, seasonally saturated, forested
    2  Saline lake                            19 Palustrine, seasonally saturated, non-forested
    3  Reservoir                              20 Ephemeral, forested
    4  Large river                            21 Ephemeral, non-forested
    5  Large estuarine river                  22 Arctic/boreal peatland, forested
    6  Other permanent waterbody              23 Arctic/boreal peatland, non-forested
    7  Small streams                          24 Temperate peatland, forested
    8  Lacustrine, forested                   25 Temperate peatland, non-forested
    9  Lacustrine, non-forested               26 Tropical/subtropical peatland, forested
    10 Riverine, regularly flooded, forested  27 Tropical/subtropical peatland, non-forested
    11 Riverine, regularly flooded, non-for.  28 Mangrove
    12 Riverine, seasonally flooded, forested 29 Saltmarsh
    13 Riverine, seasonally flooded, non-for. 30 Large river delta
    14 Riverine, seasonally saturated, for.   31 Other coastal wetland
    15 Riverine, seasonally saturated, non-f. 32 Salt pan, saline/brackish wetland
    16 Palustrine, regularly flooded, forested 33 Rice paddies
    17 Palustrine, regularly flooded, non-for.  (0 = dryland/non-wetland -> nodata 255)

task_type=classification, dense_raster. GLWD v2 is a static compilation of the recent
(~1990-2020) state, so change_time is null and the time range is a representative 1-year
window on 2020 (§5 static-label rule; the manifest [2016] is just a nominal tag).

GLOBAL derived-product => BOUNDED-TILE sampling (spec §5). We do NOT attempt global
coverage. The full global main_class raster (86400x33600, ~2.9 GB uint8) is streamed in
latitude strips and scanned on its native 15 arc-sec grid for spatially-HOMOGENEOUS 3x3
native blocks (~1.5 km) in which all 9 cells share a single dominant wetland class -- §4
guidance to prefer homogeneous/high-confidence windows for coarse derived-product maps.
Homogeneity over 1.5 km gives confidence that the reprojected 640 m (64x64 @ 10 m) output
tile is genuinely that single class despite the coarse source. Candidate block centers are
subsampled per class with a fixed-seed probability (for global geographic spread), balanced
tiles-per-class (<=1000/class, 25k total => 757/class), and each is reprojected from native
EPSG:4326 to a local UTM projection at 10 m with NEAREST resampling (categorical labels).
Non-wetland / dryland (source 0) and nodata (255) become nodata=255 (no fabricated
background), per §2.

NOTE on the 50%-threshold layer: GLWD also ships GLWD_v2_0_main_class_50pct.tif (only
pixels where total wetland extent > 50%). We deliberately use the plain dominant-class
raster instead, because the 50% layer drops linear/sparse classes entirely (e.g. small
streams -> 0 qualifying pixels) -- we want the full 33-class taxonomy. The 3x3 homogeneity
filter supplies the spatial-confidence signal instead. Caveat recorded in the summary: the
500 m label describes the *dominant* wetland type of the cell, not necessarily a >50%
areal majority.

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_lakes_and_wetlands_database_glwd_v2
"""

import argparse
import csv
import multiprocessing
import zipfile
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import tqdm
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds

from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "global_lakes_and_wetlands_database_glwd_v2"

FIGSHARE_ZIP_URL = (
    "https://ndownloader.figshare.com/files/54001814"  # combined_classes_tif.zip
)
ZIP_NAME = "GLWD_v2_0_combined_classes_tif.zip"
MAIN_CLASS_MEMBER = "GLWD_v2_0_combined_classes/GLWD_v2_0_main_class.tif"
MAIN_CLASS_TIF = "GLWD_v2_0_main_class.tif"
LEGEND_MEMBER = "GLWD_Legend_v2_0.csv"
LEGEND_CSV = "GLWD_Legend_v2_0.csv"

YEAR = 2020  # static compilation (~1990-2020); representative 1-yr window in S2 era
PER_CLASS = 1000  # balance_by_class lowers this to 25000 // 33 = 757 via total_cap
TILE = 64  # output UTM tile is 64x64 @ 10 m (= 640 m)
B = 3  # native ~500 m block => 3x3 ~= 1.5 km homogeneity neighborhood
KEEP_PER_CLASS = (
    3000  # per-class candidate cap after spatial subsampling (>= 757 for spread)
)
STRIP_ROWS = 3000  # streaming strip height (rounded down to multiple of B at read time)
PAD_DEG = 0.012  # ~1.3 km geographic pad so the reprojected UTM tile is fully covered
SEED = 42

# GLWD source value 1..33 -> output class id 0..32 ; source 0 (dryland) & 255 -> nodata.
SRC_TO_ID = {v: v - 1 for v in range(1, 34)}
WET_VALUES = tuple(SRC_TO_ID.keys())


def raw():
    return io.raw_dir(SLUG)


def main_class_path():
    return raw() / MAIN_CLASS_TIF


def _ensure_source() -> None:
    """Download the combined-classes zip (if needed) and extract main_class.tif + legend."""
    from olmoearth_pretrain.open_set_segmentation_data import download

    d = raw()
    d.mkdir(parents=True, exist_ok=True)
    tif = main_class_path()
    leg = d / LEGEND_CSV
    if tif.exists() and leg.exists():
        return
    zip_path = d / ZIP_NAME
    if not zip_path.exists():
        print(f"Downloading {ZIP_NAME} (~925 MB) from figshare...")
        download.download_http(FIGSHARE_ZIP_URL, zip_path)
    io.check_disk()
    with zipfile.ZipFile(str(zip_path)) as zf:
        for member, out_name in (
            (MAIN_CLASS_MEMBER, MAIN_CLASS_TIF),
            (LEGEND_MEMBER, LEGEND_CSV),
        ):
            out = d / out_name
            if out.exists():
                continue
            data = zf.read(member)
            tmp = d / (out_name + ".tmp")
            with tmp.open("wb") as f:
                f.write(data)
            tmp.rename(out)


def load_legend() -> dict[int, str]:
    with (raw() / LEGEND_CSV).open() as f:
        return {int(r["GLWD_ID"]): r["Class_name"] for r in csv.DictReader(f)}


def _homogeneous_blocks(arr: np.ndarray):
    """Return (brow, bcol, class) arrays for homogeneous BxB blocks in a strip array."""
    nby, nbx = arr.shape[0] // B, arr.shape[1] // B
    a = arr[: nby * B, : nbx * B].reshape(nby, B, nbx, B)
    tl = a[:, 0:1, :, 0:1]
    same = (a == tl).all(axis=(1, 3))
    c = tl[:, 0, :, 0]
    valid = same & (c != 0) & (c != io.CLASS_NODATA)
    brs, bcs = np.nonzero(valid)
    return brs, bcs, c[brs, bcs]


def count_homogeneous() -> dict[int, int]:
    """Phase A: per-class total count of homogeneous BxB blocks over the whole raster."""
    totals = np.zeros(256, np.int64)
    with rasterio.open(str(main_class_path())) as ds:
        for _r0, arr in tqdm.tqdm(list(_iter_strips_meta(ds)), desc="count"):
            _, _, vals = _homogeneous_blocks(arr)
            totals += np.bincount(vals, minlength=256)
    return {v: int(totals[v]) for v in WET_VALUES}


def _iter_strips_meta(ds):
    # tqdm needs a length; materialize strip offsets, read lazily.
    h = ds.height
    hb = (h // B) * B
    offsets = list(range(0, hb, STRIP_ROWS - STRIP_ROWS % B))
    for r0 in offsets:
        rows = min(STRIP_ROWS - STRIP_ROWS % B, hb - r0)
        rows -= rows % B
        if rows == 0:
            continue
        arr = ds.read(1, window=Window(0, r0, (ds.width // B) * B, rows))
        yield r0, arr


def collect_candidates(totals: dict[int, int]) -> list[dict[str, Any]]:
    """Phase B: stream again, subsample homogeneous block centers per class for spread."""
    keep_prob = {v: min(1.0, KEEP_PER_CLASS / max(1, totals[v])) for v in WET_VALUES}
    recs: list[dict[str, Any]] = []
    with rasterio.open(str(main_class_path())) as ds:
        st = ds.transform
        for r0, arr in tqdm.tqdm(list(_iter_strips_meta(ds)), desc="collect"):
            brs, bcs, vals = _homogeneous_blocks(arr)
            if len(vals) == 0:
                continue
            rng = np.random.default_rng(SEED + r0)
            probs = np.array([keep_prob[int(v)] for v in vals])
            take = rng.random(len(vals)) < probs
            brs, bcs, vals = brs[take], bcs[take], vals[take]
            # block center in native pixel coords (global row = r0 + brow*B)
            cx = bcs * B + B / 2.0
            cy = r0 + brs * B + B / 2.0
            lons = st.c + cx * st.a
            lats = st.f + cy * st.e
            for br, bc, v, lon, lat in zip(
                brs.tolist(), bcs.tolist(), vals.tolist(), lons.tolist(), lats.tolist()
            ):
                recs.append(
                    {
                        "lon": float(lon),
                        "lat": float(lat),
                        "label": SRC_TO_ID[int(v)],
                        "source_id": f"gr{r0 + br * B}_c{bc * B}",
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

    with rasterio.open(str(main_class_path())) as ds:
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


def _write_source_txt(totals: dict[int, int]) -> None:
    d = raw()
    d.mkdir(parents=True, exist_ok=True)
    (d / "SOURCE.txt").write_text(
        "Global Lakes and Wetlands Database (GLWD) v2.\n"
        "Lehner et al. 2025, ESSD 17, 2277-2329. doi:10.5194/essd-17-2277-2025.\n"
        "Data: figshare doi:10.6084/m9.figshare.28519994 (CC-BY-4.0), HydroSHEDS/WWF.\n"
        f"Downloaded {ZIP_NAME} (~925 MB); extracted {MAIN_CLASS_TIF} (dominant wetland\n"
        "class per 15 arc-sec pixel, EPSG:4326, uint8, 0=dryland, 255=nodata, 1..33=class)\n"
        "and the legend CSV. Bounded-tile sampling: the global raster is scanned for\n"
        "homogeneous 3x3 native (~1.5 km) single-class blocks; a per-class spatial\n"
        "subsample is reprojected to local UTM 10 m tiles. Full global mosaic not tiled.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    _ensure_source()
    io.check_disk()

    legend = load_legend()
    # Output classes in id order (id = GLWD_ID - 1), names from the GLWD legend CSV.
    classes = [
        {"id": SRC_TO_ID[gid], "name": legend[gid], "description": None}
        for gid in range(1, 34)
    ]

    print("Phase A: counting homogeneous blocks per class...")
    totals = count_homogeneous()
    print("homogeneous block totals:", totals)
    _write_source_txt(totals)

    print("Phase B: collecting spatially-subsampled candidates...")
    all_recs = collect_candidates(totals)
    print(f"collected {len(all_recs)} candidate windows")
    print(
        "candidate class counts:",
        dict(sorted(Counter(r["label"] for r in all_recs).items())),
    )

    selected = balance_by_class(all_recs, "label", per_class=PER_CLASS)
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} windows (balanced, <= {PER_CLASS}/class, 25k cap)")

    io.check_disk()
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            pass

    counts = Counter(r["label"] for r in selected)
    class_counts = {legend[gid]: counts.get(SRC_TO_ID[gid], 0) for gid in range(1, 34)}
    print("class counts:", class_counts)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "Global Lakes and Wetlands Database (GLWD) v2",
            "task_type": "classification",
            "source": "figshare / HydroSHEDS / WWF (ESSD)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.org/10.6084/m9.figshare.28519994",
                "have_locally": False,
                "annotation_method": (
                    "derived-product (GLWD v2 dominant-class map, 15 arc-sec ~500 m, "
                    "EPSG:4326, ~1990-2020 compilation)"
                ),
                "citation": (
                    "Lehner, B., Anand, M., Fluet-Chouinard, E., et al. (2025): Mapping the "
                    "world's inland surface waters: an upgrade to the Global Lakes and Wetlands "
                    "Database (GLWD v2). Earth Syst. Sci. Data 17, 2277-2329. "
                    "doi:10.5194/essd-17-2277-2025 / doi:10.6084/m9.figshare.28519994"
                ),
                "raster": MAIN_CLASS_TIF,
                "year": YEAR,
                "source_value_legend": {str(gid): legend[gid] for gid in range(0, 34)},
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": class_counts,
            "notes": (
                "Bounded-tile sampling of the global GLWD v2 dominant-class wetland map "
                "(15 arc-sec ~500 m, 33 lake/river/wetland types). Output class id = "
                "GLWD_ID - 1 (ids 0..32). The full 86400x33600 uint8 raster is streamed and "
                "scanned for homogeneous 3x3 native (~1.5 km) single-class blocks; per class "
                "a fixed-seed spatial subsample is balanced tiles-per-class (25000//33 = 757 "
                "/class) and reprojected from EPSG:4326 to local UTM at 10 m with nearest "
                "resampling. Dryland (source 0) and nodata are 255 (no fabricated background). "
                "Static ~1990-2020 compilation: change_time=null, 1-year window on 2020. "
                "CAVEAT: the coarse 500 m label describes the DOMINANT wetland type of each "
                "cell, not a >50% areal majority; each 640 m output tile is essentially one "
                "native cell so tiles are (near-)uniform in class. We use the plain "
                "dominant-class raster (not the main_class_50pct layer) so linear/sparse "
                "classes (small streams, ephemeral, palustrine reg-flooded forested) are "
                "retained; class 16 (Palustrine, regularly flooded, forested) is naturally "
                "rare (~416 homogeneous blocks) and falls below the 757 target -- kept per "
                "spec 5. Thematically overlaps GWL_FCS30 / PEATMAP but has a distinct, richer "
                "hydro-functional 33-class taxonomy (peatlands by climate zone, riverine/"
                "palustrine flooding regimes, deltas, rice paddies)."
            ),
        },
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
