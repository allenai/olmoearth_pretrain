"""Process AI4SmallFarms (crop-field boundaries, Cambodia & Vietnam) into label patches.

Source: Persello et al. (2023), "AI4SmallFarms: A Dataset for Crop Field Delineation in
Southeast Asian Smallholder Farms", IEEE GRSL 20, 2505705
(https://doi.org/10.1109/LGRS.2023.3323095). Distributed as a fiboa GeoParquet on Source
Cooperative (https://source.coop/fiboa/ai4sf, CC-BY-4.0, no credential), converted by
Matthias Mohr with fiboa-cli. We download only the single 29 MB GeoParquet
(https://data.source.coop/fiboa/ai4sf/ai4sf.parquet) -- pretraining supplies its own imagery.

The file holds 439,001 manually-digitized smallholder crop-field polygons across 62 tiles of
~5x5 km in Cambodia (318,088) and Vietnam (120,913). Every polygon carries
determination_datetime = 2021-08-01 (labeling anchored on 2021 Sentinel-2 composites),
determination_method = auto-imagery (manual digitization from imagery), a group id (0-61,
the 5x5 km tile), and a country. Geometry is stored in EPSG:32648 (UTM 48N) metres.

ENCODING DECISION (task spec §4 polygons + lines judgment):
  Field polygons are rasterized as a **field-vs-background mask**, NOT a boundary-line class:
    0 = non-field / background
    1 = field  (interior/extent of a digitized smallholder crop-field polygon)
  Why not a boundary-line class? Fields here are small: median ~1,702 m^2 (~4 px across at
  10 m), 10th pct ~412 m^2 (~2 px). The physical field boundaries (bunds/dikes/paths) are
  typically 1-5 m wide -- sub-pixel at Sentinel-2's 10 m GSD and not separable as their own
  spectral class; a dilated 1-px boundary would consume most of these small fields, leaving
  no interior. The field *extent* IS observable at 10 m, so we encode field vs background.
  Because AI4SmallFarms exhaustively digitized every field inside each 5x5 km tile, the
  non-field (0) pixels are a genuine negative (not undeclared fields), unlike AI4Boundaries.

Processing (task spec §4 polygons, §5 balancing):
  * Per group (5x5 km tile): pick the local UTM (48N for centroid lon<108, else 49N),
    reproject polygons into that UTM's 10 m pixel grid, and tile the group extent into
    non-overlapping <=64x64 windows (spec cap 64). Rasterize field polygons (value 1) onto
    each window (fill 0). Keep windows containing >=1 field pixel; drop <32 px edge slivers.
  * Tiles-per-class balanced selection: <=1000 tiles per class, rarest-first, <=25k total.

Time range: 1-year 2021 window (labels anchored on 2021 S2 composites; post-2016 Sentinel
era). Static field extent, not a change dataset (change_time=null).

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.ai4smallfarms
"""

import argparse
import multiprocessing
import pickle
from typing import Any

import geopandas as gpd
import numpy as np
import shapely
import tqdm
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import (
    io,
    manifest,
    rasterize,
    sampling,
)

SLUG = "ai4smallfarms"
NAME = "AI4SmallFarms"
SRC_EPSG = 32648  # UTM 48N metres (parquet storage CRS)
RES = 10.0
TILE = 64
MIN_TILE = 32  # drop edge slivers smaller than this on either axis
PER_CLASS = 1000
YEAR = 2021
PARQUET = io.raw_dir(SLUG) / "ai4sf.parquet"
DATA_URL = "https://data.source.coop/fiboa/ai4sf/ai4sf.parquet"

CLASSES = [
    (
        "non-field",
        "Background: land not delineated as a smallholder crop field within the AI4SmallFarms "
        "5x5 km digitization tiles. Because every field inside a tile was exhaustively digitized, "
        "this is a genuine negative (not undeclared/missing fields).",
    ),
    (
        "field",
        "Interior/extent of a manually-digitized smallholder agricultural crop-field polygon "
        "(Cambodia/Vietnam), rasterized as a field mask at 10 m. The dataset's field-boundary "
        "delineation signal encoded as field vs non-field (sub-5 m physical bunds/boundaries are "
        "not separable at Sentinel-2 10 m; see module docstring).",
    ),
]
NUM_CLASSES = len(CLASSES)


def _dst_projection(lon: float, lat: float) -> Projection:
    """Local UTM projection at 10 m for a group's centroid lon/lat."""
    return get_utm_ups_projection(lon, lat, RES, -RES)


def _group_pixel_geoms(grp: int) -> tuple[list[Any], Projection]:
    """Load a group's polygons and return (list of pixel-space shapely geoms, dst UTM proj)."""
    gdf = gpd.read_parquet(str(PARQUET), filters=[("group", "==", grp)])
    src_crs = CRS.from_epsg(SRC_EPSG)
    src_proj = Projection(
        src_crs, 1, 1
    )  # geom coords are metres -> src "pixels" = metres
    # centroid lon/lat for UTM-zone choice (reproject metres -> WGS84)
    union = shapely.union_all(list(gdf.geometry.values))
    c = union.centroid
    lon, lat = (
        STGeometry(src_proj, shapely.Point(c.x, c.y), None)
        .to_projection(WGS84_PROJECTION)
        .shp.coords[0]
    )
    dst_proj = _dst_projection(lon, lat)
    geoms = [
        rasterize.geom_to_pixels(g, src_proj, dst_proj) for g in gdf.geometry.values
    ]
    return geoms, dst_proj


def _windows_for_extent(
    minc: int, minr: int, maxc: int, maxr: int
) -> list[tuple[int, int, int, int]]:
    """Non-overlapping (c0, r0, w, h) windows of <=64 px; drop <MIN_TILE edge slivers."""
    out = []
    for r0 in range(minr, maxr, TILE):
        h = min(TILE, maxr - r0)
        if h < MIN_TILE:
            continue
        for c0 in range(minc, maxc, TILE):
            w = min(TILE, maxc - c0)
            if w < MIN_TILE:
                continue
            out.append((c0, r0, w, h))
    return out


def _rasterize_window(
    geoms: list[Any], tree: shapely.STRtree, c0: int, r0: int, w: int, h: int
) -> np.ndarray:
    """Rasterize field polygons intersecting a window into a (h, w) uint8 field mask."""
    box = shapely.box(c0, r0, c0 + w, r0 + h)
    idx = tree.query(box)
    if len(idx) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    shapes = [(geoms[i], 1) for i in idx]
    arr = rasterize.rasterize_shapes(
        shapes, (c0, r0, c0 + w, r0 + h), fill=0, dtype="uint8", all_touched=False
    )
    return arr[0]


def _scan_group(grp: int) -> list[dict[str, Any]]:
    """Emit one lightweight record per field-containing window of a group."""
    try:
        geoms, dst_proj = _group_pixel_geoms(grp)
    except Exception as e:  # noqa: BLE001
        print(f"WARN scan failed group {grp}: {e}")
        return []
    if not geoms:
        return []
    tree = shapely.STRtree(geoms)
    xs = np.concatenate([np.array(g.bounds)[[0, 2]] for g in geoms])
    ys = np.concatenate([np.array(g.bounds)[[1, 3]] for g in geoms])
    minc, maxc = int(np.floor(xs.min())), int(np.ceil(xs.max()))
    minr, maxr = int(np.floor(ys.min())), int(np.ceil(ys.max()))
    crs_str = dst_proj.crs.to_string()
    recs = []
    for c0, r0, w, h in _windows_for_extent(minc, minr, maxc, maxr):
        sub = _rasterize_window(geoms, tree, c0, r0, w, h)
        present = sorted(int(v) for v in np.unique(sub))
        if 1 not in present:
            continue
        recs.append(
            {
                "group": grp,
                "crs": crs_str,
                "c0": c0,
                "r0": r0,
                "w": w,
                "h": h,
                "classes_present": present,
                "field_px": int((sub == 1).sum()),
                "source_id": f"group{grp:02d}/c{c0}_r{r0}",
            }
        )
    return recs


def _scan_all(workers: int) -> list[dict[str, Any]]:
    cache = io.raw_dir(SLUG) / "scan_cache.pkl"
    if cache.exists():
        print(f"loading cached scan from {cache}")
        with cache.open("rb") as f:
            return pickle.load(f)
    groups = list(range(62))
    print(f"scanning {len(groups)} groups (mp, reproject+rasterize windows)")
    all_recs: list[dict[str, Any]] = []
    with multiprocessing.Pool(workers) as p:
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan_group, [dict(grp=g) for g in groups]),
            total=len(groups),
        ):
            all_recs.extend(recs)
    print(f"scanned {len(all_recs)} field-containing candidate windows")
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    tmp = io.raw_dir(SLUG) / "scan_cache.pkl.tmp"
    with tmp.open("wb") as f:
        pickle.dump(all_recs, f)
    tmp.rename(cache)
    return all_recs


def _write_group(grp: int, recs: list[dict[str, Any]]) -> None:
    """Rasterize and write all selected windows for one group (polygons loaded once)."""
    todo = [
        r
        for r in recs
        if not (io.locations_dir(SLUG) / f"{r['sample_id']}.tif").exists()
    ]
    if not todo:
        return
    geoms, dst_proj = _group_pixel_geoms(grp)
    tree = shapely.STRtree(geoms)
    proj = Projection(CRS.from_string(recs[0]["crs"]), RES, -RES)
    for r in todo:
        c0, r0, w, h = r["c0"], r["r0"], r["w"], r["h"]
        sub = _rasterize_window(geoms, tree, c0, r0, w, h)
        bounds = (c0, r0, c0 + w, r0 + h)
        io.write_label_geotiff(SLUG, r["sample_id"], sub, proj, bounds)
        io.write_sample_json(
            SLUG,
            r["sample_id"],
            proj,
            bounds,
            io.year_range(YEAR),
            source_id=r["source_id"],
            classes_present=sorted(int(v) for v in np.unique(sub)),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    if not PARQUET.exists():
        raise RuntimeError(
            f"{PARQUET} missing; download it first with:\n  curl -L {DATA_URL} -o {PARQUET}"
        )
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Source: AI4SmallFarms (Persello et al. 2023), fiboa GeoParquet on Source "
            "Cooperative (CC-BY-4.0, no credential).\n"
            f"URL: https://source.coop/fiboa/ai4sf  ->  {DATA_URL}\n"
            "Downloaded only ai4sf.parquet (29 MB, 439,001 field polygons; EPSG:32648).\n"
            "NOT downloaded: ai4sf.pmtiles (map tiles). Pretraining supplies imagery.\n"
        )

    records = _scan_all(args.workers)
    selected = sampling.select_tiles_per_class(
        records,
        classes_key="classes_present",
        per_class=PER_CLASS,
        total_cap=sampling.MAX_SAMPLES_PER_DATASET,
    )
    selected.sort(key=lambda r: r["source_id"])
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} windows (of {len(records)} candidates)")

    by_group: dict[int, list[dict[str, Any]]] = {}
    for r in selected:
        by_group.setdefault(r["group"], []).append(r)
    args_list = [dict(grp=g, recs=rs) for g, rs in by_group.items()]
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_group, args_list), total=len(args_list)
        ):
            pass

    tile_counts = {i: 0 for i in range(NUM_CLASSES)}
    country_counts: dict[str, int] = {}
    field_px_total = 0
    total_px = 0
    for r in selected:
        for c in r["classes_present"]:
            tile_counts[c] += 1
        field_px_total += r["field_px"]
        total_px += r["w"] * r["h"]
    # per-group country lookup (pandas; no geometry needed)
    import pandas as pd

    meta = pd.read_parquet(str(PARQUET), columns=["group", "country"])
    g2c = meta.groupby("group")["country"].first().to_dict()
    for r in selected:
        c = str(g2c.get(r["group"], "unknown"))
        country_counts[c] = country_counts.get(c, 0) + 1

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Source Cooperative (fiboa/ai4sf)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://source.coop/fiboa/ai4sf",
                "have_locally": False,
                "annotation_method": "manual digitization from Sentinel-2 imagery (2021)",
                "citation": (
                    "Persello, C., Grift, J., Fan, X., Paris, C., Hansch, R., Koeva, M., "
                    "& Nelson, A. (2023). AI4SmallFarms. IEEE GRSL 20, 2505705. "
                    "https://doi.org/10.1109/LGRS.2023.3323095"
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_tile_counts": {
                CLASSES[i][0]: tile_counts[i] for i in range(NUM_CLASSES)
            },
            "country_tile_counts": country_counts,
            "field_pixel_fraction": round(field_px_total / max(1, total_px), 4),
            "notes": (
                "439,001 manually-digitized smallholder crop-field polygons (Cambodia + "
                "Vietnam), 62 tiles of ~5x5 km, fiboa GeoParquet on Source Cooperative. "
                "Encoded as a 2-class field-vs-background mask (0 non-field, 1 field), NOT a "
                "boundary-line class: fields are small (median ~1,702 m^2, ~4 px at 10 m; p10 "
                "~412 m^2) and their physical boundaries (bunds/dikes, 1-5 m) are sub-pixel at "
                "Sentinel-2 10 m, so a dilated boundary line would consume the field. "
                "Exhaustive within-tile digitization makes non-field (0) a genuine negative. "
                "Per group, polygons reprojected to local UTM (48N/49N) at 10 m and rasterized "
                "into non-overlapping <=64x64 windows containing >=1 field pixel (<32 px "
                "slivers dropped). Time range = 1-year 2021 window (labels anchored on 2021 S2 "
                "composites; determination_datetime=2021-08-01). Static extent, not a change "
                "dataset. Tiles-per-class balanced <=1000/class, rarest-first, <=25k total."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print(
        "class tile counts:",
        {CLASSES[i][0]: tile_counts[i] for i in range(NUM_CLASSES)},
    )
    print("country tile counts:", country_counts)
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
