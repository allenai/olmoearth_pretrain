"""Process Global Debris-Covered Glaciers (Herreid & Pellicciotti 2020) into open-set
segmentation label patches.

Source: Zenodo record 3866466 ("Supplementary Information for Herreid and Pellicciotti,
Nature Geoscience, 2020"), a single ``SupplementaryInformation.zip`` containing nested
``S1.zip`` (per-RGI-region shapefiles -- the vector product we use), ``S2.zip``
(footprints) and ``S3.zip`` (Scherler-2018 comparison). We use S1 only.

S1 holds one folder per RGI first-order region (all 18 except Antarctica). Per region the
relevant polygon layers are:
  * ``{region}_minGl1km2.shp``            -- glacier outlines >= 1 km^2 (ice extent)
  * ``{region}_minGl1km2_debrisCover.shp``-- supraglacial debris-cover polygons
  * ``{region}_ablationZone.shp``         -- ablation-zone polygons
(``debrisExpansionLine`` / ``equilibriumLine`` are lines and are not used; ``minGl2km2``
is a coarser subset of the same glaciers.)

Class mapping (classification, 3 classes):
  0 debris-covered area  <- minGl1km2_debrisCover polygons
  1 clean ice            <- minGl1km2 glacier outline, with debris polygons subtracted
                            (burned to nodata) so only debris-free ice remains
  2 ablation zone        <- ablationZone polygons

Debris-covered area and clean ice are the two surface-cover classes and are mutually
exclusive (debris is a subset of the glacier outline); the ablation zone is an
elevation-based partition orthogonal to surface cover, so it is emitted as its own mask
rather than composited with the other two. Each output tile is therefore a single-class
positive mask: the class ID inside the polygon(s), 255 (nodata) everywhere else. This
respects the <=1000-tiles-per-class balance exactly (each tile counts toward one class).

Each polygon is rasterized into a 64x64 UTM 10 m window centered on a representative
interior point of the polygon. Large glaciers exceed the window and yield homogeneous
interior tiles; small polygons show their shape against nodata.

Sampling: round-robin across all 18 regions for geographic diversity, up to 1000 tiles
per class.

Time range: supraglacial debris and glacier outlines are slowly-changing features; the
manifest anchors this product at 2016-2017 and per-feature imagery years span ~1986-2016.
We assign a uniform 1-year window (2016) in the Sentinel era to every sample and record
the source imagery year (``img_time``) in the sample source_id.
"""

import argparse
import multiprocessing
import os
import random
import warnings
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import tqdm
from rasterio.crs import CRS as RioCRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)

warnings.filterwarnings("ignore")

SLUG = "global_debris_covered_glaciers_herreid_pellicciotti"
NAME = "Global Debris-Covered Glaciers (Herreid & Pellicciotti)"
RAW = io.raw_dir(SLUG)
S1_DIR = RAW / "SupplementaryInformation" / "S1_extracted" / "S1"
PER_CLASS = 1000
TILE = 64
YEAR = 2016

# class_id -> (name, layer suffix, description)
CLASSES = [
    (
        0,
        "debris-covered area",
        "minGl1km2_debrisCover",
        "Supraglacial debris cover (rock/sediment mantling glacier ice) mapped on "
        "glaciers >= 1 km^2, manually corrected (Herreid & Pellicciotti 2020).",
    ),
    (
        1,
        "clean ice",
        "minGl1km2",
        "Debris-free glacier ice: the RGI glacier outline (>= 1 km^2) with mapped "
        "supraglacial debris polygons removed.",
    ),
    (
        2,
        "ablation zone",
        "ablationZone",
        "Glacier ablation zone (surface below the equilibrium-line altitude, net mass "
        "loss area) delineated per glacier.",
    ),
]
ID_TO_LAYER = {cid: layer for cid, _n, layer, _d in CLASSES}
ID_TO_NAME = {cid: n for cid, n, _l, _d in CLASSES}


def regions() -> list[str]:
    return sorted(d for d in os.listdir(S1_DIR.path) if (S1_DIR / d).is_dir())


def _shp_path(region: str, layer: str) -> str:
    return (S1_DIR / region / f"{region}_{layer}.shp").path


# --------------------------------------------------------------------------- scan


def _scan_one(region: str, cid: int) -> list[dict[str, Any]]:
    """Read one (region, class) shapefile; return lightweight per-polygon records."""
    import geopandas as gpd

    layer = ID_TO_LAYER[cid]
    path = _shp_path(region, layer)
    if not os.path.exists(path):
        return []
    gdf = gpd.read_file(path)
    if len(gdf) == 0:
        return []
    reps = gdf.geometry.representative_point()
    lonlat = reps.to_crs(4326)
    has_time = "img_time" in gdf.columns
    recs = []
    for i in range(len(gdf)):
        pt = lonlat.iloc[i]
        if pt is None or pt.is_empty:
            continue
        recs.append(
            {
                "region": region,
                "cid": cid,
                "fid": i,
                "lon": float(pt.x),
                "lat": float(pt.y),
                "img_time": float(gdf["img_time"].iloc[i]) if has_time else None,
            }
        )
    return recs


def scan() -> list[dict[str, Any]]:
    jobs = [dict(region=r, cid=cid) for r in regions() for cid, *_ in CLASSES]
    out: list[dict[str, Any]] = []
    with multiprocessing.Pool(min(64, len(jobs))) as p:
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan_one, jobs), total=len(jobs), desc="scan"
        ):
            out.extend(recs)
    return out


# ---------------------------------------------------------------------- selection


def sample_round_robin(
    cands: list[dict[str, Any]], per_class: int, seed: int = 42
) -> list[dict[str, Any]]:
    """Round-robin across regions to maximize geographic diversity, up to per_class."""
    by_reg: dict[str, list] = defaultdict(list)
    for c in cands:
        by_reg[c["region"]].append(c)
    rng = random.Random(seed)
    for v in by_reg.values():
        rng.shuffle(v)
    regs = sorted(by_reg)
    out: list[dict[str, Any]] = []
    idx = {r: 0 for r in regs}
    while len(out) < per_class:
        progressed = False
        for r in regs:
            if idx[r] < len(by_reg[r]):
                out.append(by_reg[r][idx[r]])
                idx[r] += 1
                progressed = True
                if len(out) >= per_class:
                    break
        if not progressed:
            break
    return out


# --------------------------------------------------------------------------- write


def _write_region(region: str, recs: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Rasterize + write all selected samples for one region. Returns (sample_id, cid)."""
    import geopandas as gpd
    import shapely

    # Load each needed layer once.
    layer_gdf: dict[str, Any] = {}
    layer_srcproj: dict[str, Any] = {}
    needed_layers = {ID_TO_LAYER[r["cid"]] for r in recs}
    # clean-ice needs the debris layer for subtraction
    need_debris = any(r["cid"] == 1 for r in recs)
    if need_debris:
        needed_layers.add("minGl1km2_debrisCover")
    for layer in needed_layers:
        path = _shp_path(region, layer)
        if not os.path.exists(path):
            continue
        gdf = gpd.read_file(path)
        layer_gdf[layer] = gdf
        layer_srcproj[layer] = Projection(RioCRS.from_wkt(gdf.crs.to_wkt()), 1, 1)

    written: list[tuple[str, int]] = []
    for rec in recs:
        sample_id = rec["sample_id"]
        cid = rec["cid"]
        tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
        if tif.exists():
            written.append((sample_id, cid))
            continue
        layer = ID_TO_LAYER[cid]
        gdf = layer_gdf.get(layer)
        if gdf is None:
            continue
        src_proj = layer_srcproj[layer]
        geom = gdf.geometry.iloc[rec["fid"]]
        if geom is None or geom.is_empty:
            continue

        proj, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"])
        bounds = io.centered_bounds(col, row, TILE, TILE)

        geom_px = geom_to_pixels(geom, src_proj, proj)
        shapes: list[tuple[Any, int]] = [(geom_px, cid)]

        if cid == 1:
            # Subtract supraglacial debris from the glacier outline (burn to nodata).
            deb = layer_gdf.get("minGl1km2_debrisCover")
            if deb is not None and len(deb) > 0:
                deb_src = layer_srcproj["minGl1km2_debrisCover"]
                # query debris polygons near the glacier's representative point
                box = shapely.buffer(shapely.Point(*_repr_xy(geom)), 800.0).envelope
                hits = deb.sindex.query(box, predicate="intersects")
                for j in hits:
                    dg = deb.geometry.iloc[int(j)]
                    if dg is None or dg.is_empty:
                        continue
                    shapes.append((geom_to_pixels(dg, deb_src, proj), io.CLASS_NODATA))

        arr = rasterize_shapes(
            shapes, bounds, fill=io.CLASS_NODATA, dtype="uint8", all_touched=True
        )
        if not np.any(arr == cid):
            # degenerate (e.g. glacier fully debris-covered) -> skip, no valid pixels
            continue

        io.write_label_geotiff(
            SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA
        )
        src_id = f"{region}/{layer}/{rec['fid']}"
        if rec["img_time"] is not None:
            src_id += f"@img_time={rec['img_time']}"
        io.write_sample_json(
            SLUG,
            sample_id,
            proj,
            bounds,
            io.year_range(YEAR),
            source_id=src_id,
            classes_present=[cid],
        )
        written.append((sample_id, cid))
    return written


def _repr_xy(geom: Any) -> tuple[float, float]:
    p = geom.representative_point()
    return (p.x, p.y)


# ---------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    if not S1_DIR.exists():
        raise RuntimeError(f"S1 not extracted at {S1_DIR}; run download/unzip first.")

    with (RAW / "SOURCE.txt").open("w") as f:
        f.write(
            "Zenodo record 3866466 (Herreid & Pellicciotti 2020).\n"
            "SupplementaryInformation.zip -> S1.zip -> per-RGI-region shapefiles.\n"
        )

    recs = scan()
    print(f"scanned {len(recs)} polygons across {len(regions())} regions")

    # Select up to PER_CLASS per class, round-robin over regions.
    selected: list[dict[str, Any]] = []
    for cid, *_ in CLASSES:
        cands = [r for r in recs if r["cid"] == cid]
        sel = sample_round_robin(cands, PER_CLASS)
        selected.extend(sel)
        print(
            f"  class {cid} ({ID_TO_NAME[cid]}): {len(cands)} cands -> {len(sel)} selected"
        )

    # Deterministic ordering -> stable sample ids (idempotent reruns).
    selected.sort(key=lambda r: (r["cid"], r["region"], r["fid"]))
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"

    io.check_disk()

    by_region: dict[str, list] = defaultdict(list)
    for r in selected:
        by_region[r["region"]].append(r)
    jobs = [dict(region=reg, recs=rs) for reg, rs in by_region.items()]

    written: list[tuple[str, int]] = []
    with multiprocessing.Pool(min(args.workers, len(jobs))) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_region, jobs), total=len(jobs), desc="write"
        ):
            written.extend(res)

    counts = Counter(cid for _sid, cid in written)
    print(f"wrote {len(written)} samples")
    for cid, *_ in CLASSES:
        print(f"  class {cid} ({ID_TO_NAME[cid]}): {counts.get(cid, 0)}")

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Zenodo (record 3866466)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://zenodo.org/records/3866466",
                "have_locally": False,
                "annotation_method": "manual correction of automated debris mapping "
                "(Herreid & Pellicciotti, Nature Geoscience 2020)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": cid, "name": name, "description": desc}
                for cid, name, _layer, desc in CLASSES
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(written),
            "class_counts": {
                ID_TO_NAME[cid]: counts.get(cid, 0) for cid, *_ in CLASSES
            },
            "notes": (
                "Single-class 64x64 UTM 10 m polygon masks (class id inside polygon, 255 "
                "nodata elsewhere). Clean ice = glacier outline minus supraglacial debris. "
                "Round-robin sampled across all 18 RGI regions (Antarctica excluded). "
                "Uniform 2016 1-year time range; per-feature source imagery year in "
                "sample source_id."
            ),
        },
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
