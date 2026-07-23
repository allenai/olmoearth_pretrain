"""Process the RGIK Rock Glacier Inventories (RoGI) into open-set segmentation labels.

Source: Zenodo record 14501398 (concept) / 15467203 (v2.0), Rouyet et al., "Rock Glacier
Inventories (RoGI) in 12 areas worldwide using a multi-operator consensus-based procedure"
(ESA CCI Permafrost). A single ~2.4 MB archive holds one all-areas GeoPackage with layers:
  * ``..._AOI_...``  (12 area-of-interest polygons; not used)
  * ``..._GO_...``   603 geomorphological-outline MultiPolygons -- the rock-glacier
                     landform footprints we rasterize. Attributes: PolyUID, PrimaryID,
                     OutType (Extended | Restricted), RelFr/RelLeftLM/... geomorphological
                     activity indices.
  * ``..._MA_...``   575 InSAR "moving area" MultiPolygons (VelClass); not used -- moving
                     areas are a kinematic sub-delineation orthogonal to the per-landform
                     activity class, and overlap active rock glaciers.
  * ``..._PM_...``   631 primary-marker Points with the consensus activity classification
                     (``ActiCl``) used as the label. Joined to GO outlines via PrimaryID.

Task: classification of rock-glacier **activity** (per RGIK guidelines). We rasterize each
geomorphological outline polygon into a 64x64 UTM 10 m tile centered on the polygon's
representative point; pixels inside the outline carry the activity class id, everything
outside is 255 (nodata). Each tile is a single-class positive mask (tiles-per-class
balanced, one class per tile), mirroring the debris-covered-glaciers recipe.

Class mapping (3 classes), consolidating the RGIK "uncertain" qualifier into the base
class (uncertainty is recorded but does not change the activity category):
  0 active        <- ActiCl in {Active, Active uncertain}
  1 transitional  <- ActiCl == Transitional
  2 relict        <- ActiCl in {Relict, Relict uncertain}
Outlines whose linked marker has ActiCl == "Uncertain" or null are dropped (no activity).

Both outline delineations per rock glacier are used as separate tiles: the Extended
outline (full landform incl. rooting zone/talus) and, where present, the Restricted
outline (main body). They share a location/class but are distinct source delineations;
using both maximizes labels for this small, rare-class inventory. Large outlines (~21%
exceed 640 m) are clipped to the 64x64 window, yielding a homogeneous interior tile;
small outlines show their shape against nodata.

Time range: rock glaciers are slow landforms and the consensus (esp. kinematic) attribution
draws on InSAR over ~2018-2021 (manifest time_range). We assign every sample a uniform
1-year window (2019) within that observation period; the InSAR period is noted in the
summary.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.rgik_rock_glacier_inventories_rogi
"""

import argparse
import multiprocessing
import warnings
from collections import Counter
from typing import Any

import numpy as np
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)

warnings.filterwarnings("ignore")

SLUG = "rgik_rock_glacier_inventories_rogi"
NAME = "RGIK Rock Glacier Inventories (RoGI)"
RAW = io.raw_dir(SLUG)
GPKG = (
    RAW
    / "extracted"
    / "Rouyet-et-al_RoGI_Zenodo_v2.0"
    / "ESACCI-PERMAFROST_ROGI_ALL-AREAS_AOI-PM-MA-GO_2025-fv02.0.gpkg"
)
GO_LAYER = "ESACCI-PERMAFROST_ROGI_ALL-AREAS_GO_2025-fv02.0"
PM_LAYER = "ESACCI-PERMAFROST_ROGI_ALL-AREAS_PM_2025-fv02.0"

TILE = 64
YEAR = 2019
PER_CLASS = 1000

# activity class id -> (name, description)
CLASSES = [
    (
        0,
        "active",
        "Active rock glacier: creeping ice-rich permafrost landform with detectable "
        "downslope movement (RGIK consensus ActiCl 'Active'/'Active uncertain').",
    ),
    (
        1,
        "transitional",
        "Transitional rock glacier: intermediate between active and relict; degrading "
        "permafrost with weak/residual movement (RGIK ActiCl 'Transitional').",
    ),
    (
        2,
        "relict",
        "Relict rock glacier: ice-free, no longer moving fossil landform "
        "(RGIK ActiCl 'Relict'/'Relict uncertain').",
    ),
]
ID_TO_NAME = {cid: n for cid, n, _d in CLASSES}


def _acti_to_cid(acti: Any) -> int | None:
    """Map RGIK ActiCl string to an activity class id (or None to drop)."""
    if not isinstance(acti, str):
        return None
    a = acti.strip().lower()
    if a.startswith("active"):
        return 0
    if a.startswith("transitional"):
        return 1
    if a.startswith("relict"):
        return 2
    return None  # pure "Uncertain" / unknown


# --------------------------------------------------------------------------- scan


def scan() -> list[dict[str, Any]]:
    """Load GO outlines, join activity from PM markers, return per-outline records."""
    import geopandas as gpd

    pm = gpd.read_file(GPKG.path, layer=PM_LAYER)
    go = gpd.read_file(GPKG.path, layer=GO_LAYER)
    acti_by_pid = dict(zip(pm["PrimaryID"], pm["ActiCl"]))
    country_by_pid = dict(zip(pm["PrimaryID"], pm["Country"]))

    reps = go.geometry.representative_point()  # in EPSG:4326 -> lon/lat
    recs: list[dict[str, Any]] = []
    for i in range(len(go)):
        row = go.iloc[i]
        pid = row["PrimaryID"]
        cid = _acti_to_cid(acti_by_pid.get(pid))
        if cid is None:
            continue
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        pt = reps.iloc[i]
        if pt is None or pt.is_empty:
            continue
        recs.append(
            {
                "cid": cid,
                "geom": geom,
                "lon": float(pt.x),
                "lat": float(pt.y),
                "poly_uid": row["PolyUID"],
                "primary_id": pid,
                "out_type": (row["OutType"] or "").strip(),
                "country": country_by_pid.get(pid),
            }
        )
    return recs


# --------------------------------------------------------------------------- write


def _write_chunk(recs: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Rasterize + write a chunk of outline records. Returns (sample_id, cid)."""
    src_proj = WGS84_PROJECTION
    written: list[tuple[str, int]] = []
    for rec in recs:
        sample_id = rec["sample_id"]
        cid = rec["cid"]
        tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
        if tif.exists():
            written.append((sample_id, cid))
            continue
        proj, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"])
        bounds = io.centered_bounds(col, row, TILE, TILE)
        geom_px = geom_to_pixels(rec["geom"], src_proj, proj)
        arr = rasterize_shapes(
            [(geom_px, cid)],
            bounds,
            fill=io.CLASS_NODATA,
            dtype="uint8",
            all_touched=True,
        )
        if not np.any(arr == cid):
            continue  # polygon missed the window (degenerate); skip
        io.write_label_geotiff(
            SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA
        )
        src_id = f"{rec['country']}/{rec['primary_id']}/{rec['out_type']}"
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


# ---------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    if not GPKG.exists():
        raise RuntimeError(f"GeoPackage not found at {GPKG}; run download/unzip first.")

    with (RAW / "SOURCE.txt").open("w") as f:
        f.write(
            "Zenodo record 14501398 / v2.0 15467203 (Rouyet et al., RoGI, ESA CCI "
            "Permafrost).\nRouyet-et-al_RoGI_Zenodo_v2.0.zip -> all-areas GeoPackage "
            "(GO outlines + PM activity markers).\n"
        )

    recs = scan()
    counts_all = Counter(r["cid"] for r in recs)
    print(
        f"scanned {len(recs)} outline records with activity: "
        + ", ".join(f"{ID_TO_NAME[c]}={counts_all.get(c, 0)}" for c, *_ in CLASSES)
    )

    # Per-class cap (far under caps here) with deterministic ordering -> stable ids.
    from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

    selected = balance_by_class(recs, "cid", per_class=PER_CLASS)
    selected.sort(key=lambda r: (r["cid"], str(r["primary_id"]), r["out_type"]))
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"

    io.check_disk()

    # Chunk for the write pool.
    n_workers = max(1, min(args.workers, len(selected)))
    chunks: list[list[dict[str, Any]]] = [[] for _ in range(n_workers)]
    for i, r in enumerate(selected):
        chunks[i % n_workers].append(r)
    jobs = [dict(recs=c) for c in chunks if c]

    written: list[tuple[str, int]] = []
    with multiprocessing.Pool(len(jobs)) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_chunk, jobs), total=len(jobs), desc="write"
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
            "source": "Zenodo (record 14501398 / v2.0 15467203)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.org/10.5281/zenodo.14501398",
                "have_locally": False,
                "annotation_method": "multi-operator consensus geomorphological mapping "
                "+ InSAR kinematics (RGIK guidelines; Rouyet et al., ESA CCI Permafrost)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": cid, "name": name, "description": desc}
                for cid, name, desc in CLASSES
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(written),
            "class_counts": {
                ID_TO_NAME[cid]: counts.get(cid, 0) for cid, *_ in CLASSES
            },
            "notes": (
                "Geomorphological-outline polygons (GO layer) rasterized to 64x64 UTM "
                "10 m tiles; activity class inside outline, 255 nodata outside (one class "
                "per tile). Activity from linked primary-marker ActiCl (PrimaryID join); "
                "'uncertain' qualifier folded into base class, pure-uncertain/null dropped. "
                "Both Extended and Restricted outlines per landform kept as separate tiles. "
                "InSAR/consensus obs period ~2018-2021; uniform 2019 1-year time range. "
                "Moving-area (MA) and AOI layers not used."
            ),
        },
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
