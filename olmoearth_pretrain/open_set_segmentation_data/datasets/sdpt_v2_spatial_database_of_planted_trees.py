"""Process SDPT v2 (Spatial Database of Planted Trees) into label patches (rasterized polygons).

Source: WRI / Global Forest Watch, "Spatial Database of Planted Trees (SDPT Version 2.0)"
(Richter et al. 2024), the near-global compilation of planted-forest and tree-crop
polygons for 158 countries (~90% of world planted-forest area in 2020). Licensed CC-BY-4.0.
Downloaded as the public File Geodatabase from the GFW S3 bucket:
  https://gfw2-data.s3.amazonaws.com/plantations/sdpt/sdpt_v2_v11282023_public.gdb.zip
(5.5 GB zipped; ~25 GB unzipped). The GFW Data API /query endpoint requires an API key we
do not have, but this bulk archive is public/unauthenticated, so we pull it once and sample
locally. The GDB has one MultiPolygon layer per country (``{iso3}_plant_v2``), 26.6M polygons
total, sharing a harmonized attribute table.

Task: per-pixel **classification** of planted-tree species/type. The class field is
``sciName`` (scientific taxon), the SDPT harmonized species/genus name -- e.g. Hevea
brasiliensis (rubber), Elaeis guineensis (oil palm), Pinus sp., Eucalyptus sp., Prunus dulcis
(almond). Globally there are ~1180 distinct ``sciName`` values, so we honor the uint8 254-class
cap: keep the **top 254 by global frequency** (ids 0..N-1 in descending frequency) and drop the
rest (dropped count recorded in the summary). The sentinel value ``Unknown`` (and ``Unknown
mix`` / null), which covers ~65% of polygons where the source could not identify a species, is
**not a usable class** and is excluded from the class set (documented; these polygons are simply
never sampled). ``simpleName`` (13 coarse types: Oil palm / Rubber / Fruit / Wood fiber or
timber / ...) and ``simpleType`` (Planted forest / Tree crops) are recorded in the summary but
not used as the label -- the task calls for the fine species/type scheme.

Each selected polygon is rasterized into a <=64x64 UTM 10 m tile sized to the polygon footprint
(capped at 64): the polygon's class id is burned inside, 255 (nodata/ignore) outside -- SDPT
only labels planted-tree polygons, so unlabeled land is ignore, not a background class (spec 5:
no fabricated negatives; assembly step supplies negatives from other datasets).

Sampling (spec 5, bounded sampling for a large global product): tiles-per-class balanced with
the 25k per-dataset cap. With N (<=254) classes the effective per-class limit is
min(1000, 25000 // N). We do not attempt global coverage: only enough polygons per class are
read to reach the target. Rare classes are prioritized by ``balance_by_class``.

Time range: SDPT plantations are persistent land cover. We assign each sample a 1-year window
(spec 5 static-label rule) anchored on a representative Sentinel-era year parsed from the
polygon's ``imageryYear`` (the year(s) of imagery used to delineate it), clamped into the
manifest range [2016, 2020]; unparseable -> 2020.

Run (idempotent; skips already-written {sample_id}.tif):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.sdpt_v2_spatial_database_of_planted_trees
"""

import argparse
import hashlib
import multiprocessing
import random
import re
from collections import Counter
from typing import Any

import numpy as np
import pyogrio
import shapely
import tqdm
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "sdpt_v2_spatial_database_of_planted_trees"
NAME = "SDPT v2 (Spatial Database of Planted Trees)"

S3_URL = "https://gfw2-data.s3.amazonaws.com/plantations/sdpt/sdpt_v2_v11282023_public.gdb.zip"
GDB_ZIP = "sdpt_v2_v11282023_public.gdb.zip"
GDB_NAME = "sdpt_v2_v11282023_public.gdb"

CLASS_FIELD = "sciName"
# Values that are not a usable species/type class (source could not identify a taxon).
EXCLUDE_CLASSES = {"Unknown", "Unknown mix", "__NA__", "", "None", "nan"}

MAX_CLASSES = 254
PER_CLASS = 1000  # lowered automatically to 25000 // N by balance_by_class.
MAX_TILE = io.MAX_TILE  # 64
# Per-layer per-class candidate cap: reads only enough polygons to draw the target counts
# from a fair random subset (bounded sampling for a large global product; spec 5).
CAND_CAP = 400

_WGS84_SRC = Projection(CRS.from_epsg(4326), 1, 1)


# --------------------------------------------------------------------------------------
# Download + unzip.
# --------------------------------------------------------------------------------------
def ensure_data() -> str:
    """Download + unzip the SDPT v2 public File Geodatabase; return the .gdb path."""
    import zipfile
    from pathlib import Path

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / GDB_ZIP
    download.download_http(S3_URL, zip_path)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "SDPT v2 (Spatial Database of Planted Trees), WRI / Global Forest Watch, "
            "CC-BY-4.0.\n"
            f"{S3_URL}\n"
            "https://www.wri.org/research/spatial-database-planted-trees-sdpt-version-2\n"
        )
    gdb_path = Path(raw.path) / "unzip" / GDB_NAME
    if not gdb_path.exists():
        (Path(raw.path) / "unzip").mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path.path) as zf:
            zf.extractall(Path(raw.path) / "unzip")
    if not gdb_path.exists():
        raise RuntimeError(f"GDB not found after unzip: {gdb_path}")
    return str(gdb_path)


def list_layers(gdb: str) -> list[str]:
    return [str(row[0]) for row in pyogrio.list_layers(gdb)]


# --------------------------------------------------------------------------------------
# Year parsing.
# --------------------------------------------------------------------------------------
def pick_year(imagery_year: Any) -> int:
    """Representative Sentinel-era year for a polygon, clamped into [2016, 2020]."""
    yrs = [int(y) for y in re.findall(r"(?:19|20)\d{2}", str(imagery_year))]
    yrs = [y for y in yrs if 1900 <= y <= 2025]
    if yrs:
        return min(max(max(yrs), 2016), 2020)
    return 2020


# --------------------------------------------------------------------------------------
# Pass 1 worker: global class frequency (attribute-only read).
# --------------------------------------------------------------------------------------
def _freq_worker(gdb: str, layer: str) -> Counter:
    df = pyogrio.read_dataframe(
        gdb, layer=layer, read_geometry=False, columns=[CLASS_FIELD]
    )
    vals = df[CLASS_FIELD].fillna("__NA__").astype(str)
    c: Counter = Counter(vals.tolist())
    for k in list(c):
        if k in EXCLUDE_CLASSES:
            del c[k]
    return c


# --------------------------------------------------------------------------------------
# Pass 2 worker: subsample candidate fids per kept class (attribute-only read).
# --------------------------------------------------------------------------------------
def _cand_worker(
    gdb: str, layer: str, code_to_id: dict[str, int]
) -> list[dict[str, Any]]:
    df = pyogrio.read_dataframe(
        gdb,
        layer=layer,
        read_geometry=False,
        columns=[CLASS_FIELD, "imageryYear"],
        fid_as_index=True,
    )
    names = df[CLASS_FIELD].fillna("__NA__").astype(str).to_numpy()
    years = df["imageryYear"].to_numpy()
    fids = df.index.to_numpy()
    by_class: dict[int, list[int]] = {}
    for i, name in enumerate(names):
        cid = code_to_id.get(name)
        if cid is None:
            continue
        by_class.setdefault(cid, []).append(i)
    rng = random.Random(int(hashlib.md5(layer.encode()).hexdigest()[:8], 16))
    out: list[dict[str, Any]] = []
    for cid, idxs in by_class.items():
        if len(idxs) > CAND_CAP:
            idxs = rng.sample(idxs, CAND_CAP)
        for i in idxs:
            out.append(
                {
                    "layer": layer,
                    "fid": int(fids[i]),
                    "class_id": cid,
                    "year": pick_year(years[i]),
                }
            )
    return out


# --------------------------------------------------------------------------------------
# Pass 3 worker: rasterize one polygon into a <=64x64 UTM tile.
# --------------------------------------------------------------------------------------
def _write_tile(rec: dict[str, Any]) -> tuple[str, str]:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return sample_id, "skip"
    try:
        geom = shapely.from_wkb(rec["geom_wkb"])  # WGS84 lon/lat
        proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
        pix = geom_to_pixels(geom, _WGS84_SRC, proj)
        minx, miny, maxx, maxy = pix.bounds
        cx = int(round((minx + maxx) / 2))
        cy = int(round((miny + maxy) / 2))
        w = min(MAX_TILE, max(1, int(np.ceil(maxx - minx))))
        h = min(MAX_TILE, max(1, int(np.ceil(maxy - miny))))
        bounds = io.centered_bounds(cx, cy, w, h)
        arr = rasterize_shapes(
            [(pix, int(rec["class_id"]))],
            bounds,
            fill=io.CLASS_NODATA,
            dtype="uint8",
            all_touched=True,
        )
        if not (arr != io.CLASS_NODATA).any():
            return sample_id, "empty"
        io.write_label_geotiff(
            SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA
        )
        io.write_sample_json(
            SLUG,
            sample_id,
            proj,
            bounds,
            io.year_range(rec["year"]),
            source_id=rec["source_id"],
            classes_present=sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA}),
        )
        return sample_id, "ok"
    except Exception as e:  # noqa: BLE001
        print(f"error on {sample_id}: {e}")
        return sample_id, "error"


# --------------------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    gdb = ensure_data()
    layers = list_layers(gdb)
    print(f"{len(layers)} country layers")

    io.check_disk()

    # ---- Pass 1: global class frequency ------------------------------------------
    global_freq: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for c in tqdm.tqdm(
            star_imap_unordered(
                p, _freq_worker, [dict(gdb=gdb, layer=l) for l in layers]
            ),
            total=len(layers),
            desc="freq",
        ):
            global_freq += c
    ranked = [name for name, _ in global_freq.most_common()]
    kept = ranked[:MAX_CLASSES]
    dropped = ranked[MAX_CLASSES:]
    code_to_id = {name: i for i, name in enumerate(kept)}
    print(
        f"distinct usable {CLASS_FIELD}: {len(ranked)}; kept {len(kept)}; "
        f"dropped {len(dropped)}"
    )

    io.check_disk()

    # ---- Pass 2: subsample candidate fids per kept class -------------------------
    records: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for recs in tqdm.tqdm(
            star_imap_unordered(
                p,
                _cand_worker,
                [dict(gdb=gdb, layer=l, code_to_id=code_to_id) for l in layers],
            ),
            total=len(layers),
            desc="cand",
        ):
            records.extend(recs)
    print(f"candidate polygons for kept classes: {len(records)}")

    selected = balance_by_class(
        records, key="class_id", per_class=PER_CLASS, total_cap=25000
    )
    n_classes = len(code_to_id)
    eff_per_class = max(1, min(PER_CLASS, 25000 // max(1, n_classes)))
    print(f"selected {len(selected)} polygons (eff per-class cap = {eff_per_class})")

    io.check_disk()

    # ---- Read geometries for selected fids (grouped by layer) --------------------
    by_layer: dict[str, list[dict[str, Any]]] = {}
    for r in selected:
        by_layer.setdefault(r["layer"], []).append(r)

    id_to_name = {i: name for name, i in code_to_id.items()}
    tile_recs: list[dict[str, Any]] = []
    for layer, recs in by_layer.items():
        fids = sorted({r["fid"] for r in recs})
        gdf = pyogrio.read_dataframe(
            gdb, layer=layer, columns=[CLASS_FIELD], fids=fids, fid_as_index=True
        )
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)
        geom_by_fid = {int(fid): geom for fid, geom in gdf.geometry.items()}
        for r in recs:
            geom = geom_by_fid.get(int(r["fid"]))
            if geom is None or geom.is_empty:
                continue
            cent = geom.centroid
            if not np.isfinite(cent.x) or not np.isfinite(cent.y):
                continue
            tile_recs.append(
                {
                    "class_id": r["class_id"],
                    "lon": float(cent.x),
                    "lat": float(cent.y),
                    "geom_wkb": shapely.to_wkb(geom),
                    "year": r["year"],
                    "source_id": f"{layer}/{r['fid']}",
                }
            )
        io.check_disk()
    print(f"read {len(tile_recs)} geometries")

    for i, r in enumerate(tile_recs):
        r["sample_id"] = f"{i:06d}"

    # ---- Write tiles in parallel -------------------------------------------------
    results: Counter = Counter()
    written_by_class: Counter = Counter()
    id_to_rec = {r["sample_id"]: r for r in tile_recs}
    with multiprocessing.Pool(args.workers) as p:
        for sample_id, res in tqdm.tqdm(
            star_imap_unordered(p, _write_tile, [dict(rec=r) for r in tile_recs]),
            total=len(tile_recs),
            desc="write",
        ):
            results[res] += 1
            if res in ("ok", "skip"):
                written_by_class[id_to_rec[sample_id]["class_id"]] += 1
    print("write results:", dict(results))

    io.check_disk()

    # ---- Metadata ----------------------------------------------------------------
    classes = [
        {"id": cid, "name": id_to_name[cid], "description": None}
        for cid in range(n_classes)
    ]
    class_counts = {
        id_to_name[cid]: int(written_by_class.get(cid, 0)) for cid in range(n_classes)
    }
    num_written = int(results.get("ok", 0) + results.get("skip", 0))
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "WRI / Global Forest Watch (GFW S3)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://www.wri.org/research/spatial-database-planted-trees-sdpt-version-2",
                "download_url": S3_URL,
                "have_locally": False,
                "annotation_method": (
                    "compiled from national governments / NGOs / researchers; mostly "
                    "supervised classification or manual polygon delineation of Landsat / "
                    "SPOT / RapidEye imagery"
                ),
                "gdb_version": "v11282023",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": num_written,
            "class_counts": class_counts,
            "class_field": CLASS_FIELD,
            "dropped_class_count": len(dropped),
            "excluded_sentinel_classes": sorted(EXCLUDE_CLASSES),
            "notes": (
                "Planted-tree / plantation polygons from SDPT v2 (26.6M polygons across "
                "116 country layers). Class = sciName (scientific taxon); kept top "
                f"{len(kept)} of {len(ranked)} usable taxa by global frequency (dropped "
                f"{len(dropped)}); ids 0..N-1 in descending frequency. Sentinel value "
                "'Unknown' (species unidentified, ~65% of polygons) excluded from the "
                "class set. Each polygon rasterized into a <=64x64 UTM 10 m tile: class id "
                "inside the polygon, 255 (nodata/ignore) outside (no background class -- "
                "unlabeled land is ignore). Tiles-per-class balanced with the 25k cap "
                f"(eff per-class = {eff_per_class}). Bounded sampling of a large global "
                "product: only enough polygons per class were read to reach the target. "
                "Time range = 1-year window on a representative Sentinel-era year parsed "
                "from imageryYear, clamped to [2016, 2020]."
            ),
        },
    )

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=num_written
    )
    print(f"done: {num_written} samples across {n_classes} classes")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
