"""Process SentinelKilnDB into open-set-segmentation detection tiles.

Source: Hugging Face dataset ``SustainabilityLabIITGN/SentinelKilnDB`` (NeurIPS 2025
Datasets & Benchmarks). 62,671 hand-validated brick kilns across the Indo-Gangetic Plain
(India, Pakistan, Bangladesh, Afghanistan), annotated as **oriented bounding boxes (OBBs)**
on free Sentinel-2 surface-reflectance imagery. Three kiln types: FCBK (Fixed Chimney
Bull's Trench Kiln), CFCBK (Circular FCBK), Zigzag. License: CC-BY-NC-4.0 (non-commercial
research use; recorded in metadata).

On-disk form: three parquet files (train/val/test). Each row is one 128x128 px @ 10 m
Sentinel-2 patch: ``image_name`` = ``"{lat}_{lon}.png"`` (the patch CENTER lon/lat -- see
GEOREF note), ``image`` = PNG bytes (NOT used here -- we only need labels + georef), and
label lists ``dota_label`` / ``yolo_obb_label`` / ``yolo_aa_label``. We read only the
``image_name`` + ``dota_label`` columns. Each DOTA label string is
``"x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult"`` with the 8 corner coords in the patch's
128 px PIXEL space. Patches tile a lat/lon grid with a 30 px overlap (grid step 128-30 =
98 px ~= 0.0088 deg lat).

GEOREF (spec section 8.2 check): the filename lat/lon is treated as the patch CENTER. Each
patch is sampled north-up at 10 m/pixel, so image->UTM is a pure translation: we take the
patch's local UTM projection from its (lon, lat), find the UTM pixel of the center, and map
image pixel (px, py) -> UTM pixel (center_col - 64 + px, center_row - 64 + py) (image rows
run southward, matching the negative-y UTM pixel grid). The center-vs-corner convention is
verified against Sentinel-2 in the summary; ANCHOR_OFFSET below encodes it (-64 = center).

Encoding (label_type = oriented boxes -> detection, spec section 4): one 64x64 UTM 10 m
context tile centered on each (deduplicated) kiln. Each kiln's OBB footprint is rasterized
as its class id (all_touched), ringed by a BUFFER px nodata (255) band to absorb annotation
/ georef slop, with background (0) filling the rest of the tile. Any other kiln falling in
the tile (same UTM zone) is rasterized too. Kilns appearing in overlapping patches are
deduplicated by rounded UTM-pixel centroid. We also emit background-only NEGATIVE tiles from
empty patches (no kilns) so the background class has spatially-meaningful negatives (spec
section 5, detection exception).

Classes (background is a real class for detection): 0=background, 1=FCBK, 2=CFCBK, 3=Zigzag.
Selection: up to 1000 tiles per kiln class, class-balanced, prioritizing the rare CFCBK
(spec section 5), plus N_NEGATIVES background-only tiles. Time range: brick kilns are
persistent structures; imagery is Nov 2023 - Feb 2024. We use a 1-year window
[2023-11-01, 2024-11-01) (static/persistent label, change_time=null, spec section 5).

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.sentinelkilndb
"""

import argparse
import math
import multiprocessing
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import shapely
import tqdm
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.rasterize import rasterize_shapes

SLUG = "sentinelkilndb"
NAME = "SentinelKilnDB"
REPO = "SustainabilityLabIITGN/SentinelKilnDB"
SPLIT_FILES = ["train/train.parquet", "val/val.parquet", "test/test.parquet"]

PATCH = 128  # source patch size (px)
ANCHOR_OFFSET = -PATCH // 2  # filename lat/lon = patch CENTER -> top-left = center - 64
TILE = io.MAX_TILE  # 64 px @ 10 m context tile
BUFFER = 5  # nodata ring (px) around each kiln footprint

BACKGROUND_ID = 0
# Fixed ids from the manifest class order (FCBK, CFCBK, Zigzag); background prepended.
CLASS_IDS = {"FCBK": 1, "CFCBK": 2, "Zigzag": 3}
CLASS_NAMES = {0: "background", 1: "FCBK", 2: "CFCBK", 3: "Zigzag"}
CLASS_DESCRIPTIONS = {
    0: "Non-kiln background land surface within the context tile.",
    1: "Fixed Chimney Bull's Trench Kiln (FCBK): rectangular/oval trench kiln with a "
    "fixed central chimney; the most common brick-kiln type in the Indo-Gangetic Plain.",
    2: "Circular FCBK (CFCBK): a circular-plan fixed-chimney Bull's Trench kiln (rarest "
    "type in the dataset).",
    3: "Zigzag kiln: Bull's Trench kiln with a zigzag flue firing pattern (a cleaner-"
    "technology retrofit), typically rectangular with an offset chimney.",
}

# Persistent-structure label; imagery Nov 2023 - Feb 2024. 1-year static window (spec 5).
TIME_RANGE = (
    datetime(2023, 11, 1, tzinfo=UTC),
    datetime(2024, 11, 1, tzinfo=UTC),
)

PER_CLASS = 1000
N_NEGATIVES = 1500
SEED = 42
# Dedup radius (px @ 10 m) for the same physical kiln seen in overlapping patches. Integer
# rounding under-deduplicates because sub-pixel reprojection + independent per-patch
# annotation push the same kiln 1-3 px apart. Kiln footprints are ~100-150 m, so distinct
# kiln centers are never within 50 m; a 5 px (50 m) radius removes overlap dups safely.
DEDUP_TOL = 5


def _parse_patches() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read label columns from all splits. Return (positive_patches, empty_patches).

    A positive patch dict: {name, lon, lat, kilns: [(class_name, [8 floats])]}.
    An empty patch dict: {name, lon, lat}.
    """
    positives: list[dict[str, Any]] = []
    empties: list[dict[str, Any]] = []
    raw = io.raw_dir(SLUG)
    for rel in SPLIT_FILES:
        path = raw / rel
        df = pq.read_table(str(path), columns=["image_name", "dota_label"]).to_pandas()
        split = rel.split("/")[0]
        for _, row in df.iterrows():
            name = row["image_name"]
            stem = name[:-4] if name.endswith(".png") else name
            lat_s, lon_s = stem.split("_")
            lat, lon = float(lat_s), float(lon_s)
            labels = row["dota_label"]
            kilns: list[tuple[str, list[float]]] = []
            if labels is not None and len(labels) > 0:
                for lab in labels:
                    parts = lab.split()
                    if len(parts) < 9:
                        continue
                    cls = parts[8]
                    if cls not in CLASS_IDS:
                        continue
                    coords = [float(v) for v in parts[:8]]
                    kilns.append((cls, coords))
            rec = {"name": name, "lon": lon, "lat": lat, "split": split}
            if kilns:
                rec["kilns"] = kilns
                positives.append(rec)
            else:
                empties.append(rec)
    return positives, empties


def _stable_kiln_order(kilns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic order (independent of pool completion) so greedy dedup is reproducible."""
    return sorted(
        kilns,
        key=lambda k: (k["crs"], round(k["cx"]), round(k["cy"]), k["src"], k["cls"]),
    )


def _patch_kilns(patch: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute each kiln's OBB polygon in absolute UTM pixel coords for one patch."""
    proj, ccol, crow = io.lonlat_to_utm_pixel(patch["lon"], patch["lat"])
    epsg = proj.crs.to_string()
    x0 = ccol + ANCHOR_OFFSET  # UTM pixel col of patch image column 0
    y0 = crow + ANCHOR_OFFSET  # UTM pixel row of patch image row 0
    out: list[dict[str, Any]] = []
    for cls, coords in patch["kilns"]:
        xs = coords[0::2]
        ys = coords[1::2]
        pts = [(x0 + xs[i], y0 + ys[i]) for i in range(4)]
        cx = sum(p[0] for p in pts) / 4.0
        cy = sum(p[1] for p in pts) / 4.0
        out.append(
            {
                "crs": epsg,
                "cls": CLASS_IDS[cls],
                "poly": pts,
                "cx": cx,
                "cy": cy,
                "src": patch["name"],
            }
        )
    return out


def _empty_tile(patch: dict[str, Any]) -> dict[str, Any] | None:
    proj, ccol, crow = io.lonlat_to_utm_pixel(patch["lon"], patch["lat"])
    return {
        "crs": proj.crs.to_string(),
        "cx": float(ccol),
        "cy": float(crow),
        "src": patch["name"],
        "neg": True,
    }


def _projection(crs: str):
    from rasterio.crs import CRS
    from rslearn.utils.geometry import Projection

    return Projection(CRS.from_string(crs), io.RESOLUTION, -io.RESOLUTION)


def _write_tile(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        # Idempotent skip: recover realized classes from the sidecar for correct counts.
        import json as _json

        jp = io.locations_dir(SLUG) / f"{sample_id}.json"
        try:
            cp = _json.load(jp.open()).get("classes_present", [])
        except Exception:
            cp = []
        kind = "neg" if rec.get("neg") else "pos"
        return f"skip-{kind}:{','.join(str(c) for c in cp)}"
    proj = _projection(rec["crs"])
    pc = int(math.floor(rec["cx"]))
    pr = int(math.floor(rec["cy"]))
    bounds = io.centered_bounds(pc, pr, TILE, TILE)

    if rec.get("neg"):
        arr = np.zeros((1, TILE, TILE), dtype=np.uint8)
        classes_present = [BACKGROUND_ID]
    else:
        # Buffer rings first (nodata), then footprints (class id) on top so kilns win.
        shapes: list[tuple[Any, int]] = []
        polys = rec["polys"]  # list of (poly_pts, cls)
        for pts, _cls in polys:
            poly = shapely.Polygon(pts)
            shapes.append((poly.buffer(BUFFER), io.CLASS_NODATA))
        for pts, cls in polys:
            shapes.append((shapely.Polygon(pts), cls))
        arr = rasterize_shapes(
            shapes, bounds, fill=BACKGROUND_ID, dtype="uint8", all_touched=True
        )
        classes_present = sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA})

    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        TIME_RANGE,
        change_time=None,
        source_id=rec["src"],
        classes_present=classes_present,
    )
    kind = "neg" if rec.get("neg") else "pos"
    # Return the classes actually rendered (some neighbour kilns near the tile edge fall
    # outside the 64x64 bounds and are not rendered), so metadata counts reflect the tiles.
    return f"{kind}:{','.join(str(c) for c in classes_present)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    for rel in SPLIT_FILES:
        download.hf_download(REPO, rel, raw)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            f"Hugging Face dataset {REPO} (NeurIPS 2025 D&B).\n"
            "train/val/test parquet files; each row = one 128x128 @ 10 m Sentinel-2 patch, "
            "image_name='{lat}_{lon}.png' (patch center), dota_label list of OBB strings "
            "'x1 y1 x2 y2 x3 y3 x4 y4 class difficult' in 128 px pixel space.\n"
            "License: CC-BY-NC-4.0 (non-commercial).\n"
        )

    print("reading label columns from all splits...", flush=True)
    positives, empties = _parse_patches()
    print(f"patches: {len(positives)} with kilns, {len(empties)} empty", flush=True)

    io.check_disk()

    # Compute kiln geometries in parallel (per-patch UTM reprojection).
    all_kilns: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for kl in tqdm.tqdm(
            star_imap_unordered(p, _patch_kilns, [dict(patch=pt) for pt in positives]),
            total=len(positives),
            desc="kiln geom",
        ):
            all_kilns.extend(kl)
    print(
        f"total kiln annotations (with patch-overlap dups): {len(all_kilns)}",
        flush=True,
    )

    # Deduplicate kilns across overlapping patches by spatial clustering within DEDUP_TOL px
    # (a spatial hash with cell = DEDUP_TOL, checking the 3x3 neighborhood). Integer rounding
    # alone leaves the same kiln duplicated when overlapping patches place it 1-3 px apart.
    tol = DEDUP_TOL
    kept_by_cell: dict[tuple[str, int, int], list[tuple[float, float]]] = defaultdict(
        list
    )
    unique_kilns: list[dict[str, Any]] = []
    for k in _stable_kiln_order(all_kilns):
        crs, cx, cy = k["crs"], k["cx"], k["cy"]
        gx, gy = int(cx // tol), int(cy // tol)
        is_dup = False
        for dxg in (-1, 0, 1):
            for dyg in (-1, 0, 1):
                for ox, oy in kept_by_cell[(crs, gx + dxg, gy + dyg)]:
                    if abs(ox - cx) <= tol and abs(oy - cy) <= tol:
                        is_dup = True
                        break
                if is_dup:
                    break
            if is_dup:
                break
        if not is_dup:
            kept_by_cell[(crs, gx, gy)].append((cx, cy))
            unique_kilns.append(k)
    cls_counts = Counter(k["cls"] for k in unique_kilns)
    print(
        "unique kilns:",
        len(unique_kilns),
        {CLASS_NAMES[c]: cls_counts[c] for c in sorted(cls_counts)},
        flush=True,
    )

    # Spatial index by CRS so a tile can pick up neighbouring kilns.
    by_crs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for k in unique_kilns:
        by_crs[k["crs"]].append(k)

    # Class-balanced selection of kiln tiles (up to PER_CLASS per kiln class).
    from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

    selected = balance_by_class(unique_kilns, key="cls", per_class=PER_CLASS, seed=SEED)
    print(f"selected {len(selected)} positive kiln tiles", flush=True)

    # For each selected tile, attach every kiln (same crs) whose footprint may fall inside.
    half = TILE // 2
    reach = half + BUFFER + 5
    pos_records: list[dict[str, Any]] = []
    for k in selected:
        pc, pr = round(k["cx"]), round(k["cy"])
        polys: list[tuple[list[tuple[float, float]], int]] = []
        for other in by_crs[k["crs"]]:
            if abs(other["cx"] - pc) <= reach and abs(other["cy"] - pr) <= reach:
                polys.append((other["poly"], other["cls"]))
        pos_records.append(
            {
                "crs": k["crs"],
                "cx": k["cx"],
                "cy": k["cy"],
                "src": k["src"],
                "polys": polys,
            }
        )

    # Negative (background-only) tiles from empty patches.
    rng = random.Random(SEED)
    rng.shuffle(empties)
    neg_records: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for rec in star_imap_unordered(
            p, _empty_tile, [dict(patch=pt) for pt in empties[: N_NEGATIVES * 2]]
        ):
            if rec is not None:
                neg_records.append(rec)
    neg_records = neg_records[:N_NEGATIVES]

    all_records = pos_records + neg_records
    # Deterministic id assignment (stable order independent of pool completion order).
    all_records.sort(key=lambda r: (r["crs"], round(r["cx"]), round(r["cy"]), r["src"]))
    for i, r in enumerate(all_records):
        r["sample_id"] = f"{i:06d}"
    print(
        f"writing {len(pos_records)} positive + {len(neg_records)} negative "
        f"= {len(all_records)} tiles",
        flush=True,
    )

    io.check_disk()
    results: Counter = Counter()
    # Realized per-kiln-class tile counts, aggregated from what was actually rendered.
    tile_class_counts: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_tile, [dict(rec=r) for r in all_records]),
            total=len(all_records),
            desc="write",
        ):
            kind = res.split(":", 1)[0]
            results[kind] += 1
            if ":" in res and res.split(":", 1)[1]:
                for c in res.split(":", 1)[1].split(","):
                    ci = int(c)
                    if ci != BACKGROUND_ID:
                        tile_class_counts[ci] += 1
    print("write results:", dict(results), flush=True)

    io.check_disk()

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",  # detection encoded as per-pixel classes
            "source": "Hugging Face / NeurIPS (SustainabilityLabIITGN/SentinelKilnDB)",
            "license": "CC-BY-NC-4.0",
            "provenance": {
                "url": "https://huggingface.co/datasets/SustainabilityLabIITGN/SentinelKilnDB",
                "have_locally": False,
                "annotation_method": "manual / hand-validated oriented bounding boxes on "
                "Sentinel-2 imagery",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {
                    "id": cid,
                    "name": CLASS_NAMES[cid],
                    "description": CLASS_DESCRIPTIONS[cid],
                }
                for cid in sorted(CLASS_NAMES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "detection_encoding": {
                "tile_size": TILE,
                "footprint": "rasterized OBB polygon (all_touched)",
                "buffer_size": BUFFER,
                "anchor": "patch-center lon/lat; image->UTM pure translation",
            },
            "num_samples": len(all_records),
            "class_tile_counts": {
                **{
                    CLASS_NAMES[c]: tile_class_counts[c]
                    for c in sorted(tile_class_counts)
                },
                "background_negative_tiles": len(neg_records),
            },
            "notes": (
                "SentinelKilnDB brick-kiln OBB detection. label_type='polygons (oriented "
                "boxes)' -> detection encoding (spec section 4): 64x64 UTM 10 m context tile "
                "per deduplicated kiln, OBB footprint rasterized as class id (all_touched), "
                "5 px nodata (255) buffer ring, background (0) elsewhere; neighbouring kilns "
                "in the same tile are also rasterized. Georef: filename '{lat}_{lon}.png' is "
                "the patch center; image->UTM is a pure translation (patch is north-up S2 at "
                "10 m). Kilns from overlapping patches deduplicated by rounded UTM-pixel "
                "centroid. Negatives: background-only tiles from empty patches (detection "
                "exception, spec section 5). Classes 0=background,1=FCBK,2=CFCBK,3=Zigzag; "
                "up to 1000 tiles/kiln class, CFCBK (rarest) prioritized, + "
                f"{len(neg_records)} negatives. Time range = 1-year static window "
                "[2023-11-01,2024-11-01) (persistent structures; imagery Nov 2023-Feb 2024). "
                "License CC-BY-NC-4.0 (non-commercial research)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(all_records)
    )
    print(f"done: {len(all_records)} samples", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
