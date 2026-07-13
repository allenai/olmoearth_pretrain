"""Smithsonian Global Volcanism Program (GVP) -> open-set-segmentation detection tiles.

Source: Global Volcanism Program, 2024. Volcanoes of the World (VOTW) database, Smithsonian
Institution (https://volcano.si.edu/). Distributed as OGC WFS from the GVP GeoServer
(https://webservices.volcano.si.edu/geoserver/GVP-VOTW/wfs). License: free research use
(attribution to GVP). Two point layers are pulled:
  - Smithsonian_VOTW_Holocene_Volcanoes    (1,196 well-preserved Holocene edifices)
  - Smithsonian_VOTW_Pleistocene_Volcanoes (1,451 older Pleistocene edifices)
Each feature is ONE POINT at the volcano's summit location with a Primary_Volcano_Type
attribute (stratovolcano, shield, caldera, ...), plus name/number/country/elevation.

Task type / encoding (spec section 4, "points -- object detection, positive-only"):
A GVP record is a summit census point, NOT a delineated edifice polygon. A volcano edifice
IS a large landform discernible at 10-30 m, so we treat the summit point as a PRESENCE
detection (exactly like dams-as-points / mines-as-points). We keep the Primary_Volcano_Type
as a MULTI-CLASS detection label ("Adds volcano-type classes" per the manifest): each summit
becomes a 1 px positive carrying its volcano-type class id, ringed by a 10 px nodata (255)
buffer (absorbs summit-point imprecision and the fact that a summit point is a weak proxy for
the whole edifice), with background (0) filling the rest of a 32x32 (320 m) context tile.
Background-only NEGATIVE tiles away from any volcano are also emitted (spec section 4).

Observability / judgment calls (recorded in the summary):
  - ACCEPT on observability: volcano edifices are large landforms clearly visible at 10-30 m.
    Caveat: the summit POINT is a weak label for the whole edifice, and Primary_Volcano_Type
    is a property of the full edifice morphology that a 320 m summit tile only partly reveals
    -- so type labels are best-effort. Handled as detection presence + best-effort type.
  - NO dated-eruption CHANGE label. GVP eruption dates are year-resolved at best (and often
    historical / BCE); per spec section 5 a change label is only allowed when the event date
    is known to ~1-2 months. Eruptions are therefore NOT encoded as change events; volcanoes
    are treated as persistent landforms with static 1-year Sentinel-era windows.
  - Pleistocene included alongside Holocene (manifest says "Holocene/Pleistocene"). "Unknown"/
    "None" volcano types are dropped (not a real class).

Class scheme (id 0 = background; 255 = nodata/ignore buffer rings; types 1..19 by frequency):
  0 background, then volcano types ordered by descending global frequency.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.smithsonian_global_volcanism_program
"""

import argparse
import json
import multiprocessing
import random
import re
from collections import Counter
from typing import Any

import numpy as np
import tqdm
from rslearn.utils.mp import star_imap_unordered
from scipy.spatial import cKDTree

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    balance_by_class,
    encode_detection_tile,
)

SLUG = "smithsonian_global_volcanism_program"
NAME = "Smithsonian Global Volcanism Program"

WFS_BASE = (
    "https://webservices.volcano.si.edu/geoserver/GVP-VOTW/wfs"
    "?service=WFS&version=2.0.0&request=GetFeature&outputFormat=application/json&count=10000"
)
LAYERS = {
    "holocene": "GVP-VOTW:Smithsonian_VOTW_Holocene_Volcanoes",
    "pleistocene": "GVP-VOTW:Smithsonian_VOTW_Pleistocene_Volcanoes",
}
HOMEPAGE = "https://volcano.si.edu/"

CID_BACKGROUND = 0

# Sampling / encoding parameters.
PER_CLASS = 1000  # spec section 5: up to 1000 positives per volcano-type class
N_NEGATIVES = 700  # background-only tiles (spec section 4)
YEARS = list(range(2016, 2025))

DET_TILE = 32
DET_POS_SIZE = 1
DET_BUFFER = 10

NEIGHBOR_RADIUS_M = 400.0  # 3857 prefilter radius for in-tile neighbor volcanoes
NEG_MIN_DIST_M = 2000.0  # min distance a negative tile center keeps from any volcano
NEG_OFFSET_MIN_M = 5000.0
NEG_OFFSET_MAX_M = 40000.0

_TO_3857 = None
_TO_4326 = None


def _to_3857(lon: float, lat: float) -> tuple[float, float]:
    global _TO_3857
    if _TO_3857 is None:
        from pyproj import Transformer

        _TO_3857 = Transformer.from_crs(4326, 3857, always_xy=True)
    return _TO_3857.transform(lon, lat)


def _to_4326(x: float, y: float) -> tuple[float, float]:
    global _TO_4326
    if _TO_4326 is None:
        from pyproj import Transformer

        _TO_4326 = Transformer.from_crs(3857, 4326, always_xy=True)
    return _TO_4326.transform(x, y)


def normalize_type(t: str | None) -> str | None:
    """Canonical Primary_Volcano_Type: strip plural/parenthetical suffixes and '?'.

    GVP records the same type with plural markers ("Shield" vs "Shield(s)"),
    parentheticals ("Shield(pyroclastic)"), and uncertainty ("Stratovolcano?"). Collapse
    them to one canonical label. "Unknown"/"None"/empty are not real types -> None (dropped).
    """
    if t is None:
        return None
    t = re.sub(r"\([^)]*\)", "", t)  # strip any parenthetical group
    t = t.replace("?", "").strip()
    if t.lower() in ("unknown", "none", ""):
        return None
    return t


def raw_paths() -> dict[str, io.UPath]:
    return {ep: io.raw_dir(SLUG) / f"gvp_{ep}.geojson" for ep in LAYERS}


def ensure_downloaded() -> dict[str, io.UPath]:
    """Fetch the two GVP WFS point layers as GeoJSON into raw_dir (atomic, skip-existing)."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    paths = raw_paths()
    for ep, typename in LAYERS.items():
        url = f"{WFS_BASE}&typeName={typename}"
        download.download_http(url, paths[ep])
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Smithsonian Global Volcanism Program -- Volcanoes of the World (VOTW).\n"
            f"Homepage: {HOMEPAGE}\n"
            "Accessed via OGC WFS: "
            "https://webservices.volcano.si.edu/geoserver/GVP-VOTW/wfs\n"
            "Layers: Smithsonian_VOTW_Holocene_Volcanoes, "
            "Smithsonian_VOTW_Pleistocene_Volcanoes.\n"
            "License: free research use (cite GVP).\n"
        )
    return paths


def read_volcanoes() -> list[dict[str, Any]]:
    """Read both GVP layers into records with lon/lat, normalized type, provenance."""
    recs: list[dict[str, Any]] = []
    for ep, path in raw_paths().items():
        with path.open() as f:
            data = json.load(f)
        for feat in data["features"]:
            props = feat["properties"]
            vtype = normalize_type(props.get("Primary_Volcano_Type"))
            if vtype is None:
                continue
            geom = feat.get("geometry")
            if geom is None:
                lon, lat = props.get("Longitude"), props.get("Latitude")
            else:
                lon, lat = geom["coordinates"][:2]
            if lon is None or lat is None:
                continue
            vnum = props.get("Volcano_Number")
            recs.append(
                {
                    "lon": float(lon),
                    "lat": float(lat),
                    "vtype": vtype,
                    "epoch": ep,
                    "source_id": (
                        f"{ep}/VNUM/{int(vnum)}"
                        if vnum is not None
                        else f"{ep}/name/{props.get('Volcano_Name')}"
                    ),
                }
            )
    return recs


# --------------------------------------------------------------------------------------
# Writers (worker processes).
# --------------------------------------------------------------------------------------
def _write_positive(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    _, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"], proj)
    bounds = io.centered_bounds(col, row, DET_TILE, DET_TILE)
    x_min, y_min, _, _ = bounds
    positives: list[tuple[int, int, int]] = []
    cands = [(rec["lon"], rec["lat"], rec["cid"])] + rec.get("neighbors", [])
    for lon, lat, cid in cands:
        _, c, r = io.lonlat_to_utm_pixel(lon, lat, proj)
        lc, lr = c - x_min, r - y_min
        if 0 <= lc < DET_TILE and 0 <= lr < DET_TILE:
            positives.append((lr, lc, cid))
    arr = encode_detection_tile(
        positives,
        tile_size=DET_TILE,
        positive_size=DET_POS_SIZE,
        buffer_size=DET_BUFFER,
        nodata=io.CLASS_NODATA,
        background=CID_BACKGROUND,
    )[np.newaxis]
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        source_id=rec["source_id"],
        classes_present=sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA}),
    )
    return "positive"


def _write_negative(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    _, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"], proj)
    bounds = io.centered_bounds(col, row, DET_TILE, DET_TILE)
    arr = encode_detection_tile(
        [],
        tile_size=DET_TILE,
        positive_size=DET_POS_SIZE,
        buffer_size=DET_BUFFER,
        nodata=io.CLASS_NODATA,
        background=CID_BACKGROUND,
    )[np.newaxis]
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        source_id=rec["source_id"],
        classes_present=[CID_BACKGROUND],
    )
    return "negative"


def _dispatch(rec: dict[str, Any]) -> str:
    if rec["kind"] == "negative":
        return _write_negative(rec)
    return _write_positive(rec)


def make_negatives(
    tree: cKDTree, volcs: list[dict[str, Any]], n: int, seed: int = 7
) -> list[dict[str, Any]]:
    """Background-only tile centers offset from volcanoes, guaranteed volcano-free."""
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    attempts = 0
    while len(out) < n and attempts < n * 200:
        attempts += 1
        base = volcs[rng.randrange(len(volcs))]
        ang = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(NEG_OFFSET_MIN_M, NEG_OFFSET_MAX_M)
        bx, by = _to_3857(base["lon"], base["lat"])
        x, y = bx + dist * np.cos(ang), by + dist * np.sin(ang)
        if tree.query_ball_point([x, y], r=NEG_MIN_DIST_M):
            continue
        lon, lat = _to_4326(x, y)
        if not (-60 <= lat <= 78):
            continue
        out.append(
            {
                "kind": "negative",
                "lon": float(lon),
                "lat": float(lat),
                "source_id": f"negative/{len(out)}",
            }
        )
    return out


def build_classes(volcs: list[dict[str, Any]]) -> tuple[list[dict], dict[str, int]]:
    """Assign class ids 1..N to volcano types by descending global frequency (0=background)."""
    freq = Counter(v["vtype"] for v in volcs)
    ordered = [t for t, _ in freq.most_common()]
    type_to_cid = {t: i + 1 for i, t in enumerate(ordered)}
    classes = [
        {
            "id": 0,
            "name": "background",
            "description": "Negative / non-volcano land: pixels away from any GVP summit point.",
        }
    ]
    for t in ordered:
        classes.append(
            {
                "id": type_to_cid[t],
                "name": t,
                "description": f"GVP Primary_Volcano_Type '{t}' (summit-point presence).",
            }
        )
    return classes, type_to_cid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    ensure_downloaded()
    print("reading volcano points ...", flush=True)
    volcs = read_volcanoes()
    print(
        f"  {len(volcs)} volcano points (after dropping Unknown/None types)", flush=True
    )

    io.check_disk()

    classes, type_to_cid = build_classes(volcs)
    for v in volcs:
        v["cid"] = type_to_cid[v["vtype"]]

    # Global KDTree over ALL volcanoes (EPSG:3857) for negatives + in-tile neighbor marking.
    volcs_xy = np.array([_to_3857(v["lon"], v["lat"]) for v in volcs], dtype=float)
    tree = cKDTree(volcs_xy)

    # Select positive tile centers, balanced/capped per volcano-type class (spec section 5).
    selected = balance_by_class(volcs, key="vtype", per_class=PER_CLASS)

    # Mark neighboring volcanoes that fall inside each positive tile (carry their own type).
    for r in selected:
        x, y = _to_3857(r["lon"], r["lat"])
        near = tree.query_ball_point([x, y], r=NEIGHBOR_RADIUS_M)
        r["neighbors"] = [
            (volcs[i]["lon"], volcs[i]["lat"], volcs[i]["cid"])
            for i in near
            if volcs[i]["source_id"] != r["source_id"]
        ][:50]
        r["kind"] = "positive"

    negatives = make_negatives(tree, volcs, N_NEGATIVES)
    print(
        f"selected {len(selected)} positive tiles + {len(negatives)} negatives",
        flush=True,
    )

    yrng = random.Random(123)
    all_recs = selected + negatives
    for r in all_recs:
        r["year"] = YEARS[yrng.randrange(len(YEARS))]
    for i, r in enumerate(all_recs):
        r["sample_id"] = f"{i:06d}"

    results: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _dispatch, [dict(rec=r) for r in all_recs]),
            total=len(all_recs),
        ):
            results[res] += 1
    print("write results:", dict(results), flush=True)

    io.check_disk()

    # Class counts among selected positive tile centers (a tile also counts its neighbors,
    # but this is the per-center distribution used for balancing).
    center_counts = Counter(r["vtype"] for r in selected)
    class_counts = {t: center_counts.get(t, 0) for t in type_to_cid}
    class_counts["background_negative_tiles"] = len(negatives)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Smithsonian Global Volcanism Program (Volcanoes of the World, VOTW)",
            "license": "free research use (cite GVP)",
            "provenance": {
                "url": HOMEPAGE,
                "wfs": "https://webservices.volcano.si.edu/geoserver/GVP-VOTW/wfs",
                "have_locally": False,
                "annotation_method": "manual (expert-curated volcano census)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes,
            "nodata_value": io.CLASS_NODATA,
            "detection_encoding": {
                "applies_to": "volcano summit points (multi-class by Primary_Volcano_Type)",
                "tile_size": DET_TILE,
                "positive_size": DET_POS_SIZE,
                "buffer_size": DET_BUFFER,
            },
            "num_samples": len(all_recs),
            "class_counts": class_counts,
            "notes": (
                "GVP summit-point census -> positive-only volcano-type object detection. "
                "1 px positive carrying the volcano-type class id at each summit + 10 px "
                "nodata buffer ring, background fill in a 32x32 (320 m) context tile; other "
                "GVP volcanoes inside a tile are marked with their own type. Holocene (1,196) "
                "+ Pleistocene (1,451) layers; 'Unknown'/'None' types dropped (82). 19 volcano "
                "types kept, capped at 1000/class (only Stratovolcano, 1,216, is truncated). "
                f"{N_NEGATIVES} background-only negatives emitted >=2 km from any volcano. "
                "NO dated-eruption change label: GVP eruption dates are year-resolved at best "
                "(spec section 5 requires ~1-2 month precision), so volcanoes are treated as "
                "persistent landforms with static 1-year Sentinel-era windows (2016-2024). "
                "Caveat: a summit point is a weak proxy for the whole edifice and volcano type "
                "is a full-edifice morphological property only partly visible in a 320 m tile."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(all_recs)
    )
    print("done:", len(all_recs), "samples", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
