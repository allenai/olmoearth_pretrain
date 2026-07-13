"""Process DeepOWT (Global Offshore Wind Turbines) into detection label tiles.

Source: DeepOWT, Zhang et al., Earth Syst. Sci. Data (ESSD), on Zenodo
(https://doi.org/10.5281/zenodo.5933967, CC-BY-4.0). A global inventory of offshore
wind-energy infrastructure with per-quarter deployment status derived from Sentinel-1
time series with deep learning + validation. We use the main file ``DeepOWT.geojson``:
9,941 Point features, each with 20 quarterly status columns Y2016Q3 ... Y2021Q2, each
valued with DeepOWT's semantic class:
    0 = open sea, 1 = under construction, 2 = offshore wind turbine, 3 = substation.

Task type: positive-only object DETECTION, encoded as per-pixel classes (spec section 4).
Offshore turbines/substations are small (~monopile + rotor / platform) but resolvable at
10 m against open water (DeepOWT itself is derived at S-1 10 m). Class scheme keeps
DeepOWT's native ids so background = open sea:
    0 = background (open sea), 1 = under_construction, 2 = offshore_turbine, 3 = substation
    255 = nodata / ignore (detection buffer rings; also ambiguous in-tile neighbors).

Time / change handling (spec section 5). DeepOWT resolves the appearance/state of each
structure only to a QUARTER (~3 months) -- coarser than the section-5 change-timing
requirement (~1-2 months) -- so we do NOT emit dated change labels. Instead each structure
is treated as a PERSISTENT structure: a positive for class c is emitted for a point only in
a full calendar year (2017-2020) in which ALL FOUR quarters equal c, guaranteeing the state
is genuinely persistent across the whole 1-year label window. change_time is null and the
time range is that calendar year (io.year_range). This mirrors GRW / the vessel precedent's
detection encoding but with a persistent-state (not change) time model.

Encoding: one 32x32 (DET_TILE) UTM 10 m context tile per selected structure, centered on
its point; a 1 px positive (class id) ringed by a 10 px nodata buffer, rest background.
Every OTHER DeepOWT point inside the tile is also encoded by its status in the same year:
1/2/3 -> that positive class; 0 (open sea) -> left as background; transitioning that year
(not all-4-equal) -> a nodata (255) marker so an ambiguous neighbor is ignored, never a
false label. Background NEGATIVE tiles are drawn from points that are open sea (status 0)
for a full calendar year -- real, geolocated open-water sites (many are future turbine
locations), with any in-tile structures still encoded correctly.

Sampling: up to 1000 tiles per positive class, stratified across calendar years for
temporal diversity, plus up to 1000 background negatives. Under-construction and substation
are sparse (kept in full; noted in the summary) -- spec section 5 says do not drop rare
classes; downstream assembly filters too-small ones.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.deepowt_global_offshore_wind_turbines
"""

import argparse
import json
import multiprocessing
import random
from collections import Counter
from typing import Any

import numpy as np
import tqdm
from rslearn.utils.mp import star_imap_unordered
from scipy.spatial import cKDTree

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    balance_by_class,
    encode_detection_tile,
)

SLUG = "deepowt_global_offshore_wind_turbines"
NAME = "DeepOWT (Global Offshore Wind Turbines)"
ZENODO = "https://doi.org/10.5281/zenodo.5933967"
GEOJSON_URL = "https://zenodo.org/records/5933967/files/DeepOWT.geojson?download=1"
GEOJSON_FILE = "DeepOWT.geojson"

# Class scheme = DeepOWT's native status ids (so background = open sea = 0).
CID_BACKGROUND = 0
CID_UNDER_CONSTRUCTION = 1
CID_TURBINE = 2
CID_SUBSTATION = 3
POSITIVE_CIDS = (CID_UNDER_CONSTRUCTION, CID_TURBINE, CID_SUBSTATION)
CLASSES = [
    {
        "id": CID_BACKGROUND,
        "name": "background",
        "description": "Open sea / open-water ocean surface with no offshore wind "
        "infrastructure (DeepOWT status 0).",
    },
    {
        "id": CID_UNDER_CONSTRUCTION,
        "name": "under_construction",
        "description": "Offshore wind site under construction -- foundation/platform "
        "present but turbine not yet operational (DeepOWT status 1). Transient state; only "
        "sites under construction for a full calendar year are emitted (persistent window).",
    },
    {
        "id": CID_TURBINE,
        "name": "offshore_turbine",
        "description": "Installed offshore wind turbine (DeepOWT status 2), detected from "
        "Sentinel-1 time series with deep learning + validation.",
    },
    {
        "id": CID_SUBSTATION,
        "name": "substation",
        "description": "Offshore wind farm substation / transformer platform "
        "(DeepOWT status 3).",
    },
]

# Full calendar years with all four quarters present in the record (guarantees a
# persistent-across-the-window state). 2016 (Q3-Q4 only) and 2021 (Q1-Q2 only) are partial
# and excluded from the all-4-quarters rule.
YEARS = [2017, 2018, 2019, 2020]
PER_CLASS = 1000
PER_YEAR = PER_CLASS // len(YEARS)  # 250 -> up to 1000 per class
N_NEGATIVES = 1000
NEG_PER_YEAR = N_NEGATIVES // len(YEARS)
SEED = 42

# Detection encoding parameters (spec section 4). Turbines ~1 px at 10 m.
DET_TILE = 32
DET_POS_SIZE = 1
DET_BUFFER = 10

# Neighbor search radius in degrees (generous; precise filter is by tile pixel bounds).
# 32 px * 10 m = 320 m tile; ~0.01 deg (~1.1 km) safely covers the tile at all latitudes.
NEIGHBOR_RADIUS_DEG = 0.01


def _quarter_cols() -> list[str]:
    return [
        f"Y{y}Q{q}"
        for y in range(2016, 2022)
        for q in range(1, 5)
        if not (y == 2016 and q < 3) and not (y == 2021 and q > 2)
    ]


def _stable_status(props: dict[str, Any], year: int) -> int | None:
    """Status for a point in a calendar year if all four quarters agree, else None."""
    vals = {props[f"Y{year}Q{q}"] for q in range(1, 5)}
    return next(iter(vals)) if len(vals) == 1 else None


def _load_points() -> list[dict[str, Any]]:
    path = io.raw_dir(SLUG) / GEOJSON_FILE
    with path.open() as f:
        data = json.load(f)
    pts: list[dict[str, Any]] = []
    for i, feat in enumerate(data["features"]):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        lon, lat = geom["coordinates"][:2]
        props = feat["properties"]
        stable = {y: _stable_status(props, y) for y in YEARS}
        pts.append({"idx": i, "lon": float(lon), "lat": float(lat), "stable": stable})
    return pts


def _build_candidates(
    pts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (positive_candidates, negative_candidates).

    One candidate per (point, class) using a randomly chosen stable year for that class, so
    each physical point contributes at most one tile per class while spreading across years.
    """
    rng = random.Random(SEED)
    pos: list[dict[str, Any]] = []
    neg: list[dict[str, Any]] = []
    for p in pts:
        stable = p["stable"]
        for cid in POSITIVE_CIDS:
            yrs = [y for y, v in stable.items() if v == cid]
            if yrs:
                pos.append(
                    {
                        "kind": "pos",
                        "class": cid,
                        "year": rng.choice(yrs),
                        "lon": p["lon"],
                        "lat": p["lat"],
                        "source_id": f"deepowt/{p['idx']}",
                    }
                )
        neg_yrs = [y for y, v in stable.items() if v == CID_BACKGROUND]
        if neg_yrs:
            neg.append(
                {
                    "kind": "neg",
                    "class": CID_BACKGROUND,
                    "year": rng.choice(neg_yrs),
                    "lon": p["lon"],
                    "lat": p["lat"],
                    "source_id": f"deepowt/{p['idx']}",
                }
            )
    return pos, neg


def _resolve_neighbors(
    rec: dict[str, Any], pts: list[dict[str, Any]], tree: cKDTree
) -> None:
    """Attach in-tile neighbor points (lon, lat, cid_in_year) to rec['neighbors'].

    cid_in_year: 1/2/3 for a neighbor stable at that positive status; 255 for a neighbor
    transitioning that year (ambiguous -> ignore); status-0 neighbors are dropped (they are
    background). The center point itself is excluded (it is the primary positive/negative).
    """
    year = rec["year"]
    idxs = tree.query_ball_point([rec["lon"], rec["lat"]], r=NEIGHBOR_RADIUS_DEG)
    out: list[tuple[float, float, int]] = []
    for j in idxs:
        q = pts[j]
        if q["lon"] == rec["lon"] and q["lat"] == rec["lat"]:
            continue
        v = q["stable"][year]
        if v == CID_BACKGROUND:
            continue
        cid = v if v in POSITIVE_CIDS else io.CLASS_NODATA
        out.append((q["lon"], q["lat"], cid))
    rec["neighbors"] = out


def _write_tile(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    _, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"], proj)
    bounds = io.centered_bounds(col, row, DET_TILE, DET_TILE)
    x_min, y_min = bounds[0], bounds[1]

    positives: list[tuple[int, int, int]] = []
    if rec["kind"] == "pos":
        positives.append((row - y_min, col - x_min, rec["class"]))
    for lon, lat, cid in rec.get("neighbors", []):
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
    )
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        change_time=None,
        source_id=rec["source_id"],
        classes_present=sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA}),
    )
    return "pos" if rec["kind"] == "pos" else "neg"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    if not (raw / GEOJSON_FILE).exists():
        from olmoearth_pretrain.open_set_segmentation_data.download import download_http

        print("downloading DeepOWT.geojson ...", flush=True)
        download_http(GEOJSON_URL, raw / GEOJSON_FILE)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "DeepOWT (Global Offshore Wind Turbines), Zhang et al., ESSD.\n"
            f"{ZENODO}\n{GEOJSON_URL}\n"
            "Main file DeepOWT.geojson: 9941 Point features, 20 quarterly status columns "
            "Y2016Q3..Y2021Q2 (0=open sea,1=under construction,2=turbine,3=substation). "
            "License CC-BY-4.0.\n"
        )

    pts = _load_points()
    print(f"loaded {len(pts)} points", flush=True)
    tree = cKDTree(np.array([[p["lon"], p["lat"]] for p in pts], dtype=float))

    pos_cands, neg_cands = _build_candidates(pts)
    pos_by_class: dict[int, list[dict[str, Any]]] = {c: [] for c in POSITIVE_CIDS}
    for r in pos_cands:
        pos_by_class[r["class"]].append(r)
    print(
        "positive candidates: "
        + ", ".join(f"{c}={len(v)}" for c, v in pos_by_class.items())
        + f"; negative candidates={len(neg_cands)}",
        flush=True,
    )

    selected: list[dict[str, Any]] = []
    for c in POSITIVE_CIDS:
        sel = balance_by_class(pos_by_class[c], "year", per_class=PER_YEAR, seed=SEED)[
            :PER_CLASS
        ]
        selected.extend(sel)
        print(f"  class {c}: selected {len(sel)}", flush=True)
    neg_sel = balance_by_class(neg_cands, "year", per_class=NEG_PER_YEAR, seed=SEED)[
        :N_NEGATIVES
    ]
    selected.extend(neg_sel)
    print(f"  negatives: selected {len(neg_sel)}", flush=True)

    for r in selected:
        _resolve_neighbors(r, pts, tree)
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"

    io.check_disk()

    results: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_tile, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            results[res] += 1
    print("write results:", dict(results), flush=True)

    io.check_disk()

    class_counts = {
        "under_construction": sum(
            1 for r in selected if r.get("class") == CID_UNDER_CONSTRUCTION
        ),
        "offshore_turbine": sum(1 for r in selected if r.get("class") == CID_TURBINE),
        "substation": sum(1 for r in selected if r.get("class") == CID_SUBSTATION),
        "background_negative_tiles": len(neg_sel),
    }
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",  # detection encoded as per-pixel classes
            "source": "Zenodo / ESSD",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": ZENODO,
                "have_locally": False,
                "annotation_method": "derived-product (deep learning, Sentinel-1) + validation",
                "file": GEOJSON_FILE,
            },
            "sensors_relevant": ["sentinel1", "sentinel2", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "detection_encoding": {
                "tile_size": DET_TILE,
                "positive_size": DET_POS_SIZE,
                "buffer_size": DET_BUFFER,
            },
            "num_samples": len(selected),
            "class_counts": class_counts,
            "notes": (
                "Offshore wind infrastructure DETECTION. DeepOWT status ids kept as the "
                "class scheme (0=background/open sea, 1=under_construction, 2=offshore_turbine, "
                "3=substation; 255=nodata). Detection encoding: 32x32 UTM 10 m context tile "
                "per structure, 1 px positive + 10 px nodata buffer (21x21 ignore), rest "
                "background; other in-tile DeepOWT points encoded by their same-year status "
                "(1/2/3 positive, 0 background, transitioning->255 ignore). Persistent-structure "
                "time model: a positive is emitted only for a full calendar year (2017-2020) in "
                "which ALL FOUR quarters equal that class, so the state is persistent across the "
                "1-year window; change_time=null (DeepOWT timing is quarterly, coarser than the "
                "~1-2 month change-label requirement, so NOT encoded as dated change). Negatives: "
                "background tiles centered on points that are open sea for a full year (real "
                "geolocated open-water sites). up to 1000 tiles/positive-class stratified across "
                "years + up to 1000 negatives. under_construction and substation are sparse "
                "(kept in full per spec section 5; downstream assembly filters too-small classes)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print(f"done: {len(selected)} samples", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
