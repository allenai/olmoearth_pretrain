"""Process Global Fishing Watch SAR Fixed Infrastructure into detection label tiles.

Source: Global Fishing Watch / Paolo et al. 2024, Nature ("Satellite mapping reveals
extensive industrial activity at sea"), analysis-data repository on figshare
(https://doi.org/10.6084/m9.figshare.24309475, CC-BY-NC-4.0). We download only the
label file ``offshore_infrastructure_v20231106.csv.zip`` (11.4 MB) -- NO imagery; the
pretraining pipeline supplies its own S1/S2/Landsat. The CSV holds 1,441,242
detection-months of offshore fixed infrastructure from 2017-2021, detected on monthly
Sentinel-1 SAR median composites and classified with deep learning.

CSV fields (README):
  structure_id   -- unique id for all detections of the SAME physical structure (its
                    lon/lat is constant across all its detection-months, verified std=0)
  composite_date -- center date of the 6-month image composite used for detection
  lat, lon       -- structure position
  label          -- oil / probable_oil / possible_oil / lake_maracaibo (oil in Lake
                    Maracaibo, VE) ; wind / probable_wind / possible_wind ; unknown

Task type: positive-only object DETECTION, encoded as per-pixel classes (spec section 4).
Offshore platforms / wind turbines are point detections resolvable at 10 m against open
water (GFW derives them from S-1 10 m). Manifest three-class scheme (oil / wind /
other-unknown), plus background for the detection encoding:
    0 = background (open water), 1 = oil, 2 = wind, 3 = other/unknown
    255 = nodata / ignore (detection buffer rings; ambiguous in-tile neighbors).
Confidence tiers are folded into the coarse class:
    oil,probable_oil,possible_oil,lake_maracaibo -> oil(1)
    wind,probable_wind,possible_wind             -> wind(2)
    unknown                                      -> other(3)

Time / change handling (spec section 5). Fixed infrastructure is PERSISTENT, not a change
event. Detection timing is only monthly on 6-month composites (coarser than the ~1-2 month
change-timing bar), so we do NOT emit dated change labels. Instead each structure is treated
as a persistent structure: a positive is emitted for a structure only in a calendar year
(2017-2021) in which it is detected persistently across the WHOLE year -- >= 6 monthly
detections spanning both the first quarter (month <= 3) and the last quarter (month >= 10),
guaranteeing the state is genuinely present across the 1-year label window. Within a
structure-year the coarse label is 100% consistent (verified), so there is no per-year
label ambiguity. change_time is null and the time range is that calendar year
(io.year_range). This mirrors the DeepOWT persistent-structure precedent.

Encoding: one 32x32 (DET_TILE) UTM 10 m context tile per selected structure, centered on
its point; a 1 px positive of its class ringed by a 10 px nodata buffer, rest background.
Every OTHER structure detected in the same year that falls inside the tile is also encoded
by its coarse class (structures cluster in oil fields / wind farms). Background NEGATIVE
tiles are geolocated open-water sites obtained by offsetting a random real structure by
3-8 km in a random bearing and confirming (KD-tree) no structure lies within ~1.1 km, so
they are real offshore open water in the same regions as the positives.

Sampling: up to 1000 tiles per positive class, stratified across the 5 years for temporal
diversity, plus up to 1000 background negatives. "other/unknown" is comparatively sparse
(kept in full; spec section 5 -- do not drop rare classes; downstream assembly filters
too-small ones).

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_fishing_watch_sar_fixed_infrastructure
"""

import argparse
import math
import multiprocessing
import random
import zipfile
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
import tqdm
from rslearn.utils.mp import star_imap_unordered
from scipy.spatial import cKDTree

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.download import download_http
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    balance_by_class,
    encode_detection_tile,
)

SLUG = "global_fishing_watch_sar_fixed_infrastructure"
NAME = "Global Fishing Watch SAR Fixed Infrastructure"
FIGSHARE_DOI = "https://doi.org/10.6084/m9.figshare.24309475"
CSV_ZIP_URL = "https://ndownloader.figshare.com/files/43801560"
CSV_ZIP_FILE = "offshore_infrastructure_v20231106.csv.zip"
CSV_FILE = "offshore_infrastructure_v20231106.csv"

CID_BACKGROUND = 0
CID_OIL = 1
CID_WIND = 2
CID_OTHER = 3
POSITIVE_CIDS = (CID_OIL, CID_WIND, CID_OTHER)

# Map GFW confidence-tiered labels -> coarse manifest class id.
LABEL_TO_CID: dict[str, int] = {
    "oil": CID_OIL,
    "probable_oil": CID_OIL,
    "possible_oil": CID_OIL,
    "lake_maracaibo": CID_OIL,
    "wind": CID_WIND,
    "probable_wind": CID_WIND,
    "possible_wind": CID_WIND,
    "unknown": CID_OTHER,
}

CLASSES = [
    {
        "id": CID_BACKGROUND,
        "name": "background",
        "description": "Open water / ocean surface with no fixed offshore infrastructure.",
    },
    {
        "id": CID_OIL,
        "name": "oil",
        "description": "Fixed offshore oil/gas infrastructure (platforms, wellheads, "
        "related structures) detected on Sentinel-1 SAR and classified by deep learning. "
        "Includes GFW oil / probable_oil / possible_oil confidence tiers and "
        "lake_maracaibo (oil structures in Lake Maracaibo, Venezuela).",
    },
    {
        "id": CID_WIND,
        "name": "wind",
        "description": "Fixed offshore wind infrastructure (turbines, substations) "
        "detected on Sentinel-1 SAR and classified by deep learning. Includes GFW wind / "
        "probable_wind / possible_wind confidence tiers.",
    },
    {
        "id": CID_OTHER,
        "name": "other",
        "description": "Other/unknown human-made fixed offshore structure (GFW 'unknown' "
        "label): piers, bridges, power lines, aquaculture, and other man-made objects not "
        "classified as oil or wind.",
    },
]

YEARS = [2017, 2018, 2019, 2020, 2021]
PER_CLASS = 1000
PER_YEAR = PER_CLASS // len(YEARS)  # 200 -> up to 1000 per class
N_NEGATIVES = 1000
NEG_PER_YEAR = N_NEGATIVES // len(YEARS)
SEED = 42

# Persistence rule: a structure counts for a calendar year if it has >= this many monthly
# detections spanning both the first quarter and the last quarter of the year.
PERSIST_MIN_MONTHS = 6

# Detection encoding parameters (spec section 4). Platforms/turbines ~1 px at 10 m.
DET_TILE = 32
DET_POS_SIZE = 1
DET_BUFFER = 10

# Neighbor search radius (deg): 32 px * 10 m = 320 m tile; ~0.006 deg (~660 m) covers the
# tile safely at all latitudes; precise filter is by tile pixel bounds.
NEIGHBOR_RADIUS_DEG = 0.006
# Negative open-water offset from a real structure, and the min clearance from any
# structure required for a negative center (deg; ~1.1 km at the equator).
NEG_OFFSET_M = (3000.0, 8000.0)
NEG_MIN_CLEAR_DEG = 0.01


def _load_dataframe() -> pd.DataFrame:
    raw = io.raw_dir(SLUG)
    csv_path = raw / CSV_FILE
    if not csv_path.exists():
        zip_path = raw / CSV_ZIP_FILE
        if not zip_path.exists():
            print(f"downloading {CSV_ZIP_FILE} ...", flush=True)
            download_http(CSV_ZIP_URL, zip_path)
        print("extracting csv ...", flush=True)
        with zipfile.ZipFile(str(zip_path)) as zf:
            zf.extract(CSV_FILE, path=str(raw))
    df = pd.read_csv(str(csv_path))
    df["composite_date"] = pd.to_datetime(df["composite_date"])
    df["year"] = df["composite_date"].dt.year
    df["month"] = df["composite_date"].dt.month
    df = df[df["year"].isin(YEARS)].copy()
    df["cid"] = df["label"].map(LABEL_TO_CID)
    df = df.dropna(subset=["cid"])
    df["cid"] = df["cid"].astype(int)
    return df


def _build_records(
    df: pd.DataFrame,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, tuple[float, float]],
    dict[tuple[int, int], int],
]:
    """Return (positive_candidates, negative_candidates, struct_coords, struct_year_cid).

    positive_candidates: one per (structure, class) at a randomly chosen persistent year.
    struct_coords: structure_id -> (lon, lat).
    struct_year_cid: (structure_id, year) -> coarse cid for any structure detected that
      year (used for in-tile neighbor labeling; not restricted to persistent years).
    """
    # Constant coords + per-(structure,year) detection stats.
    grp = df.groupby(["structure_id", "year"], sort=False)
    agg = grp.agg(
        n=("month", "size"),
        mn=("month", "min"),
        mx=("month", "max"),
        cid=("cid", "first"),
        lat=("lat", "first"),
        lon=("lon", "first"),
    ).reset_index()

    struct_coords: dict[int, tuple[float, float]] = {}
    struct_year_cid: dict[tuple[int, int], int] = {}
    for row in agg.itertuples(index=False):
        struct_coords[int(row.structure_id)] = (float(row.lon), float(row.lat))
        struct_year_cid[(int(row.structure_id), int(row.year))] = int(row.cid)

    persist = agg[
        (agg["n"] >= PERSIST_MIN_MONTHS) & (agg["mn"] <= 3) & (agg["mx"] >= 10)
    ]

    # Group persistent years by (structure, class).
    by_sc: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row in persist.itertuples(index=False):
        by_sc[(int(row.structure_id), int(row.cid))].append(int(row.year))

    rng = random.Random(SEED)
    pos: list[dict[str, Any]] = []
    for (sid, cid), years in by_sc.items():
        lon, lat = struct_coords[sid]
        pos.append(
            {
                "kind": "pos",
                "class": cid,
                "year": rng.choice(sorted(years)),
                "lon": lon,
                "lat": lat,
                "source_id": f"gfw_infra/{sid}",
            }
        )

    # Negatives: offset a random structure into nearby open water.
    coords = np.array(list(struct_coords.values()), dtype=float)  # (N, 2) lon,lat
    tree = cKDTree(coords)
    sids = list(struct_coords.keys())
    neg: list[dict[str, Any]] = []
    n_target = N_NEGATIVES * 3
    attempts = 0
    while len(neg) < n_target and attempts < n_target * 50:
        attempts += 1
        sid = sids[rng.randrange(len(sids))]
        lon, lat = struct_coords[sid]
        dist = rng.uniform(*NEG_OFFSET_M)
        bearing = rng.uniform(0, 2 * math.pi)
        dlat = (dist * math.cos(bearing)) / 111320.0
        dlon = (dist * math.sin(bearing)) / (
            111320.0 * max(0.1, math.cos(math.radians(lat)))
        )
        nlon, nlat = lon + dlon, lat + dlat
        if not (-180 <= nlon <= 180 and -85 <= nlat <= 85):
            continue
        d, _ = tree.query([nlon, nlat], k=1)
        # d is in degrees (planar over lon/lat); require clearance in both axes.
        if d < NEG_MIN_CLEAR_DEG:
            continue
        neg.append(
            {
                "kind": "neg",
                "class": CID_BACKGROUND,
                "year": rng.choice(YEARS),
                "lon": nlon,
                "lat": nlat,
                "source_id": f"gfw_infra_neg/{sid}",
            }
        )
    return pos, neg, struct_coords, struct_year_cid


# Globals for worker processes (populated in main before the write pool).
_STRUCT_COORDS: dict[int, tuple[float, float]] = {}
_STRUCT_YEAR_CID: dict[tuple[int, int], int] = {}
_NEI_TREE: cKDTree | None = None
_NEI_SIDS: list[int] = []


def _init_worker(
    coords_items: list[tuple[int, float, float]],
    year_cid: dict[tuple[int, int], int],
) -> None:
    global _STRUCT_COORDS, _STRUCT_YEAR_CID, _NEI_TREE, _NEI_SIDS
    _STRUCT_COORDS = {sid: (lon, lat) for sid, lon, lat in coords_items}
    _STRUCT_YEAR_CID = year_cid
    _NEI_SIDS = [sid for sid, _, _ in coords_items]
    _NEI_TREE = cKDTree(
        np.array([[lon, lat] for _, lon, lat in coords_items], dtype=float)
    )


def _write_tile(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    _, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"], proj)
    bounds = io.centered_bounds(col, row, DET_TILE, DET_TILE)
    x_min, y_min = bounds[0], bounds[1]
    year = rec["year"]

    positives: list[tuple[int, int, int]] = []
    if rec["kind"] == "pos":
        positives.append((row - y_min, col - x_min, rec["class"]))

    # In-tile neighbor structures detected in the same year.
    if _NEI_TREE is not None:
        idxs = _NEI_TREE.query_ball_point(
            [rec["lon"], rec["lat"]], r=NEIGHBOR_RADIUS_DEG
        )
        for j in idxs:
            sid = _NEI_SIDS[j]
            nlon, nlat = _STRUCT_COORDS[sid]
            if nlon == rec["lon"] and nlat == rec["lat"]:
                continue
            cid = _STRUCT_YEAR_CID.get((sid, year))
            if cid is None:
                continue  # not present that year -> leave as background
            _, c, r = io.lonlat_to_utm_pixel(nlon, nlat, proj)
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
        io.year_range(year),
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
    df = _load_dataframe()
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Global Fishing Watch SAR Fixed Infrastructure (Paolo et al. 2024, Nature).\n"
            f"{FIGSHARE_DOI}\n{CSV_ZIP_URL}  ({CSV_ZIP_FILE})\n"
            "Label-only figshare analysis-data file offshore_infrastructure_v20231106.csv: "
            "detections of offshore fixed infrastructure 2017-2021 (monthly on 6-month S-1 "
            "SAR composites), fields structure_id/composite_date/lat/lon/label "
            "(oil,probable_oil,possible_oil,lake_maracaibo,wind,probable_wind,possible_wind,"
            "unknown). License CC-BY-NC-4.0. NO imagery downloaded.\n"
        )
    print(f"loaded {len(df)} detection-months for years {YEARS}", flush=True)

    pos_cands, neg_cands, struct_coords, struct_year_cid = _build_records(df)
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

    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"

    io.check_disk()

    coords_items = [(sid, lon, lat) for sid, (lon, lat) in struct_coords.items()]
    results: Counter = Counter()
    with multiprocessing.Pool(
        args.workers, initializer=_init_worker, initargs=(coords_items, struct_year_cid)
    ) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_tile, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            results[res] += 1
    print("write results:", dict(results), flush=True)

    io.check_disk()

    class_counts = {
        "oil": sum(1 for r in selected if r.get("class") == CID_OIL),
        "wind": sum(1 for r in selected if r.get("class") == CID_WIND),
        "other": sum(1 for r in selected if r.get("class") == CID_OTHER),
        "background_negative_tiles": len(neg_sel),
    }
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",  # detection encoded as per-pixel classes
            "source": "Global Fishing Watch / figshare (Paolo et al. 2024, Nature)",
            "license": "CC-BY-NC-4.0",
            "provenance": {
                "url": FIGSHARE_DOI,
                "have_locally": False,
                "annotation_method": "manual training + deep learning (Sentinel-1 SAR)",
                "file": CSV_FILE,
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
                "Offshore fixed infrastructure DETECTION from GFW SAR (Paolo et al. 2024). "
                "Class scheme: 0=background(open water), 1=oil, 2=wind, 3=other/unknown; "
                "255=nodata. GFW confidence tiers folded into coarse classes (oil/"
                "probable_oil/possible_oil/lake_maracaibo->oil; wind/probable_wind/"
                "possible_wind->wind; unknown->other). Detection encoding: 32x32 UTM 10 m "
                "context tile per structure, 1 px positive of its class + 10 px nodata "
                "buffer (21x21 ignore), rest background; other in-tile structures detected "
                "the same year encoded by their coarse class. Persistent-structure time "
                "model: a positive is emitted only for a calendar year (2017-2021) in which "
                "the structure is detected >=6 months spanning both first and last quarter, "
                "so it is present across the whole 1-year window; change_time=null "
                "(detection timing is monthly on 6-month composites, coarser than the ~1-2 "
                "month change-label bar, so NOT encoded as dated change). Coarse label is "
                "100% consistent within each structure-year. Negatives: open-water tiles "
                "offset 3-8 km from real structures with no structure within ~1.1 km. Up to "
                "1000 tiles/positive-class stratified across years + up to 1000 negatives. "
                "'other/unknown' is comparatively sparse (kept in full; spec section 5; "
                "downstream assembly filters too-small classes). All labels post-2016."
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
