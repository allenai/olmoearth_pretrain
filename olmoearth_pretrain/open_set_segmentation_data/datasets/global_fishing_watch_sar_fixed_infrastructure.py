"""Process Global Fishing Watch SAR Fixed Infrastructure into presence-only points.

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

Task type: presence-only POINTS (spec section 2a). Each selected structure is emitted as
one presence point in a dataset-wide ``points.geojson``; negatives are supplied by the
downstream assembly (no fabricated background tiles here). Three real object classes:
    0 = oil, 1 = wind, 2 = other/unknown
Confidence tiers are folded into the coarse class:
    oil,probable_oil,possible_oil,lake_maracaibo -> oil(0)
    wind,probable_wind,possible_wind             -> wind(1)
    unknown                                      -> other(2)

Time / change handling. Fixed infrastructure is PERSISTENT, not a change event. Detection
timing is only monthly on 6-month composites (coarser than the ~1-2 month change-timing
bar), so we do NOT emit dated change labels (change_time is null). Each structure is
treated as a persistent structure: a positive is emitted for a structure only in a
calendar year (2017-2021) in which it is detected persistently across the WHOLE year --
>= 6 monthly detections spanning both the first quarter (month <= 3) and the last quarter
(month >= 10), guaranteeing the state is genuinely present across the 1-year label window.
Within a structure-year the coarse label is 100% consistent (verified). The time range is
that calendar year (io.year_range).

Sampling: up to 1000 points per class (sampling.balance_by_class, default 25k total cap).

Run (reuses cached raw CSV):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_fishing_watch_sar_fixed_infrastructure
"""

import argparse
import multiprocessing
import random
import zipfile
from collections import Counter, defaultdict
from typing import Any

import pandas as pd

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.download import download_http
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "global_fishing_watch_sar_fixed_infrastructure"
NAME = "Global Fishing Watch SAR Fixed Infrastructure"
FIGSHARE_DOI = "https://doi.org/10.6084/m9.figshare.24309475"
CSV_ZIP_URL = "https://ndownloader.figshare.com/files/43801560"
CSV_ZIP_FILE = "offshore_infrastructure_v20231106.csv.zip"
CSV_FILE = "offshore_infrastructure_v20231106.csv"

CID_OIL = 0
CID_WIND = 1
CID_OTHER = 2

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
CID_TO_NAME = {c["id"]: c["name"] for c in CLASSES}

YEARS = [2017, 2018, 2019, 2020, 2021]
PER_CLASS = 1000
SEED = 42

# Persistence rule: a structure counts for a calendar year if it has >= this many monthly
# detections spanning both the first quarter and the last quarter of the year.
PERSIST_MIN_MONTHS = 6


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


def _build_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return one presence record per (structure, class) at a randomly chosen persistent year.

    A structure-year is "persistent" (state genuinely present across the whole 1-year
    label window) if it has >= PERSIST_MIN_MONTHS monthly detections spanning both the
    first quarter (month <= 3) and the last quarter (month >= 10).
    """
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
    for row in agg.itertuples(index=False):
        struct_coords[int(row.structure_id)] = (float(row.lon), float(row.lat))

    persist = agg[
        (agg["n"] >= PERSIST_MIN_MONTHS) & (agg["mn"] <= 3) & (agg["mx"] >= 10)
    ]

    # Group persistent years by (structure, class).
    by_sc: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row in persist.itertuples(index=False):
        by_sc[(int(row.structure_id), int(row.cid))].append(int(row.year))

    rng = random.Random(SEED)
    recs: list[dict[str, Any]] = []
    for (sid, cid), years in by_sc.items():
        lon, lat = struct_coords[sid]
        recs.append(
            {
                "label": cid,
                "year": rng.choice(sorted(years)),
                "lon": lon,
                "lat": lat,
                "source_id": f"gfw_infra/{sid}",
            }
        )
    return recs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.parse_args()

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

    recs = _build_records(df)
    cand_counts = Counter(r["label"] for r in recs)
    print(
        "presence candidates: "
        + ", ".join(f"{CID_TO_NAME[c]}={cand_counts[c]}" for c in sorted(cand_counts)),
        flush=True,
    )

    selected = balance_by_class(recs, "label", per_class=PER_CLASS, seed=SEED)
    print(f"selected {len(selected)} points (<= {PER_CLASS}/class)", flush=True)

    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": r["label"],
                "time_range": io.year_range(r["year"]),
                "change_time": None,
                "source_id": r["source_id"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    counts = Counter(r["label"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
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
            "num_samples": len(selected),
            "class_counts": {
                CID_TO_NAME[c]: counts.get(c, 0) for c in sorted(CID_TO_NAME)
            },
            "notes": (
                "Presence-only POINTS converted from the former detection-tile encoding; "
                "negatives are supplied by the downstream assembly. Offshore fixed "
                "infrastructure from GFW SAR (Paolo et al. 2024). Real object classes only: "
                "0=oil, 1=wind, 2=other/unknown. GFW confidence tiers folded into coarse "
                "classes (oil/probable_oil/possible_oil/lake_maracaibo->oil; wind/"
                "probable_wind/possible_wind->wind; unknown->other). Persistent-structure "
                "time model: a point is emitted only for a calendar year (2017-2021) in "
                "which the structure is detected >=6 months spanning both first and last "
                "quarter, so it is present across the whole 1-year window; change_time=null "
                "(detection timing is monthly on 6-month composites, coarser than the ~1-2 "
                "month change-label bar). Coarse label is 100% consistent within each "
                "structure-year. Up to 1000 points/class (balance_by_class). All labels "
                "post-2016."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print(f"done: {len(selected)} points", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
