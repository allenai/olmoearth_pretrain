"""Process DeepOWT (Global Offshore Wind Turbines) into presence-only points.

Source: DeepOWT, Zhang et al., Earth Syst. Sci. Data (ESSD), on Zenodo
(https://doi.org/10.5281/zenodo.5933967, CC-BY-4.0). A global inventory of offshore
wind-energy infrastructure with per-quarter deployment status derived from Sentinel-1
time series with deep learning + validation. We use the main file ``DeepOWT.geojson``:
9,941 Point features, each with 20 quarterly status columns Y2016Q3 ... Y2021Q2, each
valued with DeepOWT's semantic class:
    0 = open sea, 1 = under construction, 2 = offshore wind turbine, 3 = substation.

Task type: presence-only classification POINTS (spec section 2a). Each selected structure
is emitted as a single presence point in one dataset-wide ``points.geojson``. We keep only
the real object classes (under_construction / offshore_turbine / substation), renumbered
0..2; open sea (DeepOWT background) is dropped. Negatives are supplied downstream by the
assembly step from other datasets.

Time handling (spec section 5). DeepOWT resolves the appearance/state of each structure
only to a QUARTER (~3 months) -- coarser than the section-5 change-timing requirement
(~1-2 months) -- so we do NOT emit dated change labels. Each structure is treated as a
PERSISTENT structure: a positive for class c is emitted for a point only in a full calendar
year (2017-2020) in which ALL FOUR quarters equal c, guaranteeing the state is genuinely
persistent across the whole 1-year label window. change_time is null and the time range is
that calendar year (io.year_range).

Run (reuses cached raw):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.deepowt_global_offshore_wind_turbines
"""

import argparse
import json
import multiprocessing
import random
from collections import Counter
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "deepowt_global_offshore_wind_turbines"
NAME = "DeepOWT (Global Offshore Wind Turbines)"
ZENODO = "https://doi.org/10.5281/zenodo.5933967"
GEOJSON_URL = "https://zenodo.org/records/5933967/files/DeepOWT.geojson?download=1"
GEOJSON_FILE = "DeepOWT.geojson"

# DeepOWT native status ids for the real object classes (0 = open sea dropped).
STATUS_UNDER_CONSTRUCTION = 1
STATUS_TURBINE = 2
STATUS_SUBSTATION = 3
POSITIVE_STATUSES = (STATUS_UNDER_CONSTRUCTION, STATUS_TURBINE, STATUS_SUBSTATION)

# Class scheme = the real object classes only, renumbered 0..N-1 (no background).
STATUS_TO_CID = {
    STATUS_UNDER_CONSTRUCTION: 0,
    STATUS_TURBINE: 1,
    STATUS_SUBSTATION: 2,
}
CLASSES = [
    {
        "id": 0,
        "name": "under_construction",
        "description": "Offshore wind site under construction -- foundation/platform "
        "present but turbine not yet operational (DeepOWT status 1). Transient state; only "
        "sites under construction for a full calendar year are emitted (persistent window).",
    },
    {
        "id": 1,
        "name": "offshore_turbine",
        "description": "Installed offshore wind turbine (DeepOWT status 2), detected from "
        "Sentinel-1 time series with deep learning + validation.",
    },
    {
        "id": 2,
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
SEED = 42


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


def _build_records(pts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One presence record per (point, class), using a random stable year for that class.

    Each physical point contributes at most one point per class while spreading across
    years for temporal diversity.
    """
    rng = random.Random(SEED)
    recs: list[dict[str, Any]] = []
    for p in pts:
        stable = p["stable"]
        for status in POSITIVE_STATUSES:
            yrs = [y for y, v in stable.items() if v == status]
            if yrs:
                recs.append(
                    {
                        "label": STATUS_TO_CID[status],
                        "year": rng.choice(yrs),
                        "lon": p["lon"],
                        "lat": p["lat"],
                        "source_id": f"deepowt/{p['idx']}",
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

    recs = _build_records(pts)
    print(f"built {len(recs)} presence records", flush=True)
    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class)", flush=True)

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
            "num_samples": len(points),
            "class_counts": {
                c["name"]: counts.get(c["id"], 0) for c in CLASSES
            },
            "notes": (
                "Presence-only classification POINTS converted from the old detection-tile "
                "encoding. Each selected offshore wind structure is emitted as a single "
                "presence point (no fabricated GeoTIFF context, no background/negative tiles). "
                "Only the real object classes are kept (0=under_construction, 1=offshore_turbine, "
                "2=substation); DeepOWT open-sea background is dropped. Persistent-structure time "
                "model: a positive is emitted only for a full calendar year (2017-2020) in which "
                "ALL FOUR quarters equal that class, so the state is persistent across the 1-year "
                "window; change_time=null. up to 1000 points/class (balance_by_class). Negatives "
                "are supplied downstream by the assembly step from other datasets."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(points)
    )
    print(f"done: {len(points)} points", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
