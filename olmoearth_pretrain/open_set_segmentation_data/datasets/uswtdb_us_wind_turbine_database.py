"""Process the USWTDB (US Wind Turbine Database) into presence-only points.

Source (external, USGS / LBNL / AWEA, public domain -- a U.S. Government work):
  https://energy.usgs.gov/uswtdb/
The U.S. Wind Turbine Database is the authoritative national inventory of onshore and
offshore wind turbines in the United States and its territories, each turbine
position-verified against high-resolution aerial/satellite imagery and updated quarterly.

We download only the LABEL points (no imagery -- pretraining supplies imagery) from the
public USGS EERSC PostgREST API as one JSON array (75,727 turbines):
  https://energy.usgs.gov/api/uswtdb/v1/turbines
Each record is one turbine (unique ``case_id``) with WGS84 ``xlong``/``ylat``, project
online year ``p_year`` (year-granular), and turbine attributes: ``t_cap`` (nameplate kW),
``t_hh`` (hub height m), ``t_rd`` (rotor diameter m), ``t_model``/``t_manu``, ``t_offshore``
(0/1), and location/attribute confidence ``t_conf_loc``/``t_conf_atr`` (1-3).

Task type: presence-only POINTS (spec section 2a), single class (turbine). Each selected
turbine is emitted as one presence point in a dataset-wide ``points.geojson``; negatives
are supplied by the downstream assembly (no fabricated background tiles here). A single
turbine tower/pad is ~1 px at 10 m but a strong, detectable signature (tower shadow, gravel
pad, access roads) -> observable at 10-30 m from Sentinel-2/Sentinel-1/Landsat.

CHANGE vs PRESENCE (spec section 5 timing rule): ``p_year`` is year-granular only, so the
installation event is NOT resolvable to ~1-2 months and CANNOT be a change label. We use the
persistent post-construction STATE (a turbine stays visible for years) as presence with
change_time=null, anchoring each point's 1-year window in the Sentinel-2 era AFTER
commissioning so the turbine is present: window_year = clamp(p_year+1, 2017, 2024) (missing
p_year -> 2022). This keeps pre-2016 turbines (still standing post-2016) while honoring the
post-2016 rule.

Classes: 0 turbine.

Sampling: up to 1000 points (sampling.balance_by_class, default 25k total cap).

Run (reuses cached raw JSON):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.uswtdb_us_wind_turbine_database
"""

import argparse
import json
import multiprocessing
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "uswtdb_us_wind_turbine_database"
NAME = "USWTDB (US Wind Turbine Database)"
URL = "https://energy.usgs.gov/uswtdb/"
API = "https://energy.usgs.gov/api/uswtdb/v1/turbines"
JSON_NAME = "uswtdb_turbines.json"
UA = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

TURBINE_ID = 0
CLASSES = [
    {
        "id": TURBINE_ID,
        "name": "turbine",
        "description": "Utility-scale wind turbine (onshore or offshore), position-verified "
        "against high-resolution aerial imagery (USWTDB). Tower/pad footprint is ~1 px at "
        "10 m but a strong signature (shadow, gravel pad, access roads).",
    },
]
CID_TO_NAME = {c["id"]: c["name"] for c in CLASSES}

PER_CLASS = 1000  # turbine presence points (single class, spec section 5)
DEFAULT_YEAR = 2022  # window for turbines with a missing p_year
WINDOW_MIN, WINDOW_MAX = 2017, 2024  # S2-era post-commissioning window clamp
SEED = 42


def _download() -> str:
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    out = raw / JSON_NAME
    download.download_postgrest_json(
        API,
        out,
        select="case_id,p_name,t_state,p_year,t_cap,t_hh,t_rd,t_model,t_manu,"
        "t_conf_loc,t_conf_atr,t_img_date,t_img_src,t_offshore,xlong,ylat",
        order="case_id",
        page=20000,
        headers=UA,
    )
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "USWTDB (US Wind Turbine Database), USGS / LBNL / AWEA, public domain "
            "(U.S. Government work).\n"
            f"{URL}\n"
            "Label-only download of turbine POINTS from the public USGS EERSC PostgREST "
            f"API:\n{API}\n"
            "One JSON array of turbines (unique case_id) with WGS84 xlong/ylat, p_year "
            "(year-granular commissioning year), t_cap/t_hh/t_rd/t_model/t_manu, t_offshore, "
            "and t_conf_loc/t_conf_atr confidence. Imagery is supplied by pretraining, not "
            "downloaded here.\n"
        )
    return str(out)


def _window_year(p_year: int | None) -> int:
    if not p_year:
        return DEFAULT_YEAR
    return max(WINDOW_MIN, min(WINDOW_MAX, p_year + 1))


def _load_turbines() -> list[dict[str, Any]]:
    """Parse the JSON array into per-turbine presence records (one row = one turbine)."""
    path = io.raw_dir(SLUG) / JSON_NAME
    rows = json.load(path.open())
    out: list[dict[str, Any]] = []
    for r in rows:
        lon, lat = r.get("xlong"), r.get("ylat")
        if lon is None or lat is None:
            continue
        try:
            lon, lat = float(lon), float(lat)
        except (TypeError, ValueError):
            continue
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            continue
        p_year = int(r["p_year"]) if r.get("p_year") else None
        out.append(
            {
                "label": TURBINE_ID,
                "lon": lon,
                "lat": lat,
                "year": _window_year(p_year),
                "offshore": bool(r.get("t_offshore")),
                "source_id": f"uswtdb/case_id={r.get('case_id')}/{r.get('p_name')}",
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    _download()
    io.check_disk()

    turbs = _load_turbines()
    print(f"loaded {len(turbs)} turbines", flush=True)

    selected = balance_by_class(turbs, "label", per_class=PER_CLASS, seed=SEED)
    print(f"selected {len(selected)} points (<= {PER_CLASS})", flush=True)

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

    n_offshore = sum(1 for t in turbs if t["offshore"])
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "USGS / LBNL / AWEA (USWTDB)",
            "license": "public domain (U.S. Government work)",
            "provenance": {
                "url": URL,
                "have_locally": False,
                "annotation_method": "manual position verification against aerial imagery",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "num_samples": len(selected),
            "class_counts": {CID_TO_NAME[TURBINE_ID]: len(selected)},
            "available": {
                "total_turbines": len(turbs),
                "offshore_turbines": n_offshore,
            },
            "window_rule": f"clamp(p_year+1, {WINDOW_MIN}, {WINDOW_MAX}); missing={DEFAULT_YEAR}",
            "notes": (
                "Presence-only POINTS converted from the former detection-tile encoding; "
                "negatives are supplied by the downstream assembly. Single class: 0=turbine. "
                "USWTDB national wind-turbine POINT inventory (75,727 turbines). "
                "Presence/state, NOT change: p_year is year-granular only (not resolvable to "
                "~1-2 months per the spec change-timing rule), so the persistent "
                "post-construction state is used with change_time=null and a 1-year window "
                "anchored AFTER commissioning in the S2 era "
                f"(window_year=clamp(p_year+1,{WINDOW_MIN},{WINDOW_MAX}); missing "
                f"p_year->{DEFAULT_YEAR}); this keeps pre-2016 turbines (still standing "
                "post-2016) while honoring the post-2016 rule. Onshore and offshore "
                f"({n_offshore}) turbines both used. Up to 1000 points (balance_by_class)."
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
