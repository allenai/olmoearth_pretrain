"""Process Tick Tick Bloom (CAML) harmful-algal-bloom severity into a point table.

Source: the DrivenData / NASA "Tick Tick Bloom" competition data, now released for
open use as the Cyanobacteria Aggregated Manual Labels (CAML) dataset via the NASA
SeaBASS / OB.DAAC archive (DOI 10.5067/SeaBASS/CAML/DATA001). One SeaBASS ``.sb`` file
holds 23,570 in-situ cyanobacteria measurements at points on US inland water bodies over
2013-2021, each with uid, data_provider, region, lat, lon, date, time, abun (density),
severity (1-5), distance_to_water_m.

Encoding decision: severity CATEGORY classification (the competition's target), 5 ordinal
classes (severity 1..5 -> class ids 0..4). The raw cyanobacteria density (``abun``, SeaBASS
"cells/L"; multiply by 1000 for competition cells/mL) is carried as an auxiliary per-point
``density`` field. Sparse single-pixel water-column point measurements -> one dataset-wide
points.geojson (spec 2a), no per-point GeoTIFFs.

Time handling: blooms are transient, so each point gets a TIGHT +/-15 day window centered
on its sample date (a state at a time, not a change event -> change_time=null). Only
samples on/after 2016-01-01 are kept (Sentinel era; spec post-2016 rule).

Access: OB.DAAC getfile requires NASA Earthdata (URS) auth. Credentials are read from
.env (NASA_EARTHDATA_USERNAME / NASA_EARTHDATA_PASSWORD) and written
to ~/.netrc so the URS OAuth redirect authenticates (spec 8).

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.tick_tick_bloom_hab_severity
"""

import argparse
import os
import re
import urllib.request
from collections import Counter
from datetime import UTC, datetime

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "tick_tick_bloom_hab_severity"
NAME = "Tick Tick Bloom (HAB severity)"

# SeaBASS archive directory that lists the CAML .sb file (whose OB.DAAC getfile URL carries
# a content-hash prefix we resolve at runtime so it stays correct if the archive re-hashes).
SEABASS_ARCHIVE_DIR = (
    "https://seabass.gsfc.nasa.gov/archive/NASA_HEADQUARTERS/SGupta/CAML/"
    "CAML_2013_2021/archive"
)
SB_FILENAME = "CAML_cyanobacteria_abundance_20211229_R1.sb"
ENV_PATH = ".env"

MIN_YEAR = 2016  # spec post-2016 rule: keep only samples on/after 2016-01-01
PER_CLASS = 1000  # spec 5: up to 1000 locations per class (classification)
HALF_WINDOW_DAYS = 15  # tight +/-15d window on the sample date (transient blooms)

FIELDS = [
    "uid",
    "data_provider",
    "region",
    "lat",
    "lon",
    "date",
    "time",
    "abun",
    "severity",
    "distance_to_water_m",
]

# severity level (1..5) -> (class id, name, description). Density bands are the competition
# WHO-based thresholds in cells/mL (SeaBASS abun is those /1000, i.e. "cells/L").
SEVERITY_CLASSES = [
    (
        1,
        "severity_1_low",
        "Cyanobacteria density < 20,000 cells/mL (WHO low / non-bloom).",
    ),
    (
        2,
        "severity_2_moderate",
        "Cyanobacteria density 20,000-<100,000 cells/mL (moderate).",
    ),
    (3, "severity_3_high", "Cyanobacteria density 100,000-<1,000,000 cells/mL (high)."),
    (
        4,
        "severity_4_very_high",
        "Cyanobacteria density 1,000,000-<10,000,000 cells/mL (very high).",
    ),
    (
        5,
        "severity_5_extreme",
        "Cyanobacteria density >= 10,000,000 cells/mL (extreme bloom).",
    ),
]
SEV_TO_ID = {str(sev): i for i, (sev, _n, _d) in enumerate(SEVERITY_CLASSES)}


def _write_netrc() -> None:
    """Write ~/.netrc with NASA Earthdata (URS) creds from the project .env (spec 8)."""
    creds = {}
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    creds[k.strip()] = v.strip()
    user = creds.get("NASA_EARTHDATA_USERNAME")
    pw = creds.get("NASA_EARTHDATA_PASSWORD")
    if not user or not pw:
        raise RuntimeError(
            "needs-credential: NASA Earthdata (URS) login not found in " + ENV_PATH
        )
    netrc = os.path.expanduser("~/.netrc")
    line = f"machine urs.earthdata.nasa.gov login {user} password {pw}\n"
    existing = ""
    if os.path.exists(netrc):
        with open(netrc) as f:
            existing = f.read()
    if "urs.earthdata.nasa.gov" not in existing:
        with open(netrc, "a") as f:
            f.write(line)
    os.chmod(netrc, 0o600)


def _resolve_getfile_url() -> str:
    """Scrape the SeaBASS archive dir for the OB.DAAC getfile URL of the .sb file."""
    req = urllib.request.Request(SEABASS_ARCHIVE_DIR, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=120) as r:
        html = r.read().decode("utf-8", "replace")
    m = re.search(
        r'href="(https://oceandata\.sci\.gsfc\.nasa\.gov/ob/getfile/[^"]*'
        + re.escape(SB_FILENAME)
        + r')"',
        html,
    )
    if not m:
        raise RuntimeError(
            "could not find OB.DAAC getfile URL for CAML in SeaBASS archive listing"
        )
    return m.group(1)


def download_sb() -> str:
    """Download the CAML .sb into raw_dir (idempotent). Returns local path."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    dst = raw / SB_FILENAME
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Cyanobacteria Aggregated Manual Labels (CAML), NASA SeaBASS/OB.DAAC\n"
            "DOI: 10.5067/SeaBASS/CAML/DATA001\n"
            f"archive dir: {SEABASS_ARCHIVE_DIR}\n"
            "Originally the DrivenData/NASA 'Tick Tick Bloom' competition training+test\n"
            "labels; released for open use. Requires NASA Earthdata (URS) auth.\n"
        )
    if dst.exists() and dst.stat().st_size > 100_000:
        return dst.path
    _write_netrc()
    url = _resolve_getfile_url()
    print(f"downloading {url}")
    import requests

    tmp = raw / (SB_FILENAME + ".tmp")
    with requests.Session() as s:
        resp = s.get(url, timeout=600, stream=True)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if "text/html" in ctype:
            raise RuntimeError(
                f"Earthdata auth failed (got HTML) for {url}; check ~/.netrc"
            )
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)
    tmp.rename(dst)
    return dst.path


def parse_records(sb_path: str) -> list[dict]:
    """Parse the SeaBASS .sb body into dict records (skip the /header)."""
    recs = []
    started = False
    with open(sb_path) as f:
        for line in f:
            if started:
                parts = line.strip().split(",")
                if len(parts) == len(FIELDS):
                    recs.append(dict(zip(FIELDS, parts)))
            if line.startswith("/end_header"):
                started = True
    return recs


def _center_datetime(date_s: str, time_s: str) -> datetime | None:
    """Parse yyyymmdd (+ hh:mm:ss) into a UTC datetime, or None if unparseable."""
    if not date_s or len(date_s) != 8 or not date_s.isdigit():
        return None
    y, mo, d = int(date_s[:4]), int(date_s[4:6]), int(date_s[6:8])
    hh = mm = ss = 0
    if time_s and ":" in time_s:
        try:
            hh, mm, ss = (int(x) for x in time_s.split(":")[:3])
        except ValueError:
            hh = mm = ss = 0
    try:
        return datetime(y, mo, d, hh, mm, ss, tzinfo=UTC)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    sb_path = download_sb()
    raw = parse_records(sb_path)
    print(f"parsed {len(raw)} raw CAML records")

    # Build clean records: valid coords, severity in 1..5, parseable post-2016 date.
    records = []
    dropped_pre2016 = 0
    dropped_bad = 0
    for r in raw:
        sev = r.get("severity")
        if sev not in SEV_TO_ID:
            dropped_bad += 1
            continue
        try:
            lon = float(r["lon"])
            lat = float(r["lat"])
        except (ValueError, KeyError):
            dropped_bad += 1
            continue
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            dropped_bad += 1
            continue
        center = _center_datetime(r.get("date", ""), r.get("time", ""))
        if center is None:
            dropped_bad += 1
            continue
        if center.year < MIN_YEAR:
            dropped_pre2016 += 1
            continue
        try:
            abun = float(r["abun"])
        except (ValueError, KeyError):
            abun = None
        records.append(
            {
                "uid": r["uid"],
                "lon": lon,
                "lat": lat,
                "severity": sev,
                "center": center,
                "density": abun,  # SeaBASS "cells/L" (competition cells/mL = *1000)
                "region": r.get("region"),
            }
        )
    print(
        f"kept {len(records)} clean post-{MIN_YEAR} records "
        f"(dropped pre-{MIN_YEAR}={dropped_pre2016}, bad={dropped_bad})"
    )

    # Balance to <=1000 per severity class (subject to 25k cap; spec 5). Rare classes kept.
    selected = balance_by_class(records, "severity", per_class=PER_CLASS)
    selected.sort(key=lambda r: r["uid"])  # deterministic id assignment
    print(f"selected {len(selected)} points (<= {PER_CLASS}/class)")

    points = []
    for i, r in enumerate(selected):
        cid = SEV_TO_ID[r["severity"]]
        tr = io.centered_time_range(r["center"], half_window_days=HALF_WINDOW_DAYS)
        p = {
            "id": f"{i:06d}",
            "lon": r["lon"],
            "lat": r["lat"],
            "label": cid,
            "time_range": tr,
            "change_time": None,
            "source_id": r["uid"],
            "region": r["region"],
        }
        if r["density"] is not None:
            p["density"] = r["density"]
        points.append(p)
    io.write_points_table(SLUG, "classification", points)

    class_counts = Counter(SEV_TO_ID[r["severity"]] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "DrivenData / NASA (SeaBASS OB.DAAC CAML)",
            "license": "open (competition data released for reuse, with attribution)",
            "provenance": {
                "url": "https://www.drivendata.org/competitions/143/tick-tick-bloom/",
                "doi": "10.5067/SeaBASS/CAML/DATA001",
                "have_locally": False,
                "annotation_method": "field survey (in-situ cyanobacteria cell-count measurements)",
            },
            "sensors_relevant": ["sentinel2", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (_sev, name, desc) in enumerate(SEVERITY_CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "auxiliary_fields": {
                "density": "raw cyanobacteria density from SeaBASS 'abun' (units cells/L; "
                "multiply by 1000 for competition cells/mL). Present per point where available.",
                "region": "US region label from the competition (south/west/midwest/northeast).",
            },
            "num_samples": len(selected),
            "class_counts": {
                i: class_counts.get(i, 0) for i in range(len(SEVERITY_CLASSES))
            },
            "notes": (
                "In-situ HAB severity at points on US inland water bodies. Severity 1-5 -> "
                "class ids 0-4. Only post-2016 samples kept. Tight +/-15d window centered on "
                "each sample date (transient blooms); change_time=null. Balanced to <=1000/"
                "class. Weak label for 10-30 m optical (water color): a single-pixel "
                "water-column point measurement, not a full-water-body segmentation."
            ),
        },
    )

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    main()
