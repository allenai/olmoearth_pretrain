"""Process the PhenoCam Network (v3) site vegetation types into open-set-segmentation labels.

Source: the PhenoCam network (ORNL DAAC PhenoCam Images V3 corresponds to this network).
Rather than the (Earthdata-gated, multi-GB) image archives, we only need the *labels*: the
per-site coordinate plus the human-assigned vegetation/land-cover type. That label-only
signal is published openly by the PhenoCam project API at
``https://phenocam.nau.edu/api/cameras/`` (site name, Lat, Lon, active dates, and
``sitemetadata.primary_veg_type``). Each site is a ground camera monitoring a dominant,
relatively homogeneous vegetation stand; the PhenoCam team assigns its land-cover type by
human vegetation typing. We treat each site as a weak site-level land-cover reference
*point* (sparse point segmentation, spec §2a) -> one dataset-wide ``points.geojson``,
balanced to <=1000 per class.

Caveat (documented in the summary): the coordinate is the camera location and the field of
view is oblique/local, so the 10 m pixel at the coordinate is only an approximate stand-in
for the site's dominant land cover. The distinctions we keep (forest / grass / crop /
wetland / shrub / tundra / non-vegetated) are all resolvable at 10-30 m from S2/S1/Landsat,
and downstream assembly treats these as weak labels.
"""

import argparse
import json
import multiprocessing
import urllib.request
from collections import Counter
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "phenocam_network_v3"
API_URL = "https://phenocam.nau.edu/api/cameras/?format=json&limit=5000"
PER_CLASS = 1000
SENTINEL_ERA_START = 2016

# PhenoCam primary_veg_type code -> (class name, description). Order fixes class ids.
# Codes without a coherent overhead land-cover meaning (understory "UN", missing) are dropped.
VEG_TYPES: list[tuple[str, str, str]] = [
    (
        "DB",
        "Deciduous broadleaf forest",
        "Site dominated by deciduous broadleaf trees (e.g. temperate hardwood forest).",
    ),
    (
        "EN",
        "Evergreen needleleaf forest",
        "Site dominated by evergreen needleleaf conifers (e.g. pine/spruce/fir forest).",
    ),
    (
        "GR",
        "Grassland",
        "Site dominated by grasses / herbaceous cover (natural or managed grassland, prairie).",
    ),
    (
        "AG",
        "Agriculture",
        "Cultivated cropland / managed agricultural field (annual or perennial crops).",
    ),
    (
        "SH",
        "Shrub",
        "Site dominated by shrubs / low woody vegetation (shrubland, chaparral, sagebrush).",
    ),
    (
        "WL",
        "Wetland",
        "Wetland vegetation (marsh, bog, fen, emergent or flooded herbaceous/woody cover).",
    ),
    (
        "EB",
        "Evergreen broadleaf forest",
        "Site dominated by evergreen broadleaf trees (e.g. tropical/subtropical broadleaf forest).",
    ),
    (
        "TN",
        "Tundra",
        "Arctic/alpine tundra: low-stature herbs, mosses, lichens, dwarf shrubs.",
    ),
    (
        "NV",
        "Non-vegetated",
        "Little/no vegetation: bare soil, rock, sand, snow/ice, or built surfaces.",
    ),
    (
        "DN",
        "Deciduous needleleaf forest",
        "Site dominated by deciduous needleleaf trees (e.g. larch/tamarack forest).",
    ),
    (
        "MX",
        "Mixed forest",
        "Mixed stand of deciduous and evergreen trees with neither strongly dominant.",
    ),
]
CODE_TO_ID = {code: i for i, (code, _n, _d) in enumerate(VEG_TYPES)}


def fetch_sites() -> list[dict[str, Any]]:
    """Fetch all PhenoCam site records from the public API (paginating if needed)."""
    records: list[dict[str, Any]] = []
    url: str | None = API_URL
    while url:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.loads(r.read())
        records.extend(data.get("results", []))
        url = data.get("next")
    return records


def build_records(sites: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter]:
    """Filter/convert raw site records into flat point records; return (records, drop_reasons)."""
    out: list[dict[str, Any]] = []
    drops: Counter = Counter()
    for c in sites:
        lat, lon = c.get("Lat"), c.get("Lon")
        if lat is None or lon is None:
            drops["no_coord"] += 1
            continue
        sm = c.get("sitemetadata") or {}
        code = (sm.get("primary_veg_type") or "").strip()
        if code not in CODE_TO_ID:
            drops[f"veg:{code or 'EMPTY'}"] += 1
            continue
        # Choose a representative Sentinel-era 1-year window within the site's active span.
        # Vegetation/land-cover type is a persistent (static) label, so any 2016+ year in
        # which the site was operating is representative (spec §5 static labels).
        df = c.get("date_first") or ""
        dl = c.get("date_last") or ""
        try:
            fy = int(df[:4])
            ly = int(dl[:4])
        except ValueError:
            drops["bad_dates"] += 1
            continue
        if ly < SENTINEL_ERA_START:
            drops["pre_2016"] += 1
            continue
        year = max(SENTINEL_ERA_START, fy)
        if year > ly:
            year = ly
        out.append(
            {
                "lon": float(lon),
                "lat": float(lat),
                "code": code,
                "label": CODE_TO_ID[code],
                "year": year,
                "source_id": c.get("Sitename"),
            }
        )
    return out, drops


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    _ = args

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "PhenoCam Network (ORNL DAAC PhenoCam Images V3).\n"
            "Label-only source (site coords + human vegetation type) via public API:\n"
            f"{API_URL}\n"
            "Image archives NOT downloaded (not needed for labels).\n"
        )

    sites = fetch_sites()
    print(f"fetched {len(sites)} PhenoCam site records")
    # Cache the raw site table for provenance / reproducibility.
    with (raw / "cameras.json").open("w") as f:
        json.dump(sites, f)

    records, drops = build_records(sites)
    print(f"usable records: {len(records)}; drops: {dict(drops)}")

    selected = balance_by_class(records, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class, 25k total cap)")

    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": r["label"],
                "time_range": io.year_range(r["year"]),
                "source_id": r["source_id"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    counts = Counter(r["code"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "PhenoCam Network v3",
            "task_type": "classification",
            "source": "ORNL DAAC / PhenoCam Network",
            "license": "open (ORNL DAAC; PhenoCam data policy)",
            "provenance": {
                "url": "https://daac.ornl.gov/VEGETATION/guides/Phenocam_Images_V3.html",
                "label_source_url": API_URL,
                "have_locally": False,
                "annotation_method": "ground camera + human vegetation typing (site primary_veg_type)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc, "phenocam_code": code}
                for i, (code, name, desc) in enumerate(VEG_TYPES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {code: counts.get(code, 0) for code, _n, _d in VEG_TYPES},
            "notes": (
                "Sparse 1x1 site-level land-cover points from PhenoCam primary_veg_type. "
                "Label is the camera-site dominant vegetation type (in-situ human typing); "
                "coordinate is the camera location, so the 10 m pixel is an approximate "
                "stand-in for the site footprint (weak label). Persistent land-cover label: "
                "~1-year window (>=2016) within each site's active span. Dropped sites with "
                "no coordinate, no/empty primary_veg_type, understory-only (UN), or activity "
                "entirely pre-2016."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
