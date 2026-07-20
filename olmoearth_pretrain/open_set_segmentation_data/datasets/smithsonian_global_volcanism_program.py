"""Smithsonian Global Volcanism Program (GVP) -> presence-only volcano-type POINTS.

Source: Global Volcanism Program, 2024. Volcanoes of the World (VOTW) database, Smithsonian
Institution (https://volcano.si.edu/). Distributed as OGC WFS from the GVP GeoServer
(https://webservices.volcano.si.edu/geoserver/GVP-VOTW/wfs). License: free research use
(attribution to GVP). Two point layers are pulled:
  - Smithsonian_VOTW_Holocene_Volcanoes    (1,196 well-preserved Holocene edifices)
  - Smithsonian_VOTW_Pleistocene_Volcanoes (1,451 older Pleistocene edifices)
Each feature is ONE POINT at the volcano's summit location with a Primary_Volcano_Type
attribute (stratovolcano, shield, caldera, ...), plus name/number/country/elevation.

Task type / encoding (presence-only POINTS): a GVP record is a summit census point marking
that a volcano of a given Primary_Volcano_Type is PRESENT at that location. We emit each
summit as one presence POINT carrying its multi-class volcano-type label into a dataset-wide
``points.geojson`` (joining the other presence-only point datasets); cross-dataset negatives
are supplied by assembly. The earlier per-detection GeoTIFF tile encoding (positive square +
nodata buffer + background fill + fabricated background-only negative tiles) is dropped.

Observability / judgment calls (recorded in the summary):
  - ACCEPT on observability: volcano edifices are large landforms clearly visible at 10-30 m.
    Caveat: the summit POINT is a weak label for the whole edifice, and Primary_Volcano_Type
    is a property of the full edifice morphology -- so type labels are best-effort.
  - NO dated-eruption CHANGE label. GVP eruption dates are year-resolved at best (and often
    historical / BCE); per spec section 5 a change label is only allowed when the event date
    is known to ~1-2 months. Eruptions are therefore NOT encoded as change events; volcanoes
    are treated as persistent landforms with static 1-year Sentinel-era windows.
  - Pleistocene included alongside Holocene (manifest says "Holocene/Pleistocene"). "Unknown"/
    "None" volcano types are dropped (not a real class).

Class scheme: volcano types only, ids 0..N-1 ordered by descending global frequency.

Run (idempotent; reuses cached raw):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.smithsonian_global_volcanism_program
"""

import argparse
import json
import multiprocessing
import random
import re
from collections import Counter
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

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

# Sampling parameters.
PER_CLASS = 1000  # spec section 5: up to 1000 presence points per volcano-type class
YEARS = list(range(2016, 2025))  # persistent landforms -> static 1-year Sentinel-era windows


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


def build_classes(volcs: list[dict[str, Any]]) -> tuple[list[dict], dict[str, int]]:
    """Assign class ids 0..N-1 to volcano types by descending global frequency."""
    freq = Counter(v["vtype"] for v in volcs)
    ordered = [t for t, _ in freq.most_common()]
    type_to_cid = {t: i for i, t in enumerate(ordered)}
    classes = [
        {
            "id": type_to_cid[t],
            "name": t,
            "description": f"GVP Primary_Volcano_Type '{t}' (summit-point presence).",
        }
        for t in ordered
    ]
    return classes, type_to_cid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.parse_args()

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
        v["label"] = type_to_cid[v["vtype"]]

    # Select presence points, balanced/capped per volcano-type class (spec section 5).
    selected = balance_by_class(volcs, "label", per_class=PER_CLASS)

    # Static 1-year Sentinel-era windows (volcanoes are persistent landforms).
    yrng = random.Random(123)
    for r in selected:
        r["year"] = YEARS[yrng.randrange(len(YEARS))]

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

    center_counts = Counter(r["vtype"] for r in selected)
    class_counts = {t: center_counts.get(t, 0) for t in type_to_cid}
    print(f"selected {len(selected)} presence points", flush=True)
    for t in sorted(class_counts, key=lambda c: -class_counts[c]):
        print(f"  {class_counts[t]:5d}  {t}", flush=True)

    io.check_disk()

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
            "num_samples": len(selected),
            "class_counts": class_counts,
            "notes": (
                "GVP summit-point census -> presence-only volcano-type POINTS (converted from "
                "the earlier per-detection GeoTIFF tile encoding; negatives now come from "
                "assembly). Each summit is one presence point carrying its Primary_Volcano_Type "
                "class id in a dataset-wide points.geojson. Holocene (1,196) + Pleistocene "
                "(1,451) layers; 'Unknown'/'None' types dropped. Volcano-type classes only, "
                "ids 0..N-1 by descending frequency, capped at 1000/class. NO dated-eruption "
                "change label: GVP eruption dates are year-resolved at best (spec section 5 "
                "requires ~1-2 month precision), so volcanoes are treated as persistent "
                "landforms with static 1-year Sentinel-era windows (2016-2024). Caveat: a "
                "summit point is a weak proxy for the whole edifice and volcano type is a "
                "full-edifice morphological property."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done:", len(selected), "samples", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
