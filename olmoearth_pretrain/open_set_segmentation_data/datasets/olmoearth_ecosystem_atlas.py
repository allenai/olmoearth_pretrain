"""Process the OlmoEarth Ecosystem Atlas labels into open-set-segmentation point datasets.

Source (local artifact, nothing downloaded):
``/weka/dfive-default/rslearn-eai/artifacts/ecosystem_atlas_labels_20260716.geojson`` -- a
FeatureCollection of Point features (WGS84), each an expert-interpreted sample tagged with an
**IUCN Global Ecosystem Typology Ecosystem Functional Group (EFG)** code at two spatial
resolutions: a 10 m label and a 100 m label. High-quality, globally diverse annotations paired
with (mostly) 2025 imagery.

This produces TWO sparse-point datasets (spec 2a, one dataset-wide points.geojson each):
  - ``olmoearth_ecosystem_atlas_iucn_efg_10m``  -- label = the 10 m EFG code
  - ``olmoearth_ecosystem_atlas_iucn_efg_100m`` -- label = the 100 m EFG code
Points with no code at a given resolution are dropped for that dataset (many features are
un-annotated tasks). Per the data owner these labels are high quality and diverse, so we keep
EVERY coded point -- **no per-class balancing and no 25k cap** (both datasets are already well
under 25k anyway).

Label field coalescing (two naming conventions appear across annotation batches):
  10 m : ``iucn_efg_code_dominant_10m``  else ``iucn_efg_code_10m``
  100 m: ``iucn_efg_code_dominant_100m`` else ``iucn_efg_code_100m``
Code strings also come in two string formats that we normalize to one canonical EFG code so
they merge into the same class:
  Format A: ``"T5.4 Cool deserts and semi-deserts"``      -> ``T5.4``
  Format B: ``"T_5_5_HYPER-ARID_DESERTS"`` / ``"MT_2_1_..."`` -> ``T5.5`` / ``MT2.1``
Sentinels ``Unknown`` and ``Data deficient`` are dropped; ``Open ocean`` (no proper EFG code)
is kept as a single class ``M_OPEN_OCEAN``. Class ids are assigned by descending frequency
(spec 5); the class ``name`` is the canonical EFG code and ``description`` the human-readable
EFG name (harvested from the Format-A strings when available).

Time range: each point's ``start_time`` year (the reference-imagery year; fallback 2025),
as a 1-year window (ecosystem type is a stable/annual property); ``change_time`` = null.
Secondary EFG codes are ignored (dominant/primary only). Per-point ``homogeneity_estimate``,
``confidence_level``, ``sample_id`` and ``project_name`` are carried through as auxiliary
feature properties (not filtered on here).

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_ecosystem_atlas
"""

import argparse
import json
import re
from collections import Counter
from typing import Any

from upath import UPath

from olmoearth_pretrain.open_set_segmentation_data import io, manifest

ARTIFACT = UPath(
    "/weka/dfive-default/rslearn-eai/artifacts/ecosystem_atlas_labels_20260716.geojson"
)
SOURCE_DESC = (
    "OlmoEarth Ecosystem Atlas expert IUCN-EFG point labels; local artifact "
    "ecosystem_atlas_labels_20260716.geojson"
)
DEFAULT_YEAR = 2025

SLUG_10M = "olmoearth_ecosystem_atlas_iucn_efg_10m"
SLUG_100M = "olmoearth_ecosystem_atlas_iucn_efg_100m"

_FMT_A = re.compile(r"^([A-Z]{1,3})\s*([0-9]+)\.([0-9]+)\s*(.*)$")
_FMT_B = re.compile(r"^([A-Z]{1,3})_([0-9]+)_([0-9]+)_?(.*)$")
_OPEN_OCEAN_CODE = "M_OPEN_OCEAN"
_DROP = {"unknown", "data deficient", "data_deficient", "not sure", "not_sure"}


def _normalize(raw: str | None) -> tuple[str, str] | None:
    """Return (canonical_efg_code, display_name) or None to drop.

    Merges the two on-disk string formats to one canonical code. ``Open ocean`` is kept as a
    single marine class; ``Unknown``/``Data deficient`` are dropped.
    """
    if not raw:
        return None
    s = raw.strip()
    low = s.lower()
    if low in _DROP:
        return None
    if low in ("open ocean", "open_ocean"):
        return _OPEN_OCEAN_CODE, "Open ocean"
    m = _FMT_A.match(s)
    if m:
        code = f"{m.group(1)}{m.group(2)}.{m.group(3)}"
        return code, m.group(4).strip()
    m = _FMT_B.match(s)
    if m:
        code = f"{m.group(1)}{m.group(2)}.{m.group(3)}"
        name = m.group(4).replace("_", " ").strip().capitalize()
        return code, name
    return None  # unparseable / not an EFG code -> drop


def _code_10m(p: dict[str, Any]) -> str | None:
    return p.get("iucn_efg_code_dominant_10m") or p.get("iucn_efg_code_10m")


def _code_100m(p: dict[str, Any]) -> str | None:
    return p.get("iucn_efg_code_dominant_100m") or p.get("iucn_efg_code_100m")


def _homogeneity(p: dict[str, Any], res: str) -> Any:
    return p.get(f"homogeneity_estimate_dominant_{res}") or p.get(
        f"homogeneity_estimate_{res}"
    )


def _year(p: dict[str, Any]) -> int:
    st = p.get("start_time") or ""
    if len(st) >= 4 and st[:4].isdigit():
        return int(st[:4])
    return DEFAULT_YEAR


def _build_points(
    feats: list[dict[str, Any]], res: str
) -> tuple[list[dict[str, Any]], list[tuple[int, str, str]]]:
    """Return (point dicts, class table [(id, code, display_name)]) for a resolution.

    ``res`` is "10m" or "100m". Class ids are assigned by descending frequency.
    """
    code_fn = _code_10m if res == "10m" else _code_100m
    # First pass: normalize + frequency + best display name per code.
    freq: Counter = Counter()
    names: dict[str, str] = {}
    norm: list[tuple[dict[str, Any], str]] = []  # (feature, canonical_code)
    for f in feats:
        p = f["properties"]
        got = _normalize(code_fn(p))
        if got is None:
            continue
        code, name = got
        freq[code] += 1
        # Prefer a non-empty, properly-cased Format-A name if one shows up.
        if name and (code not in names or (name and not names[code])):
            names.setdefault(code, name)
            if name and names[code] and name[:1].isupper() and " " in name:
                names[code] = name
        norm.append((f, code))
    ordered = [c for c, _ in freq.most_common()]
    code_to_id = {c: i for i, c in enumerate(ordered)}
    class_table = [(i, c, names.get(c, "")) for i, c in enumerate(ordered)]

    points: list[dict[str, Any]] = []
    for i, (f, code) in enumerate(norm):
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"][:2]
        pt: dict[str, Any] = {
            "id": f"{i:06d}",
            "lon": lon,
            "lat": lat,
            "label": code_to_id[code],
            "time_range": io.year_range(_year(p)),
            "source_id": p.get("sample_id") or p.get("task_name"),
            "efg_code": code,
        }
        hom = _homogeneity(p, res)
        if hom is not None:
            pt["homogeneity_estimate"] = hom
        if p.get("confidence_level_dominant") is not None:
            pt["confidence_level"] = p["confidence_level_dominant"]
        if p.get("project_name"):
            pt["project_name"] = p["project_name"]
        points.append(pt)
    return points, class_table


def _write_dataset(
    slug: str, res: str, points: list[dict[str, Any]], class_table: list[tuple[int, str, str]]
) -> None:
    raw = io.raw_dir(slug)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(f"{SOURCE_DESC}\n{ARTIFACT}\n")
    io.write_points_table(slug, "classification", points)
    counts = Counter(p["label"] for p in points)
    io.write_dataset_metadata(
        slug,
        {
            "dataset": slug,
            "name": f"OlmoEarth Ecosystem Atlas IUCN EFG ({res})",
            "task_type": "classification",
            "source": "olmoearth",
            "license": "internal",
            "provenance": {
                "url": str(ARTIFACT),
                "have_locally": True,
                "annotation_method": "expert visual interpretation (IUCN EFG typology)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": code, "description": name or None}
                for i, code, name in class_table
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(points),
            "class_counts": {code: counts.get(i, 0) for i, code, _ in class_table},
            "notes": (
                f"Sparse IUCN-EFG presence points at {res} resolution from the OlmoEarth "
                "Ecosystem Atlas. Every coded point kept (no class balancing / no 25k cap, "
                "per data owner). 1-year time_range on each point's start_time year "
                "(fallback 2025); change_time=null. Two on-disk code string formats "
                "normalized to canonical EFG codes; Unknown/Data-deficient dropped; Open "
                "ocean kept as M_OPEN_OCEAN. Secondary codes ignored."
            ),
        },
    )
    manifest.write_registry_entry(
        slug, "completed", task_type="classification", num_samples=len(points)
    )
    print(f"{slug}: {len(points)} points, {len(class_table)} classes")


def main() -> None:
    argparse.ArgumentParser().parse_args()
    io.check_disk()
    for slug in (SLUG_10M, SLUG_100M):
        manifest.write_registry_entry(slug, "in_progress")

    with ARTIFACT.open() as f:
        feats = json.load(f)["features"]
    print(f"read {len(feats)} features from {ARTIFACT.name}")

    pts10, table10 = _build_points(feats, "10m")
    _write_dataset(SLUG_10M, "10m", pts10, table10)
    pts100, table100 = _build_points(feats, "100m")
    _write_dataset(SLUG_100M, "100m", pts100, table100)
    print("done")


if __name__ == "__main__":
    main()
