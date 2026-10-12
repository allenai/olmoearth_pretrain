"""UNOSAT Conflict Damage Assessments -> open-set-segmentation damage masks.

Source: UNITAR/UNOSAT via the Humanitarian Data Exchange (HDX, https://data.humdata.org/
organization/unosat). UNOSAT experts photo-interpret VHR imagery over conflict zones and
release per-structure damage assessments (points/polygons) with a damage severity class
(Destroyed / Severe / Moderate / Possible Damage) and the analysis sensor date(s). HDX is
open access; no credentials required.

Why classification (not a change label): UNOSAT comprehensive/cumulative assessments compare
a post-event image to a baseline that is often 1-3 years earlier, so *when* within that span
a given structure was damaged is not resolvable to ~1-2 months -> a dated change label would
be misaligned (spec S5 change-timing rule). BUT destroyed / heavily-damaged structures are a
**persistent post-change state**: rubble stays visible for years in these zones (no near-term
reconstruction). Per spec S5 we therefore recast this as **presence/state classification** with
``change_time=null`` and a static 1-year window anchored *forward* on the assessment date (so the
paired imagery is guaranteed to post-date the damage and show the persistent state).

Resolution handling: individual buildings are ~1 pixel at 10 m, so per spec (manifest note
"aggregate to heavily-damaged zones") we do not emit 1x1 point labels. Instead we bin all
damaged structures onto the 10 m UTM grid and cut 64x64 tiles over areas that contain a
cluster of damage (>= MIN_DAMAGE_PER_TILE structures). Each labeled pixel carries the most
severe damage class of the structures falling in it; non-damage pixels are nodata (255) --
this is a positive-only dataset (spec S5), assembly supplies negatives from other datasets.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.unosat_conflict_damage_assessments
"""

import argparse
import json
import multiprocessing
import os
import urllib.request
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from rasterio.crs import CRS
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest

SLUG = "unosat_conflict_damage_assessments"

TILE = 64
MIN_DAMAGE_PER_TILE = 3
PER_CLASS = 1000
WINDOW_DAYS = 360
MIN_YEAR = 2016  # Sentinel era; drop features whose latest assessment predates this.

# Unified 4-class damage scheme (manifest order): id 0..3, most-severe first.
CLASSES = [
    (
        "destroyed",
        "Structure collapsed / largely reduced to rubble; footprint no longer intact.",
    ),
    (
        "severely damaged",
        "Major structural damage (partial collapse, roof/walls gone) but footprint partly standing.",
    ),
    (
        "moderately damaged",
        "Visible partial damage (roof holes, blast damage) with structure largely standing.",
    ),
    (
        "possibly damaged",
        "Possible / uncertain damage flagged by the analyst (lower confidence).",
    ),
]

# UNOSAT numeric damage-class domain -> our ids. 1=Destroyed .. 4=Possible. Codes 5 (No
# Visible Damage), 6 (Not Affected), 11 (Impact Crater), etc. are not building-damage
# classes and are dropped.
CODE_MAP = {1: 0, 2: 1, 3: 2, 4: 3}
TEXT_MAP = {
    "destroyed": 0,
    "severe damage": 1,
    "severely damaged": 1,
    "severe": 1,
    "moderate damage": 2,
    "moderately damaged": 2,
    "moderate": 2,
    "possible damage": 3,
    "possibly damaged": 3,
    "possible": 3,
}

# Curated set of open (HDX) UNOSAT conflict damage packages with per-structure geodata,
# covering Gaza, Ukraine, and Syria (Iraq products are pre-2016 -> excluded). Each entry is
# an HDX package (dataset) name. The Gaza package is the whole-strip comprehensive
# assessment; the Syria "Hama" package's zip actually bundles all Syria CDA-2016 cities
# (Damascus, Daraa, Deir-ez-Zor, Hama, Homs, Idlib, Raqqa, Aleppo); the Sumy package bundles
# Sumy + Kharkiv.
PACKAGES = [
    ("unosat-gaza-strip-comprehensive-damage-assessment-11-october-2025", "gaza"),
    (
        "mariupol-updated-building-damage-assessment-overview-map-livoberezhnyi-and-zhovtnevyi-dist",
        "ukraine",
    ),
    ("sumy-rapid-damage-assessment-overview-map", "ukraine"),
    ("kremenchuk-damage-assessment-overview", "ukraine"),
    ("damage-assessement-of-hama-hama-governorate-syria", "syria"),
]

HDX_API = "https://data.humdata.org/api/3/action/package_show?id="
UA = {"User-Agent": "Mozilla/5.0 (olmoearth-open-set-seg)"}


# ---------------------------------------------------------------------------
# Field detection + value mapping (handles heterogeneous UNOSAT schemas)
# ---------------------------------------------------------------------------
def _is_damage_col(name: str) -> bool:
    n = name.lower()
    if any(
        x in n
        for x in (
            "grp",
            "group",
            "sts",
            "status",
            "_sta",
            "densit",
            "percent",
            "conf",
            "sens",
            "analyst",
            "fieldval",
        )
    ):
        return False
    if n.startswith("dmgcls") or n.startswith("dmg_cls"):
        return True
    if "main_d" in n:  # Main_Damage_Site_Class*, d_Main_Dam, d_Main_D_1, Main_Damag*
        return True
    if "damage" in n and "clas" in n:
        return True
    return False


def _is_date_col(name: str) -> bool:
    n = name.lower()
    return "sens" in n and ("da" in n or "dt" in n)


def _map_value(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if not s:
            return None
        if s in TEXT_MAP:
            return TEXT_MAP[s]
        if s.isdigit():
            return CODE_MAP.get(int(s))
        return None
    try:
        iv = int(v)
    except (ValueError, TypeError):
        return None
    return CODE_MAP.get(iv)


# ---------------------------------------------------------------------------
# Download + read
# ---------------------------------------------------------------------------
def _hdx_package(name: str) -> dict[str, Any]:
    with urllib.request.urlopen(
        urllib.request.Request(HDX_API + name, headers=UA), timeout=120
    ) as r:
        return json.load(r)["result"]


def _pick_resources(pkg: dict[str, Any]) -> list[tuple[str, str]]:
    """Return every SHP/GDB zip resource (url, filename).

    UNOSAT packages sometimes split content across resources (e.g. a partial polygon SHP
    plus the authoritative point SHP/GDB). We download all vector zips and dedup records by
    coordinate downstream, so we never miss the full-damage layer.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for res in pkg.get("resources", []):
        fmt = (res.get("format") or "").lower()
        url = res.get("url") or ""
        if not url.lower().endswith(".zip") or fmt not in ("shp", "geodatabase", "gdb"):
            continue
        fname = os.path.basename(url.split("?")[0])
        if fname in seen:
            continue
        seen.add(fname)
        out.append((url, fname))
    if not out:
        raise RuntimeError(f"no SHP/GDB zip resource in package {pkg.get('name')}")
    return out


def _download_extract(name: str) -> list[str]:
    """Download all vector zips of a package into raw_dir and extract; return extract dirs."""
    pkg = _hdx_package(name)
    raw = io.raw_dir(SLUG) / name
    raw.mkdir(parents=True, exist_ok=True)
    dirs = []
    for url, fname in _pick_resources(pkg):
        zpath = raw / fname
        try:
            download.download_http(url, zpath, headers=UA)
        except Exception as e:
            print(f"  WARN download failed {fname}: {e}")
            continue
        ext = raw / (fname + ".d")
        try:
            download.extract_zip(zpath, ext)
        except Exception as e:
            print(f"  WARN extract failed {fname}: {e}")
            continue
        dirs.append(str(ext))
    return dirs


def _iter_layer_frames(ext_dir: str):
    """Yield geopandas frames for every readable vector layer under ext_dir."""
    import geopandas as gpd
    import pyogrio

    for root, dirs, files in os.walk(ext_dir):
        if root.endswith(".gdb"):
            dirs[:] = []  # don't descend into the gdb internals
            try:
                layers = pyogrio.list_layers(root)
            except Exception:
                continue
            for lname in layers[:, 0]:
                try:
                    yield gpd.read_file(root, layer=lname)
                except Exception:
                    continue
            continue
        for f in files:
            if f.lower().endswith(".shp"):
                try:
                    yield gpd.read_file(os.path.join(root, f))
                except Exception:
                    continue


def _read_records(name: str, region: str) -> list[dict[str, Any]]:
    """Read all damage layers of a package -> deduped per-structure records.

    Each record: lon, lat (WGS84), cls (0..3), year (latest assessment), region, source.
    """
    import geopandas as gpd
    import pandas as pd

    seen: dict[tuple[int, int], dict[str, Any]] = {}
    frames = (g for ext in _download_extract(name) for g in _iter_layer_frames(ext))
    for g in frames:
        if g is None or len(g) == 0 or g.crs is None:
            continue
        dmg_cols = [c for c in g.columns if c != "geometry" and _is_damage_col(c)]
        if not dmg_cols:
            continue
        # last VALID mapped id per row (later columns are often 0/placeholder)
        ids = np.full(len(g), -1, dtype=np.int16)
        for c in dmg_cols:
            for i, v in enumerate(g[c].tolist()):
                m = _map_value(v)
                if m is not None:
                    ids[i] = m
        valid = ids >= 0
        if not valid.any():
            continue
        # latest assessment date per row
        date_cols = [c for c in g.columns if c != "geometry" and _is_date_col(c)]
        if date_cols:
            dts = g[date_cols].apply(
                lambda col: pd.to_datetime(col, errors="coerce", utc=True)
            )
            latest = dts.max(axis=1)
        else:
            latest = pd.Series([pd.NaT] * len(g))
        # representative point (inside each geometry) in native CRS, then -> WGS84 lon/lat
        try:
            reps = g.geometry.representative_point()
            g4 = gpd.GeoSeries(reps, crs=g.crs).to_crs(4326)
        except Exception:
            continue
        cx = g4.x.to_numpy()
        cy = g4.y.to_numpy()
        for i in np.nonzero(valid)[0]:
            lon, lat = float(cx[i]), float(cy[i])
            if not (np.isfinite(lon) and np.isfinite(lat)):
                continue
            ts = latest.iloc[i]
            year = int(ts.year) if ts is not None and not pd.isna(ts) else None
            if year is None or year < MIN_YEAR:
                continue
            key = (int(round(lon * 1e5)), int(round(lat * 1e5)))
            cls = int(ids[i])
            prev = seen.get(key)
            if prev is None or cls < prev["cls"]:  # keep most severe at a location
                seen[key] = {
                    "lon": lon,
                    "lat": lat,
                    "cls": cls,
                    "year": year,
                    "region": region,
                    "source": name,
                }
    return list(seen.values())


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------
def _utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _build_tiles(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bin records onto 10 m UTM grid and cut 64x64 damage-mask tiles."""
    import geopandas as gpd
    import pandas as pd

    df = pd.DataFrame(records)
    df["epsg"] = [_utm_epsg(lo, la) for lo, la in zip(df.lon, df.lat)]
    tiles: dict[tuple, dict[str, Any]] = {}
    for epsg, sub in df.groupby("epsg"):
        gser = gpd.GeoSeries(gpd.points_from_xy(sub.lon, sub.lat), crs=4326).to_crs(
            int(epsg)
        )
        cols = np.floor(gser.x.to_numpy() / io.RESOLUTION).astype(np.int64)
        rows = np.floor(-gser.y.to_numpy() / io.RESOLUTION).astype(np.int64)
        cls = sub.cls.to_numpy()
        yr = sub.year.to_numpy()
        reg = sub.region.to_numpy()
        tcol = np.floor_divide(cols, TILE)
        trow = np.floor_divide(rows, TILE)
        for i in range(len(sub)):
            key = (int(epsg), int(tcol[i]), int(trow[i]))
            t = tiles.get(key)
            if t is None:
                t = {
                    "epsg": int(epsg),
                    "tcol": int(tcol[i]),
                    "trow": int(trow[i]),
                    "arr": np.full((TILE, TILE), io.CLASS_NODATA, dtype=np.uint8),
                    "n": 0,
                    "year": 0,
                    "regions": set(),
                }
                tiles[key] = t
            lr = int(rows[i] - t["trow"] * TILE)
            lc = int(cols[i] - t["tcol"] * TILE)
            if 0 <= lr < TILE and 0 <= lc < TILE:
                t["arr"][lr, lc] = min(
                    int(t["arr"][lr, lc]), int(cls[i])
                )  # most severe
                t["n"] += 1
                t["year"] = max(t["year"], int(yr[i]))
                t["regions"].add(str(reg[i]))
    out = []
    for t in tiles.values():
        if t["n"] < MIN_DAMAGE_PER_TILE:
            continue
        present = sorted(int(v) for v in np.unique(t["arr"]) if v != io.CLASS_NODATA)
        if not present:
            continue
        t["classes_present"] = present
        out.append(t)
    return out


def _forward_window(year: int) -> tuple[datetime, datetime]:
    """1-year window anchored forward on the assessment year (mid-year start)."""
    start = datetime(year, 7, 1, tzinfo=UTC)
    return (start, start + timedelta(days=WINDOW_DAYS))


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
def _write_one(
    sample_id: str,
    epsg: int,
    tcol: int,
    trow: int,
    arr: np.ndarray,
    year: int,
    regions: str,
    classes_present: list[int],
) -> int:
    from rslearn.utils.geometry import Projection

    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return 0
    proj = Projection(CRS.from_epsg(epsg), io.RESOLUTION, -io.RESOLUTION)
    x0, y0 = tcol * TILE, trow * TILE
    bounds = (x0, y0, x0 + TILE, y0 + TILE)
    start, end = _forward_window(year)
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        (start, end),
        change_time=None,
        source_id=f"{regions}:{epsg}/{tcol}_{trow}",
        classes_present=classes_present,
    )
    return 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--probe", action="store_true", help="read+report only, no writes"
    )
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    all_records: list[dict[str, Any]] = []
    for name, region in PACKAGES:
        recs = _read_records(name, region)
        print(f"{region:8s} {name[:55]:55s} -> {len(recs):7d} damage structures")
        all_records.extend(recs)
    print(f"total damage structures (post-{MIN_YEAR}): {len(all_records)}")
    reg_counts = Counter(r["region"] for r in all_records)
    cls_counts = Counter(r["cls"] for r in all_records)
    print("  by region:", dict(reg_counts))
    print("  by class :", {CLASSES[c][0]: n for c, n in sorted(cls_counts.items())})

    tiles = _build_tiles(all_records)
    print(f"candidate tiles (>= {MIN_DAMAGE_PER_TILE} structures): {len(tiles)}")

    from olmoearth_pretrain.open_set_segmentation_data.sampling import (
        select_tiles_per_class,
    )

    selected = select_tiles_per_class(
        tiles, classes_key="classes_present", per_class=PER_CLASS
    )
    print(f"selected tiles: {len(selected)}")
    tile_cls_counts: Counter = Counter()
    tile_reg_counts: Counter = Counter()
    for t in selected:
        for c in t["classes_present"]:
            tile_cls_counts[c] += 1
        for rgn in t["regions"]:
            tile_reg_counts[rgn] += 1
    print(
        "  tiles-per-class:",
        {CLASSES[c][0]: n for c, n in sorted(tile_cls_counts.items())},
    )
    print("  tiles-per-region:", dict(tile_reg_counts))

    if args.probe:
        print("probe only; exiting before writes")
        return

    jobs = []
    for i, t in enumerate(selected):
        jobs.append(
            {
                "sample_id": f"{i:06d}",
                "epsg": t["epsg"],
                "tcol": t["tcol"],
                "trow": t["trow"],
                "arr": t["arr"],
                "year": t["year"],
                "regions": "+".join(sorted(t["regions"])),
                "classes_present": t["classes_present"],
            }
        )
    written = 0
    with multiprocessing.Pool(args.workers) as p:
        for n in star_imap_unordered(p, _write_one, jobs):
            written += n
    print(f"wrote {written} new tiles ({len(jobs)} total selected)")

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "UNOSAT Conflict Damage Assessments",
            "task_type": "classification",
            "source": "UNITAR/UNOSAT (HDX)",
            "license": "open (HDX)",
            "provenance": {
                "url": "https://data.humdata.org/organization/unosat",
                "have_locally": False,
                "annotation_method": "expert VHR photo-interpretation",
                "packages": [n for n, _ in PACKAGES],
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(jobs),
            "tiles_per_class": {
                CLASSES[c][0]: n for c, n in sorted(tile_cls_counts.items())
            },
            "tiles_per_region": dict(tile_reg_counts),
            "notes": (
                "64x64 UTM 10 m damage-severity masks aggregated from per-structure UNOSAT "
                "assessments (most-severe class per pixel; non-damage = nodata 255; "
                "positive-only). Recast as persistent-state classification: change_time=null, "
                "1-year window anchored forward on the assessment year (spec S5). Only post-"
                f"{MIN_YEAR} assessments kept; Iraq (pre-2016) excluded."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(jobs)
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
