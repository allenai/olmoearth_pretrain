"""Process INPE DETER-B near-real-time deforestation & degradation alerts.

Source: INPE TerraBrasilis DETER-B, served as WFS from the TerraBrasilis GeoServer
(https://terrabrasilis.dpi.inpe.br/geoserver/ows). DETER is a daily/near-real-time
alert system in which analysts photointerpret medium-resolution imagery (CBERS-4/4A
AWFI/WPM, Amazonia-1 WFI, and others) and hand-digitize polygons of newly detected
forest change, each tagged with a change class (``classname``) and an observation date
(``view_date``). Two biome layers are used:

* ``deter-amz:deter_amz``      -- Legal Amazon (all classes; ~451k polygons, 2016-08+).
* ``deter-cerrado-nb:deter_cerrado`` -- Cerrado (clearcut only; ~129k polygons, 2018-05+).

(No standalone DETER Pantanal layer is published on the GeoServer; Pantanal has PRODES
but not DETER, so this dataset covers Amazon + Cerrado.)

These are dated CHANGE/EVENT labels, so we use the change_time scheme (spec 5): each
sample's ``change_time`` is the alert ``view_date``, which splits the sample into two
adjacent six-month windows (via ``io.pre_post_time_ranges``): ``pre_time_range`` = the
~6 months (<=183 days) immediately before the alert and ``post_time_range`` = the ~6 months
(<=183 days) immediately after, with ``time_range`` = null. The label is a **mask of the
alert polygon** (spec: one polygon -> one tile). Deforestation/degradation/fire/mining
change persists in the imagery, so the pre/post split around the alert is well-posed;
pretraining pairs the "before" image stack with the "after" stack and probes on their
difference.

Encoding: for each selected alert polygon, center a 64x64 UTM 10 m tile on the polygon
centroid, rasterize that polygon as its class id (all_touched so small polygons register),
background (0) elsewhere. Large polygons (many exceed 640 m -- see summary) are cropped to
the central 640 m; the resulting mask ("change occurred here") is still valid. Only the
target polygon is drawn; any co-located alert of another date/class is left as background.

Class scheme (unified, uint8):
  0 background            (no detected change in-tile)
  1 clearcut             (DESMATAMENTO_CR, both biomes)
  2 deforestation_with_vegetation (DESMATAMENTO_VEG)
  3 degradation          (DEGRADACAO)
  4 selective_logging    (CS_DESORDENADO + CS_GEOMETRICO + CORTE_SELETIVO)
  5 mining               (MINERACAO)
  6 fire_scar            (CICATRIZ_DE_QUEIMADA)

Up to 1000 tiles per alert class (spec 5), sampled across years for temporal diversity.
"""

import argparse
import json
import multiprocessing
import random
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.mp import star_imap_unordered
from shapely.geometry import shape

from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)

SLUG = "deter_b_near_real_time_deforestation_degradation_alerts"
NAME = "DETER-B (near-real-time deforestation & degradation alerts)"
URL = "https://terrabrasilis.dpi.inpe.br/downloads/"
WFS = "https://terrabrasilis.dpi.inpe.br/geoserver/ows"

TILE = 64  # 64 px @ 10 m = 640 m
BACKGROUND_ID = 0
TARGET_PER_CLASS = 1000
FETCH_PER_QUERY_YEAR = 300  # candidates fetched per (layer, rawclass, year)
YEARS = list(range(2016, 2026))
SEED = 42
WINDOW_HALF_DAYS = 180  # +/- 180 d -> 360-day time range centered on the alert

# (layer, raw classname) -> unified class id.
QUERIES: list[tuple[str, str, int]] = [
    ("deter-amz:deter_amz", "DESMATAMENTO_CR", 1),
    ("deter-cerrado-nb:deter_cerrado", "DESMATAMENTO_CR", 1),
    ("deter-amz:deter_amz", "DESMATAMENTO_VEG", 2),
    ("deter-amz:deter_amz", "DEGRADACAO", 3),
    ("deter-amz:deter_amz", "CS_DESORDENADO", 4),
    ("deter-amz:deter_amz", "CS_GEOMETRICO", 4),
    ("deter-amz:deter_amz", "CORTE_SELETIVO", 4),
    ("deter-amz:deter_amz", "MINERACAO", 5),
    ("deter-amz:deter_amz", "CICATRIZ_DE_QUEIMADA", 6),
]

CLASS_DEFS = [
    (
        0,
        "background",
        "No detected DETER alert within the tile (unchanged forest, water, or other cover).",
    ),
    (
        1,
        "clearcut",
        "Clear-cut deforestation (DESMATAMENTO_CR / corte raso): complete removal of forest, soil fully exposed. Amazon and Cerrado biomes.",
    ),
    (
        2,
        "deforestation_with_vegetation",
        "Deforestation with residual vegetation (DESMATAMENTO_VEG): forest removal in progress where some vegetation/debris remains on the ground.",
    ),
    (
        3,
        "degradation",
        "Forest degradation (DEGRADACAO): opening of the canopy by repeated logging/fire without complete clearance, canopy still partly present.",
    ),
    (
        4,
        "selective_logging",
        "Selective logging (CS_DESORDENADO disordered + CS_GEOMETRICO geometric + CORTE_SELETIVO): extraction of high-value timber leaving a disturbed but forested matrix.",
    ),
    (
        5,
        "mining",
        "Mining (MINERACAO): bare-earth scars and tailings ponds from (often illegal) alluvial/open-pit mineral extraction.",
    ),
    (
        6,
        "fire_scar",
        "Fire scar (CICATRIZ_DE_QUEIMADA): area burned by wildfire, visible as a darkened/charred surface.",
    ),
]


def _layer_key(layer: str) -> str:
    return layer.split(":")[-1]


def _raw_path(layer: str, rawclass: str, year: int):
    return io.raw_dir(SLUG) / f"{_layer_key(layer)}__{rawclass}__{year}.geojson"


def _fetch(layer: str, rawclass: str, year: int) -> dict[str, Any]:
    cql = (
        f"classname='{rawclass}' AND view_date DURING "
        f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z"
    )
    q = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": layer,
        "count": str(FETCH_PER_QUERY_YEAR),
        "outputFormat": "application/json",
        "srsName": "EPSG:4326",
        "CQL_FILTER": cql,
    }
    url = WFS + "?" + urllib.parse.urlencode(q)
    last = None
    for _ in range(4):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"WFS fetch failed for {layer}/{rawclass}/{year}: {last}")


def download_all() -> None:
    """Fetch candidate GeoJSONs per (layer, rawclass, year) to raw/ (idempotent)."""
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    for layer, rawclass, _cid in QUERIES:
        for year in YEARS:
            dst = _raw_path(layer, rawclass, year)
            if dst.exists():
                continue
            data = _fetch(layer, rawclass, year)
            tmp = dst.parent / (dst.name + ".tmp")
            with tmp.open("w") as f:
                json.dump(data, f)
            tmp.rename(dst)
    with (io.raw_dir(SLUG) / "SOURCE.txt").open("w") as f:
        f.write(
            "INPE TerraBrasilis DETER-B near-real-time deforestation/degradation alerts.\n"
            f"{URL}\n"
            f"WFS: {WFS}\n"
            "Layers: deter-amz:deter_amz, deter-cerrado-nb:deter_cerrado\n"
            "Fetched per (layer, classname, year); srsName=EPSG:4326.\n"
        )


def _load_candidates() -> dict[int, list[dict[str, Any]]]:
    """Load raw GeoJSONs -> {class_id: [record, ...]} (records carry WKB geom + date)."""
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for layer, rawclass, cid in QUERIES:
        for year in YEARS:
            p = _raw_path(layer, rawclass, year)
            if not p.exists():
                continue
            with p.open() as f:
                data = json.load(f)
            for feat in data.get("features", []):
                geom = feat.get("geometry")
                vd = feat["properties"].get("view_date")
                if geom is None or not vd:
                    continue
                try:
                    g = shape(geom)
                except Exception:  # noqa: BLE001
                    continue
                if g.is_empty:
                    continue
                if not g.is_valid:
                    g = g.buffer(0)
                    if g.is_empty or not g.is_valid:
                        continue
                by_class[cid].append(
                    {
                        "wkb": shapely.to_wkb(g),
                        "class_id": cid,
                        "view_date": vd,
                        "year": int(vd[:4]),
                        "gid": feat.get("id", ""),
                        "layer": _layer_key(layer),
                        "rawclass": rawclass,
                    }
                )
    return by_class


def _sample_per_class(
    by_class: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Pick <= TARGET_PER_CLASS records per class, spread across years."""
    rng = random.Random(SEED)
    chosen: list[dict[str, Any]] = []
    for cid, recs in sorted(by_class.items()):
        by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in recs:
            by_year[r["year"]].append(r)
        for yr in by_year:
            rng.shuffle(by_year[yr])
        # Round-robin across years until we hit the target or exhaust candidates.
        picked: list[dict[str, Any]] = []
        years = sorted(by_year)
        idx = {yr: 0 for yr in years}
        while len(picked) < TARGET_PER_CLASS:
            progressed = False
            for yr in years:
                if idx[yr] < len(by_year[yr]):
                    picked.append(by_year[yr][idx[yr]])
                    idx[yr] += 1
                    progressed = True
                    if len(picked) >= TARGET_PER_CLASS:
                        break
            if not progressed:
                break
        chosen.extend(picked)
    return chosen


def _write_tile(rec: dict[str, Any]) -> int:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return rec["class_id"]
    g = shapely.from_wkb(rec["wkb"])
    cid = rec["class_id"]
    c = g.centroid
    proj, col, row = io.lonlat_to_utm_pixel(c.x, c.y)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    px = geom_to_pixels(g, WGS84_PROJECTION, proj)
    if px.is_empty:
        return -1
    if not px.is_valid:
        px = px.buffer(0)
        if px.is_empty or not px.is_valid:
            return -1
    arr = rasterize_shapes(
        [(px, cid)], bounds, fill=BACKGROUND_ID, dtype="uint8", all_touched=True
    )
    if int(arr.max()) != cid:
        return -1  # polygon fell outside the centered tile (shouldn't happen); skip
    ct = datetime.strptime(rec["view_date"], "%Y-%m-%d").replace(tzinfo=UTC)
    pre_range, post_range = io.pre_post_time_ranges(ct)
    tr = (pre_range[0], post_range[1])  # outer bounding span
    present = [BACKGROUND_ID, cid] if int((arr == BACKGROUND_ID).sum()) else [cid]
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        tr,
        change_time=ct,
        source_id=f"{rec['layer']}:{rec['rawclass']}:{rec['gid']}",
        classes_present=present,
        pre_time_range=pre_range,
        post_time_range=post_range,
    )
    return cid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    from olmoearth_pretrain.open_set_segmentation_data import manifest

    manifest.write_registry_entry(SLUG, "in_progress")
    io.check_disk()
    download_all()
    io.check_disk()

    by_class = _load_candidates()
    print(
        "candidates per class: "
        + ", ".join(f"{cid}:{len(v)}" for cid, v in sorted(by_class.items())),
        flush=True,
    )
    chosen = _sample_per_class(by_class)
    for i, r in enumerate(chosen):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(chosen)} tiles (<= {TARGET_PER_CLASS}/class)", flush=True)

    io.check_disk()
    with multiprocessing.Pool(args.workers) as p:
        results = list(
            tqdm.tqdm(
                star_imap_unordered(p, _write_tile, [dict(rec=r) for r in chosen]),
                total=len(chosen),
                desc="tiles",
            )
        )

    written = Counter(r for r in results if r >= 0)
    n_degenerate = sum(1 for r in results if r == -1)
    id_to_name = {cid: name for cid, name, _ in CLASS_DEFS}
    class_counts = {
        id_to_name[cid]: written.get(cid, 0) for cid, _, _ in CLASS_DEFS if cid != 0
    }
    total = int(sum(written.values()))

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "INPE TerraBrasilis (DETER-B)",
            "license": "CC-BY-SA-4.0",
            "provenance": {
                "url": URL,
                "have_locally": False,
                "annotation_method": "manual photointerpretation (INPE analysts)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": cid, "name": name, "description": desc}
                for cid, name, desc in CLASS_DEFS
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": total,
            "class_counts": class_counts,
            "tile_size": TILE,
            "change_time_scheme": True,
            "time_range_days": 2 * WINDOW_HALF_DAYS,
            "notes": (
                "64x64 UTM 10 m tiles centered on each alert polygon centroid; the polygon is "
                "rasterized (all_touched) as its class id, background=0 elsewhere. Dated change "
                "labels: change_time=view_date, time_range=+/-180 d (360 d) centered on it. "
                "Classes unify DETER classnames: clearcut=DESMATAMENTO_CR (Amazon+Cerrado); "
                "deforestation_with_vegetation=DESMATAMENTO_VEG; degradation=DEGRADACAO; "
                "selective_logging=CS_DESORDENADO+CS_GEOMETRICO+CORTE_SELETIVO; mining=MINERACAO; "
                "fire_scar=CICATRIZ_DE_QUEIMADA. Only the target polygon is drawn per tile "
                "(co-located alerts of other dates left as background). Many polygons exceed "
                "640 m and are cropped to the central 640 m tile. Source CRS EPSG:4674 (SIRGAS "
                "2000), reprojected to EPSG:4326 by the WFS then to local UTM. "
                f"{n_degenerate} tiles dropped as degenerate."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=total
    )
    print(
        f"done: {total} tiles; class_counts={class_counts}; dropped {n_degenerate}",
        flush=True,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
