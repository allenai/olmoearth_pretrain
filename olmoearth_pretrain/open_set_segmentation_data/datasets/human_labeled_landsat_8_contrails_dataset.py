"""Process the Human-Labeled Landsat-8 Contrails Dataset into contrail segmentation tiles.

Source: Google Research "A human-labeled Landsat-8 contrails dataset" (McCloskey et al.,
ICML Climate Change AI workshop 2021), released CC BY 4.0 at
``gs://landsat_contrails_dataset`` (we use the 2023-01-20 version). The distribution is a
set of 100 JSON-lines shards; each line is ONE Landsat-8 scene:

    {"filename": "LC08_L1TP_036037_20180406_20180417_01_T1_B10.TIF",
     "polygons": [[[x, y], ...], ...],           # human contrail annotations
     "advected_flight_waypoints": {...},          # (ignored) flight context
     "advected_flight_density": [[...]]}          # (ignored) flight context

The ``polygons`` are vertex lists in the pixel grid of the **10x-downsampled** Landsat-8
thermal band that the labelers viewed (see the released notebook: the false-color image is
``gdal ReadAsArray(buf=shape/10)`` of the 30 m band, so 1 downsampled pixel ~= 300 m). The
dataset itself carries no lon/lat, but each scene's georeferencing is recoverable from the
Landsat-8 L1 MTL metadata on the public bucket ``gs://gcp-public-data-landsat`` (UTM zone,
UL corner projection coords, 30 m thermal grid). We convert every polygon vertex
downsampled-pixel -> scene UTM metres -> WGS84 lon/lat, then rasterize into local-UTM 10 m
label tiles like the other polygon datasets (cf. cal_fire_frap_fire_perimeters.py).

Task: **binary contrail segmentation** (label_type dense_raster; single manifest class
"contrail"):

    0 = no_contrail   (observed pixel with no contrail annotation)
    1 = contrail      (inside a human contrail polygon)

Each Landsat scene was exhaustively annotated for contrails, so non-contrail pixels are
real observed negatives (as in cabuar_california_burned_areas), not fabricated ones; we
therefore keep a real background class 0 rather than nodata. nodata 255 is reserved/unused.

Time range: a contrail is a **specific-image** feature valid only at the exact Landsat
overpass (spec §5 specific-image rule), NOT a seasonal/annual label. We set ``time_range``
to a 1-hour window centered on the scene's DATE_ACQUIRED + SCENE_CENTER_TIME (from MTL) and
leave ``change_time`` null (it is a single-instant presence label, not a change event). All
scenes are 2017-2020 (post-2016), so no pre-2016 filtering is needed.

Tiling / sampling: contrails span whole 185 km scenes but the label footprint per tile is
capped at 64x64 @ 10 m (640 m). To maximize spatial/temporal diversity of this global
dataset we take, per scene, up to ``N_PER_SCENE`` of its largest contrail polygons as
candidate tiles (each 64x64 tile centered on a polygon centroid, with ALL of that scene's
contrail polygons rasterized into the tile so neighbouring contrails are labeled too), then
select **round-robin across scenes** (one tile per scene per round) up to
``TARGET_SAMPLES`` (1000, the per-class cap for the single "contrail" class). This yields
~1000 tiles drawn from ~1000 distinct scenes.

Caveat: the annotations were drawn at ~300 m (10x-downsampled 30 m) resolution, so contrail
mask boundaries are coarse (~±300 m) when upsampled to 10 m; the where-mask is valid but
the exact edge is approximate. Contrails are resolvable in Landsat-8 thermal/cirrus bands
(how they were labeled); they pair best with the Landsat modality in pretraining.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.human_labeled_landsat_8_contrails_dataset
Idempotent: existing locations/{id}.tif are skipped; MTLs are cached under raw/mtl/.
"""

import argparse
import glob
import json
import multiprocessing
import os
import random
import re
import urllib.request
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import shapely
import shapely.geometry
import shapely.ops
import shapely.wkb
import tqdm
from pyproj import Transformer
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest

SLUG = "human_labeled_landsat_8_contrails_dataset"
NAME = "Human-Labeled Landsat-8 Contrails Dataset"

BUCKET_VERSION = "2023_01_20_1674247800"
DATA_URL = (
    f"https://storage.googleapis.com/landsat_contrails_dataset/{BUCKET_VERSION}/data"
)
PAPER_URL = "https://research.google/pubs/a-human-labeled-landsat-contrails-dataset/"
LANDSAT_BUCKET = "https://storage.googleapis.com/gcp-public-data-landsat"

SHARDS_DIR = io.raw_dir(SLUG) / "shards"
MTL_DIR = io.raw_dir(SLUG) / "mtl"

DOWNSAMPLE = 10  # labelers viewed the 30 m band downsampled 10x (see notebook)
THERMAL_M = 30.0  # Landsat-8 thermal grid cell size (m)
TILE = 64  # 64x64 @ 10 m = 640 m label tile (hard cap)
N_PER_SCENE = 3  # candidate contrail tiles kept per scene (largest polygons)
TARGET_SAMPLES = 1000  # per-class cap for the single "contrail" class (spec §5)
MIN_POLY_AREA_DS = 1.0  # drop degenerate polygons < 1 downsampled px^2 (~0.09 km^2)
HALF_WINDOW = timedelta(minutes=30)  # +/-30 min => 1-hour specific-image window

NO_CONTRAIL, CONTRAIL = 0, 1
CLASSES = [
    (
        "no_contrail",
        "Observed Landsat-8 pixel with no human contrail annotation. Each scene was "
        "exhaustively labeled for contrails, so out-of-polygon pixels are genuine "
        "non-contrail context (clear sky, natural/other clouds, or surface).",
    ),
    (
        "contrail",
        "Condensation trail: a line-shaped ice cloud produced by aircraft, hand-annotated "
        "as a polygon on the false-color Landsat-8 thermal image (11 um - 12 um brightness "
        "temperature difference, 1.37 um cirrus reflectance, 12 um brightness temperature).",
    ),
]

FN_RE = re.compile(r'"filename"\s*:\s*"([^"]+)"')
MTL_KEYS = re.compile(
    r"^\s*(MAP_PROJECTION|UTM_ZONE|CORNER_UL_PROJECTION_X_PRODUCT|"
    r"CORNER_UL_PROJECTION_Y_PRODUCT|CORNER_UL_LAT_PRODUCT|THERMAL_SAMPLES|"
    r"THERMAL_LINES|GRID_CELL_SIZE_THERMAL|DATE_ACQUIRED|SCENE_CENTER_TIME)\s*=\s*(.+?)\s*$",
    re.M,
)


# ------------------------------------------------------------------- shard scanning


def _scan_shard(path: str) -> list[tuple[str, list]]:
    """Parse a JSON-lines shard, returning (filename, polygons) for positive scenes only.

    Only the prefix of each line (filename + polygons, which come before the large flight
    arrays) is parsed, so we never load gigabytes of flight data.
    """
    out: list[tuple[str, list]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = FN_RE.search(line)
            if not m:
                continue
            fn = m.group(1)
            key = '"polygons":'
            i = line.find(key)
            if i < 0:
                continue
            j = line.find('"advected_flight', i)
            seg = (
                line[i + len(key) : (j if j > 0 else len(line))]
                .rstrip()
                .rstrip(",")
                .strip()
            )
            try:
                polys = json.loads(seg)
            except Exception:
                continue
            if polys:
                out.append((fn, polys))
    return out


# ------------------------------------------------------------------- MTL / georef


def _mtl_paths(filename: str) -> tuple[str, str, str]:
    """Return (scene_id, remote MTL url, local cache path) for a *_B10.TIF filename."""
    scene_id = filename[: -len("_B10.TIF")]
    parts = scene_id.split("_")
    path, row = parts[2][:3], parts[2][3:]
    sensor = parts[0]  # LC08 / LT08
    url = f"{LANDSAT_BUCKET}/{sensor}/01/{path}/{row}/{scene_id}/{scene_id}_MTL.txt"
    local = str(MTL_DIR / f"{scene_id}_MTL.txt")
    return scene_id, url, local


def _fetch_mtl(url: str, local: str, retries: int = 4) -> str | None:
    """Download an MTL to the local cache (atomic, idempotent). None on 404/failure."""
    if os.path.exists(local):
        with open(local) as f:
            return f.read()
    last: Exception | None = None
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                text = r.read().decode("utf-8", "replace")
            tmp = local + ".tmp"
            with open(tmp, "w") as f:
                f.write(text)
            os.rename(tmp, local)
            return text
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last = e
        except Exception as e:  # noqa: BLE001
            last = e
        import time as _t

        _t.sleep(2**a)
    print(f"  MTL fetch failed: {url}: {last}")
    return None


def _parse_mtl(text: str) -> dict[str, Any] | None:
    md: dict[str, str] = {}
    for m in MTL_KEYS.finditer(text):
        md[m.group(1)] = m.group(2).strip().strip('"')
    if md.get("MAP_PROJECTION") != "UTM":
        return None
    try:
        zone = int(float(md["UTM_ZONE"]))
        ul_lat = float(md["CORNER_UL_LAT_PRODUCT"])
        info = {
            "epsg": (32600 if ul_lat >= 0 else 32700) + zone,
            "ul_e": float(md["CORNER_UL_PROJECTION_X_PRODUCT"]),
            "ul_n": float(md["CORNER_UL_PROJECTION_Y_PRODUCT"]),
            "samples": int(float(md["THERMAL_SAMPLES"])),
            "lines": int(float(md["THERMAL_LINES"])),
            "cell": float(md.get("GRID_CELL_SIZE_THERMAL", THERMAL_M)),
            "date": md["DATE_ACQUIRED"],
            "time": md["SCENE_CENTER_TIME"],
        }
    except (KeyError, ValueError):
        return None
    return info


def _acq_time(info: dict[str, Any]) -> datetime:
    """Scene acquisition time (UTC) from DATE_ACQUIRED + SCENE_CENTER_TIME."""
    hh, mm, ss = info["time"].rstrip("Zz").split(":")
    sec = int(float(ss))
    y, mo, d = (int(x) for x in info["date"].split("-"))
    return datetime(y, mo, d, int(hh), int(mm), min(sec, 59), tzinfo=UTC)


def _prep_scene(filename: str, polygons: list) -> dict[str, Any] | None:
    """Fetch/parse MTL and convert all contrail polygons to WGS84 shapely polygons.

    Returns a scene record with the WGS84 MultiPolygon (all contrails) plus per-polygon
    centroids/areas, or None if the scene is unusable (no MTL / non-UTM / no valid polygon).
    """
    scene_id, url, local = _mtl_paths(filename)
    text = _fetch_mtl(url, local)
    if text is None:
        return None
    info = _parse_mtl(text)
    if info is None:
        return None

    ds_w = max(1, int(info["samples"] / DOWNSAMPLE))
    ds_h = max(1, int(info["lines"] / DOWNSAMPLE))
    sx = info["samples"] / ds_w * info["cell"]  # metres per downsampled pixel (x)
    sy = info["lines"] / ds_h * info["cell"]  # metres per downsampled pixel (y)
    ul_e, ul_n = info["ul_e"], info["ul_n"]

    to_wgs = Transformer.from_crs(info["epsg"], 4326, always_xy=True)

    wgs_polys: list[Any] = []
    anchors: list[tuple[float, float]] = []
    areas: list[float] = []
    for poly in polygons:
        if not isinstance(poly, list) or len(poly) < 3:
            continue
        try:
            xy = np.asarray(poly, dtype=float)
        except Exception:
            continue
        if xy.ndim != 2 or xy.shape[1] != 2:
            continue
        # downsampled px -> scene UTM metres
        east = ul_e + xy[:, 0] * sx
        north = ul_n - xy[:, 1] * sy
        ring_utm = shapely.geometry.Polygon(np.column_stack([east, north]))
        area_ds = float(ring_utm.area) / (sx * sy)  # area in downsampled px^2
        if area_ds < MIN_POLY_AREA_DS:
            continue
        lon, lat = to_wgs.transform(east, north)
        g = shapely.geometry.Polygon(np.column_stack([lon, lat]))
        if not g.is_valid:
            g = g.buffer(0)
        if g.is_empty or g.area <= 0:
            continue
        # Anchor tiles on a point ON the contrail boundary (not the interior centroid) so
        # each 640 m tile straddles a contrail edge -> both contrail and background present
        # (centering inside a wide contrail would fill the whole small tile with class 1).
        try:
            ap = g.exterior.interpolate(0.5, normalized=True)
        except Exception:
            ap = g.centroid
        wgs_polys.append(g)
        anchors.append((float(ap.x), float(ap.y)))
        areas.append(area_ds)

    if not wgs_polys:
        return None

    union = shapely.ops.unary_union(wgs_polys)
    acq = _acq_time(info)
    return {
        "scene_id": scene_id,
        "union_wkb": shapely.wkb.dumps(union),
        "anchors": anchors,
        "areas": areas,
        "time": acq.isoformat(),
        "n_polys": len(wgs_polys),
    }


# ------------------------------------------------------------------- tile writing


def _write_one(rec: dict[str, Any]) -> str | None:
    from shapely.geometry import box

    from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
        geom_to_pixels,
        rasterize_shapes,
    )

    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return None

    lon, lat = rec["center"]
    proj = io.utm_projection_for_lonlat(lon, lat)
    union = shapely.wkb.loads(rec["union_wkb"])
    px = geom_to_pixels(union, WGS84_PROJECTION, proj)
    if px.is_empty or px.area <= 0:
        return None

    # centre the tile on the chosen polygon's centroid (in proj pixel coords)
    cpx = geom_to_pixels(shapely.geometry.Point(lon, lat), WGS84_PROJECTION, proj)
    col, row = round(cpx.x), round(cpx.y)
    bounds = io.centered_bounds(col, row, TILE, TILE)

    clip = px.intersection(box(*bounds))
    if clip.is_empty or clip.area <= 0:
        return None
    label = rasterize_shapes(
        [(clip, CONTRAIL)], bounds, fill=NO_CONTRAIL, dtype="uint8", all_touched=True
    )[0]

    t = datetime.fromisoformat(rec["time"])
    time_range = (t - HALF_WINDOW, t + HALF_WINDOW)
    present = sorted(int(v) for v in np.unique(label))

    # proj is already at 10 m/pixel (io.utm_projection_for_lonlat), a local UTM.
    io.write_label_geotiff(SLUG, sample_id, label, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        time_range,
        change_time=None,
        source_id=rec["source_id"],
        classes_present=present,
    )
    return "with_bg" if NO_CONTRAIL in present else "contrail_only"


# ------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")
    MTL_DIR.mkdir(parents=True, exist_ok=True)

    shard_files = sorted(glob.glob(str(SHARDS_DIR / "landsat_contrails.json-*")))
    if not shard_files:
        raise RuntimeError(
            f"no shards under {SHARDS_DIR}; download first (see SOURCE.txt)"
        )

    # ---- Phase A: scan shards for positive scenes (parallel)
    with multiprocessing.Pool(args.workers) as p:
        results = list(
            tqdm.tqdm(
                p.imap_unordered(_scan_shard, shard_files),
                total=len(shard_files),
                desc="scan shards",
            )
        )
    scene_polys: dict[str, list] = {}
    for chunk in results:
        for fn, polys in chunk:
            scene_polys.setdefault(fn, []).extend(polys)
    print(
        f"positive scenes: {len(scene_polys)}; "
        f"total polygons: {sum(len(v) for v in scene_polys.values())}"
    )

    # ---- Phase B: fetch MTLs + build WGS84 polygons per scene (parallel)
    io.check_disk()
    scenes: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for rec in tqdm.tqdm(
            star_imap_unordered(
                p,
                _prep_scene,
                [
                    dict(filename=fn, polygons=polys)
                    for fn, polys in scene_polys.items()
                ],
            ),
            total=len(scene_polys),
            desc="georef scenes",
        ):
            if rec is not None:
                scenes.append(rec)
    print(
        f"georeferenced scenes: {len(scenes)} "
        f"({len(scene_polys) - len(scenes)} dropped: no MTL / non-UTM / no valid polygon)"
    )

    # ---- Phase C: up to N_PER_SCENE largest-polygon candidate tiles per scene
    per_scene: list[list[dict[str, Any]]] = []
    for s in scenes:
        order = sorted(
            range(len(s["areas"])), key=lambda i: s["areas"][i], reverse=True
        )
        cands = []
        for k in order[:N_PER_SCENE]:
            cands.append(
                {
                    "union_wkb": s["union_wkb"],
                    "center": s["anchors"][k],
                    "time": s["time"],
                    "source_id": f"{s['scene_id']}#poly{k}",
                }
            )
        if cands:
            per_scene.append(cands)

    # ---- Phase D: round-robin across scenes (max diversity), cap TARGET_SAMPLES
    rng = random.Random(42)
    per_scene.sort(key=lambda lst: lst[0]["source_id"])  # deterministic base order
    for lst in per_scene:
        rng.shuffle(lst)
    rng.shuffle(per_scene)
    selected: list[dict[str, Any]] = []
    active = [lst for lst in per_scene if lst]
    i = 0
    while active and len(selected) < TARGET_SAMPLES:
        lst = active[i % len(active)]
        selected.append(lst.pop())
        i += 1
        if i % len(active) == 0:
            active = [lst for lst in active if lst]
    for j, r in enumerate(selected):
        r["sample_id"] = f"{j:06d}"
    print(
        f"selected {len(selected)} contrail tiles from {len(per_scene)} scenes "
        f"(cap {TARGET_SAMPLES})"
    )

    # ---- Phase E: write tiles (parallel)
    io.check_disk()
    counts: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write tiles",
        ):
            if res is not None:
                counts[res] += 1

    n_written = len(list(io.locations_dir(SLUG).glob("*.tif")))
    years = Counter(int(r["source_id"][17:21]) for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Google Research (McCloskey et al., ICML CCAI 2021)",
            "license": "CC BY 4.0",
            "provenance": {
                "url": PAPER_URL,
                "data": f"gs://landsat_contrails_dataset/{BUCKET_VERSION}/",
                "have_locally": False,
                "annotation_method": "manual (human pixel-level contrail polygons on "
                "false-color Landsat-8 thermal imagery)",
                "georeferencing": "recovered from Landsat-8 L1 MTL (gs://gcp-public-data-"
                "landsat): downsampled-pixel -> scene UTM -> WGS84 -> local UTM 10 m",
            },
            "sensors_relevant": ["landsat", "sentinel2"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_written,
            "tile_counts": {
                "tiles_with_background": counts.get("with_bg", 0),
                "contrail_only_tiles": counts.get("contrail_only", 0),
            },
            "samples_per_year": dict(sorted(years.items())),
            "is_change_dataset": False,
            "notes": (
                "Binary contrail segmentation (0 no_contrail, 1 contrail) from the Google "
                "Human-Labeled Landsat-8 Contrails Dataset. 64x64 uint8 tiles, local UTM at "
                "10 m; nodata 255 reserved/unused. Contrail is a SPECIFIC-IMAGE feature: "
                "time_range = 1-hour window centered on the Landsat overpass "
                "(DATE_ACQUIRED + SCENE_CENTER_TIME from MTL), change_time null. All scenes "
                "2017-2020 (post-2016). Annotations drawn at ~300 m (10x-downsampled 30 m) "
                "resolution, so mask boundaries are coarse (~+/-300 m) when upsampled to "
                "10 m. One tile per scene centered on a contrail polygon (all of the "
                "scene's polygons rasterized into the tile); round-robin across scenes for "
                f"diversity, capped at {TARGET_SAMPLES} (single-class per-class cap). "
                "Contrails resolvable in Landsat-8 thermal/cirrus bands; pair best with the "
                "Landsat modality."
            ),
        },
    )
    print("tile counts:", dict(counts))
    print("samples per year:", dict(sorted(years.items())))
    print("total tif on disk:", n_written)

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=n_written
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
