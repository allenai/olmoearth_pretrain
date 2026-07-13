"""Process OlmoEarth HLS Burn Scars into open-set-segmentation label patches.

Source: NASA/IBM HLS Burn Scars (HuggingFace `ibm-nasa-geospatial/hls_burn_scars`),
binary burn-scar segmentation over 512x512 Harmonized Landsat-Sentinel (HLS) 30 m scenes
across the CONUS, 2018-2021, with per-pixel masks derived from MTBS (Monitoring Trends in
Burn Severity). We consume the internally-staged rslearn copy at
`/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/hls_burn_scars/`
(`have_locally: true`), so nothing is downloaded — `raw/{slug}/SOURCE.txt` points at it.

The staged copy already resampled the native 30 m masks to **10 m** (nearest) in local
UTM and split each 512x512 (30 m) scene into a 6x6 grid of 256x256 (10 m) windows named
`HLS_S30_{MGRS}_{YEAR}{DOY}_r{r}_c{c}`. The `label_raster` layer is int16 with values
0 = not burned, 1 = burned, -1 = nodata; each window's metadata.json carries a real UTM
projection at 10 m and a tight (~2-day) acquisition `time_range` around the HLS scene date.

Class scheme (dense per-pixel CLASSIFICATION, matching the manifest's 2 classes; ids
follow the fire-dataset convention, cf. cabuar_california_burned_areas / floga):
    id 0 = unburned   (label == 0, observed non-burnt)
    id 1 = burned     (label == 1, inside the HLS burn-scar mask)
    255  = nodata/ignore  (source -1: unobserved / outside-scene fill)

Processing (label_type = dense_raster): each staged 256x256 (10 m) window is cut into a
4x4 grid of 64x64 (10 m) tiles (the source is already UTM 10 m, so no resampling here).
Sampling is **tiles-per-class balanced** (spec 5): a tile counts toward every class present
in it (>= MIN_CLASS_PX px), rarer class (burned) filled first, up to PER_CLASS tiles/class
under the 25k cap. Tiles that are mostly nodata are skipped.

Time range / change label: a burn scar is a change/event label. The HLS scene is acquired
shortly after the fire (MTBS-derived), so `change_time` is set to the HLS acquisition date
(midpoint of the window's ~2-day acquisition range) and `time_range` is a 360-day window
centered on it (spec 5). The fire ignition falls a few weeks-to-months before the
acquisition, comfortably inside the centered window, so the pairing imagery brackets the
forest->burned transition. (Same convention as cabuar/floga.)

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_hls_burn_scars
"""

import argparse
import json
import multiprocessing
import os
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import rasterio
import tqdm
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest, sampling

SLUG = "olmoearth_hls_burn_scars"
NAME = "OlmoEarth HLS burn scars"

SRC = "/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/hls_burn_scars"
GROUPS = ["train", "val"]

TILE = 64  # output tile edge (px) at 10 m => 640 m
SRC_TILE = 256  # staged window edge (px) at 10 m
GRID = SRC_TILE // TILE  # 4 x 4 sub-tiles per staged window
PER_CLASS = 1000
MIN_CLASS_PX = 32  # a tile counts toward a class only with >= this many px
MAX_NODATA_FRAC = 0.5  # skip tiles that are more than half nodata

SRC_NODATA = -1
UNBURNED, BURNED = 0, 1
CLASSES = [
    (
        "unburned",
        "Not a burn scar: HLS pixel outside the NASA/IBM HLS Burn Scars mask (MTBS-derived) "
        "for the fire captured in this scene, among observed pixels.",
    ),
    (
        "burned",
        "Burn scar: HLS pixel inside the NASA/IBM HLS Burn Scars mask, derived from MTBS "
        "(Monitoring Trends in Burn Severity), i.e. area burned by the fire captured in this "
        "HLS scene.",
    ),
]


def _win_dir(group: str, name: str) -> str:
    return os.path.join(SRC, "windows", group, name)


def _label_tif(group: str, name: str) -> str:
    return os.path.join(
        _win_dir(group, name), "layers", "label_raster", "label", "geotiff.tif"
    )


def _read_label(group: str, name: str) -> np.ndarray:
    """Read a staged 256x256 window label as uint8 (0 unburned / 1 burned / 255 nodata)."""
    with rasterio.open(_label_tif(group, name)) as ds:
        a = ds.read(1)
    out = np.full(a.shape, io.CLASS_NODATA, dtype=np.uint8)
    out[a == UNBURNED] = UNBURNED
    out[a == BURNED] = BURNED
    # source -1 (and anything else) stays nodata (255)
    return out


def _block(label: np.ndarray, ti: int, tj: int) -> np.ndarray:
    return label[ti * TILE : (ti + 1) * TILE, tj * TILE : (tj + 1) * TILE]


# --------------------------------------------------------------------------- scan phase
def _scan_window(group: str, name: str) -> list[dict[str, Any]]:
    """One candidate record per non-mostly-nodata 64x64 sub-tile of a staged window."""
    try:
        label = _read_label(group, name)
    except Exception:
        return []
    total = TILE * TILE
    recs: list[dict[str, Any]] = []
    for ti in range(GRID):
        for tj in range(GRID):
            b = _block(label, ti, tj)
            nod = int((b == io.CLASS_NODATA).sum())
            if nod > MAX_NODATA_FRAC * total:
                continue
            present = [
                c for c in (UNBURNED, BURNED) if int((b == c).sum()) >= MIN_CLASS_PX
            ]
            if not present:
                continue
            recs.append(
                {
                    "group": group,
                    "name": name,
                    "ti": ti,
                    "tj": tj,
                    "classes_present": present,
                }
            )
    return recs


# --------------------------------------------------------------------------- write phase
def _window_geo(
    group: str, name: str
) -> tuple[Projection, list[int], datetime, tuple[datetime, datetime]]:
    """(canonical-UTM projection, staged pixel bounds, change_time, 360-day window).

    The staged CRS is a non-EPSG WGS84-UTM WKT; we re-derive the canonical EPSG UTM
    projection from the window centroid (numerically identical WGS84 UTM at 10 m), keeping
    the staged pixel bounds. change_time = midpoint of the ~2-day HLS acquisition range.
    """
    m = json.load(open(os.path.join(_win_dir(group, name), "metadata.json")))
    wkt = m["projection"]["crs"]
    bounds = m["bounds"]
    lon, lat = io.pixel_center_lonlat(wkt, bounds)
    proj = io.utm_projection_for_lonlat(lon, lat)
    t0 = datetime.fromisoformat(m["time_range"][0])
    t1 = datetime.fromisoformat(m["time_range"][1])
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=UTC)
    if t1.tzinfo is None:
        t1 = t1.replace(tzinfo=UTC)
    change_time = t0 + (t1 - t0) / 2
    tr = (change_time - timedelta(days=180), change_time + timedelta(days=180))
    return proj, bounds, change_time, tr


def _write_window(group: str, name: str, tiles: list[dict[str, Any]]) -> None:
    """Write all selected sub-tiles of one staged window (idempotent)."""
    remaining = [
        t
        for t in tiles
        if not (io.locations_dir(SLUG) / f"{t['sample_id']}.tif").exists()
    ]
    if not remaining:
        return
    proj, bounds, change_time, tr = _window_geo(group, name)
    x_min, y_min = bounds[0], bounds[1]
    label = _read_label(group, name)
    for t in remaining:
        ti, tj = t["ti"], t["tj"]
        b = _block(label, ti, tj)
        c0 = x_min + tj * TILE
        r0 = y_min + ti * TILE
        out_bounds = (c0, r0, c0 + TILE, r0 + TILE)
        io.write_label_geotiff(
            SLUG, t["sample_id"], b, proj, out_bounds, nodata=io.CLASS_NODATA
        )
        present = sorted(int(v) for v in np.unique(b) if v != io.CLASS_NODATA)
        io.write_sample_json(
            SLUG,
            t["sample_id"],
            proj,
            out_bounds,
            tr,
            change_time=change_time,
            source_id=f"{group}/{name}_r{ti}_c{tj}",
            classes_present=present,
        )


def _ensure_raw() -> None:
    RAW = io.raw_dir(SLUG)
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "SOURCE.txt").write_text(
        "OlmoEarth HLS Burn Scars labels: internally-staged rslearn dataset at\n"
        f"{SRC}\n"
        "(have_locally=true; not copied). Public source: HuggingFace\n"
        "ibm-nasa-geospatial/hls_burn_scars (NASA/IBM HLS Burn Scars, MTBS-derived).\n"
        "Only the `label_raster` layer (int16: 0 not-burned, 1 burned, -1 nodata) is used;\n"
        "the co-located HLS imagery bands are ignored (pretraining supplies its own imagery).\n"
        "Windows: windows/{train,val}/HLS_S30_{MGRS}_{YEAR}{DOY}_r{r}_c{c} (256x256 @ 10 m,\n"
        "local UTM). The native 30 m masks were resampled to 10 m (nearest) in the staged copy.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")
    _ensure_raw()

    windows: list[tuple[str, str]] = []
    for g in GROUPS:
        for name in os.listdir(os.path.join(SRC, "windows", g)):
            windows.append((g, name))
    print(f"{len(windows)} staged windows (256x256 @ 10 m) across {GROUPS}")

    print("Scanning windows into 64x64 tiles...")
    scan_args = [{"group": g, "name": n} for g, n in windows]
    all_recs: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan_window, scan_args), total=len(scan_args)
        ):
            all_recs.extend(recs)
    print(f"  {len(all_recs)} candidate tiles")

    selected = sampling.select_tiles_per_class(
        all_recs, classes_key="classes_present", per_class=PER_CLASS
    )
    selected.sort(key=lambda r: (r["group"], r["name"], r["ti"], r["tj"]))
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(
        f"  selected {len(selected)} tiles (tiles-per-class balanced, <= {PER_CLASS}/class)"
    )

    tile_class_counts = {name: 0 for name, _ in CLASSES}
    for r in selected:
        for c in r["classes_present"]:
            tile_class_counts[CLASSES[c][0]] += 1
    print("tiles containing each class:", tile_class_counts)

    by_window: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in selected:
        by_window[(r["group"], r["name"])].append(r)

    io.check_disk()
    print(f"Writing tiles for {len(by_window)} windows...")
    write_args = [
        {"group": g, "name": n, "tiles": ts} for (g, n), ts in by_window.items()
    ]
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_window, write_args), total=len(write_args)
        ):
            pass

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "OlmoEarth (staged rslearn copy of NASA/IBM HLS Burn Scars)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars",
                "staged_path": SRC,
                "have_locally": True,
                "annotation_method": "derived (NASA/IBM HLS Burn Scars, from MTBS)",
            },
            "sensors_relevant": ["sentinel2", "landsat", "sentinel1"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "tile_class_counts": tile_class_counts,
            "notes": (
                "Binary HLS burn-scar masks (MTBS-derived) over the CONUS, 2018-2021, from "
                "the internally-staged rslearn dataset (have_locally). The native 30 m masks "
                "were resampled to 10 m (nearest) in the staged copy and split into 256x256 "
                "local-UTM windows; here each window is cut into 64x64 (10 m) tiles (no "
                "further resampling). Classes: 0 unburned, 1 burned, 255 nodata (source -1). "
                "Tiles-per-class balanced (<=1000/class), burned filled first. Burn is an "
                "event label: change_time = HLS acquisition date (midpoint of the ~2-day "
                "window metadata range), time_range = 360-day window centered on it; the "
                "fire ignition falls a few weeks-to-months before acquisition, inside the "
                "window. CRS re-derived to canonical EPSG UTM from each window centroid "
                "(the staged CRS is a numerically-identical non-EPSG WGS84-UTM WKT)."
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
