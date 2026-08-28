"""Process S1S2-Water into open-set-segmentation label patches.

Source: S1S2-Water (Wieland et al. 2024, IEEE JSTARS), Zenodo record 11278238
(https://zenodo.org/records/11278238). A global dataset for semantic segmentation
of water bodies from Sentinel-1 and Sentinel-2. It provides 65 globally distributed
(29 countries) ~100x100 km scenes, each a Sentinel-1 / Sentinel-2 pair with
quality-checked **binary water masks**. All 65 scenes are non-flood (permanent /
static water; the release's ``flood`` flag is False for every scene).

Per scene the archive stores (as cloud-optimized GeoTIFFs, in per-scene STAC items):
  * ``s2_msk`` (uint8): Sentinel-2-derived binary water mask, native **UTM, 10 m/px**,
    10980x10980. 0 = no-water, 1 = water.
  * ``s2_valid`` (uint8): S2 validity mask (1 = valid pixel, 0 = invalid/nodata).
  * ``s1_msk`` / ``s1_valid``: the S1 counterpart (9 m/px) -- NOT used here.
  * ``s2_img`` / ``s1_img`` / DEM: imagery + Copernicus DEM (elevation, slope) -- not used.

We use the **S2 mask** because it is already in the target projection/resolution
(local UTM at 10 m), so no reprojection is needed. It is remapped to the manifest's
2-class scheme (dense per-pixel CLASSIFICATION):
    id 0 = water       (s2_msk == 1)
    id 1 = no-water    (s2_msk == 0)
    255  = nodata      (s2_valid == 0, i.e. outside the valid swath / no data)

Processing (label_type = dense_raster): each 10980x10980 UTM 10 m scene is cut,
without reprojection, into 64x64 tiles aligned to the source grid. Sampling is
**tiles-per-class balanced** (spec 5): a tile counts toward every class present in it
(>= MIN_CLASS_PX pixels); the rare class (water) is filled first up to PER_CLASS
tiles. No-water co-occurs in nearly every water tile; a small deterministic sample of
land-only tiles per scene is also emitted so no-water can reach its target even if some
water tiles are pure water.

Time range: each scene has a Sentinel-2 acquisition date (parsed from the source
product id in the STAC ``properties.s2_srcids``; the STAC ``datetime`` field is a
placeholder 2020-01-01). The masks are static water (not a dated event), so we set no
``change_time`` and use a 1-year window centered on the S2 acquisition date (spec 5,
seasonal/annual). All acquisition dates are 2018-2020 (post-2016 / Sentinel era).

Download: only the S2 mask, S2 validity mask and per-scene meta.json are pulled
(via HTTP range requests against the Zenodo zip parts using ``remotezip``); the large
imagery/DEM assets (~163 GB across the 6 zip parts) are skipped -- the needed files are
~2.3 MB/scene (~150 MB total).

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.s1s2_water
"""

import argparse
import json
import multiprocessing
import random
import re
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import rasterio
import tqdm
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io

SLUG = "s1s2_water"
NAME = "S1S2-Water"
ZENODO_RECORD = "11278238"
ZIP_URL = "https://zenodo.org/api/records/%s/files/part{}.zip/content" % ZENODO_RECORD
CATALOG_URL = (
    "https://zenodo.org/api/records/%s/files/catalog.json/content" % ZENODO_RECORD
)

TILE = 64
PER_CLASS = 1000
MIN_CLASS_PX = 32  # a tile counts toward a class only with >= this many px of it
MAX_NODATA_FRAC = 0.5  # skip tiles that are more than half nodata
LAND_PER_SCENE = 50  # deterministic land-only tiles per scene (no-water fill safety)

# Manifest class order -> id. Water (the phenomenon of interest) is 0; no-water is 1.
CLASSES = [
    (
        "water",
        "Open water / water bodies (rivers, lakes, reservoirs, coastal water) per the "
        "quality-checked Sentinel-2-derived binary water mask (s2_msk == 1).",
    ),
    (
        "no-water",
        "Land / non-water: Sentinel-2 binary water mask labeled not-water (s2_msk == 0).",
    ),
]
WATER, NOWATER = 0, 1

# Which zip part each scene id lives in (from the record's zip central directories).
SCENE_TO_PART = {
    "1": 1,
    "5": 1,
    "6": 1,
    "7": 1,
    "8": 1,
    "9": 1,
    "10": 1,
    "11": 1,
    "12": 1,
    "13": 1,
    "15": 1,
    "16": 1,
    "17": 2,
    "19": 2,
    "21": 2,
    "22": 2,
    "23": 2,
    "25": 2,
    "26": 2,
    "27": 2,
    "28": 2,
    "29": 2,
    "30": 2,
    "31": 2,
    "32": 3,
    "33": 3,
    "35": 3,
    "36": 3,
    "37": 3,
    "39": 3,
    "40": 3,
    "41": 3,
    "47": 3,
    "48": 3,
    "49": 3,
    "51": 3,
    "52": 4,
    "53": 4,
    "54": 4,
    "55": 4,
    "57": 4,
    "59": 4,
    "62": 4,
    "64": 4,
    "65": 4,
    "67": 4,
    "68": 4,
    "69": 4,
    "71": 5,
    "73": 5,
    "75": 5,
    "76": 5,
    "77": 5,
    "78": 5,
    "80": 5,
    "82": 5,
    "83": 6,
    "84": 6,
    "85": 6,
    "86": 6,
    "87": 6,
    "88": 6,
    "89": 6,
    "91": 6,
    "93": 6,
}


def raw_root():
    return io.raw_dir(SLUG)


def scene_dir(scene: str):
    return raw_root() / scene


def msk_path(scene: str):
    return scene_dir(scene) / f"sentinel12_s2_{scene}_msk.tif"


def valid_path(scene: str):
    return scene_dir(scene) / f"sentinel12_s2_{scene}_valid.tif"


def meta_path(scene: str):
    return scene_dir(scene) / f"sentinel12_{scene}_meta.json"


def _scene_files(scene: str) -> list[str]:
    return [
        f"{scene}/sentinel12_s2_{scene}_msk.tif",
        f"{scene}/sentinel12_s2_{scene}_valid.tif",
        f"{scene}/sentinel12_{scene}_meta.json",
    ]


def download_raw(scenes: list[str]) -> None:
    """Extract only the S2 mask, S2 validity mask and meta.json per scene (idempotent).

    Uses ``remotezip`` HTTP range requests so the huge imagery/DEM assets in each zip
    part are never downloaded. Files land in raw/{slug}/{scene}/.
    """
    import remotezip

    io.check_disk()
    by_part: dict[int, list[str]] = defaultdict(list)
    for s in scenes:
        # Skip scenes whose 3 target files already exist.
        if msk_path(s).exists() and valid_path(s).exists() and meta_path(s).exists():
            continue
        by_part[SCENE_TO_PART[s]].append(s)
    if not by_part:
        return

    for part in sorted(by_part):
        url = ZIP_URL.format(part)
        members: list[str] = []
        for s in by_part[part]:
            scene_dir(s).mkdir(parents=True, exist_ok=True)
            members.extend(_scene_files(s))
        # Retry the remote open a few times for transient Zenodo hiccups.
        last_err = None
        for attempt in range(4):
            try:
                with remotezip.RemoteZip(url) as z:
                    for m in members:
                        dst = raw_root() / m
                        if dst.exists():
                            continue
                        with z.open(m) as f:
                            data = f.read()
                        tmp = dst.parent / (dst.name + ".tmp")
                        with tmp.open("wb") as g:
                            g.write(data)
                        tmp.rename(dst)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                print(f"  part {part} attempt {attempt} failed: {e}")
        else:
            raise RuntimeError(f"failed to fetch part {part}: {last_err}")
        io.check_disk()


def scene_time(scene: str):
    """1-year window centered on the S2 acquisition date parsed from the STAC meta."""
    with meta_path(scene).open() as f:
        meta = json.load(f)
    srcids = meta["properties"].get("s2_srcids", [])
    m = re.search(r"_(\d{8})T", srcids[0]) if srcids else None
    if not m:
        # Fallback: use the (placeholder) datetime year.
        dt = datetime.fromisoformat(
            meta["properties"]["datetime"].replace("Z", "+00:00")
        )
    else:
        dt = datetime.strptime(m.group(1), "%Y%m%d").replace(tzinfo=UTC)
    return dt, (dt - timedelta(days=182), dt + timedelta(days=183))


def _load_label(scene: str) -> tuple[np.ndarray, Projection, int, int]:
    """Return (label array HxW uint8, rslearn Projection, origin_col_px, origin_row_px).

    Label: 0 = water, 1 = no-water, 255 = nodata (invalid). The array is cropped to a
    whole number of TILE-sized tiles. Pixel (origin_col_px + tj, origin_row_px + ti) is
    the top-left of tile (ti, tj) under the returned projection.
    """
    with rasterio.open(str(msk_path(scene))) as d:
        msk = d.read(1)
        crs = d.crs
        transform = d.transform
    with rasterio.open(str(valid_path(scene))) as d:
        valid = d.read(1)

    label = np.full(msk.shape, io.CLASS_NODATA, dtype=np.uint8)
    ok = valid == 1
    label[ok & (msk == 1)] = WATER
    label[ok & (msk == 0)] = NOWATER

    h = (label.shape[0] // TILE) * TILE
    w = (label.shape[1] // TILE) * TILE
    label = label[:h, :w]

    proj = Projection(crs, io.RESOLUTION, -io.RESOLUTION)
    origin_col = int(round(transform.c / io.RESOLUTION))  # left / 10
    origin_row = int(round(transform.f / -io.RESOLUTION))  # top / -10
    return label, proj, origin_col, origin_row


def _tile_counts(label: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized per-tile pixel counts. Returns (water, nowater, nodata) arrays
    each of shape (nty, ntx).
    """
    nty, ntx = label.shape[0] // TILE, label.shape[1] // TILE

    def block_count(mask: np.ndarray) -> np.ndarray:
        return mask.reshape(nty, TILE, ntx, TILE).sum(axis=(1, 3))

    water = block_count(label == WATER)
    nowater = block_count(label == NOWATER)
    nodata = block_count(label == io.CLASS_NODATA)
    return water, nowater, nodata


def _scan_scene(scene: str) -> list[dict[str, Any]]:
    """Emit candidate tile records for a scene: all water tiles + a deterministic
    land-only sample. Each record has scene/ti/tj and count_classes (>= MIN_CLASS_PX).
    """
    label, _proj, _oc, _or = _load_label(scene)
    water, nowater, nodata = _tile_counts(label)
    total = TILE * TILE
    water_recs: list[dict[str, Any]] = []
    land_recs: list[dict[str, Any]] = []
    nty, ntx = water.shape
    for ti in range(nty):
        for tj in range(ntx):
            if nodata[ti, tj] > MAX_NODATA_FRAC * total:
                continue
            has_w = water[ti, tj] >= MIN_CLASS_PX
            has_nw = nowater[ti, tj] >= MIN_CLASS_PX
            if not (has_w or has_nw):
                continue
            classes = []
            if has_w:
                classes.append(WATER)
            if has_nw:
                classes.append(NOWATER)
            rec = {
                "scene": scene,
                "ti": int(ti),
                "tj": int(tj),
                "count_classes": classes,
            }
            if has_w:
                water_recs.append(rec)
            else:
                land_recs.append(rec)
    # Deterministic land-only subsample (sort then seeded shuffle).
    land_recs.sort(key=lambda r: (r["ti"], r["tj"]))
    rng = random.Random(1000 + int(scene))
    rng.shuffle(land_recs)
    return water_recs + land_recs[:LAND_PER_SCENE]


def _select_tiles_per_class(all_recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tiles-per-class balanced selection (spec 5). Rare class (water) filled first.

    Candidates are sorted deterministically before the seeded shuffle so re-runs are
    reproducible regardless of multiprocessing completion order.
    """
    all_recs = sorted(all_recs, key=lambda r: (int(r["scene"]), r["ti"], r["tj"]))
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, rec in enumerate(all_recs):
        for c in rec["count_classes"]:
            by_class[c].append(i)
    order = sorted(by_class, key=lambda c: len(by_class[c]))  # rarest first
    rng = random.Random(42)
    selected_idx: set[int] = set()
    counts: dict[int, int] = defaultdict(int)
    for c in order:
        idxs = by_class[c][:]
        rng.shuffle(idxs)
        for i in idxs:
            if counts[c] >= PER_CLASS:
                break
            if i in selected_idx:
                continue
            selected_idx.add(i)
            for cc in all_recs[i]["count_classes"]:
                counts[cc] += 1
    return [all_recs[i] for i in sorted(selected_idx)]


def _write_scene(scene: str, tiles: list[dict[str, Any]]) -> None:
    """Load one scene's label once and write all its selected tiles."""
    label, proj, origin_col, origin_row = _load_label(scene)
    change_time = None  # static water masks, not dated events
    _dt, tr = scene_time(scene)
    for t in tiles:
        sample_id = t["sample_id"]
        if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
            continue
        ti, tj = t["ti"], t["tj"]
        sub = label[ti * TILE : (ti + 1) * TILE, tj * TILE : (tj + 1) * TILE].copy()
        x_min = origin_col + tj * TILE
        y_min = origin_row + ti * TILE
        bounds = (x_min, y_min, x_min + TILE, y_min + TILE)
        io.write_label_geotiff(
            SLUG, sample_id, sub, proj, bounds, nodata=io.CLASS_NODATA
        )
        present = sorted(int(x) for x in np.unique(sub) if x != io.CLASS_NODATA)
        io.write_sample_json(
            SLUG,
            sample_id,
            proj,
            bounds,
            tr,
            change_time=change_time,
            source_id=f"scene{scene}_r{ti}_c{tj}",
            classes_present=present,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    from olmoearth_pretrain.open_set_segmentation_data import manifest

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    scenes = sorted(SCENE_TO_PART, key=int)
    print(
        f"Extracting S2 masks/valid/meta for {len(scenes)} scenes (range requests)..."
    )
    download_raw(scenes)
    io.check_disk()

    print("Scanning scenes into 64x64 tiles...")
    with multiprocessing.Pool(args.workers) as p:
        all_recs: list[dict[str, Any]] = []
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan_scene, [dict(scene=s) for s in scenes]),
            total=len(scenes),
        ):
            all_recs.extend(recs)
    print(f"  {len(all_recs)} candidate tiles")

    selected = _select_tiles_per_class(all_recs)
    selected.sort(key=lambda r: (int(r["scene"]), r["ti"], r["tj"]))
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(
        f"  selected {len(selected)} tiles (tiles-per-class balanced, <= {PER_CLASS}/class)"
    )

    tile_class_counts = {name: 0 for name, _ in CLASSES}
    for r in selected:
        for c in r["count_classes"]:
            tile_class_counts[CLASSES[c][0]] += 1
    print("tiles containing each class:", tile_class_counts)

    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in selected:
        by_scene[r["scene"]].append(r)

    io.check_disk()
    print(f"Writing tiles for {len(by_scene)} scenes...")
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(
                p,
                _write_scene,
                [dict(scene=s, tiles=ts) for s, ts in by_scene.items()],
            ),
            total=len(by_scene),
        ):
            pass

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Zenodo / IEEE JSTARS (Wieland et al. 2024)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://zenodo.org/records/11278238",
                "have_locally": False,
                "annotation_method": "manual / quality-checked binary water masks",
                "citation": "Wieland et al. 2024, IEEE JSTARS (S1S2-Water)",
                "used_asset": "s2_msk (Sentinel-2 binary water mask, native UTM 10 m)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "tile_class_counts": tile_class_counts,
            "notes": (
                "65 globally distributed ~100x100 km scenes (29 countries). We use the "
                "Sentinel-2 binary water mask (s2_msk), which is already in local UTM at "
                "10 m/px, so no reprojection is needed; the scene is cut into 64x64 tiles "
                "aligned to the source grid. Classes: 0=water (msk==1), 1=no-water "
                "(msk==0), 255=nodata (s2_valid==0). Tiles-per-class balanced (<=1000/class); "
                "water is the rare class and is filled first, no-water co-occurs widely. "
                "All 65 scenes are non-flood/static water (release flood flag False), so no "
                "change_time is set; time_range is a 1-year window centered on the S2 "
                "acquisition date (from properties.s2_srcids; dates 2018-2020, all post-2016). "
                "Only the S2 mask/valid/meta files were downloaded via zip range requests; the "
                "S1 mask (9 m), imagery and Copernicus DEM assets were not used."
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
