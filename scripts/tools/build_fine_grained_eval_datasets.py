"""Build the fine-grained segmentation eval datasets.

Two rslearn eval datasets that keep an ordinary training split but score val/test
only on small objects and on a sample of the pixels bordering them, so they
measure whether a model recovers detail finer than the surrounding land cover:

* ``worldcover_fine_grained``: Geo-Wiki 10 m land-cover reference blocks (the
  ``worldcover`` rslearn dataset: one 10x10 photointerpreted block at the center
  of each 53x53 window). Objects are 8-connected same-class components of at
  most 10 px.
* ``pastis_fine_grained``: PASTIS (``pastis_rslearn``). Objects are crop parcels
  (PASTIS-R ``ParcelIDs``) of at most 50 px.

An object is scored only when its full extent is known: worldcover components
whose 8-neighbour ring touches unlabeled pixels or the raster edge are dropped
(they may continue outside the labeled block), as are PASTIS parcels that touch
the patch edge. The scored border is the union of the objects' 8-neighbour rings
(minus the objects and unlabeled pixels), randomly subsampled per window to at
most that window's object-pixel count. Scoring the objects alone rewards a model
that paints a blob over each object; scoring the full ring rewards one that
ignores objects (worldcover objects are mostly single pixels, so the full ring
is ~4.5x the object pixels); the capped ring penalizes both about equally.

The worldcover imagery is copied as is: it is 2016 (Sentinel-2A only, no cloud
filter), so about 63% of windows have 1-7 all-zero monthly mosaics, which the eval
loader feeds as data rather than as missing timesteps.

Usage (from outside the repo root, see the scan cache note in each builder):

    python scripts/tools/build_fine_grained_eval_datasets.py worldcover
    python scripts/tools/build_fine_grained_eval_datasets.py pastis
"""

import argparse
import json
import logging
import multiprocessing
import random
import shutil
import zlib
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window
from scipy import ndimage
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)
EARTH_RADIUS_KM = 6371.0


def ring(mask: np.ndarray) -> np.ndarray:
    """The 8-neighbour ring around a boolean mask (excluding the mask)."""
    return ndimage.binary_dilation(mask, EIGHT_CONNECTED) & ~mask


def touches_edge(mask: np.ndarray) -> bool:
    """Whether a boolean mask touches the raster edge."""
    return bool(
        mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
    )


def scored_mask(
    objects: list[np.ndarray], valid: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Objects plus a capped random sample of their bordering pixels.

    Args:
        objects: boolean masks of the scored objects.
        valid: boolean mask of pixels that carry a usable label.
        rng: generator for the border subsample.

    Returns:
        (object mask, sampled border mask).
    """
    obj = np.zeros(valid.shape, dtype=bool)
    border = np.zeros(valid.shape, dtype=bool)
    for m in objects:
        obj |= m
        border |= ring(m)
    border &= valid & ~obj
    candidates = np.flatnonzero(border)
    keep = rng.choice(
        candidates, size=min(len(candidates), int(obj.sum())), replace=False
    )
    sampled = np.zeros(valid.shape, dtype=bool)
    sampled.flat[keep] = True
    return obj, sampled


def window_rng(name: str) -> np.random.Generator:
    """Deterministic per-window generator."""
    return np.random.default_rng(zlib.crc32(name.encode()))


def to_unit_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Lat/lon in degrees to unit-sphere xyz, for chord-distance KD-trees."""
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    return np.stack(
        [np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)],
        axis=1,
    )


def km_to_chord(km: float) -> float:
    """Great-circle distance in km to unit-sphere chord length."""
    return 2 * np.sin(km / EARTH_RADIUS_KM / 2)


def write_geotiff(
    path: Path, array: np.ndarray, profile: dict[str, Any], transform: Any
) -> None:
    """Write a (C, H, W) array as an LZW GeoTIFF with the given georeferencing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = {
        "driver": "GTiff",
        "dtype": array.dtype,
        "count": array.shape[0],
        "height": array.shape[1],
        "width": array.shape[2],
        "crs": profile["crs"],
        "transform": transform,
        "compress": "lzw",
    }
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(array)


def write_layer(layer_dir: Path, band_dir: str, array: np.ndarray, src: Path) -> None:
    """Write a single-image layer, georeferenced like the source raster.

    The source's rslearn metadata.json sidecar (timestamps) is copied along.
    """
    with rasterio.open(src) as s:
        profile, transform = s.profile, s.transform
    write_geotiff(layer_dir / band_dir / "geotiff.tif", array, profile, transform)
    if (src.parent / "metadata.json").exists():
        shutil.copy(
            src.parent / "metadata.json", layer_dir / band_dir / "metadata.json"
        )
    (layer_dir / "completed").touch()


# ---------------------------------------------------------------------------
# worldcover_fine_grained
# ---------------------------------------------------------------------------

WC_SOURCE = Path("/weka/dfive-default/rslearn-eai/datasets/worldcover")
WC_GROUP = "20260109"
WC_OUT = Path(
    "/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/worldcover_fine_grained"
)
# Open-set pretraining sample of the same Geo-Wiki reference data; val/test
# windows containing one of its points are excluded.
WC_OPEN_SET_POINTS = Path(
    "/weka/dfive-default/helios/dataset_creation/open_set_segmentation/datasets/"
    "geo_wiki_global_10_m_land_cover_reference_2015/points.geojson"
)
WC_S2_LAYERS = ["sentinel2"] + [f"sentinel2.{i}" for i in range(1, 12)]
WC_S2_BANDS_DIR = "B01_B02_B03_B04_B05_B06_B07_B08_B8A_B09_B11_B12"
# Source class codes kept, in output order (0..8). Dropped: 0 no_data,
# 2 burnt, 6 lichen and moss, 8 snow and ice (too rare to score).
WC_KEPT_CODES = [1, 3, 4, 5, 7, 9, 10, 11, 12]
WC_DROPPED_CODES = [2, 6, 8]
WC_NODATA = 255
WC_MAX_OBJECT_PX = 10
WC_TRAIN_PER_CLASS = 1000
WC_TRAIN_MIN_CLASS_PX = 10
WC_EVAL_PER_SPLIT = 5000
WC_BUFFER_KM = 1.0
# Windows are 530 m wide, so a point within 400 m of the center may be inside.
WC_OPEN_SET_RADIUS_KM = 0.4
# The 53x53 source windows are center-cropped to 32x32 (offset 10), which always
# contains the labeled block (offsets 21-22, 10 px).
WC_CROP_OFFSET = 10
WC_CROP_SIZE = 32


def wc_remap(raw: np.ndarray) -> np.ndarray:
    """Source class codes to output ids (0..8), WC_NODATA elsewhere."""
    lut = np.full(256, WC_NODATA, dtype=np.uint8)
    for new_id, code in enumerate(WC_KEPT_CODES):
        lut[code] = new_id
    return lut[raw]


def wc_objects(raw: np.ndarray) -> list[np.ndarray]:
    """Fully observed small components of the kept classes.

    The fully-observed test uses the source nodata (0) only: a component next
    to a dropped-class pixel still has a known extent.
    """
    objects = []
    for code in WC_KEPT_CODES:
        labeled, _ = ndimage.label(raw == code, structure=EIGHT_CONNECTED)
        counts = np.bincount(labeled.ravel())
        for idx in np.flatnonzero(counts <= WC_MAX_OBJECT_PX):
            if idx == 0:
                continue
            m = labeled == idx
            if touches_edge(m) or (raw[ring(m)] == 0).any():
                continue
            objects.append(m)
    return objects


def wc_eval_label(name: str, raw: np.ndarray) -> np.ndarray:
    """Val/test label: objects + capped border, WC_NODATA elsewhere."""
    objects = wc_objects(raw)
    valid = np.isin(raw, WC_KEPT_CODES)
    obj, border = scored_mask(objects, valid, window_rng(name))
    out = np.full(raw.shape, WC_NODATA, dtype=np.uint8)
    keep = obj | border
    out[keep] = wc_remap(raw)[keep]
    return out


def wc_scan_one(name: str) -> dict[str, Any] | None:
    """Per-window facts needed for selection."""
    wdir = WC_SOURCE / "windows" / WC_GROUP / name
    try:
        meta = json.loads((wdir / "metadata.json").read_text())
        with rasterio.open(wdir / "layers/label_raster/label/geotiff.tif") as f:
            raw = f.read(1)
    except Exception as e:  # noqa: BLE001
        logger.warning("skipping %s: %s", name, e)
        return None
    s2_ok = all(
        (wdir / "layers" / layer / "completed").exists() for layer in WC_S2_LAYERS
    )
    proj = meta["projection"]
    b = meta["bounds"]
    cx = (b[0] + b[2]) / 2 * proj["x_resolution"]
    cy = (b[1] + b[3]) / 2 * proj["y_resolution"]
    lon, lat = Transformer.from_crs(proj["crs"], "EPSG:4326", always_xy=True).transform(
        cx, cy
    )
    objects = wc_objects(raw)
    return {
        "name": name,
        "split": meta["options"].get("split"),
        "s2_ok": s2_ok,
        "lat": lat,
        "lon": lon,
        "class_counts": np.bincount(raw.ravel(), minlength=13)[:13],
        "object_px": int(sum(m.sum() for m in objects)),
    }


def wc_scan(cache: Path, workers: int) -> list[dict[str, Any]]:
    """Scan every source window (cached, the scan reads ~166k label rasters)."""
    if cache.exists():
        logger.info("loading scan cache %s", cache)
        rows = json.loads(cache.read_text())
        for r in rows:
            r["class_counts"] = np.array(r["class_counts"])
        return rows
    names = sorted(p.name for p in (WC_SOURCE / "windows" / WC_GROUP).iterdir())
    logger.info("scanning %d windows", len(names))
    with multiprocessing.Pool(workers) as pool:
        rows = [r for r in pool.imap(wc_scan_one, names, chunksize=64) if r is not None]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps([r | {"class_counts": r["class_counts"].tolist()} for r in rows])
    )
    return rows


def wc_open_set_xyz() -> np.ndarray:
    """Unit-sphere xyz of the open-set pretraining Geo-Wiki points."""
    features = json.loads(WC_OPEN_SET_POINTS.read_text())["features"]
    lon = np.array([f["geometry"]["coordinates"][0] for f in features])
    lat = np.array([f["geometry"]["coordinates"][1] for f in features])
    return to_unit_xyz(lat, lon)


def wc_select(rows: list[dict[str, Any]], seed: int) -> dict[str, list[str]]:
    """Pick the train subset and the val/test windows."""
    rng = random.Random(seed)
    usable = [r for r in rows if r["s2_ok"] and r["class_counts"][1:].sum() > 0]
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for r in usable:
        if r["split"] in by_split:
            by_split[r["split"]].append(r)
    logger.info("usable windows: %s", {k: len(v) for k, v in by_split.items()})

    train: set[str] = set()
    for code in WC_KEPT_CODES:
        candidates = sorted(
            r["name"]
            for r in by_split["train"]
            if r["class_counts"][code] >= WC_TRAIN_MIN_CLASS_PX
        )
        picked = rng.sample(candidates, min(WC_TRAIN_PER_CLASS, len(candidates)))
        logger.info(
            "class %d: %d candidates, picked %d", code, len(candidates), len(picked)
        )
        train.update(picked)
    train_rows = [r for r in by_split["train"] if r["name"] in train]
    train_tree = cKDTree(
        to_unit_xyz(
            np.array([r["lat"] for r in train_rows]),
            np.array([r["lon"] for r in train_rows]),
        )
    )
    open_set_tree = cKDTree(wc_open_set_xyz())

    selection = {"train": sorted(train)}
    for split in ("val", "test"):
        rs = [r for r in by_split[split] if r["object_px"] > 0]
        xyz = to_unit_xyz(
            np.array([r["lat"] for r in rs]), np.array([r["lon"] for r in rs])
        )
        near_train = train_tree.query(xyz, k=1)[0] <= km_to_chord(WC_BUFFER_KM)
        near_open_set = open_set_tree.query(xyz, k=1)[0] <= km_to_chord(
            WC_OPEN_SET_RADIUS_KM
        )
        eligible = sorted(
            r["name"] for r, a, b in zip(rs, near_train, near_open_set) if not (a or b)
        )
        logger.info(
            "%s: %d with objects, %d near train, %d near open-set, %d eligible",
            split,
            len(rs),
            int(near_train.sum()),
            int(near_open_set.sum()),
            len(eligible),
        )
        if len(eligible) < WC_EVAL_PER_SPLIT:
            raise ValueError(f"only {len(eligible)} eligible {split} windows")
        selection[split] = sorted(rng.sample(eligible, WC_EVAL_PER_SPLIT))
    return selection


def wc_write_one(item: tuple[str, str], out: Path) -> dict[str, Any]:
    """Copy one window, center-cropped to 32x32, with its new labels."""
    name, split = item
    src = WC_SOURCE / "windows" / WC_GROUP / name
    dst = out / "windows" / WC_GROUP / name
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "layers").mkdir(parents=True)
    crop = Window(WC_CROP_OFFSET, WC_CROP_OFFSET, WC_CROP_SIZE, WC_CROP_SIZE)
    off, size = WC_CROP_OFFSET, WC_CROP_SIZE

    meta = json.loads((src / "metadata.json").read_text())
    b = meta["bounds"]
    meta["bounds"] = [b[0] + off, b[1] + off, b[0] + off + size, b[1] + off + size]
    meta["options"] = {
        "split": split,
        "eval_split": split,
        "source_split": meta["options"].get("split"),
    }
    (dst / "metadata.json").write_text(json.dumps(meta))
    items = json.loads((src / "items.json").read_text())
    (dst / "items.json").write_text(
        json.dumps([x for x in items if x["layer_name"] == "sentinel2"])
    )

    for layer in WC_S2_LAYERS:
        tif = src / "layers" / layer / WC_S2_BANDS_DIR / "geotiff.tif"
        with rasterio.open(tif) as f:
            arr = f.read(window=crop)
            transform = f.window_transform(crop)
            profile = f.profile
        write_geotiff(
            dst / "layers" / layer / WC_S2_BANDS_DIR / "geotiff.tif",
            arr,
            profile,
            transform,
        )
        (dst / "layers" / layer / "completed").touch()

    label_tif = src / "layers/label_raster/label/geotiff.tif"
    with rasterio.open(label_tif) as f:
        raw = f.read(1)
        transform = f.window_transform(crop)
        profile = f.profile
    full = wc_remap(raw)
    label = full if split == "train" else wc_eval_label(name, raw)
    sl = (slice(off, off + size), slice(off, off + size))
    # Scored pixels all lie in the labeled block, which the crop contains.
    assert (label != WC_NODATA).sum() == (label[sl] != WC_NODATA).sum()
    for layer, arr in (("label_raster", label), ("label_raster_full", full)):
        write_geotiff(
            dst / "layers" / layer / "label" / "geotiff.tif",
            arr[sl][None],
            profile,
            transform,
        )
        (dst / "layers" / layer / "completed").touch()
    return {
        "name": name,
        "split": split,
        "scored_px": int((label != WC_NODATA).sum()),
        "class_px": np.bincount(label[label != WC_NODATA], minlength=9).tolist(),
    }


def wc_config() -> dict[str, Any]:
    """Dataset config.json: the source sentinel2 layer plus the new labels."""
    src = json.loads((WC_SOURCE / "config.json").read_text())
    class_names = [
        src["layers"]["label_raster"]["class_names"][c] for c in WC_KEPT_CODES
    ]
    label_layer = {
        "type": "raster",
        "band_sets": [
            {"bands": ["label"], "dtype": "uint8", "nodata_vals": [WC_NODATA]}
        ],
        "class_names": class_names,
    }
    return {
        "layers": {
            "sentinel2": src["layers"]["sentinel2"],
            "label_raster": label_layer,
            "label_raster_full": label_layer,
        }
    }


def build_worldcover(args: argparse.Namespace) -> None:
    """Build worldcover_fine_grained."""
    out = Path(args.out or WC_OUT)
    rows = wc_scan(
        Path(args.scan_cache or out.parent / "worldcover_fine_grained_scan.json"),
        args.workers,
    )
    selection = wc_select(rows, args.seed)
    if args.dry_run:
        logger.info("dry run: %s", {k: len(v) for k, v in selection.items()})
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(wc_config(), indent=2))
    items = [(n, s) for s, names in selection.items() for n in names]
    with multiprocessing.Pool(args.workers) as pool:
        written = list(
            pool.imap_unordered(partial(wc_write_one, out=out), items, chunksize=16)
        )
    summarize_and_save(out, written, selection, args, num_classes=9)


# ---------------------------------------------------------------------------
# pastis_fine_grained
# ---------------------------------------------------------------------------

PA_SOURCE = Path("/weka/dfive-default/olmoearth/eval_datasets/pastis_rslearn")
PA_GROUP = "pastis"
PA_ANNOTATIONS = Path("/weka/dfive-default/rslearn-eai/artifacts/PASTIS-R/ANNOTATIONS")
PA_OUT = Path(
    "/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/pastis_fine_grained"
)
PA_LAYERS = [f"sentinel2_l2a_mo{i:02d}" for i in range(1, 13)] + [
    f"sentinel1_mo{i:02d}" for i in range(1, 13)
]
PA_BACKGROUND = 0
PA_VOID = 19
PA_MAX_PARCEL_PX = 50


def pa_eval_label(name: str, target: np.ndarray, parcel_ids: np.ndarray) -> np.ndarray:
    """Val/test label: small crop parcels + capped border, PA_VOID elsewhere."""
    edge_ids = set(
        np.unique(
            np.concatenate(
                [parcel_ids[0], parcel_ids[-1], parcel_ids[:, 0], parcel_ids[:, -1]]
            )
        ).tolist()
    )
    ids, counts = np.unique(parcel_ids, return_counts=True)
    objects = []
    for pid, count in zip(ids, counts):
        if pid == 0 or pid in edge_ids or count > PA_MAX_PARCEL_PX:
            continue
        m = (parcel_ids == pid) & (target != PA_VOID)
        if not m.any() or np.bincount(target[m]).argmax() == PA_BACKGROUND:
            continue
        objects.append(m)
    obj, border = scored_mask(objects, target != PA_VOID, window_rng(name))
    out = np.full(target.shape, PA_VOID, dtype=target.dtype)
    keep = obj | border
    out[keep] = target[keep]
    return out


def pa_write_one(name: str, out: Path) -> dict[str, Any]:
    """Copy one PASTIS window with its new labels."""
    src = PA_SOURCE / "windows" / PA_GROUP / name
    dst = out / "windows" / PA_GROUP / name
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "layers").mkdir(parents=True)
    meta = json.loads((src / "metadata.json").read_text())
    split = meta["options"]["eval_split"]
    meta["options"] = {"split": split, "eval_split": split}
    (dst / "metadata.json").write_text(json.dumps(meta))
    if (src / "items.json").exists():
        items = json.loads((src / "items.json").read_text())
        (dst / "items.json").write_text(
            json.dumps([x for x in items if x["layer_name"] in PA_LAYERS])
        )
    # Some pastis_rslearn windows lack a monthly Sentinel-1 mosaic; mirror what
    # the source has.
    for layer in PA_LAYERS:
        if (src / "layers" / layer).exists():
            shutil.copytree(src / "layers" / layer, dst / "layers" / layer)

    label_tif = src / "layers/label/label/geotiff.tif"
    with rasterio.open(label_tif) as f:
        target = f.read(1)
    if split == "train":
        label = target
    else:
        parcel_ids = np.load(PA_ANNOTATIONS / f"ParcelIDs_{name}.npy")
        reference = np.load(PA_ANNOTATIONS / f"TARGET_{name}.npy")[0]
        # The rslearn export must be on the PASTIS-R patch grid for the parcel
        # IDs to apply.
        if not np.array_equal(reference, target):
            raise ValueError(f"{name}: label raster does not match PASTIS-R TARGET")
        label = pa_eval_label(name, target, parcel_ids)
    write_layer(dst / "layers/label", "label", label[None], label_tif)
    write_layer(dst / "layers/label_full", "label", target[None], label_tif)
    scored = label != PA_VOID
    return {
        "name": name,
        "split": split,
        "scored_px": int(scored.sum()),
        "class_px": np.bincount(label[scored], minlength=19).tolist(),
    }


def pa_config() -> dict[str, Any]:
    """Dataset config.json: the source S2/S1 monthly layers plus the labels."""
    src = json.loads((PA_SOURCE / "config.json").read_text())
    layers = {name: src["layers"][name] for name in PA_LAYERS}
    layers["label"] = src["layers"]["label"]
    layers["label_full"] = src["layers"]["label"]
    return {k: v for k, v in src.items() if k != "layers"} | {"layers": layers}


def build_pastis(args: argparse.Namespace) -> None:
    """Build pastis_fine_grained (every pastis_rslearn window, same splits)."""
    out = Path(args.out or PA_OUT)
    names = sorted(p.name for p in (PA_SOURCE / "windows" / PA_GROUP).iterdir())
    if args.dry_run:
        logger.info("dry run: %d windows", len(names))
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(pa_config(), indent=2))
    with multiprocessing.Pool(args.workers) as pool:
        written = list(
            pool.imap_unordered(partial(pa_write_one, out=out), names, chunksize=4)
        )
    # Like worldcover, val/test only keep windows with at least one object.
    for w in written:
        if w["split"] != "train" and w["scored_px"] == 0:
            shutil.rmtree(out / "windows" / PA_GROUP / w["name"])
    written = [w for w in written if w["split"] == "train" or w["scored_px"] > 0]
    selection: dict[str, list[str]] = {}
    for w in written:
        selection.setdefault(w["split"], []).append(w["name"])
    summarize_and_save(out, written, selection, args, num_classes=19)


# ---------------------------------------------------------------------------


def summarize_and_save(
    out: Path,
    written: list[dict[str, Any]],
    selection: dict[str, list[str]],
    args: argparse.Namespace,
    num_classes: int,
) -> None:
    """Log per-split scored-pixel stats and record the build in selection.json."""
    stats = {}
    for split in ("train", "val", "test"):
        ws = [w for w in written if w["split"] == split]
        class_px = np.sum([w["class_px"] for w in ws], axis=0).tolist() if ws else []
        stats[split] = {
            "windows": len(ws),
            "windows_with_scored_px": sum(w["scored_px"] > 0 for w in ws),
            "scored_px": sum(w["scored_px"] for w in ws),
            "class_px": class_px,
        }
        logger.info("%s: %s", split, stats[split])
    (out / "selection.json").write_text(
        json.dumps(
            {
                "args": {k: v for k, v in vars(args).items() if k != "func"},
                "stats": stats,
                "num_classes": num_classes,
                "windows": {k: sorted(v) for k, v in selection.items()},
            },
            indent=1,
        )
    )


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(required=True)
    for name, func in (("worldcover", build_worldcover), ("pastis", build_pastis)):
        p = sub.add_parser(name)
        p.set_defaults(func=func)
        p.add_argument("--out", default=None, help="output dataset root")
        p.add_argument("--workers", type=int, default=32)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--dry_run", action="store_true")
        if name == "worldcover":
            p.add_argument(
                "--scan_cache",
                default=None,
                help="scan cache JSON (default: next to the output dataset)",
            )
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
