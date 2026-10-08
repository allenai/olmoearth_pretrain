"""Process OlmoEarth Fields of the World into open-set-segmentation label patches.

Source: local rslearn dataset at
``/weka/dfive-default/rslearn-eai/datasets/fields_of_the_world/rslearn_dataset_utm/``
(the ingested Fields of the World / FTW benchmark). Each window is a ~154 x ~120 px
chip already in a local UTM projection at 10 m/pixel, with a dense categorical
``label`` raster layer (``layers/label/label/geotiff.tif``, dtype uint8).

Source label values (verified over the whole dataset):
  0 = background (non-field)
  1 = field (crop field interior)
  2 = field_boundary (field edge)
  3 = nodata / unlabeled  (source geotiff nodata = 3)

We keep 0/1/2 as class ids 0/1/2 and remap source value 3 -> 255 (CLASS_NODATA).

Task family is polygon field-boundary segmentation, but on disk it is a pre-rasterized
dense raster, so we treat it as a dense-raster classification task: tile each window into
<=64x64 patches (edge-aligned so every tile is a full 64x64 where the window allows) and
select with tiles-per-class balanced sampling (<=1000 tiles per class; a tile counts
toward every class present in it, rarest classes prioritized).

The source is already UTM at 10 m/pixel, so we read the label at its native
projection/bounds with no reprojection (an identity read, equivalent to nearest
resampling -- no interpolation of the categorical labels). Each window carries its own
~240-day time_range in metadata (well under 1 year), which we use directly.

Run:
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_fields_of_the_world
"""

import argparse
import json
import multiprocessing
import os
import random
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

import numpy as np
import rasterio
import tqdm
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io

try:
    from rasterio.crs import CRS
except ImportError:  # pragma: no cover
    from rasterio import CRS  # type: ignore

SLUG = "olmoearth_fields_of_the_world"
NAME = "OlmoEarth Fields of the World"
SOURCE = (
    "/weka/dfive-default/rslearn-eai/datasets/fields_of_the_world/rslearn_dataset_utm/"
)
LABEL_TIF = "layers/label/label/geotiff.tif"

PER_CLASS = 1000
TILE = io.MAX_TILE  # 64
SOURCE_NODATA = 3  # source geotiff nodata value -> remapped to CLASS_NODATA (255)
# Windows sampled per country/group for scanning; small groups are fully included.
WINDOWS_PER_GROUP = 300
SEED = 42

CLASSES = [
    ("background", "Non-field land: pixels outside any agricultural field parcel."),
    ("field", "Interior of a cultivated agricultural field parcel."),
    (
        "field_boundary",
        "Field edge/boundary pixels separating adjacent parcels (from national LPIS "
        "polygon boundaries, rasterized at 10 m).",
    ),
]


def _tile_offsets(n: int, size: int) -> list[tuple[int, int]]:
    """Offsets (start, length) tiling [0, n) with full-``size`` tiles, edge-aligned.

    If ``n <= size`` a single tile of length ``n`` is returned; otherwise tiles of
    length ``size`` covering the whole extent, the last one aligned to ``n - size``
    (may overlap the previous one).
    """
    if n <= size:
        return [(0, n)]
    starts = list(range(0, n - size + 1, size))
    if starts[-1] != n - size:
        starts.append(n - size)
    return [(s, size) for s in starts]


def _list_windows() -> list[tuple[str, str]]:
    windows_root = os.path.join(SOURCE, "windows")
    out: list[tuple[str, str]] = []
    for group in sorted(os.listdir(windows_root)):
        gd = os.path.join(windows_root, group)
        if not os.path.isdir(gd):
            continue
        names = sorted(os.listdir(gd))
        out.extend((group, n) for n in names)
    return out


def _sample_windows(rng: random.Random) -> list[tuple[str, str]]:
    """Sample up to WINDOWS_PER_GROUP windows per group for geographic diversity."""
    by_group: dict[str, list[str]] = defaultdict(list)
    for group, name in _list_windows():
        by_group[group].append(name)
    picked: list[tuple[str, str]] = []
    for group in sorted(by_group):
        names = by_group[group]
        if len(names) > WINDOWS_PER_GROUP:
            names = rng.sample(names, WINDOWS_PER_GROUP)
        picked.extend((group, n) for n in names)
    return picked


def _scan_window(group: str, name: str) -> list[dict[str, Any]]:
    """Read one window's metadata + label raster and return one record per tile.

    Records are lightweight (no arrays); the write phase re-reads and slices.
    """
    wdir = os.path.join(SOURCE, "windows", group, name)
    try:
        with open(os.path.join(wdir, "metadata.json")) as f:
            md = json.load(f)
    except FileNotFoundError:
        return []
    tif = os.path.join(wdir, LABEL_TIF)
    if not os.path.exists(tif):
        return []
    with rasterio.open(tif) as ds:
        arr = ds.read(1)
    proj = md["projection"]
    bounds = md["bounds"]  # [x_min, y_min(top row), x_max, y_max]
    tr = md.get("time_range")
    h, w = arr.shape
    recs: list[dict[str, Any]] = []
    for r0, th in _tile_offsets(h, TILE):
        for c0, tw in _tile_offsets(w, TILE):
            sub = arr[r0 : r0 + th, c0 : c0 + tw]
            present = [int(v) for v in np.unique(sub) if v != SOURCE_NODATA]
            if not present:  # entirely nodata
                continue
            recs.append(
                {
                    "group": group,
                    "name": name,
                    "r0": r0,
                    "c0": c0,
                    "th": th,
                    "tw": tw,
                    "crs": proj["crs"],
                    "win_x0": bounds[0],
                    "win_y0": bounds[1],
                    "time_range": tr,
                    "classes_present": present,
                }
            )
    return recs


def _balance_multilabel(
    tiles: list[dict[str, Any]], per_class: int, seed: int
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Tiles-per-class balanced selection.

    A tile counts toward every class present in it. Greedily serves the currently
    rarest under-target class; a tile is selected iff it helps at least one class still
    below ``per_class``. Stops when no class remains servable.
    """
    rng = random.Random(seed)
    tiles = list(tiles)
    rng.shuffle(tiles)
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(tiles):
        for c in t["classes_present"]:
            by_class[c].append(i)
    classes = sorted(by_class)
    ptr = {c: 0 for c in classes}
    counts: dict[int, int] = {c: 0 for c in classes}
    chosen = [False] * len(tiles)
    selected: list[dict[str, Any]] = []
    while True:
        cand = [
            c for c in classes if counts[c] < per_class and ptr[c] < len(by_class[c])
        ]
        if not cand:
            break
        c = min(cand, key=lambda c: counts[c])
        idxs = by_class[c]
        while ptr[c] < len(idxs) and chosen[idxs[ptr[c]]]:
            ptr[c] += 1
        if ptr[c] >= len(idxs):
            continue
        i = idxs[ptr[c]]
        ptr[c] += 1
        chosen[i] = True
        selected.append(tiles[i])
        for cc in tiles[i]["classes_present"]:
            counts[cc] += 1
    return selected, counts


def _write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    wdir = os.path.join(SOURCE, "windows", rec["group"], rec["name"])
    with rasterio.open(os.path.join(wdir, LABEL_TIF)) as ds:
        arr = ds.read(1)
    r0, c0, th, tw = rec["r0"], rec["c0"], rec["th"], rec["tw"]
    sub = arr[r0 : r0 + th, c0 : c0 + tw].astype(np.uint8).copy()
    sub[sub == SOURCE_NODATA] = io.CLASS_NODATA
    proj = Projection(CRS.from_string(rec["crs"]), io.RESOLUTION, -io.RESOLUTION)
    x0 = rec["win_x0"] + c0
    y0 = rec["win_y0"] + r0
    bounds = (x0, y0, x0 + tw, y0 + th)
    io.write_label_geotiff(SLUG, sample_id, sub, proj, bounds, nodata=io.CLASS_NODATA)
    tr = rec["time_range"]
    time_range = (
        (datetime.fromisoformat(tr[0]), datetime.fromisoformat(tr[1])) if tr else None
    )
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        time_range,
        source_id=f"{rec['group']}/{rec['name']}#{r0}_{c0}",
        classes_present=sorted(set(rec["classes_present"])),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(f"local rslearn dataset: {SOURCE}\n")

    rng = random.Random(SEED)
    windows = _sample_windows(rng)
    print(f"scanning {len(windows)} windows (<= {WINDOWS_PER_GROUP}/group)")

    with multiprocessing.Pool(args.workers) as p:
        tile_lists = list(
            tqdm.tqdm(
                star_imap_unordered(
                    p, _scan_window, [dict(group=g, name=n) for g, n in windows]
                ),
                total=len(windows),
            )
        )
    tiles = [t for lst in tile_lists for t in lst]
    print(f"collected {len(tiles)} candidate tiles")

    selected, avail_counts = _balance_multilabel(tiles, PER_CLASS, SEED)
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} tiles")

    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            pass

    # Report per-class tile counts among selected (tiles-per-class semantics).
    sel_counts: Counter = Counter()
    for r in selected:
        for c in set(r["classes_present"]):
            sel_counts[c] += 1
    group_counts = Counter(r["group"] for r in selected)
    print(
        "selected per-class:",
        {CLASSES[c][0]: sel_counts[c] for c in sorted(sel_counts)},
    )
    print("selected per-group:", dict(group_counts))

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "olmoearth",
            "license": "CC-BY (mixed)",
            "provenance": {
                "url": SOURCE,
                "have_locally": True,
                "annotation_method": "national LPIS + manual QC (Fields of the World benchmark)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {
                CLASSES[c][0]: sel_counts.get(c, 0) for c in range(len(CLASSES))
            },
            "notes": (
                "Dense field-boundary segmentation from the Fields of the World (FTW) "
                "benchmark ingested as an rslearn dataset. Windows tiled into <=64x64 "
                "patches; tiles-per-class balanced (a tile counts toward each class in "
                "it, <=1000/class). Source value 3 (nodata) remapped to 255. Source is "
                "already UTM 10 m so labels read natively (no reprojection). Per-window "
                "~240-day time_range used directly. Sampled up to "
                f"{WINDOWS_PER_GROUP} windows per country across 25 countries."
            ),
        },
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
