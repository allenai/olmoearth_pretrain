"""Build the per-window, per-month Landsat calibration sidecar for a dataset.

The eval loader reads Landsat straight out of the rslearn GeoTIFFs as raw
Collection-2 DN (``passthrough: true``), while a model pretrained on the
reflectance h5 expects TOA reflectance / brightness temperature. Converting at
read time needs each month's base scene ``SUN_ELEVATION`` and platform, which
live in the scene MTL.txt on S3 -- far too slow to fetch per sample. This
script precomputes them into::

    {
      "meta": {...},
      "windows": {"<group>/<name>": {"mo01": {"sun_elevation": 54.3,
                                              "platform": "LC08"}, ...}}
    }

which feeds ``landsat_reflectance`` on ``RslearnToOlmoEarthDataset``. A month
with no scene, or whose MTL could not be read, records ``null``; the loader
turns those timesteps into MISSING rather than feeding DN on a reflectance
scale.

Scenes are shared across windows, so unique blob paths are fetched once and
reused -- the S3 pass is over scenes, not window-months.

Requires AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY: ``s3://usgs-landsat`` is
requester-pays.

TWO-TREE NOTE: as with the cloud-cover sidecar, the Landsat items live in the
STAGING tree's items.json while the eval loader reads the sidecar from the
REGISTERED dataset root. Point --ds_path at the tree that has the items and
--out at the tree evals read::

    python scripts/tools/build_landsat_calibration_sidecar.py \
        --ds_path /weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/ethiopia_crops_year_aligned \
        --out /weka/dfive-default/olmoearth/eval_datasets/ethiopia_crops_year_aligned
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.landsat_calibration import (
    fetch_sun_elevation,
    platform_from_scene_id,
)

LAYER_PREFIX = "landsat_mo"
MONTHS = 12
OUTPUT_NAME = "landsat_calibration.json"
FETCH_GROUP_SUFFIX = "_tessera_v2"


def read_window_scenes(window_dir: Path) -> dict[str, dict[str, str]] | None:
    """Map "moNN" -> the month's base scene {name, blob_path}, or None if unprepared.

    Mirrors the cloud-cover sidecar: the first item of the month's mosaic group
    is the one that dominates the composite, and is what the h5 conversion
    calibrates against.
    """
    items_path = window_dir / "items.json"
    if not items_path.exists():
        return None
    scenes: dict[str, dict[str, str]] = {}
    saw_landsat = False
    for entry in json.loads(items_path.read_text()):
        layer_name = entry.get("layer_name", "")
        if not layer_name.startswith(LAYER_PREFIX):
            continue
        saw_landsat = True
        groups = entry.get("serialized_item_groups") or []
        if not (groups and groups[0]):
            continue
        base = groups[0][0]
        blob_path = base.get("blob_path")
        if not blob_path:
            continue
        # The layer's period start, recorded so the loader can match a
        # timestep to its month by date. These layers are a 30-day grid, not
        # calendar months (moNN can start in month NN-1), so the loader must
        # not infer the month from the date itself.
        time_ranges = entry.get("group_time_ranges") or []
        start = time_ranges[0][0] if time_ranges and time_ranges[0] else None
        if not start:
            continue
        scenes[f"mo{int(layer_name[-2:]):02d}"] = {
            "name": base.get("name", ""),
            "blob_path": blob_path,
            "start": start,
        }
    return scenes if saw_landsat else None


def main() -> int:
    """Build the sidecar and print calibration coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds_path", required=True, help="Tree holding the items.json.")
    parser.add_argument(
        "--out",
        default=None,
        help="Dataset root to write the sidecar to (default ds_path).",
    )
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    root = Path(args.ds_path)
    window_dirs = [
        p
        for p in (root / "windows").glob("*/*")
        if p.is_dir() and not p.parent.name.endswith(FETCH_GROUP_SUFFIX)
    ]
    if not window_dirs:
        print(f"no windows under {root}/windows")
        return 1
    print(f"scanning {len(window_dirs)} windows in {root.name}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        scanned = list(pool.map(read_window_scenes, window_dirs))
    per_window = {
        f"{d.parent.name}/{d.name}": scenes
        for d, scenes in zip(window_dirs, scanned)
        if scenes is not None
    }
    skipped = len(window_dirs) - len(per_window)
    if skipped:
        print(f"skipped {skipped} windows without landsat items")

    blob_paths = sorted(
        {s["blob_path"] for months in per_window.values() for s in months.values()}
    )
    print(f"fetching SUN_ELEVATION for {len(blob_paths)} unique scenes")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        elevations = list(pool.map(fetch_sun_elevation, blob_paths))
    sun_by_blob = dict(zip(blob_paths, elevations))
    failed = sum(1 for e in elevations if e is None)
    if failed:
        print(f"WARNING: {failed}/{len(blob_paths)} scenes had no readable MTL")

    windows: dict[str, dict[str, dict[str, object] | None]] = {}
    for key, months in per_window.items():
        entry: dict[str, dict[str, object] | None] = {
            f"mo{m:02d}": None for m in range(1, MONTHS + 1)
        }
        for month, scene in months.items():
            sun_elevation = sun_by_blob.get(scene["blob_path"])
            if sun_elevation is None:
                continue
            entry[month] = {
                "sun_elevation": sun_elevation,
                "platform": platform_from_scene_id(scene["name"]),
                "start": scene["start"],
            }
        windows[key] = entry

    out_root = Path(args.out) if args.out else root
    out_path = out_root / OUTPUT_NAME
    out_path.write_text(
        json.dumps(
            {
                "meta": {
                    "generated_by": "scripts/tools/build_landsat_calibration_sidecar.py",
                    "items_from": str(root),
                    "value": "sun_elevation (degrees), platform and layer "
                    "period start of the first (top) item in each month's "
                    "landsat mosaic group; null = no scene or unreadable MTL",
                    "windows_scanned": len(windows),
                    "windows_without_landsat_items": skipped,
                    "unique_scenes": len(blob_paths),
                    "scenes_without_mtl": failed,
                },
                "windows": windows,
            }
        )
    )
    print(f"wrote {out_path}")

    total = MONTHS * len(windows)
    calibrated = sum(1 for m in windows.values() for v in m.values() if v is not None)
    print(
        f"\nwindow-months: {total}; calibrated: {calibrated} "
        f"({calibrated / total:.1%}); MISSING at eval: {total - calibrated}"
    )
    fully_missing = sum(
        1 for m in windows.values() if all(v is None for v in m.values())
    )
    if fully_missing:
        print(
            f"windows with NO calibrated month: {fully_missing} "
            f"({fully_missing / len(windows):.1%}) -- these run landsat-all-MISSING"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
