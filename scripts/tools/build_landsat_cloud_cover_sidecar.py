"""Build the per-window, per-month Landsat cloud-cover sidecar for a dataset.

Unlike the Sentinel-2 case (whose STAC source discards eo:cloud_cover unless
properties_to_record is set), rslearn's ``LandsatOliTirsItem`` persists each
scene's ``cloud_cover`` into items.json natively -- so this is a purely local
walk, no network. The output feeds the eval loader's scene-level Landsat
cloud mask (``landsat_cloud_cover_max`` on ``RslearnToOlmoEarthDataset``):

    {
      "meta": {...},
      "windows": {"<group>/<name>": {"mo01": 43.2, ..., "mo12": null}, ...}
    }

The recorded value is the cloud_cover of the FIRST item in the month's
mosaic group (least-cloudy first via sort_by, so it dominates the
composite). ``null`` = month matched no scene; ``-1`` = the scene's metadata
carried no cover (never masked by the loader).

TWO-TREE NOTE: the Landsat items live in the STAGING tree's items.json (the
raster copy to the registered tree carries layer dirs only), while the eval
loader reads the sidecar from the REGISTERED dataset root. Point --ds_path
at the tree that has the items and --out at the tree evals read::

    python scripts/tools/build_landsat_cloud_cover_sidecar.py \
        --ds_path /weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/ethiopia_crops_year_aligned \
        --out /weka/dfive-default/olmoearth/eval_datasets/ethiopia_crops_year_aligned
"""

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LAYER_PREFIX = "landsat_mo"
MONTHS = 12
OUTPUT_NAME = "landsat_cloud_cover.json"
FETCH_GROUP_SUFFIX = "_tessera_v2"


def read_window_covers(window_dir: Path) -> dict[str, float | None] | None:
    """Map "moNN" -> top mosaic scene's cloud_cover, or None if unprepared."""
    items_path = window_dir / "items.json"
    if not items_path.exists():
        return None
    covers: dict[str, float | None] = {f"mo{m:02d}": None for m in range(1, MONTHS + 1)}
    saw_landsat = False
    for entry in json.loads(items_path.read_text()):
        layer_name = entry.get("layer_name", "")
        if not layer_name.startswith(LAYER_PREFIX):
            continue
        saw_landsat = True
        groups = entry.get("serialized_item_groups") or []
        if groups and groups[0]:
            covers[f"mo{int(layer_name[-2:]):02d}"] = groups[0][0].get("cloud_cover")
    return covers if saw_landsat else None


def main() -> int:
    """Build the sidecar and print the cover distribution."""
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
        scanned = list(pool.map(read_window_covers, window_dirs))
    windows = {
        f"{d.parent.name}/{d.name}": covers
        for d, covers in zip(window_dirs, scanned)
        if covers is not None
    }
    skipped = len(window_dirs) - len(windows)
    if skipped:
        print(f"skipped {skipped} windows without landsat items")

    out_root = Path(args.out) if args.out else root
    out_path = out_root / OUTPUT_NAME
    out_path.write_text(
        json.dumps(
            {
                "meta": {
                    "generated_by": "scripts/tools/build_landsat_cloud_cover_sidecar.py",
                    "items_from": str(root),
                    "value": "cloud_cover of the first (top) item in each "
                    "month's landsat mosaic group; -1 = unknown",
                    "windows_scanned": len(windows),
                    "windows_without_landsat_items": skipped,
                },
                "windows": windows,
            }
        )
    )
    print(f"wrote {out_path}")

    covers = [
        c for months in windows.values() for c in months.values() if c is not None
    ]
    total = sum(len(m) for m in windows.values())
    if not covers:
        print("no covers recorded")
        return 0
    known = [c for c in covers if c >= 0]
    print(
        f"\nwindow-months: {total}; no scene: {total - len(covers)}; unknown (-1): {len(covers) - len(known)}"
    )
    if known:
        known.sort()
        print(
            f"cloud_cover of top scene: mean {sum(known) / len(known):.1f}, median {known[len(known) // 2]:.1f}"
        )
        for threshold in (30, 50, 70):
            share = sum(1 for c in known if c >= threshold) / len(known)
            print(f"  >= {threshold}%: {share:.1%}  (masked at threshold {threshold})")
    counts = Counter(
        sum(1 for c in m.values() if c is not None and c >= 50)
        for m in windows.values()
    )
    fully = counts.get(MONTHS, 0)
    print(
        f"windows with ALL 12 months >= 50%: {fully} ({fully / len(windows):.1%}) -- these run landsat-all-MISSING"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
