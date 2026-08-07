"""Verify a dataset's SCL layers after setup_extra_layers.py + materialize.

Checks, over a sample of windows:

1. every ``sentinel2_scl_moNN`` items entry exists and names EXACTLY the same
   scenes as its ``sentinel2_l2a_moNN`` sibling -- the copy is only a valid
   cloud mask if it describes the imagery it masks
2. SCL materialized wherever the imagery did (completed markers per month) --
   a gap means the eval loader silently skips masking for that window
3. with ``--pixels``, reads the SCL rasters and reports the class histogram
   and cloudy fraction (SCL in {0,1,3,8,9,10}) overall and at the window
   center pixel -- the diagnostic the whole exercise exists for: descals
   should be far cloudier than lcmap/canada

Usage::

    python scripts/tools/check_scl_layers.py \
        --ds_path /weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/descals_year_aligned

    # every dataset at once
    for n in africa_crop_mask canada_crops_coarse canada_crops_fine descals \
             ethiopia_crops glance lcmap_lu us_trees pastis; do
        python scripts/tools/check_scl_layers.py --pixels --ds_path \
            /weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/${n}_year_aligned
    done
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

MONTHS = 12
S2_PREFIX = "sentinel2_l2a_mo"
SCL_PREFIX = "sentinel2_scl_mo"
CLOUD_CLASSES = (0, 1, 3, 8, 9, 10)
SCL_CLASS_NAMES = {
    0: "nodata",
    1: "saturated",
    2: "dark",
    3: "shadow",
    4: "vegetation",
    5: "bare",
    6: "water",
    7: "unclassified",
    8: "cloud-med",
    9: "cloud-high",
    10: "cirrus",
    11: "snow",
}


def item_names(entry: dict) -> list[str]:
    """Scene names in an items.json entry, in mosaic order."""
    groups = entry.get("serialized_item_groups") or [[]]
    return [item["name"] for item in (groups[0] if groups else [])]


def check_window(window_dir: Path) -> tuple[Counter, dict[str, str]]:
    """Run the structural checks on one window."""
    problems: Counter = Counter()
    examples: dict[str, str] = {}

    def note(kind: str, detail: str) -> None:
        problems[kind] += 1
        examples.setdefault(kind, detail)

    items_path = window_dir / "items.json"
    if not items_path.exists():
        note("no items.json", window_dir.name)
        return problems, examples

    entries = {e["layer_name"]: e for e in json.loads(items_path.read_text())}
    for month in range(1, MONTHS + 1):
        s2 = entries.get(f"{S2_PREFIX}{month:02d}")
        scl = entries.get(f"{SCL_PREFIX}{month:02d}")
        if s2 is None:
            continue  # window not fully prepared; required inputs drop it
        if scl is None:
            note("scl items entry missing", f"{window_dir.name} mo{month:02d}")
            continue
        if item_names(s2) != item_names(scl):
            note("scl/s2 scenes differ", f"{window_dir.name} mo{month:02d}")

        s2_done = (
            window_dir / "layers" / f"{S2_PREFIX}{month:02d}" / "completed"
        ).exists()
        scl_done = (
            window_dir / "layers" / f"{SCL_PREFIX}{month:02d}" / "completed"
        ).exists()
        if s2_done and not scl_done:
            note("scl not materialized", f"{window_dir.name} mo{month:02d}")

    return problems, examples


def pixel_stats(window_dirs: list[Path]) -> None:
    """Read sampled SCL rasters and print class/cloud statistics."""
    import numpy as np
    import rasterio

    class_counts: Counter = Counter()
    total = 0
    center_cloudy = 0
    center_months = 0
    for window_dir in window_dirs:
        for month in range(1, MONTHS + 1):
            layer_dir = window_dir / "layers" / f"{SCL_PREFIX}{month:02d}"
            tifs = list(layer_dir.glob("*/*.tif")) if layer_dir.exists() else []
            if not tifs:
                continue
            with rasterio.open(tifs[0]) as src:
                scl = src.read(1)
            values, counts = np.unique(scl, return_counts=True)
            class_counts.update(dict(zip(values.tolist(), counts.tolist())))
            total += scl.size
            center_months += 1
            if scl[scl.shape[0] // 2, scl.shape[1] // 2] in CLOUD_CLASSES:
                center_cloudy += 1

    if not total:
        print("\npixel stats: no materialized SCL rasters in the sample yet")
        return
    print("\nSCL class histogram (sampled windows, all pixels):")
    for value, count in sorted(class_counts.items()):
        name = SCL_CLASS_NAMES.get(value, f"class {value}")
        flag = "  <- cloudy" if value in CLOUD_CLASSES else ""
        print(f"  {value:3d} {name:12s} {count / total:6.1%}{flag}")
    cloudy = sum(class_counts.get(v, 0) for v in CLOUD_CLASSES)
    print(f"\ncloudy fraction, all pixels:    {cloudy / total:.1%}")
    print(f"cloudy fraction, center pixel:  {center_cloudy / center_months:.1%}")


def main() -> int:
    """Run the checks and return nonzero if any structural check fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds_path", required=True)
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pixels",
        action="store_true",
        help="Also read sampled SCL rasters and report class/cloud stats.",
    )
    args = parser.parse_args()

    root = Path(args.ds_path)
    # Exclude tessera_v2 fetch groups (no monthly layers, not eval windows);
    # same exclusion as setup_extra_layers.py.
    window_dirs = [
        p
        for p in (root / "windows").glob("*/*")
        if p.is_dir() and not p.parent.name.endswith("_tessera_v2")
    ]
    if not window_dirs:
        print(f"no windows under {root}/windows")
        return 1
    random.seed(args.seed)
    sample = random.sample(window_dirs, min(args.sample, len(window_dirs)))

    problems: Counter = Counter()
    examples: dict[str, str] = {}
    for window_dir in sample:
        window_problems, window_examples = check_window(window_dir)
        problems.update(window_problems)
        for kind, detail in window_examples.items():
            examples.setdefault(kind, detail)

    print(f"dataset: {root.name}")
    print(f"windows sampled: {len(sample)}/{len(window_dirs)}")

    if args.pixels:
        pixel_stats(sample)

    if not problems:
        print("\nall structural checks passed: SCL items mirror S2, materialized")
        print("wherever the imagery is")
        return 0
    print("\nPROBLEMS:")
    for kind, count in problems.most_common():
        print(f"  {kind}: {count}")
        print(f"      e.g. {examples[kind]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
