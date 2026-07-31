"""Verify a *_year_aligned dataset's imagery after prepare, before materialize.

Checks, over a sample of windows, everything the re-export was supposed to fix:

1. windows are anchored to a calendar year -- (Jan 1 Y, Jan 1 Y+1)
2. all 24 monthly layers are present (i.e. prepare finished)
3. how many timesteps matched no scene, per sensor -- "prepared" counts windows
   that were *attempted*, so a clean prepare summary does not prove S1 landed
4. acquisition dates ASCEND from mo01 to mo12 -- the original export stored them
   descending (per_period_mosaic_reverse_time_order defaults to True), which is
   the defect the monthly layer scheme removes structurally
5. dates fall inside the window's own calendar year
6. S1 and S2 timestep i cover the same 30-day period, since both use identical
   time_offsets

Dates come from each item's ``geometry.time_range`` rather than from parsing
scene names, so it does not depend on a sensor's naming convention.

Usage::

    python scripts/tools/check_year_aligned_imagery.py \
        --ds_path /weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/pastis_year_aligned

    # every dataset at once
    for n in africa_crop_mask canada_crops_coarse canada_crops_fine descals \
             ethiopia_crops glance lcmap_lu us_trees pastis; do
        python scripts/tools/check_year_aligned_imagery.py --ds_path \
            /weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/${n}_year_aligned
    done
"""

import argparse
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

MONTHS = 12
SENSORS = {"sentinel2_l2a_mo": "s2", "sentinel1_mo": "s1"}


def sensor_of(layer_name: str) -> str | None:
    """Map a layer name to 's2'/'s1', or None if it is not monthly imagery."""
    for prefix, sensor in SENSORS.items():
        if layer_name.startswith(prefix):
            return sensor
    return None


def first_date(entry: dict) -> datetime | None:
    """Start of the first matched item's time range, or None if nothing matched."""
    groups = entry.get("serialized_item_groups") or []
    if not groups or not groups[0]:
        return None
    time_range = (groups[0][0].get("geometry") or {}).get("time_range")
    if not time_range:
        return None
    return datetime.fromisoformat(time_range[0])


def main() -> int:
    """Run the checks and return a nonzero exit code if any fail."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds_path", required=True)
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.ds_path)
    window_dirs = [p for p in (root / "windows").glob("*/*") if p.is_dir()]
    if not window_dirs:
        print(f"no windows under {root}/windows")
        return 1
    random.seed(args.seed)
    sample = random.sample(window_dirs, min(args.sample, len(window_dirs)))

    empty: Counter = Counter()
    total: Counter = Counter()
    problems: Counter = Counter()
    examples: dict[str, str] = {}
    checked = 0

    def note(kind: str, detail: str) -> None:
        problems[kind] += 1
        examples.setdefault(kind, detail)

    for window_dir in sample:
        items_path = window_dir / "items.json"
        meta_path = window_dir / "metadata.json"
        if not items_path.exists():
            note("no items.json", str(window_dir))
            continue
        checked += 1

        year = None
        if meta_path.exists():
            time_range = json.loads(meta_path.read_text()).get("time_range")
            if time_range:
                start = datetime.fromisoformat(time_range[0])
                end = datetime.fromisoformat(time_range[1])
                year = start.year
                if (start.month, start.day) != (1, 1) or (end.month, end.day) != (1, 1):
                    note(
                        "window not calendar-anchored",
                        f"{window_dir.name} {time_range}",
                    )

        # layer -> date, per sensor, indexed by month number
        dates: dict[str, dict[int, datetime | None]] = {"s1": {}, "s2": {}}
        for entry in json.loads(items_path.read_text()):
            sensor = sensor_of(entry["layer_name"])
            if sensor is None:
                continue
            month = int(entry["layer_name"][-2:])
            date = first_date(entry)
            dates[sensor][month] = date
            total[sensor] += 1
            if date is None:
                empty[sensor] += 1

        for sensor in ("s2", "s1"):
            got = dates[sensor]
            if len(got) != MONTHS:
                note(
                    f"{sensor}: missing layers", f"{window_dir.name} has {len(got)}/12"
                )
                continue
            present = [(m, d) for m, d in sorted(got.items()) if d is not None]
            ordered = [d for _, d in present]
            if ordered != sorted(ordered):
                note(f"{sensor}: dates not ascending", f"{window_dir.name} {ordered}")
            if year is not None:
                outside = [d.isoformat()[:10] for d in ordered if d.year != year]
                if outside:
                    note(
                        f"{sensor}: dates outside window year",
                        f"{window_dir.name} {outside}",
                    )

        # co-registration: same month index should sit in the same 30-day period
        for month in range(1, MONTHS + 1):
            d1, d2 = dates["s1"].get(month), dates["s2"].get(month)
            if d1 is None or d2 is None:
                continue
            if abs((d1 - d2).days) > 31:
                note(
                    "s1/s2 timestep misaligned",
                    f"{window_dir.name} mo{month:02d}: s1={d1.date()} s2={d2.date()}",
                )

    print(f"dataset: {root.name}")
    print(f"windows sampled: {checked}/{len(sample)} (of {len(window_dirs)} total)\n")

    print("empty timesteps (matched no scene):")
    for sensor in ("s2", "s1"):
        if total[sensor]:
            pct = empty[sensor] / total[sensor]
            flag = "  <-- LOOK" if pct > 0.25 else ""
            print(
                f"  {sensor}: {empty[sensor]:6d}/{total[sensor]:6d}  {pct:6.1%}{flag}"
            )
        else:
            print(f"  {sensor}: no layers found  <-- LOOK")
    print()

    if not problems:
        print("all structural checks passed: calendar-anchored, 24 layers, dates")
        print("ascending within the window's year, S1/S2 timesteps co-registered")
        return 0

    print("PROBLEMS:")
    for kind, count in problems.most_common():
        print(f"  {kind}: {count} window(s)")
        print(f"      e.g. {examples[kind]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
