"""Create the extra-period copies of PASTIS and PLANTEUR for the window-retrieval evals.

Window retrieval composes stored annual embeddings across years, e.g.
Sep 2018-Aug 2019 from z_2018 and z_2019, and is scored against fresh
inference on the crop year itself. The eval datasets only have calendar-2019
imagery (``pastis_year_aligned``, PLANTEUR's ``pastis2_drom_2019``), so this
builds NEW staging datasets holding the same windows for other periods:

- ``pastis_cal2018`` / ``planteur_cal2018``: re-anchored to calendar 2018.
  Window metadata is copied with only ``time_range`` changed, the label layers
  as-is; imagery is fetched fresh (``items.json`` is not copied).
- ``pastis_sepaug``: ``pastis_rslearn``'s windows (Sep 2018 - Aug 2019) with
  its already-materialized S2/S1 monthlies and ``items.json`` copied, so only
  the Landsat layers need fetching. ``pastis_rslearn`` itself has no Landsat
  and is a registered eval dataset whose config.json must not change.

Window names, bounds, projection and split / LOIO options are unchanged, so
windows pair 1:1 with the 2019 datasets by name. config.json holds the 24
S2/S1 monthly layer definitions from the seed and the 12 Landsat monthlies
from ``landsat_from`` (the seed itself, or ``pastis_year_aligned`` for
``pastis_sepaug``), verbatim, plus the label layers: the same sources,
offsets and sorting the 2019 floor was built with. Embedding products, SCL,
Landsat QA and the *_all fetch layers are dropped.

The seeds are only ever read. Destinations must not be under
``eval_datasets``; ``apply`` refuses an existing destination unless
``--resume`` (which rewrites config.json and skips windows already copied).
After ``apply``, run rslearn prepare + materialize on the destination, then
ingest it under a new eval name.

Stdlib only, so it runs in any image with weka mounted::

    python scripts/tools/setup_time_window_exports.py plan
    python scripts/tools/setup_time_window_exports.py apply --only pastis_cal2018
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

STAGE_ROOT = Path("/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals")
EVAL_DATASETS = Path("/weka/dfive-default/olmoearth/eval_datasets")

S2_S1_PREFIXES = ("sentinel2_l2a_mo", "sentinel1_mo")
LANDSAT_PREFIX = "landsat_mo"
MONTHS = 12

# name -> seed dataset, window group, label layers, new (start, end) or None to
# keep the seed's, where the Landsat layer definitions come from, and whether
# the seed's S2/S1 imagery is copied. Each end follows its seed's convention:
# pastis_year_aligned windows run to Jan 1 of the next year, PLANTEUR's to
# Dec 27 (12 x 30 days).
TARGETS = {
    "pastis_cal2018": {
        "seed": EVAL_DATASETS / "pastis_year_aligned",
        "group": "pastis",
        "label_layers": ("label",),
        "time_range": (
            datetime(2018, 1, 1, tzinfo=UTC),
            datetime(2019, 1, 1, tzinfo=UTC),
        ),
        "landsat_from": EVAL_DATASETS / "pastis_year_aligned",
        "copy_imagery": False,
    },
    "planteur_cal2018": {
        "seed": EVAL_DATASETS / "pastis2_drom_2019",
        "group": "rpg_2019_cal",
        "label_layers": ("label", "label_bgdown"),
        "time_range": (
            datetime(2018, 1, 1, tzinfo=UTC),
            datetime(2018, 12, 27, tzinfo=UTC),
        ),
        "landsat_from": EVAL_DATASETS / "pastis2_drom_2019",
        "copy_imagery": False,
    },
    "pastis_sepaug": {
        "seed": EVAL_DATASETS / "pastis_rslearn",
        "group": "pastis",
        "label_layers": ("label",),
        "time_range": None,
        "landsat_from": EVAL_DATASETS / "pastis_year_aligned",
        "copy_imagery": True,
    },
}


def build_config(
    seed_config: dict, landsat_config: dict, label_layers: tuple[str, ...]
) -> tuple[dict, list[str]]:
    """S2/S1 monthlies and labels from the seed, Landsat monthlies from landsat_config."""
    layers = seed_config["layers"]
    keep = {
        name: layer
        for name, layer in layers.items()
        if name.startswith(S2_S1_PREFIXES) or name in label_layers
    }
    s2_s1 = [n for n in keep if n.startswith(S2_S1_PREFIXES)]
    if len(s2_s1) != 2 * MONTHS:
        raise ValueError(
            f"expected 24 S2/S1 monthly layers in the seed, found {len(s2_s1)}"
        )
    missing = [n for n in label_layers if n not in keep]
    if missing:
        raise ValueError(f"label layers missing from seed config: {missing}")
    landsat = {
        n: v
        for n, v in landsat_config["layers"].items()
        if n.startswith(LANDSAT_PREFIX)
    }
    if len(landsat) != MONTHS:
        raise ValueError(f"expected 12 Landsat monthly layers, found {sorted(landsat)}")
    keep.update(landsat)
    config = {k: v for k, v in seed_config.items() if k != "layers"}
    config["layers"] = keep
    dropped = sorted(n for n in layers if n not in keep)
    return config, dropped


def target_config(target: dict) -> tuple[dict, list[str]]:
    """The destination config.json and the seed layers it drops."""
    seed_config = json.loads((target["seed"] / "config.json").read_text())
    landsat_config = json.loads((target["landsat_from"] / "config.json").read_text())
    return build_config(seed_config, landsat_config, target["label_layers"])


def check_destination(name: str, seed: Path, dest: Path, resume: bool) -> None:
    """Refuse destinations that could touch an existing dataset."""
    if dest.resolve() == seed.resolve():
        raise SystemExit(f"{name}: destination is the seed")
    if EVAL_DATASETS in dest.resolve().parents:
        raise SystemExit(f"{name}: destination {dest} is under {EVAL_DATASETS}")
    if dest.exists() and not resume:
        raise SystemExit(
            f"{name}: {dest} already exists (pass --resume to continue a partial apply)"
        )


def window_dirs(seed: Path, group: str) -> list[Path]:
    """Window directories of one group, sorted by name."""
    return sorted(p for p in (seed / "windows" / group).iterdir() if p.is_dir())


def iso(dt: datetime) -> str:
    """Timestamp in the form rslearn writes to metadata.json."""
    return dt.isoformat()


def copied_layers(target: dict) -> list[str]:
    """Layer directories copied from each seed window."""
    layers = list(target["label_layers"])
    if target["copy_imagery"]:
        layers += [f"{p}{m:02d}" for p in S2_S1_PREFIXES for m in range(1, MONTHS + 1)]
    return layers


def plan(name: str, target: dict, dest: Path, resume: bool) -> None:
    """Report what apply would do. Reads the seeds only."""
    seed, group = target["seed"], target["group"]
    print(f"\n=== {name}")
    print(f"  seed: {seed} (group {group})")
    print(f"  dest: {dest}")
    check_destination(name, seed, dest, resume)
    print(f"  dest exists: {dest.exists()}")

    config, dropped = target_config(target)
    print(f"  layers kept ({len(config['layers'])}): {sorted(config['layers'])}")
    print(f"  layers dropped ({len(dropped)}): {dropped}")
    for layer in ("sentinel2_l2a_mo01", "sentinel1_mo01", "landsat_mo01"):
        src = config["layers"][layer]["data_source"]
        print(
            f"  {layer}: {src['class_path']} offset {src.get('time_offset')} duration {src.get('duration')}"
        )
    print(f"  Landsat layer definitions from: {target['landsat_from']}")
    print(
        f"  S2/S1 imagery: {'copied from seed' if target['copy_imagery'] else 'fetched fresh'}"
    )

    windows = window_dirs(seed, group)
    ranges: Counter = Counter()
    splits: Counter = Counter()
    incomplete = 0
    has_items = 0
    for w in windows:
        meta = json.loads((w / "metadata.json").read_text())
        tr = meta.get("time_range")
        ranges[f"{tr[0][:10]}..{tr[1][:10]}" if tr else "None"] += 1
        splits[(meta.get("options") or {}).get("eval_split", "-")] += 1
        if not all((w / "layers" / layer).is_dir() for layer in copied_layers(target)):
            incomplete += 1
        has_items += (w / "items.json").exists()
    print(f"  windows: {len(windows)}  eval_split {dict(splits)}")
    print(f"  time_range now: {dict(ranges)}")
    if target["time_range"] is None:
        print("  time_range new: unchanged")
    else:
        start, end = target["time_range"]
        print(
            f"  time_range new: {iso(start)[:10]}..{iso(end)[:10]} for all {len(windows)}"
        )
    print(f"  windows missing a copied layer: {incomplete}")
    print(
        f"  seed windows with items.json: {has_items} ({'copied' if target['copy_imagery'] else 'NOT copied'})"
    )


def copy_window(src: Path, dest_group: Path, target: dict) -> bool:
    """Copy one window's metadata and layers. False if already done."""
    dst = dest_group / src.name
    marker = dst / "metadata.json"
    if marker.exists():
        return False
    (dst / "layers").mkdir(parents=True, exist_ok=True)
    for layer in copied_layers(target):
        if (src / "layers" / layer).is_dir():
            shutil.copytree(
                src / "layers" / layer, dst / "layers" / layer, dirs_exist_ok=True
            )
    if target["copy_imagery"] and (src / "items.json").exists():
        shutil.copyfile(src / "items.json", dst / "items.json")
    meta = json.loads((src / "metadata.json").read_text())
    if target["time_range"] is not None:
        meta["time_range"] = [
            iso(target["time_range"][0]),
            iso(target["time_range"][1]),
        ]
    # metadata.json last, so its presence marks a finished window for --resume.
    tmp = dst / "metadata.json.tmp"
    tmp.write_text(json.dumps(meta))
    os.replace(tmp, marker)
    return True


def apply(name: str, target: dict, dest: Path, resume: bool, workers: int) -> None:
    """Create (or complete) the destination dataset. Writes only under dest."""
    seed, group = target["seed"], target["group"]
    check_destination(name, seed, dest, resume)
    config, _ = target_config(target)
    dest_group = dest / "windows" / group
    dest_group.mkdir(parents=True, exist_ok=True)
    (dest / "config.json").write_text(json.dumps(config, indent=2))

    windows = window_dirs(seed, group)
    with ThreadPoolExecutor(workers) as pool:
        done = list(pool.map(lambda w: copy_window(w, dest_group, target), windows))
    written = sum(done)
    total = sum(1 for p in dest_group.iterdir() if (p / "metadata.json").exists())
    print(
        f"=== {name}: wrote {written} windows, {total}/{len(windows)} present at {dest}"
    )
    if total != len(windows):
        raise SystemExit(f"{name}: incomplete copy")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["plan", "apply"])
    parser.add_argument("--only", choices=sorted(TARGETS), default=None)
    parser.add_argument("--stage_root", type=Path, default=STAGE_ROOT)
    parser.add_argument(
        "--resume", action="store_true", help="continue a partial apply"
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    names = [args.only] if args.only else sorted(TARGETS)
    for name in names:
        dest = args.stage_root / name
        if args.command == "plan":
            plan(name, TARGETS[name], dest, args.resume)
        else:
            apply(name, TARGETS[name], dest, args.resume, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
