"""Re-anchor an eval dataset onto the calendar year of its labels (weka-side).

Companion to build_year_aligned_eval_configs.py. Given a copy of an existing
eval dataset, this rewrites its rslearn config.json to use the twelve ascending
30-day Sentinel-1 + Sentinel-2 layers from pretraining, and moves every
window's time range to the calendar year of its label so the imagery spans
Jan-Dec of that year -- the same support AEF and Tessera are built over.

The label year is taken from the midpoint of the window's *current* time range,
which is what ``embedding_materializer/providers.get_target_year`` already used
to fetch AEF/Tessera. Since the new range is (Jan 1 Y, Jan 1 Y+1) its midpoint
stays in year Y, so **the existing AEF and Tessera layers remain correct and do
not need re-fetching**. ``plan`` prints the year histogram so you can confirm
that mapping per dataset before writing (it is exact for canada, whose windows
are (DATE_COLL - 1yr, DATE_COLL); verify ethiopia_crops and us_trees, whose
anchors are not mid-year).

Runbook, per dataset::

    SRC=/weka/dfive-default/olmoearth/eval_datasets/canada_crops_coarse
    DST=${SRC}_year_aligned

    # 1. copy everything except the imagery we are replacing
    rsync -a --info=progress2 \
        --exclude='layers/sentinel2' --exclude='layers/sentinel2_l2a_mo*' \
        --exclude='layers/sentinel1_mo*' \
        "$SRC/" "$DST/"

    # 2. inspect, then apply
    python scripts/tools/reanchor_year_aligned_dataset.py plan --ds_path "$DST"
    python scripts/tools/reanchor_year_aligned_dataset.py apply --ds_path "$DST"

    # 3. fetch the new imagery
    rslearn dataset prepare  --root "$DST" --workers 64
    rslearn dataset materialize --root "$DST" --workers 64

    # 4. register the new eval task
    python -m olmoearth_pretrain.evals.studio_ingest.cli ingest \
        --name canada_crops_coarse_year_aligned --source "$DST" \
        --olmoearth-run-config-path \
            data/rslearn_dataset_configs/canada_crops_coarse_year_aligned \
        --start-time <YYYY>-01-01 --end-time <YYYY>-12-31 --register

Step 4's start/end are only a fallback for timestamp synthesis, and a single
dataset-level range cannot describe a multi-year dataset at all; prefer landing
the per-window timestamp fix so the real acquisition dates are used.

NOTE, unresolved: the AEF/Tessera layers live in the *eval_datasets* copy (the
embedding materializer and wire_embedding_modalities.py both run there), not in
the rslearn-eai source that `ingest --source` reads from. So whether the
existing embedding layers survive depends on which copy you rsync from and
whether `ingest` re-copies over them. Settle this on the first dataset before
scaling out -- see the two options in the handover notes.
"""

import argparse
import json
import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from rslearn.dataset import Dataset
from upath import UPath

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "data" / "rslearn_dataset_configs"

# The pretraining layer definitions. Copied verbatim so the eval imagery is
# fetched exactly the way the training data was.
PRETRAIN_CONFIGS = ("config_sentinel2_l2a.json", "config_sentinel1.json")

# Offsets in the pretraining configs are centred on the window start
# (-180d .. +150d). We want the twelve layers to tile the calendar year
# forward from Jan 1, so they are rewritten to 0d .. 330d -- the convention
# config_pastis_rslearn.json already uses for an eval export. Layer names,
# band sets and data source args are otherwise untouched.
MONTHLY_PREFIXES = ("sentinel2_l2a_mo", "sentinel1_mo")
DROP_LAYERS = {"sentinel2", "sentinel1"}


def is_monthly(layer_name: str) -> bool:
    """Whether a layer belongs to the twelve-month imagery scheme."""
    return layer_name.startswith(MONTHLY_PREFIXES)


def build_monthly_layers() -> dict:
    """Load the pretraining monthly layers, re-offset to tile Jan 1 -> Dec 27."""
    layers: dict = {}
    for filename in PRETRAIN_CONFIGS:
        config = json.loads((CONFIG_ROOT / filename).read_text())
        for name, layer in config["layers"].items():
            if not is_monthly(name):
                continue  # skip the "_freq" INTERSECTS layers
            month = int(name[-2:])
            layer = json.loads(json.dumps(layer))  # deep copy
            layer["data_source"]["time_offset"] = f"{(month - 1) * 30}d"
            layer["data_source"]["duration"] = "30d"
            layers[name] = layer
    expected = 24
    if len(layers) != expected:
        raise ValueError(f"expected {expected} monthly layers, built {len(layers)}")
    return layers


def label_years(dataset: Dataset) -> tuple[Counter, int]:
    """Histogram of label years (current time-range midpoint) and range-less count."""
    years: Counter = Counter()
    missing = 0
    for window in dataset.storage.get_windows():
        if window.time_range is None:
            missing += 1
            continue
        start, end = window.time_range
        years[(start + (end - start) / 2).year] += 1
    return years, missing


def plan(ds_path: UPath) -> None:
    """Report what apply() would change, without writing anything."""
    dataset = Dataset(ds_path)
    config = json.loads((ds_path / "config.json").read_text())
    existing = config.get("layers", {})

    dropped = [n for n in existing if is_monthly(n) or n in DROP_LAYERS]
    kept = [n for n in existing if n not in dropped]
    added = sorted(build_monthly_layers())

    print(f"dataset: {ds_path}")
    print(f"  layers dropped ({len(dropped)}): {', '.join(sorted(dropped)) or '-'}")
    print(f"  layers kept    ({len(kept)}): {', '.join(sorted(kept)) or '-'}")
    print(f"  layers added   ({len(added)}): {added[0]} .. {added[-1]}")

    years, missing = label_years(dataset)
    total = sum(years.values())
    print(f"\n  label year (from current midpoint), {total} windows:")
    for year in sorted(years):
        print(f"    {year} -> ({year}-01-01, {year + 1}-01-01)   n={years[year]}")
    if missing:
        print(f"  !! {missing} windows have no time range and will be skipped")
    print("\n  AEF/Tessera target year is the new midpoint (Jul 1) -> unchanged,")
    print("  so existing embedding layers stay valid. Confirm the mapping above")
    print("  looks like this dataset's label years before running apply.")


def apply(ds_path: UPath) -> None:
    """Rewrite config.json and re-anchor every window to its label year."""
    config_path = ds_path / "config.json"
    config = json.loads(config_path.read_text())

    layers = {
        name: layer
        for name, layer in config.get("layers", {}).items()
        if not is_monthly(name) and name not in DROP_LAYERS
    }
    layers.update(build_monthly_layers())
    config["layers"] = layers
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"wrote {config_path} with {len(layers)} layers")

    dataset = Dataset(ds_path)
    moved = skipped = 0
    for window in dataset.storage.get_windows():
        if window.time_range is None:
            skipped += 1
            continue
        start, end = window.time_range
        year = (start + (end - start) / 2).year
        window.time_range = (
            datetime(year, 1, 1, tzinfo=UTC),
            datetime(year + 1, 1, 1, tzinfo=UTC),
        )
        window.save()
        moved += 1
    logger.info(f"re-anchored {moved} windows, skipped {skipped} without a time range")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "apply"])
    parser.add_argument(
        "--ds_path",
        required=True,
        help="Path to the DESTINATION dataset copy (never the original).",
    )
    args = parser.parse_args()

    ds_path = UPath(args.ds_path)
    if args.command == "plan":
        plan(ds_path)
    else:
        apply(ds_path)


if __name__ == "__main__":
    main()
