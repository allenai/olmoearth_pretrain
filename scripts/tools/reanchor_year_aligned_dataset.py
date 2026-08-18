"""Re-anchor an eval dataset onto the calendar year of its labels (weka-side).

Companion to build_year_aligned_eval_configs.py. Given a copy of an existing
eval dataset, this rewrites its rslearn config.json to use the twelve ascending
30-day Sentinel-1 + Sentinel-2 layers from pretraining, and moves every
window's time range to the calendar year of its label so the imagery spans
Jan-Dec of that year -- the same support AEF and Tessera are built over.

How the label year is read off the window's current time range is per-dataset --
see YEAR_RULES. Most datasets use "midpoint", matching what
``embedding_materializer/providers.get_target_year`` used to fetch AEF/Tessera,
so their existing embedding layers stay valid and need no re-fetching. canada
uses "end" instead, because its windows are (DATE_COLL - 1yr, DATE_COLL) and a
June collection date puts the midpoint in the previous year -- 221 of 16 079
coarse windows. ``plan`` prints the year histogram under the chosen rule and
counts how many windows it moves off the midpoint year; those are exactly the
windows whose AEF/Tessera layers must be re-materialized, since the baselines
would otherwise read a different year than OlmoEarth.

Runbook, per dataset::

    NAME=canada_crops_coarse
    # Seed from the eval_datasets copy, NOT the rslearn-eai source: only the
    # former carries the materialized gse/tessera layers, and reusing them is
    # what avoids re-fetching AEF/Tessera over every window.
    SEED=/weka/dfive-default/olmoearth/eval_datasets/$NAME
    DST=/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/${NAME}_year_aligned

    # 1. copy everything except the imagery we are replacing. rslearn stores
    # item group N of layer L as a SIBLING directory "L.N" (sentinel2,
    # sentinel2.1, ... sentinel2.11), so the groups need their own pattern.
    # Slash-free patterns match on the final path component only, which is
    # what we want -- these names occur nowhere else in the tree.
    mkdir -p "$DST"
    rsync -a --info=progress2 \
        --exclude='sentinel2' --exclude='sentinel2.*' \
        --exclude='sentinel1' --exclude='sentinel1.*' \
        --exclude='sentinel2_l2a_mo*' --exclude='sentinel1_mo*' \
        "$SEED/" "$DST/"

    # 2. inspect, then apply. --seed_path is optional the first time but
    # REQUIRED to re-run after changing a dataset's year rule: apply overwrites
    # the observation dates the rule needs, so re-reading them off the staging
    # copy would shift every window by a year. Passing it always is safe.
    python scripts/tools/reanchor_year_aligned_dataset.py plan \
        --ds_path "$DST" --seed_path "$SEED"
    python scripts/tools/reanchor_year_aligned_dataset.py apply \
        --ds_path "$DST" --seed_path "$SEED"

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
dataset-level range cannot describe a multi-year dataset at all. Since the
per-window timestamp fix landed (``rslearn_dataset._build_timestamps``) the
loader reads each timestep's real acquisition date off the imagery, so these
flags no longer matter for datasets whose rasters carry time ranges.

The AEF/Tessera layers live only in the *eval_datasets* copy -- the embedding
materializer and wire_embedding_modalities.py both run there, not on the
rslearn-eai source. That is why step 1 seeds from eval_datasets: `ingest` copies
with an unfiltered tree tar (studio_ingest/ingest.py:_tar_copy_cmd), so anything
in the seed rides through to the registered dataset, embedding layers included,
and nothing needs re-fetching.
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

# How to read a window's label year off its current time range.
#
# "midpoint" is what embedding_materializer/providers.get_target_year() uses, so
# it is the rule that keeps a window's existing AEF/Tessera layer valid. But it
# is only *correct* when the midpoint lands in the label's own year, and for
# observation-anchored windows -- range (obs - 1yr, obs) -- the midpoint is the
# observation minus ~6 months. So it is right for Jul-Dec observations and off by
# one for Jan-Jun ones.
#
# Measured 2026-07-31 on the seeds:
#   canada_crops_coarse  15858/16079 agree (DATE_COLL is Jun-Oct; the 210 June
#                        windows land in the previous year) -> use "end"
#   us_trees             23520/45382 agree (GBIF eventdates run year-round)
#   everything else      midpoint verified correct via `plan`: the four
#                        calendar-aligned datasets have range (Jan 1 Y, Jan 1
#                        Y+1) so the midpoint is always Jul 1 of Y, and
#                        ethiopia_crops / pastis each resolve to their single
#                        known label year.
#
# us_trees deliberately stays on "midpoint": correcting it would re-year ~22k
# windows and require re-materializing AEF for all of them, and tree genus does
# not change between adjacent years, so the offset costs almost nothing. Record
# it as "aligned to AEF" rather than "aligned to the label".
YEAR_RULES = {
    "canada_crops_coarse": "end",
    "canada_crops_fine": "end",
}
DEFAULT_YEAR_RULE = "midpoint"
YEAR_RULE_CHOICES = ("midpoint", "start", "end")


def resolve_year_rule(ds_path: UPath, override: str | None = None) -> str:
    """Pick the label-year rule for a dataset, by name unless overridden."""
    if override is not None:
        return override
    name = ds_path.name.removesuffix("_year_aligned")
    return YEAR_RULES.get(name, DEFAULT_YEAR_RULE)


def label_year(time_range: tuple[datetime, datetime], rule: str) -> int:
    """Read the label year off a window's current time range."""
    start, end = time_range
    if rule == "start":
        return start.year
    if rule == "end":
        return end.year
    if rule == "midpoint":
        return (start + (end - start) / 2).year
    raise ValueError(f"unknown year rule {rule!r}; expected one of {YEAR_RULE_CHOICES}")


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


def seed_time_ranges(seed_path: UPath) -> dict[str, tuple[datetime, datetime]]:
    """Map "group/name" -> original time range from an untouched seed dataset."""
    return {
        f"{w.group}/{w.name}": w.time_range
        for w in Dataset(seed_path).storage.get_windows()
        if w.time_range is not None
    }


def looks_reanchored(time_range: tuple[datetime, datetime]) -> bool:
    """Whether a range is already (Jan 1 Y, Jan 1 Y+1), i.e. apply() has run."""
    start, end = time_range
    return (start.month, start.day) == (1, 1) and (end.month, end.day) == (1, 1)


def source_ranges(
    dataset: Dataset, seed: dict[str, tuple[datetime, datetime]] | None
) -> list[tuple[object, tuple[datetime, datetime] | None]]:
    """Pair each staging window with the range the year rule should read.

    Uses the seed's original range when a seed is given, so re-running apply
    after a rule change is safe. Without a seed the staging window's own range
    is used, which is only correct the first time apply runs -- afterwards the
    range is (Jan 1 Y, Jan 1 Y+1) and every rule but "start" reads it wrongly.

    Args:
        dataset: the staging dataset being re-anchored.
        seed: original ranges by "group/name", or None.

    Returns:
        list of (window, range-to-read), with None where no range is available.

    Raises:
        SystemExit: if no seed was given and the staging windows already look
            re-anchored, which would silently shift every window by a year.
        KeyError: if a staging window is missing from the seed.
    """
    pairs = []
    already = 0
    for window in dataset.storage.get_windows():
        if seed is not None:
            key = f"{window.group}/{window.name}"
            if key not in seed:
                raise KeyError(f"window {key} not found in the seed dataset")
            pairs.append((window, seed[key]))
            continue
        if window.time_range is not None and looks_reanchored(window.time_range):
            already += 1
        pairs.append((window, window.time_range))

    if seed is None and already > 0:
        raise SystemExit(
            f"{already} windows already have (Jan 1, Jan 1) ranges, so apply() has "
            "run before and their original observation dates are gone. Re-reading "
            "the year off these would shift every window. Pass --seed_path "
            "pointing at the untouched eval_datasets copy."
        )
    return pairs


def label_years(pairs: list, rule: str) -> tuple[Counter, int, int]:
    """Histogram of label years under `rule`, plus range-less and re-yeared counts.

    Args:
        pairs: (window, source range) pairs from source_ranges().
        rule: one of YEAR_RULE_CHOICES.

    Returns:
        (year histogram, windows without a time range, windows whose year
        differs from the midpoint rule). The last is the scope of any AEF /
        Tessera re-materialization, since those layers were fetched on the
        midpoint year.
    """
    years: Counter = Counter()
    missing = 0
    differs = 0
    for _window, time_range in pairs:
        if time_range is None:
            missing += 1
            continue
        year = label_year(time_range, rule)
        years[year] += 1
        if year != label_year(time_range, "midpoint"):
            differs += 1
    return years, missing, differs


def plan(ds_path: UPath, rule: str, seed_path: UPath | None) -> None:
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

    seed = seed_time_ranges(seed_path) if seed_path else None
    years, missing, differs = label_years(source_ranges(dataset, seed), rule)
    total = sum(years.values())
    print(f"\n  label year (rule: {rule}), {total} windows:")
    for year in sorted(years):
        print(f"    {year} -> ({year}-01-01, {year + 1}-01-01)   n={years[year]}")
    if missing:
        print(f"  !! {missing} windows have no time range and will be skipped")

    if differs == 0:
        print("\n  Every window's year matches the midpoint rule, which is what")
        print("  get_target_year used, so existing AEF/Tessera layers stay valid.")
    else:
        print(f"\n  !! {differs} of {total} windows get a DIFFERENT year than the")
        print("  midpoint rule. Their existing AEF/Tessera layers were fetched on")
        print("  the midpoint year and are now off by one, so re-materialize those")
        print("  products for the affected windows or the baselines will be reading")
        print("  a different year than OlmoEarth. Windows that gapped under the old")
        print("  year may also gain a layer and re-enter the dataset, changing the")
        print("  window count relative to previously recorded numbers.")


def apply(ds_path: UPath, rule: str, seed_path: UPath | None) -> None:
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
    seed = seed_time_ranges(seed_path) if seed_path else None
    moved = skipped = differs = 0
    for window, time_range in source_ranges(dataset, seed):
        if time_range is None:
            skipped += 1
            continue
        year = label_year(time_range, rule)
        if year != label_year(time_range, "midpoint"):
            differs += 1
        window.time_range = (
            datetime(year, 1, 1, tzinfo=UTC),
            datetime(year + 1, 1, 1, tzinfo=UTC),
        )
        window.save()
        moved += 1
    logger.info(
        f"re-anchored {moved} windows using rule {rule!r}, "
        f"skipped {skipped} without a time range"
    )
    if differs:
        logger.warning(
            f"{differs} windows moved to a year other than the midpoint year; "
            "re-materialize AEF/Tessera for them so the baselines read the same "
            "year as OlmoEarth"
        )


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
    parser.add_argument(
        "--year_from",
        choices=YEAR_RULE_CHOICES,
        default=None,
        help=(
            "Override how the label year is read off the current time range. "
            "Defaults to this dataset's entry in YEAR_RULES, else "
            f"{DEFAULT_YEAR_RULE!r}."
        ),
    )
    parser.add_argument(
        "--seed_path",
        default=None,
        help=(
            "Untouched seed dataset to read original window time ranges from. "
            "Required to re-run apply after a year-rule change, since apply "
            "overwrites the observation dates it would otherwise need."
        ),
    )
    args = parser.parse_args()

    ds_path = UPath(args.ds_path)
    seed_path = UPath(args.seed_path) if args.seed_path else None
    rule = resolve_year_rule(ds_path, args.year_from)
    logger.info(f"label-year rule: {rule}")
    if args.command == "plan":
        plan(ds_path, rule, seed_path)
    else:
        apply(ds_path, rule, seed_path)


if __name__ == "__main__":
    main()
