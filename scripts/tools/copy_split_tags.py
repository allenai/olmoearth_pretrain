"""Transplant an eval dataset's train/val/test partition onto its year-aligned copy.

``studio_ingest.scan_windows_and_splits`` reads ``window.options["split"]`` --
NOT the ``eval_split`` key these datasets actually carry. So on ingest the tags
are ignored, windows fall through to the group-name branches, and
``create_missing_splits`` re-derives val/test by shuffling. The shuffle is
seeded, but it is applied to the list returned by ``load_windows``, which uses
``imap_unordered`` and so comes back in a different order every run. Counts are
therefore stable while *membership* is not: the year-aligned val set overlaps
the original's by roughly half.

Writing the original's split into ``options["split"]`` makes the scan find all
three splits populated, so ``create_missing_splits`` takes PATH 1 ("all splits
present - no splitting needed") and the partition is preserved exactly.

Run BEFORE ingest -- afterwards means redoing a multi-hour copy.

Usage::

    NAME=canada_crops_coarse
    python scripts/tools/copy_split_tags.py \
        --source_path /weka/dfive-default/olmoearth/eval_datasets/$NAME \
        --ds_path /weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/${NAME}_year_aligned
    # add --write once the dry-run summary looks right
"""

import argparse
import logging
from collections import Counter

from rslearn.dataset import Dataset
from upath import UPath

logger = logging.getLogger(__name__)

# What studio_ingest.scan_windows_and_splits actually reads, versus what the
# datasets carry. Copying one into the other is the whole job.
INGEST_KEY = "split"
SOURCE_KEY = "eval_split"
VALID = ("train", "val", "test")


def read_splits(source_path: UPath, key: str) -> dict[str, str]:
    """Map "group/name" -> split value from an untouched source dataset."""
    splits: dict[str, str] = {}
    for window in Dataset(source_path).storage.get_windows():
        value = (window.options or {}).get(key)
        if value:
            splits[f"{window.group}/{window.name}"] = value
    return splits


def main() -> int:
    """Copy split tags across, or report what would change."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_path", required=True, help="Original dataset holding the partition."
    )
    parser.add_argument("--ds_path", required=True, help="Year-aligned staging copy.")
    parser.add_argument(
        "--source_key", default=SOURCE_KEY, help=f"Tag to read (default {SOURCE_KEY})."
    )
    parser.add_argument(
        "--write", action="store_true", help="Actually write. Default is a dry run."
    )
    args = parser.parse_args()

    splits = read_splits(UPath(args.source_path), args.source_key)
    logger.info(f"source: {len(splits)} windows tagged {args.source_key!r}")
    if not splits:
        raise SystemExit(
            f"no {args.source_key!r} tags found in {args.source_path}; nothing to copy"
        )
    logger.info(f"  distribution: {dict(Counter(splits.values()))}")

    assigned: Counter = Counter()
    unmatched = 0
    already = 0
    dataset = Dataset(UPath(args.ds_path))
    for window in dataset.storage.get_windows():
        value = splits.get(f"{window.group}/{window.name}")
        if value is None:
            unmatched += 1
            continue
        if value not in VALID:
            raise SystemExit(
                f"unexpected split value {value!r}; expected one of {VALID}"
            )
        options = dict(window.options or {})
        if options.get(INGEST_KEY) == value:
            already += 1
        elif args.write:
            options[INGEST_KEY] = value
            window.options = options
            window.save()
        assigned[value] += 1

    verb = "wrote" if args.write else "would write"
    logger.info(
        f"{verb} {INGEST_KEY!r} on {sum(assigned.values())} windows: {dict(assigned)}"
    )
    if already:
        logger.info(f"  ({already} already correct)")
    if unmatched:
        logger.warning(
            f"{unmatched} staging windows had no counterpart in the source and were "
            "left untagged; ingest will place them by group name and PATH 1 will not "
            "apply. Investigate before ingesting."
        )
    if not args.write:
        logger.info("dry run -- re-run with --write, then ingest")
    else:
        logger.info(
            "now ingest; expect 'PATH 1: All splits present' and split counts "
            "matching the source exactly"
        )
    return 1 if unmatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
