"""Delete embedding layers whose product year no longer matches their window.

``reanchor_year_aligned_dataset.py`` moves each window to the calendar year of
its label. For canada that uses the "end" rule rather than "midpoint", which
re-years the windows whose DATE_COLL fell in June -- 217 of 14 566 on
canada_crops_fine. Those windows already carry an AEF (or Tessera) layer
fetched for the *midpoint* year, so it is now one year off from the imagery.

The materializer cannot fix this on its own: ``is_layer_written`` only checks
whether the layer exists, so it skips them, and ``--overwrite`` would re-fetch
every window in the dataset. This script deletes just the stale layers, after
which a plain materializer run re-fetches exactly those windows.

Staleness is detected against the *seed* dataset, which still holds the
original (DATE_COLL - 1yr, DATE_COLL) ranges -- ``apply`` overwrote them in the
staging copy, so the staging dataset alone cannot tell you which windows moved.

Usage::

    NAME=canada_crops_fine
    SEED=/weka/dfive-default/olmoearth/eval_datasets/$NAME
    DST=/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/${NAME}_year_aligned

    python scripts/tools/clear_stale_embedding_layers.py \
        --seed_path "$SEED" --ds_path "$DST" --products gse          # dry run
    python scripts/tools/clear_stale_embedding_layers.py \
        --seed_path "$SEED" --ds_path "$DST" --products gse --delete

    python -m olmoearth_pretrain.evals.embedding_materializer \
        --dataset_path "$DST" --products aef

Note the naming mismatch: the *layer* is ``gse`` while the materializer
*product* is ``aef``. Pass layer names here and product names there.
"""

import argparse
import logging
import shutil

from rslearn.dataset import Dataset
from rslearn.dataset.storage.file import FileWindowStorage
from upath import UPath

logger = logging.getLogger(__name__)

# Must match reanchor_year_aligned_dataset.MONTHLY_PREFIXES.
MONTHLY_PREFIXES = ("sentinel2_l2a_mo", "sentinel1_mo")


def stale_windows(seed_path: UPath, rule: str = "end") -> set[str]:
    """Names of seed windows whose `rule` year differs from their midpoint year.

    Args:
        seed_path: the untouched seed dataset, still holding original ranges.
        rule: the year rule now in force ("end" for canada).

    Returns:
        set of "group/name" identifiers whose embedding year is now wrong.
    """
    stale: set[str] = set()
    for window in Dataset(seed_path).storage.get_windows():
        if window.time_range is None:
            continue
        start, end = window.time_range
        midpoint_year = (start + (end - start) / 2).year
        new_year = end.year if rule == "end" else start.year
        if new_year != midpoint_year:
            stale.add(f"{window.group}/{window.name}")
    return stale


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed_path", required=True, help="Untouched seed dataset.")
    parser.add_argument("--ds_path", required=True, help="Re-anchored staging dataset.")
    parser.add_argument(
        "--products",
        default="gse",
        help="Comma-separated LAYER names to clear (e.g. 'gse,tessera').",
    )
    parser.add_argument(
        "--rule",
        default="end",
        choices=["end", "start"],
        help="Year rule now in force.",
    )
    parser.add_argument(
        "--clear_items",
        action="store_true",
        help=(
            "Also reset the monthly imagery for the re-yeared windows: drop their "
            "sentinel*_mo* entries from items.json and remove any materialized "
            "monthly layers. Needed when prepare already ran under the old year, "
            "since prepare and materialize both skip work they think is done."
        ),
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete. Without this, only reports what would be removed.",
    )
    args = parser.parse_args()

    layer_names = [n.strip() for n in args.products.split(",") if n.strip()]
    stale = stale_windows(UPath(args.seed_path), args.rule)
    logger.info(f"{len(stale)} windows changed year under rule {args.rule!r}")

    dataset = Dataset(UPath(args.ds_path))
    # The "completed" marker is a file inside the layer directory for file-based
    # storage (storage/file.py:mark_layer_completed), so removing the directory
    # also clears the marker. Sqlite storage keeps it in a table instead, and
    # rslearn exposes no unmark API, so a deletion there would leave the layer
    # marked complete and the materializer would still skip the window.
    if not isinstance(dataset.storage, FileWindowStorage):
        raise SystemExit(
            f"{type(dataset.storage).__name__} keeps completion markers outside the "
            "layer directory; deleting rasters would not un-mark the layer. Use the "
            "materializer's --overwrite over the whole dataset instead."
        )

    removed = missing = items_reset = monthly_removed = 0
    for window in dataset.storage.get_windows():
        if f"{window.group}/{window.name}" not in stale:
            continue

        for layer_name in layer_names:
            if not window.is_layer_completed(layer_name):
                missing += 1
                continue
            if args.delete:
                shutil.rmtree(str(window.get_layer_dir(layer_name)))
            removed += 1

        if not args.clear_items:
            continue
        # Drop only the monthly imagery entries rather than deleting items.json:
        # the label layers have no data source, so prepare would never
        # regenerate their entries if the whole file went.
        layer_datas = window.load_layer_datas()
        stale_keys = [n for n in layer_datas if n.startswith(MONTHLY_PREFIXES)]
        if stale_keys:
            if args.delete:
                for key in stale_keys:
                    del layer_datas[key]
                window.save_layer_datas(layer_datas)
            items_reset += 1
        for layer_name, group_idx in window.list_completed_layers():
            if not layer_name.startswith(MONTHLY_PREFIXES):
                continue
            if args.delete:
                shutil.rmtree(str(window.get_layer_dir(layer_name, group_idx)))
            monthly_removed += 1

    verb = "deleted" if args.delete else "would delete"
    logger.info(f"{verb} {removed} embedding layer(s); {missing} already absent")
    if args.clear_items:
        logger.info(
            f"{verb.replace('delete', 'reset')} monthly items on {items_reset} "
            f"window(s) and {verb} {monthly_removed} materialized monthly layer(s)"
        )
    if not args.delete:
        logger.info("re-run with --delete, then re-prepare and run the materializer")
    else:
        logger.info(
            "now run: python -m olmoearth_pretrain.evals.embedding_materializer "
            f"--dataset_path {args.ds_path} --products aef"
        )
        if args.clear_items:
            logger.info("and re-run `rslearn dataset prepare` to refetch their items")


if __name__ == "__main__":
    main()
