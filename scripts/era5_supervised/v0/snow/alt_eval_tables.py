"""Independent eval draws of the snow windows, for replicating eval statistics.

A second draw of (cell, water year) keys with ``write_windows.select_eval_keys``
(another seed, the primary ``meta/eval_keys.parquet`` keys excluded), written
as window tables only (``meta/windows_<name>_<family>.parquet``): the embedding
dumps (``dump_embeddings.py --table-prefix windows_<name>_ --out
embeddings_<name>``) and probes read the cell series, not the rslearn windows,
so a statistic (floor, gain, between-arm F) can be checked on disjoint windows
without writing any.

``--union a,b`` instead writes the union of existing key sets (e.g. the
primary ``eval`` and ``evalb`` draws as ``evalx``: twice the windows, for evals
whose between-arm F is limited by the eval sample);
``write_windows windows --eval-group evalx`` then materializes its windows.

Usage::

    python scripts/era5_supervised/v0/snow/alt_eval_tables.py --name evalb --seed 1
    python scripts/era5_supervised/v0/snow/alt_eval_tables.py --name evalx --union eval,evalb
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from olmoearth_pretrain.dataset_creation.snow.write_windows import (
    FAMILIES,
    select_eval_keys,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default="/weka/dfive-default/rslearn-eai/datasets/snow")
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--union", default=None, help="comma list of key sets to merge")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=1500)
    ap.add_argument("--n-test", type=int, default=1500)
    args = ap.parse_args()
    meta = Path(args.root) / "meta"
    candidates = pd.read_parquet(meta / "candidates.parquet")
    if args.union:
        parts = [
            pd.read_parquet(meta / f"{name}_keys.parquet")
            for name in args.union.split(",")
        ]
        keys = pd.concat(parts, ignore_index=True)
        if keys.duplicated(["cell_id", "wy"]).any():
            raise ValueError("the key sets overlap")
    else:
        primary = pd.read_parquet(meta / "eval_keys.parquet")[["cell_id", "wy"]]
        taken = candidates.merge(
            primary.assign(_taken=True), on=["cell_id", "wy"], how="left"
        )
        pool = candidates[taken["_taken"].isna().to_numpy()]
        quotas = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
        keys = select_eval_keys(pool, quotas, args.seed)
    keys.to_parquet(meta / f"{args.name}_keys.parquet")
    for family in FAMILIES:
        rows = candidates[candidates["family"] == family].merge(
            keys[["cell_id", "wy"]], on=["cell_id", "wy"]
        )
        rows.to_parquet(meta / f"windows_{args.name}_{family}.parquet")
        logger.info("%s %s: %d windows %s", args.name, family, len(rows),
                    rows["split"].value_counts().to_dict())  # fmt: skip


if __name__ == "__main__":
    main()
