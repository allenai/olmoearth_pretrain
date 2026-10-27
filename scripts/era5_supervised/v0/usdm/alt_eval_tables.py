"""Independent alternative eval-window tables, to check that eval results replicate.

Draws a window table from the same candidates and splits as the ``eval`` group,
excluding its windows (and those of any ``--exclude`` tables), so a probe result
can be re-measured on an independent sample (embeddings dumped with
``dump_embeddings.py --table``, probed with ``emb_probe.py --table``):

* ``--mode stratified``: the ``eval`` recipe (``write_windows.select_eval``,
  stratified by drought state x dynamics);
* ``--mode uniform``: no stratification (cell- and water-year-balanced
  round-robin over all candidates of each split), i.e. the natural mix.

No rslearn windows are written; the tables carry ``t0`` so the dump reads the
cell series directly. Written so far: ``windows_eval_s1`` (stratified, seed 1),
``windows_eval_u0`` (uniform, seed 0), ``windows_eval_u1`` (uniform, seed 1,
excluding ``u0``).

Usage (repo root)::

    python scripts/era5_supervised/v0/usdm/alt_eval_tables.py --name windows_eval_u1 --mode uniform --seed 1 --exclude windows_eval_u0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from olmoearth_pretrain.dataset_creation.usdm.common import DEFAULT_ROOT
from olmoearth_pretrain.dataset_creation.usdm.write_windows import (
    balanced_subset,
    select_eval,
    stratum_of,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

QUOTAS = {"train": 4000, "val": 1500, "test": 1500}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--name", required=True, help="output table name under meta")
    ap.add_argument("--mode", choices=["stratified", "uniform"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--exclude", default="", help="comma list of other tables whose windows to skip"
    )
    args = ap.parse_args()
    meta = Path(args.root) / "meta"
    candidates = pd.read_parquet(meta / "candidates.parquet")
    excluded = [pd.read_parquet(meta / "eval_keys.parquet")[["cell_id", "anchor"]]]
    excluded += [
        pd.read_parquet(meta / f"{name}.parquet")[["cell_id", "anchor"]]
        for name in args.exclude.split(",")
        if name
    ]
    keys = pd.concat(excluded).drop_duplicates()
    used = candidates.merge(keys.assign(_used=1), on=["cell_id", "anchor"], how="left")
    pool = candidates[used["_used"].isna().to_numpy()]
    if args.mode == "stratified":
        frame = select_eval(pool, QUOTAS, seed=args.seed)
    else:
        frame = pd.concat(
            [
                balanced_subset(pool[pool["split"] == split], n, seed=args.seed)
                for split, n in QUOTAS.items()
            ],
            ignore_index=True,
        )
        frame = frame.assign(stratum=stratum_of(frame))
    frame.to_parquet(meta / f"{args.name}.parquet")
    logger.info(
        "%s: %s; strata %s",
        args.name,
        frame["split"].value_counts().to_dict(),
        frame["stratum"].value_counts(normalize=True).round(3).to_dict(),
    )


if __name__ == "__main__":
    main()
