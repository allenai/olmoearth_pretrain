"""Independent alternative eval-window tables, to check that eval results replicate.

Draws window tables from the same candidates and splits as the ``eval`` group,
excluding its windows, so a probe result can be re-measured on an independent
sample (embeddings dumped with ``dump_embeddings.py --table``, probed with
``emb_probe.py --table``):

* ``windows_eval_s1``: the ``eval`` recipe (``write_windows.select_eval``,
  stratified by drought state x dynamics) with another seed;
* ``windows_eval_u0``: no stratification (cell- and water-year-balanced
  round-robin over all candidates of each split), i.e. the natural mix.

No rslearn windows are written; the tables carry ``t0`` so the dump reads the
cell series directly.

Usage (repo root)::

    python scripts/era5_supervised/v0/usdm/alt_eval_tables.py
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
    args = ap.parse_args()
    meta = Path(args.root) / "meta"
    candidates = pd.read_parquet(meta / "candidates.parquet")
    keys = pd.read_parquet(meta / "eval_keys.parquet")[["cell_id", "anchor"]]
    used = candidates.merge(keys.assign(_used=1), on=["cell_id", "anchor"], how="left")
    pool = candidates[used["_used"].isna().to_numpy()]
    stratified = select_eval(pool, QUOTAS, seed=1)
    uniform = pd.concat(
        [
            balanced_subset(pool[pool["split"] == split], n, seed=0)
            for split, n in QUOTAS.items()
        ],
        ignore_index=True,
    )
    uniform = uniform.assign(stratum=stratum_of(uniform))
    for name, frame in (("windows_eval_s1", stratified), ("windows_eval_u0", uniform)):
        frame.to_parquet(meta / f"{name}.parquet")
        logger.info(
            "%s: %s; strata %s",
            name,
            frame["split"].value_counts().to_dict(),
            frame["stratum"].value_counts(normalize=True).round(3).to_dict(),
        )


if __name__ == "__main__":
    main()
