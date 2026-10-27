"""Dump frozen ERA5 encoder embeddings of the snow eval windows (CPU).

The snow counterpart of ``usdm/dump_embeddings.py`` (same encoder pass,
poolings, preprocessing and output format), looping over the window families: each
checkpoint (or random init) of a queue is built once and every family's
``eval`` windows go through it, saving the ``mean`` (the pooled vector every
eval probes), ``last1`` and ``last4`` poolings, so candidate targets and leads
can be probed offline (``target_lab.py``, ``emb_probe.py``).

Output: ``<root>/embeddings/<family>/<name>.npz`` and
``<root>/embeddings/<family>/windows.parquet`` (row order: ``cell_id, anchor,
split``).

Usage (repo root)::

    python scripts/era5_supervised/v0/snow/dump_embeddings.py --queue <queue.json> --shard 0/2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

V0_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V0_DIR / "fluxnet"))
from cpu_probe import CKPT_ROOT, load_weights, model_config  # noqa: E402
from dump_embeddings import EPOCH, NODATA, WINDOW_DAYS, embed  # noqa: E402
from olmo_core.utils import seed_all  # noqa: E402

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.data.normalize import Normalizer, Strategy  # noqa: E402
from olmoearth_pretrain.dataset_creation.snow.write_windows import (  # noqa: E402
    FAMILIES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SNOW_ROOT = Path("/weka/dfive-default/rslearn-eai/datasets/snow")


def eval_inputs(
    root: Path, table: str
) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preprocessed encoder inputs of every window of a window table (as USDM).

    Args:
        root: Dataset root.
        table: ``meta/<table>.parquet`` with ``cell_id, anchor, t0, split``.

    Returns:
        ``(windows, era5 [N, 448, 14], timestamps [N, 448, 3], valid [N, 448, 14])``.
    """
    windows = pd.read_parquet(root / "meta" / f"{table}.parquet").reset_index(drop=True)
    normalizer = Normalizer(Strategy.ZSCORE)
    era5 = np.zeros((len(windows), WINDOW_DAYS, 14), np.float32)
    valid = np.zeros_like(era5, dtype=bool)
    for cell_id, rows in windows.groupby("cell_id"):
        series = np.load(root / "meta" / "era5" / f"{cell_id}.npy", mmap_mode="r")
        for i, t0 in zip(rows.index, rows["t0"]):
            raw = np.array(series[int(t0) : int(t0) + WINDOW_DAYS])
            mask = raw != NODATA
            norm = np.asarray(
                normalizer.normalize(Modality.ERA5L_DAY_10, raw.copy()), np.float32
            )
            norm[~mask] = 0.0
            era5[i], valid[i] = norm, mask
    stamps = np.zeros((len(windows), WINDOW_DAYS, 3), np.int64)
    for i, anchor in enumerate(windows["anchor"]):
        start = EPOCH + timedelta(days=int(anchor) - WINDOW_DAYS)
        days = [start + timedelta(days=k) for k in range(WINDOW_DAYS)]
        stamps[i] = [[d.timetuple().tm_yday, d.month - 1, d.year] for d in days]
    return (
        windows,
        torch.from_numpy(era5),
        torch.from_numpy(stamps),
        torch.from_numpy(valid),
    )


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--queue", required=True)
    ap.add_argument("--root", default=str(SNOW_ROOT))
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--table-prefix", default="windows_eval_")
    ap.add_argument("--out", default="embeddings")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--threads", type=int, default=56)
    ap.add_argument("--batch", type=int, default=256)
    # Accepted for launcher compatibility; unused.
    ap.add_argument("--tasks", default="")
    ap.add_argument("--lr-grid", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    root = Path(args.root)
    inputs = {}
    for family in args.families.split(","):
        out_dir = root / args.out / family
        out_dir.mkdir(parents=True, exist_ok=True)
        windows, era5, stamps, valid = eval_inputs(root, f"{args.table_prefix}{family}")
        windows[["cell_id", "anchor", "split"]].to_parquet(out_dir / "windows.parquet")
        inputs[family] = (out_dir, era5, stamps, valid)
    index, count = (int(v) for v in args.shard.split("/"))
    for entry in json.loads(Path(args.queue).read_text())[index::count]:
        todo = {
            f: v
            for f, v in inputs.items()
            if not (v[0] / f"{entry['name']}.npz").exists()
        }
        if not todo:
            continue
        run_dir = CKPT_ROOT / entry["source"]
        config = model_config(
            run_dir / f"step{int(entry.get('load_step', 50000))}" / "config.json"
        )
        seed_all(int(entry.get("init_seed", 0)))
        model = config.build()
        if not entry.get("random_init", False):
            load_weights(model, run_dir / f"step{int(entry['step'])}")
        for family, (out_dir, era5, stamps, valid) in todo.items():
            target = out_dir / f"{entry['name']}.npz"
            # Written to a side file, then renamed: readers glob *.npz while dumps run.
            part = target.with_suffix(".npz.part")
            with open(part, "wb") as f:
                np.savez(f, **embed(model.encoder, era5, stamps, valid, args.batch))
            part.rename(target)
        logger.info("wrote %s (%s)", entry["name"], ",".join(todo))


if __name__ == "__main__":
    main()
