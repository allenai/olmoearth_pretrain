"""Offline target design for the snow eval on dumped encoder embeddings.

The snow counterpart of ``usdm/target_lab.py`` (same closed-form proxy probes
and summaries), over the window families: every target of every family's
``eval`` windows is scored with ridge on standardized embeddings (alpha by
5-fold CV over train blocks; val / test R2) on each embedding set written by
``dump_embeddings.py`` and each pooling. Targets undefined on some windows
(``sad_anom``, ``season_anom``) use the windows where they are defined.

The summary per (family, target, pooling) gives the random-projection floor
(``init_*`` sets), 1601 (seed mean +- SD), the gain over the floor, the arm
range, the between-arm F (between-arm over seed variance, arms with >= 2
seeds) and the 1306 trajectory (step 0..50k).

Writes ``<root>/<emb dir>/target_lab.json`` (``{family: {target: {pooling:
{set: scores}}}}``).

Usage::

    python scripts/era5_supervised/v0/snow/target_lab.py --root /weka/.../snow
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

V0_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V0_DIR / "usdm"))
from target_lab import regression_scores, summarize  # noqa: E402

from olmoearth_pretrain.dataset_creation.snow.write_windows import (  # noqa: E402
    FAMILIES,
    TARGETS,
    family_kind,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def family_windows(
    root: Path, emb_dir: Path, family: str, prefix: str = "windows_eval_"
) -> pd.DataFrame:
    """A family's window table in the embedding row order.

    Args:
        root: Dataset root.
        emb_dir: The family's embedding directory.
        family: Window family.
        prefix: Window table prefix under ``meta``.

    Returns:
        Windows with targets, ``split`` and ``block``.
    """
    order = pd.read_parquet(emb_dir / "windows.parquet")
    table = pd.read_parquet(root / "meta" / f"{prefix}{family}.parquet")
    windows = order.merge(
        table.drop(columns=["split"]), on=["cell_id", "anchor"], how="left"
    )
    assert len(windows) == len(order)
    return windows


def score_job(job: tuple) -> tuple[str, str, str, dict]:
    """Worker: score every target of a family on one (embedding set, pooling).

    Args:
        job: ``(family, npz path, pooling, targets, windows, threads)``.

    Returns:
        ``(family, set name, pooling, {target: scores})``.
    """
    family, path, pooling, todo, windows, threads = job
    split = windows["split"].to_numpy()
    blocks = windows["block"].to_numpy()
    out = {}
    with threadpool_limits(threads):
        x = np.load(path)[pooling].astype(np.float32)
        for target in todo:
            y = windows[target].to_numpy(np.float64)
            ok = np.isfinite(y)
            xs = StandardScaler().fit(x[ok & (split == "train")]).transform(x[ok])
            train_idx = np.flatnonzero(split[ok] == "train")
            folds = list(GroupKFold(5).split(train_idx, groups=blocks[ok][train_idx]))
            out[target] = regression_scores(xs, y[ok], split[ok], folds)
    return family, Path(path).stem, pooling, out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default="/weka/dfive-default/rslearn-eai/datasets/snow")
    ap.add_argument("--emb-dir", default="embeddings")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--table-prefix", default="windows_eval_")
    ap.add_argument("--poolings", default="mean,last4")
    ap.add_argument("--sets", default=None, help="regex on embedding set names")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    root = Path(args.root)
    emb_root = root / args.emb_dir
    out_path = emb_root / "target_lab.json"
    results: dict = json.loads(out_path.read_text()) if out_path.exists() else {}
    jobs = []
    for family in args.families.split(","):
        emb_dir = emb_root / family
        windows = family_windows(root, emb_dir, family, args.table_prefix)
        targets = TARGETS[family_kind(family)]
        for path in sorted(emb_dir.glob("*.npz")):
            if args.sets and not re.search(args.sets, path.stem):
                continue
            for pooling in args.poolings.split(","):
                done = results.get(family, {})
                todo = [t for t in targets
                        if path.stem not in done.get(t, {}).get(pooling, {})]  # fmt: skip
                if todo:
                    jobs.append(
                        (family, str(path), pooling, todo, windows, args.threads)
                    )
    logger.info("%d (family, set, pooling) jobs", len(jobs))
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        for k, (family, stem, pooling, scores) in enumerate(
            pool.imap_unordered(score_job, jobs), start=1
        ):
            for target, sc in scores.items():
                (results.setdefault(family, {}).setdefault(target, {})
                 .setdefault(pooling, {}))[stem] = sc  # fmt: skip
            if k % 20 == 0 or k == len(jobs):
                logger.info(
                    "[%d/%d] scored %s %s %s", k, len(jobs), family, stem, pooling
                )
                out_path.write_text(json.dumps(results, indent=1))
    out_path.write_text(json.dumps(results, indent=1))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    for metric in ("val", "test"):
        frames = []
        for family, by_target in results.items():
            frame = summarize(by_target, metric)
            frame.insert(0, "family", family)
            frames.append(frame)
        print(f"\n== {metric}\n", pd.concat(frames).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
