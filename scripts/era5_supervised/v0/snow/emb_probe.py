"""The in-loop linear probe of the snow eval tasks, run on dumped embeddings.

The snow counterpart of ``usdm/emb_probe.py``: each ``snow_<target>_<family>_eval``
task probes the family's dumped embeddings (``dump_embeddings.py``) with the
evaluator's own ``train_and_eval_probe`` (task configs resolved by
``script._resolve_eval_task_configs``: registry LR, probe seed, epochs, batch
size, the LR grid, train-split target normalization inside the probe). Labels
come from the window table; windows where a target is undefined are left out,
as the registry's ``has_sad`` tag does in the loop. The only differences from
the in-loop path are float16 storage of the embeddings and CPU execution.

Each record is appended to ``<results_dir>/<set>.jsonl`` in the evaluator's
format (``run``, ``step``, ``task``, ``probe_lr``, ``val``, ``test``,
``lrsweep``), so ``fluxnet/analyze_probes.py`` reads them.

Usage (repo root)::

    python scripts/era5_supervised/v0/snow/emb_probe.py --tasks "[snow_sdd_anom_l30_eval]" --pooling last4
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

V0_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V0_DIR / "fluxnet"))
from cpu_probe import task_configs  # noqa: E402

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.evals.datasets.configs import (  # noqa: E402
    EvalDatasetConfig,
    TaskType,
)
from olmoearth_pretrain.evals.linear_probe import (  # noqa: E402
    ProbeType,
    train_and_eval_probe,
)
from olmoearth_pretrain.train.callbacks.era5_evaluator_callback import (  # noqa: E402
    _append_results_jsonl,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SNOW_ROOT = Path("/weka/dfive-default/rslearn-eai/datasets/snow")
TASK_RE = re.compile(r"snow_(.+)_(l\d+|p\d+|post)_eval")
DEVICE = torch.device("cpu")


def run_and_step(name: str) -> tuple[str, int]:
    """Run name and step of an embedding set.

    Args:
        name: ``<run>@<step>`` or ``init_<x>``.

    Returns:
        ``(run, step)``.
    """
    if "@" in name:
        run, step = name.split("@")
        return run, int(step)
    return name, 0


def probe_set(job: tuple[str, list[Any], str, str, int, str]) -> str:
    """Worker: probe every task on one embedding set name (all its families).

    Args:
        job: ``(set name, task configs, embeddings dir, results dir, threads,
            pooling)``.

    Returns:
        The set name.
    """
    name, tasks, emb_root, results_dir, threads, pooling = job
    torch.set_num_threads(threads)
    run, step = run_and_step(name)
    out = Path(results_dir) / f"{name}.jsonl"
    done = (
        {json.loads(line)["task"] for line in out.read_text().splitlines()}
        if out.exists()
        else set()
    )
    for task in tasks:
        if task.name in done:
            continue
        target, family = TASK_RE.fullmatch(task.name).groups()
        windows = pd.read_parquet(Path(results_dir) / f"_windows_{family}.parquet")
        emb = torch.from_numpy(np.load(Path(emb_root) / family / f"{name}.npz")[pooling]
                               .astype(np.float32))  # fmt: skip
        labels = torch.tensor(windows[target].to_numpy(), dtype=torch.float32)
        ok = torch.isfinite(labels).numpy()
        split = windows["split"].to_numpy()
        rows = {s: torch.from_numpy(np.flatnonzero((split == s) & ok))
                for s in ("train", "val", "test")}  # fmt: skip
        task_type = TaskType(task.task_type)
        features = emb
        # As in cpu_probe / the evaluator: "regression" is PER_PIXEL_REGRESSION,
        # probed as a 1x1 map (the probe auto-upgrades to BILINEAR_CONV).
        if task_type in (TaskType.SEGMENTATION, TaskType.PER_PIXEL_REGRESSION):
            features = emb.unsqueeze(1).unsqueeze(1)
            labels = labels.unsqueeze(-1).unsqueeze(-1)
        config = EvalDatasetConfig(
            task_type=task_type, num_classes=task.num_classes or 2, is_multilabel=False,
            imputes=[], supported_modalities=[Modality.ERA5L_DAY_10.name],
            height_width=task.height_width,
        )  # fmt: skip

        def probe(lr: float) -> Any:
            """One probe at ``lr`` with the pinned probe seed.

            Args:
                lr: Probe LR.

            Returns:
                Val / test results.
            """
            torch.manual_seed(task.probe_seed)
            return train_and_eval_probe(
                config=config, lr=lr,
                train_embeddings=features[rows["train"]], train_labels=labels[rows["train"]],
                val_embeddings=features[rows["val"]], val_labels=labels[rows["val"]],
                test_embeddings=features[rows["test"]], test_labels=labels[rows["test"]],
                device=DEVICE, batch_size=task.probe_batch_size, epochs=task.probe_epochs,
                eval_interval=task.probe_epochs, probe_type=ProbeType.LINEAR,
            )  # fmt: skip

        result = probe(task.probe_lr)
        sweep = {lr: probe(lr) for lr in task.probe_lr_grid}
        _append_results_jsonl(str(out), {
            "run": run, "step": step, "task": task.name, "probe_lr": task.probe_lr,
            "val": result.val_result.metrics if result.val_result else None,
            "test": result.test_result.metrics if result.test_result else None,
            "lrsweep": {f"{lr:.0e}": {"val": r.val_result.metrics if r.val_result else None,
                                      "test": r.test_result.metrics if r.test_result else None}
                        for lr, r in sweep.items()},
            "device": "cpu", "source": "emb_probe", "pooling": pooling,
        })  # fmt: skip
    return name


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--lr-grid", default="[1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1]")
    ap.add_argument("--root", default=str(SNOW_ROOT))
    ap.add_argument("--emb-dir", default="embeddings")
    ap.add_argument("--table-prefix", default="windows_eval_")
    ap.add_argument(
        "--results-dir", default=None, help="default <root>/probe_results[_<pooling>]"
    )
    ap.add_argument(
        "--pooling", default="mean", help="mean (what the evaluator probes) or last4"
    )
    ap.add_argument("--sets", default=None, help="regex on embedding set names")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    root = Path(args.root)
    emb_root = root / args.emb_dir
    suffix = "" if args.pooling == "mean" else f"_{args.pooling}"
    results_dir = Path(args.results_dir or root / f"probe_results{suffix}")
    results_dir.mkdir(parents=True, exist_ok=True)
    tasks = task_configs(args.tasks, args.lr_grid)
    families = sorted({TASK_RE.fullmatch(t.name).group(2) for t in tasks})
    names: set[str] | None = None
    for family in families:
        order = pd.read_parquet(emb_root / family / "windows.parquet")
        table = pd.read_parquet(root / "meta" / f"{args.table_prefix}{family}.parquet")
        windows = order.merge(
            table.drop(columns=["split"]), on=["cell_id", "anchor"], how="left"
        )
        windows.to_parquet(results_dir / f"_windows_{family}.parquet")
        stems = {p.stem for p in (emb_root / family).glob("*.npz")}
        names = stems if names is None else names & stems
    index, count = (int(v) for v in args.shard.split("/"))
    todo = [n for n in sorted(names or [])[index::count]
            if not args.sets or re.search(args.sets, n)]  # fmt: skip
    jobs = [(n, tasks, str(emb_root), str(results_dir), args.threads, args.pooling)
            for n in todo]  # fmt: skip
    logger.info("%d embedding sets to probe on %d tasks", len(jobs), len(tasks))
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        for name in pool.imap_unordered(probe_set, jobs):
            logger.info("probed %s", name)


if __name__ == "__main__":
    main()
