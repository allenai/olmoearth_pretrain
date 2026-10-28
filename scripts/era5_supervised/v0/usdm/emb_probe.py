"""The in-loop linear probe of the USDM eval tasks, run on dumped embeddings.

Every ``usdm_*_eval`` task shares the same ``eval`` windows, so instead of
re-reading the rslearn windows and re-running the encoder once per task (as
``cpu_probe.py`` does), this probes the ``mean`` pooling saved by
``dump_embeddings.py`` (the encoder's pooled vector, what the evaluator probes)
with the evaluator's own ``train_and_eval_probe``: task configs resolved by
``script._resolve_eval_task_configs`` (registry LR, probe seed, epochs, batch
size), the same LR grid, train-split target normalization inside the probe.
Labels come from the window table (class indices in the task YAML's order).
The only differences from the in-loop path are float16 storage of the
embeddings and CPU execution.

Each record is appended to ``<results_dir>/<set>.jsonl`` in the evaluator's
format (``run``, ``step``, ``task``, ``probe_lr``, ``val``, ``test``,
``lrsweep``), so ``fluxnet/analyze_probes.py`` reads them: ``<run>@<step>``
sets map to (run, step), ``init_*`` sets to step 0.

Usage (repo root)::

    python scripts/era5_supervised/v0/usdm/emb_probe.py --tasks "[usdm_sev_eval,usdm_dpast8_eval]" --shard 0/2
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

USDM_ROOT = Path("/weka/dfive-default/rslearn-eai/datasets/usdm")
CLASSES = {
    "tpast8": ["improve", "stable", "worsen"],
    "tfut8": ["improve", "stable", "worsen"],
    "drought": ["no", "yes"],
}
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


def labels_of(windows: pd.DataFrame, target: str) -> torch.Tensor:
    """Probe labels of a target, as the eval dataset would yield them.

    Args:
        windows: Window table.
        target: Property name.

    Returns:
        Float labels (regression) or class indices.
    """
    if target in CLASSES:
        index = {c: k for k, c in enumerate(CLASSES[target])}
        return torch.tensor([index[v] for v in windows[target]], dtype=torch.long)
    return torch.tensor(windows[target].to_numpy(), dtype=torch.float32)


def probe_set(job: tuple[str, list[Any], str, str, int]) -> str:
    """Worker: probe every task on one embedding set.

    Args:
        job: ``(npz path, task configs, windows parquet, results dir, threads,
            pooling)``.

    Returns:
        The set name.
    """
    path, tasks, windows_path, results_dir, threads, pooling = job
    torch.set_num_threads(threads)
    name = Path(path).stem
    run, step = run_and_step(name)
    windows = pd.read_parquet(windows_path)
    emb = torch.from_numpy(np.load(path)[pooling].astype(np.float32))
    split = windows["split"].to_numpy()
    rows = {
        s: torch.from_numpy(np.flatnonzero(split == s))
        for s in ("train", "val", "test")
    }
    out = Path(results_dir) / f"{name}.jsonl"
    done = (
        {json.loads(line)["task"] for line in out.read_text().splitlines()}
        if out.exists()
        else set()
    )
    for task in tasks:
        if task.name in done:
            continue
        target = re.fullmatch(r"usdm_(.+)_eval", task.name).group(1)
        labels = labels_of(windows, target)
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
            "device": "cpu", "source": "emb_probe",
        })  # fmt: skip
    return name


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--lr-grid", default="[1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1]")
    ap.add_argument("--root", default=str(USDM_ROOT))
    ap.add_argument("--emb-dir", default="embeddings")
    ap.add_argument(
        "--table", default="windows_eval", help="window table under meta (labels)"
    )
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
    emb_dir = root / args.emb_dir
    suffix = "" if args.pooling == "mean" else f"_{args.pooling}"
    results_dir = Path(args.results_dir or root / f"probe_results{suffix}")
    results_dir.mkdir(parents=True, exist_ok=True)
    tasks = task_configs(args.tasks, args.lr_grid)
    order = pd.read_parquet(emb_dir / "windows.parquet")
    table = pd.read_parquet(root / "meta" / f"{args.table}.parquet").drop(
        columns=["split"]
    )
    windows = order.merge(table, on=["cell_id", "anchor"], how="left")
    windows_path = results_dir / "_windows.parquet"
    windows.to_parquet(windows_path)
    index, count = (int(v) for v in args.shard.split("/"))
    paths = sorted(emb_dir.glob("*.npz"))[index::count]
    todo = [p for p in paths if not args.sets or re.search(args.sets, p.stem)]
    jobs = [
        (str(p), tasks, str(windows_path), str(results_dir), args.threads, args.pooling)
        for p in todo
    ]
    logger.info("%d embedding sets to probe on %d tasks", len(jobs), len(tasks))
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        for name in pool.imap_unordered(probe_set, jobs):
            logger.info("probed %s", name)


if __name__ == "__main__":
    main()
