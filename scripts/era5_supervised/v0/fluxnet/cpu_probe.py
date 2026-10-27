"""Linear probes of saved ERA5 encoder checkpoints on CPU, without the trainer.

For when no GPU is free: runs the same queue entries as ``probe_queue.py`` with
the in-loop evaluator's own pieces (``script._resolve_eval_task_configs`` for
the task configs, the callback's dataset, batch, embedding and geometry helpers,
``train_and_eval_probe`` with the pinned probe seed), so registry LRs, label
extractors, epochs and seeds match the GPU path; only the device differs
(bf16 autocast on CPU instead of GPU, so values differ in floating-point
detail).

Each source run's model is rebuilt from its ``step<N>/config.json`` (fields
older configs lack get the behaviour of the code that trained them, as in
``probe_queue.source_flags``), each checkpoint is unsharded to local disk and
loaded strictly, and every eval is appended to ``<results_dir>/<name>.jsonl``
in the evaluator's format. Besides the ``probe_queue.py`` keys, an entry may
list explicit ``steps`` (to split one run's checkpoints across jobs) and a
``run`` name to record (default: the entry name), so the parts merge. ``--shard i/n`` splits a queue across nodes; a
finished entry leaves ``<done_dir>/<name>.done``.

Usage (repo root)::

    python scripts/era5_supervised/v0/fluxnet/cpu_probe.py --queue <queue.json> \
        --tasks "[fluxnet_gpp_eval,lfmc_woody_eval]" --shard 0/4 --threads 60
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from olmo_core.config import Config
from olmo_core.distributed.checkpoint import unshard_checkpoint
from olmo_core.utils import seed_all

from olmoearth_pretrain.data.constants import ERA5_INPUT_SEQUENCE_LENGTH, Modality
from olmoearth_pretrain.evals.datasets.configs import EvalDatasetConfig, TaskType
from olmoearth_pretrain.evals.linear_probe import ProbeType, train_and_eval_probe
from olmoearth_pretrain.train.callbacks.era5_evaluator_callback import (
    Era5LinearProbeTaskConfig,
    _append_results_jsonl,
    _build_eval_dataset,
    _embedding_geometry_metrics,
    _extract_embeddings,
    _lr_sweep_log_dict,
    _materialize_batches,
)

V0_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V0_DIR))
from script import (  # noqa: E402
    Era5SupervisedCommonComponents,
    _resolve_eval_task_configs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CKPT_ROOT = Path("/weka/dfive-default/olmoearth_pretrain/checkpoints/hadriens")
FLUX_ROOT = Path("/weka/dfive-default/rslearn-eai/datasets/fluxnet")
DEVICE = torch.device("cpu")


def model_config(config_path: Path) -> Config:
    """The source run's full model config, with legacy fields made explicit.

    Args:
        config_path: The run's ``step<N>/config.json``.

    Returns:
        The ``Era5MultiObjectiveModelConfig``.
    """
    model = json.loads(config_path.read_text())["model"]
    enc = model["encoder_config"]
    enc["position_embedding"] = enc.get("position_embedding") or "none"
    enc["patch_day_of_year"] = enc.get("patch_day_of_year") or "mean"
    enc["pooled_norm"] = enc.get("pooled_norm") or "none"
    recon = model.get("reconstruction_objective")
    if recon is not None:
        dec = recon["decoder"]
        dec["key_position_embedding"] = dec.get("key_position_embedding") or "none"
        dec["patch_kernel_size"] = (
            dec.get("patch_kernel_size") or enc["patch_kernel_size"]
        )
        dec["patch_stride"] = dec.get("patch_stride") or enc["patch_stride"]
    return Config.from_dict(model)


def checkpoint_steps(run_dir: Path, interval: int) -> list[int]:
    """Saved ``step*`` checkpoints, filtered like the evaluator's sweep.

    Args:
        run_dir: Run directory.
        interval: Keep steps that are multiples of this (the last is always kept).

    Returns:
        Sorted steps.
    """
    steps = sorted(
        int(m.group(1))
        for p in run_dir.iterdir()
        if (m := re.fullmatch(r"step(\d+)", p.name)) and p.is_dir()
    )
    return [s for s in steps if s % interval == 0 or s == steps[-1]]


def load_weights(model: torch.nn.Module, step_dir: Path) -> None:
    """Unshard a checkpoint to local disk and load its model weights strictly.

    Args:
        model: The rebuilt model.
        step_dir: ``<run>/step<N>``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        model_path, _ = unshard_checkpoint(
            step_dir / "model_and_optim", tmp, optim=False, quiet=True
        )
        state = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)


class Prober:
    """Probe tasks on one encoder at a time, reusing materialized batches."""

    def __init__(self, tasks: list[Era5LinearProbeTaskConfig], workers: int) -> None:
        """Keep the task configs; batches are built on first use.

        Args:
            tasks: Probe task configs.
            workers: DataLoader workers for batch materialization.
        """
        self.tasks = tasks
        self.workers = workers
        self.batches: dict[str, tuple[list[Any], list[Any], list[Any] | None]] = {}

    def _batches(
        self, task: Era5LinearProbeTaskConfig
    ) -> tuple[list[Any], list[Any], list[Any] | None]:
        """Materialized train / val / test batches of a task (cached).

        Args:
            task: Probe task config.

        Returns:
            The three batch lists (test None when the task has no test split).
        """
        if task.name not in self.batches:
            splits = ["train", "val"] + (["test"] if task.has_test else [])
            built = [
                _materialize_batches(
                    _build_eval_dataset(task, split, ERA5_INPUT_SEQUENCE_LENGTH),
                    task.embedding_batch_size, self.workers,
                )
                for split in splits
            ]  # fmt: skip
            self.batches[task.name] = (
                built[0],
                built[1],
                built[2] if len(built) > 2 else None,
            )
        return self.batches[task.name]

    def run(
        self, encoder: torch.nn.Module, run_name: str, step: int, results: Path
    ) -> None:
        """Probe every task on ``encoder`` and append the records to ``results``.

        Args:
            encoder: The encoder (eval mode is set by the embedding helper).
            run_name: Name recorded in each line.
            step: Checkpoint step recorded in each line.
            results: JSONL file.
        """
        for task in self.tasks:
            start = time.monotonic()
            train_b, val_b, test_b = self._batches(task)
            embs = [_extract_embeddings(b, encoder, DEVICE) if b else (None, None)
                    for b in (train_b, val_b, test_b)]  # fmt: skip
            geometry = _embedding_geometry_metrics(task.name, embs[0][0])
            task_type = TaskType(task.task_type)
            spatial = task_type in (
                TaskType.SEGMENTATION,
                TaskType.PER_PIXEL_REGRESSION,
            )
            inputs = []
            for emb, labels in embs:
                if emb is not None and spatial:
                    emb = emb.unsqueeze(1).unsqueeze(1)
                    labels = (
                        labels.unsqueeze(-1).unsqueeze(-1)
                        if labels.ndim == 1
                        else labels
                    )
                inputs += [emb, labels]
            eval_config = EvalDatasetConfig(
                task_type=task_type, num_classes=task.num_classes or 2,
                is_multilabel=task.is_multilabel, imputes=[],
                supported_modalities=[Modality.ERA5L_DAY_10.name], height_width=task.height_width,
            )  # fmt: skip

            def probe(lr: float) -> Any:
                """Train one probe at ``lr`` with the pinned probe seed.

                Args:
                    lr: Probe LR.

                Returns:
                    The probe's val / test results.
                """
                torch.manual_seed(task.probe_seed)
                return train_and_eval_probe(
                    config=eval_config, lr=lr,
                    train_embeddings=inputs[0], train_labels=inputs[1],
                    val_embeddings=inputs[2], val_labels=inputs[3],
                    test_embeddings=inputs[4], test_labels=inputs[5],
                    device=DEVICE, batch_size=task.probe_batch_size, epochs=task.probe_epochs,
                    eval_interval=task.probe_epochs, probe_type=ProbeType.LINEAR,
                )  # fmt: skip

            result = probe(task.probe_lr)
            sweep = {lr: probe(lr) for lr in task.probe_lr_grid}
            geometry.update(
                _lr_sweep_log_dict(
                    task.name, {lr: r.val_result for lr, r in sweep.items()}
                )
            )
            _append_results_jsonl(str(results), {
                "run": run_name, "step": step, "task": task.name, "probe_lr": task.probe_lr,
                "val": result.val_result.metrics if result.val_result else None,
                "test": result.test_result.metrics if result.test_result else None,
                "lrsweep": {f"{lr:.0e}": {"val": r.val_result.metrics if r.val_result else None,
                                          "test": r.test_result.metrics if r.test_result else None}
                            for lr, r in sweep.items()},
                "geometry": geometry, "eval_time": time.monotonic() - start, "device": "cpu",
            })  # fmt: skip
            logger.info("%s step %d %s: val %s (%.0fs)", run_name, step, task.name,
                        result.val_result.primary if result.val_result else None,
                        time.monotonic() - start)  # fmt: skip


def task_configs(tasks: str, lr_grid: str) -> list[Era5LinearProbeTaskConfig]:
    """Probe task configs exactly as ``script.py`` resolves them for an eval run.

    Args:
        tasks: ``[name,...]``.
        lr_grid: ``[lr,...]``.

    Returns:
        Task configs (registry LRs, probe seed 1202, 50 epochs).
    """
    common = Era5SupervisedCommonComponents(
        run_name="cpu_probe",
        save_folder=tempfile.mkdtemp(prefix="cpu_probe_"),
        training_modalities=[Modality.ERA5L_DAY_10.name],
        eval_tasks=tasks.strip("[]").split(","),
        eval_probe_lr_grid=[float(v) for v in lr_grid.strip("[]").split(",") if v],
        eval_run_on_test=True,
    )
    return _resolve_eval_task_configs(common)


def run_entry(entry: dict[str, Any], args: argparse.Namespace) -> None:
    """Probe every selected checkpoint of one queue entry.

    Args:
        entry: Queue entry (see ``probe_queue.py``).
        args: CLI options.
    """
    run_dir = CKPT_ROOT / entry["source"]
    load_step = int(entry.get("load_step", 50000))
    config = model_config(run_dir / f"step{load_step}" / "config.json")
    seed_all(int(entry.get("init_seed", 0)))
    model = config.build()
    prober = Prober(
        task_configs(entry.get("tasks", args.tasks), args.lr_grid), args.workers
    )
    run_name = entry.get("run", entry["name"])
    results = Path(args.results_dir) / f"{entry['name']}.jsonl"
    if entry.get("random_init", False):
        prober.run(model.encoder, run_name, 0, results)
        return
    steps = entry.get("steps") or checkpoint_steps(
        run_dir, int(entry.get("interval", 50000))
    )
    for step in steps:
        load_weights(model, run_dir / f"step{step}")
        prober.run(model.encoder, run_name, step, results)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--queue", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--lr-grid", default="[1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1]")
    ap.add_argument(
        "--shard", default="0/1", help="i/n: this job takes every n-th entry from i"
    )
    ap.add_argument("--threads", type=int, default=60)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--results-dir", default=str(FLUX_ROOT / "probe_results"))
    ap.add_argument("--done-dir", default=str(FLUX_ROOT / "probe_runs"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    index, count = (int(v) for v in args.shard.split("/"))
    queue = json.loads(Path(args.queue).read_text())[index::count]
    done_dir = Path(args.done_dir)
    done_dir.mkdir(parents=True, exist_ok=True)
    for k, entry in enumerate(queue, start=1):
        marker = done_dir / f"{entry['name']}.done"
        if marker.exists():
            logger.info("[%d/%d] %s: done already", k, len(queue), entry["name"])
            continue
        start = time.monotonic()
        run_entry(entry, args)
        minutes = (time.monotonic() - start) / 60
        marker.write_text(f"{minutes:.1f} min (cpu)\n")
        logger.info(
            "[%d/%d] %s finished in %.1f min", k, len(queue), entry["name"], minutes
        )


if __name__ == "__main__":
    main()
