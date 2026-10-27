"""Probe frozen ERA5 encoders with other poolings of the same token sequence (CPU).

The in-loop probe reads the mean of all 63 patch tokens (448 days). Targets
that hinge on the last weeks (8-day fluxes) may lose that signal in the mean,
so this re-probes the same checkpoints with poolings of the final tokens:

* ``mean``: the encoder's own pooled vector (what every eval uses);
* ``last1``: the last token (its patch spans the final 14 days);
* ``last4``: the mean of the last 4 tokens (the final 35 days);
* ``mean_last4``: ``mean`` and ``last4`` concatenated (2x width).

Token poolings get the same parameter-free LayerNorm as the pooled vector.
No retraining: the probe, seeds, LRs and splits are those of ``cpu_probe.py``.
Records land in ``<results_dir>/<name>.jsonl`` with task ``<task>@<pooling>``.

Usage (repo root)::

    python scripts/era5_supervised/v0/fluxnet/pooling_probe.py --queue <queue.json> --tasks "[...]"
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from cpu_probe import (
    CKPT_ROOT,
    DEVICE,
    FLUX_ROOT,
    Prober,
    load_weights,
    model_config,
    task_configs,
)
from olmo_core.utils import seed_all

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.datasets.configs import EvalDatasetConfig, TaskType
from olmoearth_pretrain.evals.linear_probe import ProbeType, train_and_eval_probe
from olmoearth_pretrain.train.callbacks.era5_evaluator_callback import (
    _append_results_jsonl,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

POOLINGS = ("mean", "last1", "last4", "mean_last4")


def pooled_variants(
    batches: list[Any], encoder: torch.nn.Module
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Embeddings of every pooling for all batches.

    Args:
        batches: Materialized eval batches.
        encoder: Frozen encoder.

    Returns:
        ``({pooling: [N, D']}, labels [N])``.
    """
    parts: dict[str, list[torch.Tensor]] = {p: [] for p in POOLINGS}
    labels = []
    encoder.eval()
    with torch.no_grad():
        for batch in batches:
            with torch.amp.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
                out = encoder(
                    era5=batch.era5, timestamps=batch.timestamps,
                    valid_mask=getattr(batch, "valid_mask", None),
                )  # fmt: skip
            tokens = out["tokens"].float()  # [B, T, D], no CLS / prior tokens
            mean = out["pooled"].float()
            width = tokens.shape[-1]
            last1 = F.layer_norm(tokens[:, -1], (width,))
            last4 = F.layer_norm(tokens[:, -4:].mean(dim=1), (width,))
            parts["mean"].append(mean)
            parts["last1"].append(last1)
            parts["last4"].append(last4)
            parts["mean_last4"].append(torch.cat([mean, last4], dim=-1))
            labels.append(batch.labels)
    return {p: torch.cat(v) for p, v in parts.items()}, torch.cat(labels)


def probe_all(
    prober: Prober, encoder: torch.nn.Module, run_name: str, step: int, results: Path
) -> None:
    """Probe every task with every pooling and append the records.

    Args:
        prober: Holds the task configs and cached batches.
        encoder: Frozen encoder.
        run_name: Name recorded in each line.
        step: Checkpoint step recorded in each line.
        results: JSONL file.
    """
    for task in prober.tasks:
        start = time.monotonic()
        batches = prober._batches(task)
        splits = [pooled_variants(b, encoder) if b else None for b in batches]
        task_type = TaskType(task.task_type)
        spatial = task_type in (TaskType.SEGMENTATION, TaskType.PER_PIXEL_REGRESSION)
        eval_config = EvalDatasetConfig(
            task_type=task_type, num_classes=task.num_classes or 2,
            is_multilabel=task.is_multilabel, imputes=[],
            supported_modalities=[Modality.ERA5L_DAY_10.name], height_width=task.height_width,
        )  # fmt: skip
        for pooling in POOLINGS:
            inputs: list[torch.Tensor | None] = []
            for split in splits:
                if split is None:
                    inputs += [None, None]
                    continue
                emb, labels = split[0][pooling], split[1]
                if spatial:
                    emb = emb.unsqueeze(1).unsqueeze(1)
                    labels = (
                        labels.unsqueeze(-1).unsqueeze(-1)
                        if labels.ndim == 1
                        else labels
                    )
                inputs += [emb, labels]
            sweep = {}
            for lr in [task.probe_lr, *task.probe_lr_grid]:
                torch.manual_seed(task.probe_seed)
                sweep[lr] = train_and_eval_probe(
                    config=eval_config, lr=lr,
                    train_embeddings=inputs[0], train_labels=inputs[1],
                    val_embeddings=inputs[2], val_labels=inputs[3],
                    test_embeddings=inputs[4], test_labels=inputs[5],
                    device=DEVICE, batch_size=task.probe_batch_size, epochs=task.probe_epochs,
                    eval_interval=task.probe_epochs, probe_type=ProbeType.LINEAR,
                )  # fmt: skip
            result = sweep[task.probe_lr]
            _append_results_jsonl(str(results), {
                "run": run_name, "step": step, "task": f"{task.name}@{pooling}", "probe_lr": task.probe_lr,
                "val": result.val_result.metrics if result.val_result else None,
                "test": result.test_result.metrics if result.test_result else None,
                "lrsweep": {f"{lr:.0e}": {"val": r.val_result.metrics if r.val_result else None,
                                          "test": r.test_result.metrics if r.test_result else None}
                            for lr, r in sweep.items() if lr in task.probe_lr_grid},
                "geometry": {}, "eval_time": time.monotonic() - start, "device": "cpu",
            })  # fmt: skip
        logger.info("%s step %d %s: %d poolings (%.0fs)", run_name, step, task.name,
                    len(POOLINGS), time.monotonic() - start)  # fmt: skip


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--queue", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--lr-grid", default="[1e-3,1e-2,3e-2,1e-1,3e-1]")
    ap.add_argument("--threads", type=int, default=56)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--results-dir", default=str(FLUX_ROOT / "probe_results"))
    ap.add_argument("--shard", default="0/1", help="i/n: every n-th entry from i")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    index, count = (int(v) for v in args.shard.split("/"))
    for entry in json.loads(Path(args.queue).read_text())[index::count]:
        run_dir = CKPT_ROOT / entry["source"]
        load_step = int(entry.get("load_step", 50000))
        config = model_config(run_dir / f"step{load_step}" / "config.json")
        seed_all(int(entry.get("init_seed", 0)))
        model = config.build()
        prober = Prober(
            task_configs(entry.get("tasks", args.tasks), args.lr_grid), args.workers
        )
        results = Path(args.results_dir) / f"{entry['name']}.jsonl"
        if not entry.get("random_init", False):
            load_weights(model, run_dir / f"step{load_step}")
        probe_all(prober, model.encoder, entry.get("run", entry["name"]),
                  0 if entry.get("random_init", False) else load_step, results)  # fmt: skip


if __name__ == "__main__":
    main()
