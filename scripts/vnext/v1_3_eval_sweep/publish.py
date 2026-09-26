"""Merge every v1.3 eval-sweep result into one W&B run.

The sweep jobs (launch.py) write one JSON per (task, checkpoint) to
``RESULTS_DIR/step{N}/{task}.json``; each holds the metrics the harness would have
logged (``eval/<task>``, ``eval/test/<task>``, ``eval_other/...``,
``eval_embed_diagnostics/...``, ``eval_time/<task>``) plus ``checkpoint_step``.
This logs every result not published yet to a single run, with ``checkpoint_step``
as the x-axis. It is the run's only writer, so re-run it as results come in; the
run id and the published files are recorded under ``RESULTS_DIR/_publish``.

    python scripts/vnext/v1_3_eval_sweep/publish.py
"""

import collections
import json
from pathlib import Path

import wandb
from launch import RESULTS_DIR

from olmoearth_pretrain.train.callbacks.evaluator_callback import (
    define_checkpoint_step_metrics,
)

ENTITY = "eai-ai2"
PROJECT = "2026_09_26_v1_3_eval_sweep"
RUN_NAME = "v13_eval_sweep"


def main() -> None:
    """Log every unpublished result file to the sweep's W&B run."""
    results = Path(RESULTS_DIR)
    state = results / "_publish"
    state.mkdir(exist_ok=True)
    ledger = state / "published.txt"
    published = set(ledger.read_text().split()) if ledger.exists() else set()

    by_step = collections.defaultdict(list)
    for path in sorted(results.glob("step*/*.json")):
        key = f"{path.parent.name}/{path.name}"
        if key not in published:
            by_step[int(path.parent.name[len("step") :])].append((key, path))
    if not by_step:
        print("nothing new to publish")
        return

    id_file = state / "wandb_run_id.txt"
    run_id = (
        id_file.read_text().strip() if id_file.exists() else wandb.util.generate_id()
    )
    id_file.write_text(run_id + "\n")
    run = wandb.init(
        entity=ENTITY, project=PROJECT, name=RUN_NAME, id=run_id, resume="allow"
    )
    define_checkpoint_step_metrics(wandb)
    keys = []
    for step in sorted(by_step):
        row: dict[str, float] = {}
        for key, path in by_step[step]:
            row.update(json.loads(path.read_text()))
            keys.append(key)
        wandb.log(row)
    url = run.url
    run.finish()  # flushes the upload; only then are the results marked published
    with ledger.open("a") as f:
        f.writelines(key + "\n" for key in keys)
    print(f"published {len(keys)} results over {len(by_step)} checkpoints to {url}")


if __name__ == "__main__":
    main()
