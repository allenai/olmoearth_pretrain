"""Collect every metric the v1.3 eval sweep logged into one long CSV.

Each Beaker job logs to its own W&B run (see launch.py); every logged row carries
``checkpoint_step``. This flattens all runs of the sweep group into
``run,checkpoint_step,key,value`` rows, where ``key`` is the W&B key as logged:
``eval/<task>`` (val primary metric), ``eval/test/<task>`` (test primary metric),
``eval_other/...`` (secondary metrics), ``eval_embed_diagnostics/<task>/<stat>`` and
``eval_time/<task>``.

    python scripts/vnext/v1_3_eval_sweep/collect.py --out sweep_metrics.csv
"""

import argparse
import csv
import math

import wandb
from launch import WANDB_GROUP, WANDB_PROJECT

ENTITY = "eai-ai2"


def main() -> None:
    """Write the long-format metrics CSV."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    api = wandb.Api(timeout=180)
    runs = api.runs(f"{ENTITY}/{WANDB_PROJECT}", filters={"group": WANDB_GROUP})
    n = 0
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "state", "checkpoint_step", "key", "value"])
        for run in runs:
            for row in run.scan_history(page_size=1000):
                step = row.get("checkpoint_step")
                if step is None:
                    continue
                for key, value in row.items():
                    if key.startswith("_") or key == "checkpoint_step":
                        continue
                    if not isinstance(value, int | float) or math.isnan(value):
                        continue
                    writer.writerow([run.name, run.state, step, key, value])
                    n += 1
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
