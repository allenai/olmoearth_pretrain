"""Merge the v1 and v2 v1.3 eval-sweep results into one canonical set and one W&B run.

Canonical result for (task, step):
- the v2 re-run (``RESULTS_DIR_V2/step{N}/{task}.json``, launch_v2.py) if there is one;
- else the v1 result (``RESULTS_DIR``), with the primary metric re-selected where
  v2 changed only the task's primary metric (``PRIMARY_V2``). Every metric is
  already in the v1 file and those probes evaluate once (epochs == eval
  interval), so this is exactly what the fixed config logs, without a re-run.

The canonical files are written to ``RESULTS_DIR_V2/canonical/step{N}/{task}.json``
(metrics only) with ``canonical/MANIFEST.json`` recording each file's source, then
logged to the single W&B run ``v13_eval_sweep_v2`` (x-axis ``checkpoint_step``),
which supersedes v1's run hviuiw30. This script is the run's only writer.

    python scripts/vnext/v1_3_eval_sweep/publish_v2.py [--no-wandb]
"""

import argparse
import collections
import json
import os
from pathlib import Path

from launch import RESULTS_DIR
from launch_v2 import RESULTS_DIR_V2

ENTITY = "eai-ai2"
PROJECT = "2026_09_26_v1_3_eval_sweep"
RUN_NAME = "v13_eval_sweep_v2"
CHECKPOINT_STEP_METRIC = "checkpoint_step"
# Same prefixes as evaluator_callback.EVAL_METRIC_PREFIXES (kept here so this
# script needs only wandb).
EVAL_METRIC_PREFIXES = (
    "eval/*",
    "eval/test/*",
    "eval_other/*",
    "eval_other/test/*",
    "eval_time/*",
    "eval_embed_diagnostics/*",
)

_AEF = {
    "africa_crop_mask": "auroc",
    "descals": "auroc",
    "ethiopia_crops": "auroc",
    "canada_crops_coarse": "macro_acc",
    "canada_crops_fine": "macro_acc",
    "glance": "macro_acc",
    "lcmap_lu": "macro_acc",
    "us_trees": "macro_acc",
}
# Task -> (v1 primary metric, v2 primary metric) for tasks whose fix is only the
# primary metric (all_evals.py): the 8 AEF tasks and their year-aligned siblings.
PRIMARY_V2 = {
    **{name: ("overall_acc", m) for name, m in _AEF.items()},
    **{
        f"{name}_year_aligned_sentinel1_sentinel2_landsat": ("overall_acc", m)
        for name, m in _AEF.items()
    },
}


def relabel(task: str, record: dict) -> dict:
    """Re-select the primary metric of a v1 record per PRIMARY_V2."""
    old, new = PRIMARY_V2[task]
    out = dict(record)
    for split in ("", "test/"):
        primary = f"eval/{split}{task}"
        other = f"eval_other/{split}{task}"
        if primary not in out or f"{other}/{new}" not in out:
            raise KeyError(f"{task}: missing {primary} or {other}/{new}")
        out[f"{other}/{old}"] = out[primary]
        out[primary] = out.pop(f"{other}/{new}")
    return out


def canonical_records() -> tuple[dict, dict]:
    """{(step, task): record} and {step/task: source} for the merged sweep."""
    records, sources = {}, {}
    for root, tag in ((Path(RESULTS_DIR), "v1"), (Path(RESULTS_DIR_V2), "v2")):
        for path in sorted(root.glob("step*/*.json")):
            step = int(path.parent.name[len("step") :])
            task = path.name[: -len(".json")]
            record = json.loads(path.read_text())
            source = tag
            if tag == "v1" and task in PRIMARY_V2:
                record, source = relabel(task, record), "v1, primary re-selected"
            records[(step, task)] = record  # v2 is read second and wins
            sources[f"step{step}/{task}"] = source
    return records, sources


def main() -> None:
    """Write the canonical files, then log them all to the v2 W&B run."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    records, sources = canonical_records()
    out = Path(RESULTS_DIR_V2) / "canonical"
    for (step, task), record in records.items():
        path = out / f"step{step}" / f"{task}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record))
        os.replace(tmp, path)
    counts = collections.Counter(sources.values())
    (out / "MANIFEST.json").write_text(
        json.dumps({"counts": counts, "sources": sources}, indent=1) + "\n"
    )
    print(f"{len(records)} canonical results -> {out}: {dict(counts)}")
    if args.no_wandb:
        return

    import wandb

    state = Path(RESULTS_DIR_V2) / "_publish"
    state.mkdir(exist_ok=True)
    id_file = state / "wandb_run_id.txt"
    if id_file.exists():
        raise SystemExit(
            f"{id_file} exists: the v2 run was already published; delete the run "
            "and this file to republish from scratch"
        )
    run_id = wandb.util.generate_id()
    id_file.write_text(run_id + "\n")
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=RUN_NAME,
        id=run_id,
        notes=(
            "v1.3 eval sweep v2: all 98 evals x the v1 checkpoint grid, with the v2 "
            "fixes (branch joer/v1_3-eval-sweep-v2). Supersedes hviuiw30. "
            "Canonical files: " + str(out)
        ),
        config={
            "sources": dict(counts),
            "results_v1": RESULTS_DIR,
            "results_v2": RESULTS_DIR_V2,
        },
    )
    wandb.define_metric(CHECKPOINT_STEP_METRIC)
    for prefix in EVAL_METRIC_PREFIXES:
        wandb.define_metric(prefix, step_metric=CHECKPOINT_STEP_METRIC)
    by_step = collections.defaultdict(dict)
    for (step, _), record in records.items():
        by_step[step].update(record)
    for step in sorted(by_step):
        wandb.log(by_step[step])
    url = run.url
    run.finish()
    (state / "published.txt").write_text(
        "\n".join(sorted(f"step{s}/{t}" for s, t in records)) + "\n"
    )
    print(f"published {len(records)} results over {len(by_step)} checkpoints to {url}")


if __name__ == "__main__":
    main()
