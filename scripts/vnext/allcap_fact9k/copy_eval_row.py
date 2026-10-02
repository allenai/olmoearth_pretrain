"""Copy one checkpoint's eval metrics from an eval-job W&B run into the training run.

Eval jobs (``checkpoint_sweep_evals.py``) log to their own W&B run on the
``checkpoint_step`` x-axis. This copies the rows for one ``checkpoint_step`` into
the (possibly still running) training run as a W&B *shared-mode secondary
writer*: one step-less ``log`` call, no config upload, no change to the run's
finished state, so the live training writer is unaffected. The training run's
in-loop evals plot ``eval/*`` against ``checkpoint_step`` too, so the copied
point lands on the same curves.

Usage:
    python scripts/vnext/allcap_fact9k/copy_eval_row.py --source-run 70a87be1 \
        --target-run 42u0k3ju --checkpoint-step 5000 [--dry-run]
"""

import argparse

import wandb

EVAL_PREFIXES = ("eval/", "eval_other/", "eval_time/", "eval_embed_diagnostics/")


def main() -> None:
    """Read the source rows for one checkpoint step and log them to the target run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="eai-ai2")
    parser.add_argument("--project", default="2026_09_30_allcap_fact9k")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--target-run", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = wandb.Api().run(f"{args.entity}/{args.project}/{args.source_run}")
    row: dict[str, float | int] = {}
    for record in source.scan_history():
        if record.get("checkpoint_step") != args.checkpoint_step:
            continue
        row.update(
            {
                k: v
                for k, v in record.items()
                if k.startswith(EVAL_PREFIXES) and v is not None
            }
        )
    if not row:
        raise SystemExit(
            f"no eval rows at checkpoint_step={args.checkpoint_step} in {source.path}"
        )
    row["checkpoint_step"] = args.checkpoint_step
    for k in sorted(row):
        print(f"{k} = {row[k]}")
    if args.dry_run:
        return

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.target_run,
        settings=wandb.Settings(
            mode="shared",
            x_primary=False,
            x_label=f"eval_copy_step{args.checkpoint_step}",
            x_update_finish_state=False,
        ),
    )
    run.log(row)
    run.finish()
    print(f"logged {len(row)} values to {args.entity}/{args.project}/{args.target_run}")


if __name__ == "__main__":
    main()
