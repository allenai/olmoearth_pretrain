"""Collect runs from wandb and launch evals for checkpoints.

Example:
python scripts/get_and_run_phase1_evals.py \
    --wandb_project ai2/helios \
    --min_steps 100000 \
    --eval_step 450000 \
    --cluster ai2/saturn-cirrascale \
    --filter_tag phase1
"""

import argparse
import subprocess  # nosec
from logging import getLogger

import wandb
from olmo_core.utils import prepare_cli_environment

logger = getLogger(__name__)
WANDB_ENTITY = "eai-ai2"


def get_runs_with_min_steps(
    wandb_project: str,
    min_steps: int,
    filter_tag: str | None = None,
    filter_name_prefix: str | None = None,
) -> list[tuple[str, str, str, int]]:
    """Get all runs that have completed at least min_steps.

    Returns:
        List of tuples (run_name, run_id, run_project, max_step).
    """
    api = wandb.Api()

    # Build filters
    filters = {}
    if filter_tag:
        filters["tags"] = {"$in": [filter_tag]}

    logger.info(f"Fetching runs from {wandb_project} with filters: {filters}")
    runs = api.runs(wandb_project, filters=filters)
    logger.info(f"Found {len(runs)} runs in project {wandb_project}")
    qualifying_runs = []
    for run in runs:
        run_name = run.name
        run_id = run.id  # This is the unique identifier

        # Apply name prefix filter if specified
        if filter_name_prefix and not run_name.startswith(filter_name_prefix):
            continue

        # Get the max step from the run's summary
        max_step = run.summary.get("_step", 0)

        if max_step >= min_steps:
            run_project = run.project
            qualifying_runs.append((run_id, run_project, max_step))
            logger.info(
                f"Found qualifying run: {run_name} (id={run_id}) in {run_project} (max_step={max_step})"
            )
        else:
            logger.debug(f"Skipping {run_name}: only {max_step} steps")

    return qualifying_runs


def get_eval_step_for_run(max_step: int, target_step: int) -> int:
    """Get the step to evaluate: target_step if available, else highest step < target_step."""
    if max_step >= target_step:
        return target_step
    else:
        # Return the highest step available that is divisible by 5000
        logger.warning(
            f"Run only has {max_step} steps, using that instead of {target_step}"
        )
        return max_step - (max_step % 5000)


def get_checkpoint_path_from_wandb_run_name_and_step(
    wandb_project: str, wandb_run_name: str, step: int
) -> str:
    """Get the checkpoint path from the wandb run name and step."""
    import wandb

    api = wandb.Api()
    try:
        run = api.run(f"{WANDB_ENTITY}/{wandb_project}/{wandb_run_name}")
    except Exception as e:
        raise RuntimeError(f"Could not fetch wandb run {wandb_run_name}: {e}")

    work_dir = run.config["trainer"]["checkpointer"]["work_dir"]
    return f"{work_dir}/step{step}"


def get_module_path_from_wandb(
    wandb_project: str,
    wandb_run_name: str,
) -> str:
    """Get the module path from the run project."""
    import wandb

    api = wandb.Api()
    try:
        run = api.run(f"{WANDB_ENTITY}/{wandb_project}/{wandb_run_name}")
    except Exception as e:
        raise RuntimeError(f"Could not fetch wandb run {wandb_run_name}: {e}")
    return run.config["launch"]["cmd"][0]


def run_eval_for_checkpoint(
    run_project: str,
    run_id: str,
    step: int,
    cluster: str,
    extra_args: list[str],
    dry_run: bool = False,
) -> None:
    """Run eval for a specific checkpoint using full_eval_sweep.py."""
    cmd_parts = [
        "python3",
        "helios/internal/full_eval_sweep.py",
        f"--cluster={cluster}",
        f"--checkpoint_path={get_checkpoint_path_from_wandb_run_name_and_step(run_project, run_id, step)}",
        f"--module_path={get_module_path_from_wandb(run_project, run_id)}",
    ]

    # Add any extra arguments passed through
    cmd_parts.extend(extra_args)

    cmd = " ".join(cmd_parts)
    logger.info(f"Running: {cmd}")

    if not dry_run:
        subprocess.run(cmd, shell=True, check=True)  # nosec
    else:
        logger.info("DRY RUN - would execute above command")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Collect wandb runs and launch evals for their checkpoints"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        required=True,
        help="Wandb project name (e.g., ai2/helios)",
    )
    parser.add_argument(
        "--min_steps",
        type=int,
        required=True,
        help="Minimum number of steps a run must have",
    )
    parser.add_argument(
        "--eval_step",
        type=int,
        required=True,
        help="Step to evaluate (or highest step below this)",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        required=True,
        help="Cluster to run evals on (e.g., ai2/saturn-cirrascale)",
    )
    parser.add_argument(
        "--filter_tag",
        type=str,
        default=None,
        help="Only include runs with this tag",
    )
    parser.add_argument(
        "--filter_name_prefix",
        type=str,
        default=None,
        help="Only include runs whose names start with this prefix",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them",
    )

    args, extra_args = parser.parse_known_args()

    # Get qualifying runs
    runs = get_runs_with_min_steps(
        args.wandb_project,
        args.min_steps,
        args.filter_tag,
        args.filter_name_prefix,
    )
    logger.info(f"Runs: {runs}")
    logger.info(f"Found {len(runs)} qualifying runs")

    # Process each run
    for run_id, run_project, max_step in runs:
        eval_step = get_eval_step_for_run(max_step, args.eval_step)
        logger.info(f"Processing {run_id} at step {eval_step}")

        run_eval_for_checkpoint(
            run_project,
            run_id,
            eval_step,
            args.cluster,
            extra_args,
            args.dry_run,
        )

    logger.info(f"Completed processing {len(runs)} runs")


if __name__ == "__main__":
    prepare_cli_environment()
    main()
