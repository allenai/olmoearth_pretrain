"""Launch in-loop downstream evals as separate Beaker jobs.

Used by :class:`~olmoearth_pretrain.train.callbacks.evaluator_callback.DownstreamEvaluatorCallback`
when ``run_as_beaker_job`` is enabled. Instead of pausing training to evaluate the
*live* model, we launch a Beaker job that evaluates the just-saved checkpoint via
``checkpoint_sweep_evals.py``. Training continues uninterrupted while the eval job
runs (and queues) independently.

This reuses the existing checkpoint-eval machinery end-to-end:
``checkpoint_sweep_evals.py`` already loads ``step{N}/`` checkpoints from a directory,
runs the same downstream evaluators, and logs metrics on a ``checkpoint_step`` wandb
x-axis -- so eval metrics are plotted against the training step they correspond to,
regardless of how far training has advanced by the time the eval job finishes.

Notes / constraints:
* The eval job reads a *saved checkpoint*, so the eval step must coincide with a
  checkpoint save (see ``launch_checkpoint_eval_job``'s existence check). In practice,
  set the evaluator ``eval_interval`` to a multiple of the checkpointer's
  ``save_interval`` (and prefer permanent over ephemeral saves so the checkpoint is
  not deleted before the eval job loads it).
* The spawned launch reuses the parent job's environment (``REPO_URL``, ``GIT_REF``,
  ``BEAKER_TOKEN``, ``WANDB_API_KEY``, ...), exactly like a manual ``launch_evaluate``.
"""

import logging
import os
import secrets
import subprocess  # nosec
import sys
from pathlib import Path

from upath import UPath

import olmoearth_pretrain
from olmoearth_pretrain.internal.constants import CHECKPOINT_SWEEP_LAUNCH_PATH

logger = logging.getLogger(__name__)

# Mirrors full_eval_sweep.LAUNCH_OVERRIDES (minus priority and num_gpus, which we
# resolve per-call -- num_gpus varies for rank-max LR sweeps).
_EVAL_LAUNCH_OVERRIDES = ["--launch.task_name=eval"]


def _repo_root() -> Path:
    """Return the repository root (parent of the ``olmoearth_pretrain`` package)."""
    return Path(olmoearth_pretrain.__file__).resolve().parent.parent


def resolve_parent_priority(default: str = "high") -> str:
    """Best-effort lookup of the *running* training job's Beaker priority.

    Returns the parent job's priority so the eval job can be launched at the same
    priority (consideration #2). Falls back to ``default`` when not running in Beaker
    or when the lookup fails for any reason.
    """
    job_id = os.environ.get("BEAKER_JOB_ID")
    if not job_id:
        return default
    try:
        from beaker import Beaker

        beaker = Beaker.from_env()
        priority = beaker.job.get(job_id).priority
        if priority is not None:
            # Priority is a StrEnum -> str() yields e.g. "high".
            return str(priority)
        logger.warning(
            "Parent Beaker job %s has no priority set; using %r", job_id, default
        )
    except Exception as e:  # noqa: BLE001 - best-effort, never block training
        logger.warning(
            "Could not resolve parent Beaker priority (%s); using %r", e, default
        )
    return default


def launch_checkpoint_eval_job(
    *,
    module_path: str,
    checkpoint_dir: str,
    step: int,
    cluster: str,
    run_name: str,
    priority: str,
    num_gpus: int = 1,
    tasks_to_run: list[str] | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    wandb_run_name: str | None = None,
    extra_overrides: list[str] | None = None,
    log_dir: str | None = None,
) -> bool:
    """Launch a Beaker job that evaluates ``checkpoint_dir/step{step}``.

    The job is submitted via a non-blocking ``subprocess.Popen`` invoking
    ``checkpoint_sweep_evals.py launch_evaluate``; this function returns as soon as
    the launcher subprocess is spawned so the training loop is not blocked.

    Returns ``True`` if a launcher subprocess was started, ``False`` if it was skipped
    (e.g. the checkpoint for ``step`` does not exist yet).
    """
    checkpoint_step_dir = UPath(checkpoint_dir) / f"step{step}"
    if not checkpoint_step_dir.exists():
        logger.warning(
            "Skipping Beaker eval launch for step %d: checkpoint %s does not exist. "
            "Ensure the evaluator eval_interval is a multiple of the checkpointer "
            "save_interval so a checkpoint is saved at the eval step.",
            step,
            checkpoint_step_dir,
        )
        return False

    script_path = _repo_root() / CHECKPOINT_SWEEP_LAUNCH_PATH

    env = dict(os.environ)
    env["TRAIN_SCRIPT_PATH"] = module_path
    env["CHECKPOINT_DIR"] = checkpoint_dir
    env["CHECKPOINT_STEPS"] = str(step)

    cmd: list[str] = [
        sys.executable,
        str(script_path),
        "launch_evaluate",
        run_name,
        cluster,
        f"--launch.priority={priority}",
        f"--launch.num_gpus={num_gpus}",
        *_EVAL_LAUNCH_OVERRIDES,
    ]
    # Tie the eval run to the training run's wandb project/entity/group so the
    # eval-over-training-steps curve lands alongside the training run.
    if wandb_project is not None:
        cmd.append(f"--trainer.callbacks.wandb.project={wandb_project}")
    if wandb_entity is not None:
        cmd.append(f"--trainer.callbacks.wandb.entity={wandb_entity}")
    if wandb_group is not None:
        cmd.append(f"--trainer.callbacks.wandb.group={wandb_group}")
    if wandb_run_name is not None:
        cmd.append(f"--trainer.callbacks.wandb.name={wandb_run_name}")
    # All in-loop eval jobs for this training run resume one shared wandb run id,
    # generated once here (in the single-threaded training process) and stored in
    # a fixed file under the checkpoint dir. Pre-creating it means each eval job
    # only ever reads + resumes this id -- so per-step metrics consolidate into a
    # single wandb run (keyed on checkpoint_step) with no race even when eval jobs
    # overlap, instead of creating a separate run per eval step.
    shared_runid_file = UPath(checkpoint_dir) / "loop_eval_wandb_runid.txt"
    if not shared_runid_file.exists():
        shared_runid_file.write_text(secrets.token_hex(4))
    cmd.append(f"--trainer.callbacks.wandb.runid_path={shared_runid_file}")
    if tasks_to_run:
        cmd.append(
            "--trainer.callbacks.downstream_evaluator.tasks_to_run="
            + "["
            + ",".join(tasks_to_run)
            + "]"
        )
    if extra_overrides:
        cmd.extend(extra_overrides)

    stdout = subprocess.DEVNULL
    stderr = subprocess.STDOUT
    log_handle = None
    if log_dir is not None:
        log_path = UPath(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / f"loop_eval_launch_step{step}.log"
        log_handle = log_file.open("w")
        stdout = log_handle
        stderr = subprocess.STDOUT

    logger.info(
        "Launching Beaker eval job for step %d (priority=%s, cluster=%s): %s",
        step,
        priority,
        cluster,
        " ".join(cmd),
    )
    # Non-blocking: spawn the launcher and return. The launcher itself submits the
    # Beaker experiment (follow=False) and exits within a few seconds.
    subprocess.Popen(  # nosec B603 - args are constructed from trusted config
        cmd,
        cwd=str(_repo_root()),
        env=env,
        stdout=stdout,
        stderr=stderr,
    )
    return True
