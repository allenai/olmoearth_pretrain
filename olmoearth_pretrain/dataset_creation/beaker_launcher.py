"""Generic Beaker task launcher for the dataset-creation orchestrator.

This module owns everything we need for launching Beaker experiments from within
the dataset-creation pipeline and waiting for them to finish. It is intentionally
small: every stage (prepare/ingest/materialize/export) just calls
`launch_beaker_task(...)` with its modality-specific command.

Design notes:

* Names: Beaker experiment names must be unique per workspace. We auto-dedupe by
  appending a short hash of (name, time) if needed.
* Weka: always mounts the requested weka buckets read/write at
  `/weka/{bucket_name}` (matches the convention in-repo).
* PC rate limiting: callers can pass `PC_SDK_SUBSCRIPTION_KEY` as a Beaker secret
  via `env_secrets`; rslearn honors it automatically.
* Waiting: `wait_for_experiments` polls every `poll_interval_s` seconds and
  reports terminal state per experiment. The optional `on_poll` hook is how the
  orchestrator injects the heartbeat-based progress updates.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from beaker import (
    Beaker,
    DataMount,
    DataSource,
    EnvVar,
    ExperimentSpec,
    Priority,
    TaskResources,
    TaskSpec,
)
from beaker.data_model.job import CanceledCode
from beaker.exceptions import ExperimentConflict

logger = logging.getLogger(__name__)

# Beaker considers these states terminal (experiment has stopped running).
# Per beaker-py, job statuses include: created, scheduled, started, running,
# finalized, failed, canceled, preempted, succeeded. We group them.
_TERMINAL_SUCCESS = {"succeeded"}
_TERMINAL_FAILURE = {"failed", "canceled", "preempted"}
_TERMINAL_STATES = _TERMINAL_SUCCESS | _TERMINAL_FAILURE

# Beaker experiment names: lowercase alnum + dashes/underscores/dots, <= 128 chars.
# We enforce a simple subset: [a-z0-9_-] with dashes replacing anything else.
_VALID_NAME_RE = re.compile(r"[^a-z0-9_\-\.]")


def _sanitize_name(name: str, max_len: int = 120) -> str:
    """Coerce `name` into a Beaker-legal experiment/task name."""
    cleaned = _VALID_NAME_RE.sub("-", name.lower())
    if len(cleaned) > max_len:
        # Keep a prefix for readability + short hash to preserve uniqueness.
        h = hashlib.blake2b(cleaned.encode(), digest_size=4).hexdigest()
        cleaned = f"{cleaned[: max_len - 9]}-{h}"
    return cleaned


DEFAULT_BEAKER_IMAGE = "ai2/cuda12.8-ubuntu22.04-torch2.7.1"

# Setup steps run before the task command. Mirrors the pattern in
# olmoearth_pretrain/internal/common.py: clone repo at a pinned git ref,
# install deps with uv, activate the venv.
SETUP_STEPS = (
    "set -euxo pipefail",
    # gh CLI for private repo clone — conda is available on ai2 base images.
    "conda install gh --channel conda-forge -y -q",
    'echo "$GITHUB_TOKEN" | gh auth login --with-token',
    "gh auth status",
    # Clone repo + checkout exact ref. Matches the pattern in common.py.
    'gh repo clone "$REPO_URL" .',
    'git checkout "$GIT_REF"',
    "pip install -q uv",
    'export PATH="/root/.local/bin:$PATH"',
    "uv sync --locked --extra dataset-creation",
    # Activate the uv-managed venv for the actual command.
    "venv_path=$(uv run python -c 'import sys; print(sys.executable)')",
    'source "$(dirname "$venv_path")/activate"',
)


def build_setup_command(command: Sequence[str]) -> list[str]:
    """Wrap `command` with the repo-clone + uv-install setup steps.

    Returns a ``["bash", "-c", "..."]`` invocation so Beaker runs
    everything in a single shell.
    """
    script_lines = list(SETUP_STEPS) + [" ".join(command)]
    return ["bash", "-c", "\n".join(script_lines)]


@dataclass(frozen=True)
class BeakerJobConfig:
    """Shared Beaker settings every orchestrator-launched task uses.

    Pulled into its own dataclass so the orchestrator builds it once and passes
    it to every `launch_beaker_task` call, rather than threading ~10 kwargs.
    """

    beaker_image: str = DEFAULT_BEAKER_IMAGE
    clusters: Sequence[str] = ()
    workspace: str = "ai2/earth-systems"  # matches WORKSPACE in common.py
    budget: str = "ai2/es-platform"  # matches BUDGET in common.py
    priority: str = "high"
    preemptible: bool = True
    weka_buckets: Sequence[str] = ("dfive-default",)
    git_ref: str = ""
    git_repo_url: str = ""
    git_branch: str = ""
    env_vars: Mapping[str, str] = field(default_factory=dict)
    env_secrets: Mapping[str, str] = field(default_factory=dict)


def _build_mounts(weka_buckets: Sequence[str]) -> list[DataMount]:
    return [
        DataMount(
            source=DataSource(weka=bucket),
            mount_path=f"/weka/{bucket}",
        )
        for bucket in weka_buckets
    ]


def _build_env_vars(job: BeakerJobConfig) -> list[EnvVar]:
    out: list[EnvVar] = []
    # Git context for the setup steps.
    out.append(EnvVar(name="REPO_URL", value=job.git_repo_url))
    out.append(EnvVar(name="GIT_REF", value=job.git_ref))
    out.append(EnvVar(name="GIT_BRANCH", value=job.git_branch))
    for k, v in job.env_vars.items():
        out.append(EnvVar(name=k, value=v))
    for k, secret in job.env_secrets.items():
        out.append(EnvVar(name=k, secret=secret))
    return out


def launch_beaker_task(
    beaker: Beaker,
    *,
    name: str,
    description: str,
    command: Sequence[str],
    job: BeakerJobConfig,
    gpu_count: int = 0,
    cpu_count: float | None = None,
    shared_memory: str | None = None,
    extra_datasets: Iterable[DataMount] | None = None,
) -> str:
    """Submit a single Beaker task and return its experiment id.

    Idempotent-ish: if an experiment with the derived unique name already
    exists, we reuse it rather than erroring. Callers wanting true resume
    semantics should track the returned id in their progress file.

    Args:
        beaker: a Beaker client (construct with `Beaker.from_env(...)`).
        name: human-readable identifier (e.g. "sentinel2_l2a-materialize-0").
            Will be sanitized and, if needed, truncated + hash-suffixed.
        description: longer description shown in the Beaker UI. Orchestrator
            progress updates rewrite this later.
        command: argv to run inside the container.
        job: shared Beaker config.
        gpu_count: defaults to 0 (CPU-only, appropriate for rslearn stages).
        cpu_count: optional CPU request.
        shared_memory: optional shared-memory size, e.g. "2GiB".
        extra_datasets: additional DataMount objects beyond the weka buckets.

    Returns:
        Beaker experiment id.
    """
    task_name = _sanitize_name(name)
    resources = TaskResources(
        gpu_count=gpu_count,
        cpu_count=cpu_count,
        shared_memory=shared_memory,
    )
    datasets = _build_mounts(job.weka_buckets)
    if extra_datasets:
        datasets.extend(extra_datasets)

    # Wrap the actual command with repo-clone + uv-install setup.
    full_command = build_setup_command(command)

    spec = ExperimentSpec(
        budget=job.budget,
        description=description,
        tasks=[
            TaskSpec.new(
                name=task_name,
                beaker_image=job.beaker_image,
                command=full_command,
                cluster=list(job.clusters),
                priority=Priority(job.priority),
                preemptible=job.preemptible,
                resources=resources,
                datasets=datasets,
                env_vars=_build_env_vars(job),
            )
        ],
    )

    try:
        experiment = beaker.experiment.create(task_name, spec, workspace=job.workspace)
    except ExperimentConflict:
        # Name collision -> treat as resume: reuse the existing experiment.
        existing = beaker.experiment.get(task_name, workspace=job.workspace)
        logger.info("Reusing existing Beaker experiment %s (%s)", existing.id, task_name)
        return existing.id

    logger.info("Launched Beaker experiment %s (%s)", experiment.id, task_name)
    return experiment.id


def get_experiment_state(beaker: Beaker, exp_id: str) -> str:
    """Return a simplified state string for the latest job of `exp_id`.

    We don't use `JobStatus.current` because its precedence puts `finalized`
    ahead of `failed`, so a finalized-but-failed job reports as `finalized`. We
    want the true outcome, so we inspect the status fields directly.

    Returns one of:
        "succeeded", "failed", "canceled", "preempted" (terminal)
        "running", "idle", "scheduled", "created" (non-terminal)
    """
    exp = beaker.experiment.get(exp_id)
    if not exp.jobs:
        return "scheduled"
    # Most recent job handles retries / re-runs.
    latest = max(exp.jobs, key=lambda j: j.status.created)
    s = latest.status

    # Failure signals first, since a finalized + failed job is a failure.
    if s.failed is not None:
        return "failed"
    if s.canceled is not None:
        if s.canceled_code in (
            CanceledCode.system_preemption,
            CanceledCode.user_preemption,
        ):
            return "preempted"
        return "canceled"
    if s.finalized is not None:
        # Finalized without failure markers => success. Exit code gives us a
        # belt-and-suspenders check in case Beaker ever reports exit_code != 0
        # alongside finalized (shouldn't happen, but cheap to guard).
        if s.exit_code is not None and s.exit_code != 0:
            return "failed"
        return "succeeded"
    if s.exited is not None:
        # Exited but not yet finalized -> still in the finalization window.
        return "running"
    if s.started is not None:
        return "running"
    if s.scheduled is not None:
        return "scheduled"
    return "created"


def set_experiment_description(beaker: Beaker, exp_id: str, description: str) -> None:
    """Update the description shown in the Beaker UI for `exp_id`.

    Errors are logged and swallowed: a failed description update should never
    take down the orchestrator.
    """
    try:
        beaker.experiment.set_description(exp_id, description)
    except Exception as e:  # pragma: no cover - transient Beaker API failures
        logger.warning("Failed to update description for %s: %s", exp_id, e)


def wait_for_experiments(
    beaker: Beaker,
    experiment_ids: Sequence[str],
    *,
    poll_interval_s: float = 30.0,
    on_poll: Callable[[Mapping[str, str]], None] | None = None,
) -> dict[str, str]:
    """Block until every experiment in `experiment_ids` reaches a terminal state.

    Args:
        beaker: Beaker client.
        experiment_ids: experiment ids to watch.
        poll_interval_s: how often to query Beaker.
        on_poll: optional hook called after every poll with the current
            {exp_id: state} map. Used by the orchestrator to update heartbeats
            and Beaker descriptions.

    Returns:
        {exp_id: terminal_state} for every experiment in the input.
    """
    if not experiment_ids:
        return {}
    terminal: dict[str, str] = {}
    remaining = list(experiment_ids)
    while remaining:
        states: dict[str, str] = {}
        for exp_id in list(remaining):
            state = get_experiment_state(beaker, exp_id)
            states[exp_id] = state
            if state in _TERMINAL_STATES:
                terminal[exp_id] = state
                remaining.remove(exp_id)

        # Include already-terminal experiments in the on_poll view.
        if on_poll is not None:
            on_poll({**terminal, **states})

        if remaining:
            time.sleep(poll_interval_s)

    return terminal


def classify_terminal_states(
    terminal: Mapping[str, str],
) -> tuple[list[str], list[str]]:
    """Split a {exp_id: state} mapping into (succeeded, failed) id lists."""
    succeeded = [e for e, s in terminal.items() if s in _TERMINAL_SUCCESS]
    failed = [e for e, s in terminal.items() if s in _TERMINAL_FAILURE]
    return succeeded, failed
