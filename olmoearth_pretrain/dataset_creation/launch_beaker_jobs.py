"""Launch Beaker jobs to run rslearn dataset prepare/ingest/materialize.

Distributes modalities across hosts so each Beaker job handles only its assigned
layers, using rslearn's --disabled-layers flag.

Usage:
    python -m olmoearth_pretrain.dataset_creation.launch_beaker_jobs \
        list-modalities --ds-path /weka/dfive-default/helios/dataset_creation/candidates

    python -m olmoearth_pretrain.dataset_creation.launch_beaker_jobs launch \
        --ds-path /weka/dfive-default/helios/dataset_creation/candidates \
        --stage materialize --group res_10 \
        --hosts host1 host2 host3 --workers 64

    When launching from a machine without the dataset mounted, pass a local copy of
    config.json via --local-config-path; rslearn in Beaker still uses --ds-path.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import uuid

import tqdm
from beaker import Beaker
from beaker.data_model.experiment_spec import (
    Constraints,
    DataMount,
    DataSource,
    EnvVar,
    ExperimentSpec,
    Priority,
)

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE = "ai2/earth-systems"
DEFAULT_BUDGET = "ai2/es-platform"
WEKA_BUCKET = "dfive-default"

DEFAULT_BASE_IMAGE = (
    "pytorch/pytorch:2.5.0-cuda11.8-cudnn9-runtime"
    "@sha256:d15e9803095e462e351f097fb1f5e7cdaa4f5e855d7ff6d6f36ec4c2aa2938ea"
)

SETUP_COMMANDS = (
    "apt-get update && "
    "apt-get install -y --no-install-recommends libpq-dev ffmpeg libsm6 libxext6 git wget && "
    "apt-get clean && rm -rf /var/lib/apt/lists/* && "
    "pip install --no-cache-dir 'rslearn[extra]{version_spec}'"
)

STAGES = ("prepare", "ingest", "materialize")


def _get_env_vars() -> list[EnvVar]:
    """Build the environment variables for dataset creation Beaker jobs."""
    env_vars = [
        EnvVar(
            name="GOOGLE_APPLICATION_CREDENTIALS",  # nosec
            value="/etc/credentials/gcp_credentials.json",  # nosec
        ),
        EnvVar(
            name="GCLOUD_PROJECT",  # nosec
            value="earthsystem-dev-c3po",  # nosec
        ),
        EnvVar(
            name="GOOGLE_CLOUD_PROJECT",  # nosec
            value="earthsystem-dev-c3po",  # nosec
        ),
        EnvVar(
            name="WEKA_ACCESS_KEY_ID",  # nosec
            secret="RSLEARN_WEKA_KEY",  # nosec
        ),
        EnvVar(
            name="WEKA_SECRET_ACCESS_KEY",  # nosec
            secret="RSLEARN_WEKA_SECRET",  # nosec
        ),
        EnvVar(
            name="WEKA_ENDPOINT_URL",  # nosec
            value="https://weka-aus.beaker.org:9000",  # nosec
        ),
        EnvVar(
            name="MKL_THREADING_LAYER",
            value="GNU",
        ),
        EnvVar(
            name="EARTHDATAHUB_TOKEN",
            secret="RSLEARN_EARTHDATAHUB_TOKEN",  # nosec
        ),
    ]

    if "NASA_EARTHDATA_USERNAME" in os.environ:
        env_vars += [
            EnvVar(
                name="NASA_EARTHDATA_USERNAME",
                value=os.environ["NASA_EARTHDATA_USERNAME"],
            ),
            EnvVar(
                name="NASA_EARTHDATA_PASSWORD",
                value=os.environ["NASA_EARTHDATA_PASSWORD"],
            ),
        ]
    if "HTTP_PROXY" in os.environ:
        env_vars.append(EnvVar(name="HTTP_PROXY", value=os.environ["HTTP_PROXY"]))
    if "HTTPS_PROXY" in os.environ:
        env_vars.append(EnvVar(name="HTTPS_PROXY", value=os.environ["HTTPS_PROXY"]))

    return env_vars


def _create_gcp_credentials_mount(
    secret: str = "RSLEARN_GCP_CREDENTIALS",
    mount_path: str = "/etc/credentials/gcp_credentials.json",
) -> DataMount:
    return DataMount(
        source=DataSource(secret=secret),  # nosec
        mount_path=mount_path,  # nosec
    )


def _create_gee_credentials_mount(
    mount_path: str = "/etc/credentials/gee_credentials.json",
) -> DataMount:
    secret = os.environ.get(
        "GEE_CREDENTIALS_MOUNT_SECRET", "GCP_HELIOS_SERVICE_ACCOUNT"
    )
    return _create_gcp_credentials_mount(secret, mount_path)


def load_layer_names(ds_path: str, config_json_path: str | None = None) -> list[str]:
    """Read layer names from the dataset's config.json.

    Args:
        ds_path: path to the rslearn dataset root (used to default config location).
        config_json_path: if set, read this file instead of ``{ds_path}/config.json``.
            Use when planning jobs locally while ``ds_path`` only exists on cluster mounts.

    Returns:
        sorted list of layer names defined in config.json.
    """
    config_path = config_json_path or os.path.join(ds_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    return sorted(config["layers"].keys())


def group_layers_into_modalities(layer_names: list[str]) -> dict[str, list[str]]:
    """Auto-group layer names into modalities by shared prefix.

    Layers whose names follow the pattern ``<prefix>_mo01``, ``<prefix>_mo02``, ...
    are grouped under ``<prefix>``.  Layers that don't match a multi-layer pattern
    become their own single-layer modality.  The ``tc-*`` layers (WorldCereal) are
    grouped under ``worldcereal``.

    Args:
        layer_names: all layer names from config.json.

    Returns:
        dict mapping modality name to the list of layer names it contains.
    """
    modalities: dict[str, list[str]] = {}
    for name in layer_names:
        if name.startswith("tc-"):
            modalities.setdefault("worldcereal", []).append(name)
        elif "_mo" in name:
            prefix = name[: name.index("_mo")]
            modalities.setdefault(prefix, []).append(name)
        elif "_freq" in name:
            prefix = name[: name.index("_freq")]
            modalities.setdefault(prefix, []).append(name)
        else:
            modalities.setdefault(name, []).append(name)
    return modalities


def build_disabled_layers(all_layers: list[str], assigned_layers: list[str]) -> str:
    """Return a comma-separated string of layers to disable.

    Args:
        all_layers: every layer in the config.
        assigned_layers: the layers this job should process.

    Returns:
        comma-separated disabled layer names for ``--disabled-layers``.
    """
    assigned_set = set(assigned_layers)
    disabled = [layer for layer in all_layers if layer not in assigned_set]
    return ",".join(disabled)


def build_rslearn_commands(
    stages: list[str],
    ds_path: str,
    disabled_layers: str,
    group: str | None = None,
    workers: int = 64,
    extra_args: list[str] | None = None,
) -> str:
    """Build a shell command string that runs one or more rslearn stages.

    Args:
        stages: list of stages to run (subset of prepare/ingest/materialize).
        ds_path: dataset root path.
        disabled_layers: comma-separated layers to disable.
        group: optional rslearn group argument.
        workers: number of parallel workers.
        extra_args: additional CLI flags for rslearn.

    Returns:
        a shell command string chaining the requested stages with ``&&``.
    """
    parts = []
    for stage in stages:
        cmd = [
            "rslearn",
            "dataset",
            stage,
            "--root",
            ds_path,
            "--workers",
            str(workers),
            "--no-use-initial-job",
            "--retry-max-attempts",
            "8",
            "--retry-backoff-seconds",
            "60",
        ]
        if stage == "materialize":
            cmd.append("--ignore-errors")
        if disabled_layers:
            cmd += ["--disabled-layers", disabled_layers]
        if group:
            cmd += ["--group", group]
        if extra_args:
            cmd += extra_args
        parts.append(" ".join(cmd))
    return " && ".join(parts)


def assign_modalities_to_slots(
    modalities: dict[str, list[str]],
    num_slots: int,
    selected_modalities: list[str] | None = None,
) -> list[tuple[str, list[str]]]:
    """Assign modalities to job slots (hosts or cluster jobs) round-robin.

    Args:
        modalities: full modality-to-layers mapping.
        num_slots: number of available slots (hosts or num_jobs).
        selected_modalities: optional subset of modality names to use.

    Returns:
        list of (modality_name, layer_names) tuples, one per job to launch.
        If num_slots > len(selected_modalities), extra slots get duplicate
        assignments for parallelism within a single modality.
    """
    if selected_modalities:
        unknown = set(selected_modalities) - set(modalities.keys())
        if unknown:
            raise ValueError(
                f"Unknown modalities: {unknown}. Available: {sorted(modalities.keys())}"
            )
        mod_list = [(name, modalities[name]) for name in selected_modalities]
    else:
        mod_list = sorted(modalities.items())

    assignments: list[tuple[str, list[str]]] = []
    for i in range(max(num_slots, len(mod_list))):
        idx = i % len(mod_list)
        assignments.append(mod_list[idx])
    return assignments


def launch_beaker_job(
    *,
    command_str: str,
    modality_name: str,
    base_image: str,
    rslearn_version: str | None,
    hostname: str | None = None,
    clusters: list[str] | None = None,
    priority: Priority = Priority.normal,
) -> str:
    """Create and submit a single Beaker experiment.

    Args:
        command_str: the shell command to execute after setup.
        modality_name: used in the experiment name for identification.
        base_image: Docker image to use as the base.
        rslearn_version: optional version pin for rslearn (e.g. "==0.1.3").
        hostname: pin to a specific Beaker host.
        clusters: target Beaker clusters.
        priority: job priority.

    Returns:
        the experiment name that was created.
    """
    version_spec = f"=={rslearn_version}" if rslearn_version else ""
    setup = SETUP_COMMANDS.format(version_spec=version_spec)
    full_command = f"{setup} && {command_str}"

    beaker = Beaker.from_env(default_workspace=DEFAULT_WORKSPACE)
    experiment_name = f"ds-{modality_name}-{str(uuid.uuid4())[:8]}"

    weka_mount = DataMount(
        source=DataSource(weka=WEKA_BUCKET),
        mount_path=f"/weka/{WEKA_BUCKET}",
    )

    env_vars = _get_env_vars()

    resources: dict | None
    constraints: Constraints
    if hostname is None:
        resources = {"gpuCount": 0}
        constraints = Constraints(cluster=clusters)
    else:
        resources = None
        constraints = Constraints(hostname=[hostname])

    experiment_spec = ExperimentSpec.new(
        budget=DEFAULT_BUDGET,
        task_name=experiment_name,
        docker_image=base_image,
        priority=priority,
        command=["bash", "-c", full_command],
        datasets=[
            weka_mount,
            _create_gcp_credentials_mount(),
            _create_gee_credentials_mount(),
        ],
        resources=resources,
        preemptible=True,
        constraints=constraints,
        env_vars=env_vars,
    )
    beaker.experiment.create(name=experiment_name, spec=experiment_spec)
    return experiment_name


def launch_jobs(
    ds_path: str,
    stages: list[str],
    group: str | None = None,
    hosts: list[str] | None = None,
    clusters: list[str] | None = None,
    num_jobs: int | None = None,
    modalities_filter: list[str] | None = None,
    workers: int = 64,
    base_image: str = DEFAULT_BASE_IMAGE,
    rslearn_version: str | None = None,
    priority: Priority = Priority.normal,
    extra_args: list[str] | None = None,
    local_config_path: str | None = None,
) -> None:
    """Launch Beaker jobs for rslearn dataset creation, one modality per job.

    Args:
        ds_path: path to the rslearn dataset on workers (and default config location
            for the launcher if ``local_config_path`` is not set).
        stages: list of stages to run (prepare, ingest, materialize).
        group: optional rslearn --group value (e.g. "res_10").
        hosts: list of Beaker hostnames to pin jobs to.
        clusters: list of Beaker clusters to target.
        num_jobs: number of jobs when using cluster mode.
        modalities_filter: optional subset of modality names to process.
        workers: number of rslearn workers per job.
        base_image: Docker base image for the Beaker job.
        rslearn_version: optional rslearn version pin.
        priority: Beaker job priority.
        extra_args: extra CLI flags forwarded to every rslearn command.
        local_config_path: optional path to config.json on the launch machine for
            layer/modality planning only; ``ds_path`` is still passed to rslearn in jobs.
    """
    if (clusters is not None and hosts is not None) or (
        clusters is None and hosts is None
    ):
        raise ValueError("exactly one of --hosts or --clusters must be set")
    if clusters is not None and num_jobs is None:
        raise ValueError("--num-jobs is required when using --clusters")

    all_layers = load_layer_names(ds_path, config_json_path=local_config_path)
    modalities = group_layers_into_modalities(all_layers)

    logger.info(
        "Detected %d modalities from %d layers: %s",
        len(modalities),
        len(all_layers),
        sorted(modalities.keys()),
    )

    num_slots = len(hosts) if hosts else num_jobs
    assert num_slots is not None
    assignments = assign_modalities_to_slots(modalities, num_slots, modalities_filter)

    logger.info("Launching %d jobs:", len(assignments))
    for i, (mod_name, layers) in enumerate(assignments):
        target = hosts[i % len(hosts)] if hosts else "cluster"
        logger.info(
            "  Job %d: modality=%s (%d layers) -> %s", i, mod_name, len(layers), target
        )

    for i, (mod_name, layers) in enumerate(
        tqdm.tqdm(assignments, desc="Launching jobs")
    ):
        disabled = build_disabled_layers(all_layers, layers)
        command_str = build_rslearn_commands(
            stages=stages,
            ds_path=ds_path,
            disabled_layers=disabled,
            group=group,
            workers=workers,
            extra_args=extra_args,
        )

        hostname = hosts[i % len(hosts)] if hosts else None
        launch_beaker_job(
            command_str=command_str,
            modality_name=mod_name,
            base_image=base_image,
            rslearn_version=rslearn_version,
            hostname=hostname,
            clusters=clusters,
            priority=priority,
        )


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Launch Beaker jobs for rslearn dataset creation, one modality per job.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- list-modalities subcommand ---
    list_parser = subparsers.add_parser(
        "list-modalities",
        help="Print detected modalities from config.json and exit.",
    )
    list_parser.add_argument(
        "--ds-path",
        required=True,
        help="Path to the rslearn dataset root (used only to default config.json location).",
    )
    list_parser.add_argument(
        "--local-config-path",
        default=None,
        help="Path to config.json for layer/modality detection (overrides {ds-path}/config.json).",
    )

    # --- launch subcommand ---
    launch_parser = subparsers.add_parser(
        "launch",
        help="Launch Beaker jobs for dataset creation.",
    )
    launch_parser.add_argument(
        "--ds-path",
        required=True,
        help="Path to the rslearn dataset on Beaker workers (rslearn uses this path).",
    )
    launch_parser.add_argument(
        "--local-config-path",
        default=None,
        metavar="PATH",
        help=(
            "Optional path to config.json on this machine for layer/modality planning and "
            "--disabled-layers. If omitted, reads {ds-path}/config.json (must be reachable here)."
        ),
    )
    launch_parser.add_argument(
        "--stage",
        required=True,
        choices=[*STAGES, "all"],
        help="Which rslearn stage(s) to run. 'all' runs prepare -> ingest -> materialize.",
    )
    launch_parser.add_argument(
        "--group",
        default=None,
        help="rslearn --group value (e.g. res_10).",
    )

    target_group = launch_parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--hosts",
        nargs="+",
        help="Beaker hostnames to pin jobs to (one modality per host, round-robin).",
    )
    target_group.add_argument(
        "--clusters",
        nargs="+",
        help="Beaker clusters to target. Requires --num-jobs.",
    )
    launch_parser.add_argument(
        "--num-jobs",
        type=int,
        default=None,
        help="Number of jobs to launch in cluster mode.",
    )
    launch_parser.add_argument(
        "--modalities",
        nargs="+",
        default=None,
        help=(
            "Subset of modalities to process. If omitted, all modalities in "
            "config.json are used. Use 'list-modalities' subcommand to see available names."
        ),
    )
    launch_parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Number of rslearn workers per job (default: 64).",
    )
    launch_parser.add_argument(
        "--base-image",
        default=DEFAULT_BASE_IMAGE,
        help="Docker base image for the Beaker job.",
    )
    launch_parser.add_argument(
        "--rslearn-version",
        default=None,
        help="Pin rslearn to a specific version (e.g. 0.1.3). Default: latest.",
    )
    launch_parser.add_argument(
        "--priority",
        default="normal",
        choices=["low", "normal", "high", "urgent"],
        help="Beaker job priority (default: normal).",
    )
    launch_parser.add_argument(
        "--extra-rslearn-args",
        nargs="*",
        default=None,
        help="Additional arguments forwarded to every rslearn command.",
    )

    args = parser.parse_args()

    if args.command == "list-modalities":
        all_layers = load_layer_names(
            args.ds_path, config_json_path=args.local_config_path
        )
        modalities = group_layers_into_modalities(all_layers)
        print(f"Detected {len(modalities)} modalities from {len(all_layers)} layers:\n")
        for name, layers in sorted(modalities.items()):
            print(f"  {name}: {layers}")
        return

    stages = list(STAGES) if args.stage == "all" else [args.stage]

    priority_map = {
        "low": Priority.low,
        "normal": Priority.normal,
        "high": Priority.high,
        "urgent": Priority.urgent,
    }

    launch_jobs(
        ds_path=args.ds_path,
        stages=stages,
        group=args.group,
        hosts=args.hosts,
        clusters=args.clusters,
        num_jobs=args.num_jobs,
        modalities_filter=args.modalities,
        workers=args.workers,
        base_image=args.base_image,
        rslearn_version=args.rslearn_version,
        priority=priority_map[args.priority],
        extra_args=args.extra_rslearn_args,
        local_config_path=args.local_config_path,
    )


if __name__ == "__main__":
    main()
