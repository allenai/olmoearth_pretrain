"""Launch Beaker jobs for rslearn-to-OlmoEarth geotiff conversion and meta-summary CSV creation.

Distributes modules across hosts round-robin so each Beaker job processes
its assigned modality conversion(s).  Modalities are discovered from the
dataset's config.json, not hardcoded.

Usage:
    python -m olmoearth_pretrain.dataset_creation.launch_geotiff_jobs \
        list-modalities --ds-path /weka/dfive-default/helios/dataset_creation/candidates

    python -m olmoearth_pretrain.dataset_creation.launch_geotiff_jobs launch-geotiff \
        --ds-path /weka/dfive-default/helios/dataset_creation/candidates \
        --olmoearth-path /weka/dfive-default/helios/olmoearth_dataset \
        --hosts host1 host2 host3 --workers 32 \
        --modules sentinel1 --dry-run

    python -m olmoearth_pretrain.dataset_creation.launch_geotiff_jobs launch-meta-summary \
        --ds-path /weka/dfive-default/helios/dataset_creation/candidates \
        --olmoearth-path /weka/dfive-default/helios/olmoearth_dataset \
        --hosts host1 host2 host3 \
        --modules landsat sentinel2 \
        --time-spans year \
        --dry-run \

"""

from __future__ import annotations

import argparse
import logging
import uuid
from pathlib import Path
from typing import TypeVar

import tqdm
from beaker import Beaker
from beaker.data_model.experiment_spec import (
    Constraints,
    DataMount,
    DataSource,
    ExperimentSpec,
    Priority,
)

from olmoearth_pretrain.dataset_creation.launch_beaker_jobs import (
    DEFAULT_BASE_IMAGE,
    DEFAULT_BUDGET,
    DEFAULT_WORKSPACE,
    WEKA_BUCKET,
    _create_gcp_credentials_mount,
    _create_gee_credentials_mount,
    _get_env_vars,
    group_layers_into_modalities,
    load_layer_names,
)

logger = logging.getLogger(__name__)

_PACKAGE_ROOT = str(Path(__file__).resolve().parents[2])

SETUP_COMMANDS = (
    "apt-get update && "
    "apt-get install -y --no-install-recommends libpq-dev ffmpeg libsm6 libxext6 git wget && "
    "apt-get clean && rm -rf /var/lib/apt/lists/* && "
    "pip install --no-cache-dir 'rslearn[extra]{version_spec}' && "
    "pip install --no-cache-dir -e {olmoearth_install_path}"
)

# Multitemporal modalities need separate make_meta_summary runs for each
# time_span.  All other modalities get a single run without --time_span.
MULTITEMPORAL_MODALITIES = {
    "era5",
    "era5_10",
    "landsat",
    "sentinel1",
    "sentinel2",
    "sentinel2_l2a",
}
MULTITEMPORAL_TIME_SPANS = ("two_week", "year")


def discover_modalities(
    ds_path: str,
    local_config_path: str | None = None,
) -> list[str]:
    """Discover modality names from a dataset's config.json."""
    all_layers = load_layer_names(ds_path, config_json_path=local_config_path)
    modalities = group_layers_into_modalities(all_layers)
    return sorted(modalities.keys())


def build_meta_summary_jobs(
    modalities: list[str],
    time_spans: tuple[str, ...] | None = None,
) -> list[tuple[str, str | None]]:
    """Build the list of (modality, time_span) meta-summary jobs.

    Multitemporal modalities get one job per time_span; others get a single
    job with time_span=None.

    Args:
        modalities: modality names to process.
        time_spans: override which time spans to use for multitemporal
            modalities. If None, uses :data:`MULTITEMPORAL_TIME_SPANS`.
    """
    spans = time_spans if time_spans is not None else MULTITEMPORAL_TIME_SPANS
    jobs: list[tuple[str, str | None]] = []
    for mod in modalities:
        if mod in MULTITEMPORAL_MODALITIES:
            for ts in spans:
                jobs.append((mod, ts))
        else:
            jobs.append((mod, None))
    return jobs


def _meta_job_label(modality: str, time_span: str | None) -> str:
    return f"{modality}-{time_span}" if time_span else modality


_T = TypeVar("_T")


def assign_jobs_to_slots(
    jobs: list[_T],
    num_slots: int,
) -> list[list[_T]]:
    """Distribute *jobs* across *num_slots* round-robin.

    Returns a list of length ``min(num_slots, len(jobs))`` where each element
    is the list of job names assigned to that slot.
    """
    effective = min(num_slots, len(jobs))
    slots: list[list[_T]] = [[] for _ in range(effective)]
    for i, job in enumerate(jobs):
        slots[i % effective].append(job)
    return slots


def build_geotiff_command(
    module: str,
    ds_path: str,
    olmoearth_path: str,
    workers: int = 32,
    group: str | None = None,
    extra_args: list[str] | None = None,
) -> str:
    """Build a shell command to run one rslearn_to_olmoearth module."""
    cmd = [
        "python",
        "-m",
        f"olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.{module}",
        "--ds_path",
        ds_path,
        "--olmoearth_path",
        olmoearth_path,
        "--workers",
        str(workers),
    ]
    if group:
        cmd += ["--group", group]
    if extra_args:
        cmd += extra_args
    return " ".join(cmd)


def build_meta_summary_command(
    modality: str,
    olmoearth_path: str,
    time_span: str | None = None,
) -> str:
    """Build a shell command to run make_meta_summary for one (modality, time_span)."""
    cmd = [
        "python",
        "-m",
        "olmoearth_pretrain.dataset_creation.make_meta_summary",
        "--olmoearth_path",
        olmoearth_path,
        "--modality",
        modality,
    ]
    if time_span:
        cmd += ["--time_span", time_span]
    return " ".join(cmd)


def launch_beaker_job(
    *,
    command_str: str,
    job_name: str,
    base_image: str,
    rslearn_version: str | None,
    olmoearth_install_path: str,
    hostname: str | None = None,
    clusters: list[str] | None = None,
    priority: Priority = Priority.normal,
) -> str:
    """Create and submit a single Beaker experiment.

    Similar to :func:`launch_beaker_jobs.launch_beaker_job` but uses
    :data:`SETUP_COMMANDS` which also installs ``olmoearth_pretrain``.
    """
    version_spec = f"=={rslearn_version}" if rslearn_version else ""
    setup = SETUP_COMMANDS.format(
        version_spec=version_spec,
        olmoearth_install_path=olmoearth_install_path,
    )
    full_command = f"{setup} && {command_str}"

    beaker = Beaker.from_env(default_workspace=DEFAULT_WORKSPACE)
    experiment_name = f"geotiff-{job_name}-{str(uuid.uuid4())[:8]}"

    weka_mount = DataMount(
        source=DataSource(weka=WEKA_BUCKET),
        mount_path=f"/weka/{WEKA_BUCKET}",
    )

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
        env_vars=_get_env_vars(),
    )
    beaker.experiment.create(name=experiment_name, spec=experiment_spec)
    return experiment_name


def _print_slot_plan(
    slots: list[list[str]],
    hosts: list[str] | None,
    commands_per_slot: list[str],
) -> None:
    """Print the distribution plan to stdout."""
    print(f"\n{'=' * 72}")
    print(
        f"  {len(slots)} Beaker jobs distributing {sum(len(s) for s in slots)} job units"
    )
    print(f"{'=' * 72}\n")
    for i, slot_items in enumerate(slots):
        target = hosts[i] if hosts else f"cluster-job-{i}"
        print(f"  Job {i} -> {target}")
        for item in slot_items:
            print(f"    - {item}")
        print(f"    Command: {commands_per_slot[i]}\n")


def launch_geotiff_jobs(
    ds_path: str,
    olmoearth_path: str,
    hosts: list[str] | None = None,
    clusters: list[str] | None = None,
    num_jobs: int | None = None,
    modules: list[str] | None = None,
    workers: int = 32,
    group: str | None = None,
    base_image: str = DEFAULT_BASE_IMAGE,
    rslearn_version: str | None = None,
    olmoearth_install_path: str = _PACKAGE_ROOT,
    priority: Priority = Priority.normal,
    extra_args: list[str] | None = None,
    dry_run: bool = False,
    local_config_path: str | None = None,
) -> None:
    """Launch Beaker jobs for geotiff conversion, one or more modules per job."""
    if (clusters is not None) == (hosts is not None):
        raise ValueError("exactly one of --hosts or --clusters must be set")
    if clusters is not None and num_jobs is None:
        raise ValueError("--num-jobs is required when using --clusters")

    available = discover_modalities(ds_path, local_config_path=local_config_path)

    if modules:
        unknown = set(modules) - set(available)
        if unknown:
            raise ValueError(
                f"Unknown modules: {sorted(unknown)}. "
                f"Available (from config.json): {available}"
            )
        module_list = modules
    else:
        module_list = available

    num_slots = len(hosts) if hosts else num_jobs
    assert num_slots is not None
    slots = assign_jobs_to_slots(module_list, num_slots)

    commands_per_slot = []
    for slot_modules in slots:
        commands = [
            build_geotiff_command(
                module=mod,
                ds_path=ds_path,
                olmoearth_path=olmoearth_path,
                workers=workers,
                group=group,
                extra_args=extra_args,
            )
            for mod in slot_modules
        ]
        commands_per_slot.append(" && ".join(commands))

    _print_slot_plan(slots, hosts, commands_per_slot)

    if dry_run:
        print("Dry run -- no jobs submitted.")
        return

    for i, slot_modules in enumerate(tqdm.tqdm(slots, desc="Launching geotiff jobs")):
        hostname = hosts[i] if hosts else None
        launch_beaker_job(
            command_str=commands_per_slot[i],
            job_name="_".join(slot_modules),
            base_image=base_image,
            rslearn_version=rslearn_version,
            olmoearth_install_path=olmoearth_install_path,
            hostname=hostname,
            clusters=clusters,
            priority=priority,
        )


def launch_meta_summary_jobs(
    ds_path: str,
    olmoearth_path: str,
    hosts: list[str] | None = None,
    clusters: list[str] | None = None,
    num_jobs: int | None = None,
    modules: list[str] | None = None,
    time_spans: tuple[str, ...] | None = None,
    base_image: str = DEFAULT_BASE_IMAGE,
    rslearn_version: str | None = None,
    olmoearth_install_path: str = _PACKAGE_ROOT,
    priority: Priority = Priority.normal,
    dry_run: bool = False,
    local_config_path: str | None = None,
) -> None:
    """Launch Beaker jobs for meta-summary CSV creation."""
    if (clusters is not None) == (hosts is not None):
        raise ValueError("exactly one of --hosts or --clusters must be set")
    if clusters is not None and num_jobs is None:
        raise ValueError("--num-jobs is required when using --clusters")

    available = discover_modalities(ds_path, local_config_path=local_config_path)

    if modules:
        unknown = set(modules) - set(available)
        if unknown:
            raise ValueError(
                f"Unknown modules: {sorted(unknown)}. "
                f"Available (from config.json): {available}"
            )
        modality_list = modules
    else:
        modality_list = available

    all_jobs = build_meta_summary_jobs(modality_list, time_spans=time_spans)

    labels = [_meta_job_label(mod, ts) for mod, ts in all_jobs]

    num_slots = len(hosts) if hosts else num_jobs
    assert num_slots is not None
    slots = assign_jobs_to_slots(list(range(len(all_jobs))), num_slots)

    slot_labels: list[list[str]] = []
    commands_per_slot: list[str] = []
    for slot_indices in slots:
        sl = [labels[j] for j in slot_indices]
        slot_labels.append(sl)
        commands = [
            build_meta_summary_command(
                modality=all_jobs[j][0],
                olmoearth_path=olmoearth_path,
                time_span=all_jobs[j][1],
            )
            for j in slot_indices
        ]
        commands_per_slot.append(" && ".join(commands))

    _print_slot_plan(slot_labels, hosts, commands_per_slot)

    if dry_run:
        print("Dry run -- no jobs submitted.")
        return

    for i, slot_indices in enumerate(
        tqdm.tqdm(slots, desc="Launching meta-summary jobs")
    ):
        hostname = hosts[i] if hosts else None
        launch_beaker_job(
            command_str=commands_per_slot[i],
            job_name="_".join(slot_labels[i]),
            base_image=base_image,
            rslearn_version=rslearn_version,
            olmoearth_install_path=olmoearth_install_path,
            hostname=hostname,
            clusters=clusters,
            priority=priority,
        )


def _add_common_launch_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments shared by both launch subcommands."""
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--hosts",
        nargs="+",
        help="Beaker hostnames to pin jobs to (modules distributed round-robin).",
    )
    target_group.add_argument(
        "--clusters",
        nargs="+",
        help="Beaker clusters to target. Requires --num-jobs.",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=None,
        help="Number of jobs to launch in cluster mode.",
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        default=None,
        help=(
            "Subset of modalities to process. If omitted, all modalities "
            "discovered from config.json are used. "
            "Use 'list-modalities --ds-path ...' to see available names."
        ),
    )
    parser.add_argument(
        "--local-config-path",
        default=None,
        metavar="PATH",
        help=(
            "Path to config.json on this machine for modality discovery. "
            "If omitted, reads {ds-path}/config.json (must be reachable here)."
        ),
    )
    parser.add_argument(
        "--base-image",
        default=DEFAULT_BASE_IMAGE,
        help="Docker base image for the Beaker job.",
    )
    parser.add_argument(
        "--rslearn-version",
        default=None,
        help="Pin rslearn to a specific version (e.g. 0.1.3). Default: latest.",
    )
    parser.add_argument(
        "--priority",
        default="normal",
        choices=["low", "normal", "high", "urgent"],
        help="Beaker job priority (default: normal).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the distribution plan and commands without submitting jobs.",
    )


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Launch Beaker jobs for rslearn-to-OlmoEarth geotiff conversion "
            "and meta-summary CSV creation."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- list-modalities ---
    list_parser = subparsers.add_parser(
        "list-modalities",
        help="Discover and print modalities from config.json.",
    )
    list_parser.add_argument(
        "--ds-path",
        required=True,
        help="Path to the rslearn dataset root.",
    )
    list_parser.add_argument(
        "--local-config-path",
        default=None,
        help="Path to config.json (overrides {ds-path}/config.json).",
    )

    # --- launch-geotiff ---
    geotiff_parser = subparsers.add_parser(
        "launch-geotiff",
        help="Launch Beaker jobs for rslearn-to-OlmoEarth geotiff conversion.",
    )
    geotiff_parser.add_argument(
        "--ds-path",
        required=True,
        help="Path to the rslearn dataset on Beaker workers.",
    )
    geotiff_parser.add_argument(
        "--olmoearth-path",
        required=True,
        help="Path to the OlmoEarth dataset on Beaker workers.",
    )
    geotiff_parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of workers per module (default: 32).",
    )
    geotiff_parser.add_argument(
        "--group",
        default=None,
        help="rslearn window group(s) to convert (forwarded to modules).",
    )
    geotiff_parser.add_argument(
        "--extra-args",
        nargs="*",
        default=None,
        help="Additional arguments forwarded to every conversion module.",
    )
    _add_common_launch_args(geotiff_parser)

    # --- launch-meta-summary ---
    meta_parser = subparsers.add_parser(
        "launch-meta-summary",
        help="Launch Beaker jobs for meta-summary CSV creation.",
    )
    meta_parser.add_argument(
        "--ds-path",
        required=True,
        help="Path to the rslearn dataset (for modality discovery from config.json).",
    )
    meta_parser.add_argument(
        "--olmoearth-path",
        required=True,
        help="Path to the OlmoEarth dataset on Beaker workers.",
    )
    meta_parser.add_argument(
        "--time-spans",
        nargs="+",
        default=None,
        choices=list(MULTITEMPORAL_TIME_SPANS),
        help=(
            "Time spans to generate for multitemporal modalities. "
            f"Choices: {list(MULTITEMPORAL_TIME_SPANS)}. "
            "If omitted, all time spans are used."
        ),
    )
    _add_common_launch_args(meta_parser)

    args = parser.parse_args()

    priority_map = {
        "low": Priority.low,
        "normal": Priority.normal,
        "high": Priority.high,
        "urgent": Priority.urgent,
    }

    if args.command == "list-modalities":
        modalities = discover_modalities(
            args.ds_path, local_config_path=args.local_config_path
        )
        meta_jobs = build_meta_summary_jobs(modalities)
        print(f"Discovered {len(modalities)} modalities from config.json:\n")
        for mod in modalities:
            print(f"  {mod}")
        print(f"\nMeta-summary jobs ({len(meta_jobs)}):")
        for mod, ts in meta_jobs:
            print(f"  {_meta_job_label(mod, ts)}")
        return

    if args.command == "launch-geotiff":
        launch_geotiff_jobs(
            ds_path=args.ds_path,
            olmoearth_path=args.olmoearth_path,
            hosts=args.hosts,
            clusters=args.clusters,
            num_jobs=args.num_jobs,
            modules=args.modules,
            workers=args.workers,
            group=args.group,
            base_image=args.base_image,
            rslearn_version=args.rslearn_version,
            priority=priority_map[args.priority],
            extra_args=args.extra_args,
            dry_run=args.dry_run,
            local_config_path=args.local_config_path,
        )
    elif args.command == "launch-meta-summary":
        ts_arg: tuple[str, ...] | None = (
            tuple(args.time_spans) if args.time_spans else None
        )
        launch_meta_summary_jobs(
            ds_path=args.ds_path,
            olmoearth_path=args.olmoearth_path,
            hosts=args.hosts,
            clusters=args.clusters,
            num_jobs=args.num_jobs,
            modules=args.modules,
            time_spans=ts_arg,
            base_image=args.base_image,
            rslearn_version=args.rslearn_version,
            priority=priority_map[args.priority],
            dry_run=args.dry_run,
            local_config_path=args.local_config_path,
        )


if __name__ == "__main__":
    main()
