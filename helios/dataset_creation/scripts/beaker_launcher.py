"""Launch Beaker jobs to parallelize data materialization."""

import argparse
import uuid

import tqdm
from beaker import (
    Beaker,
    Constraints,
    DataMount,
    DataSource,
    ExperimentSpec,
    Priority,
)

PLANETARY_COMPUTER_COMMAND = [
    "rslearn",
    "dataset",
    "materialize",
    "--root",
    "{ds_path}",
    "--workers",
    "64",
    "--no-use-initial-job",
    "--retry-max-attempts",
    "8",
    "--retry-backoff-seconds",
    "60",
    "--ignore-errors",
]

OPENSTREETMAP_COMMAND = [
    "rslearn",
    "dataset",
    "ingest",
    "--root",
    "{ds_path}",
    "--group",
    "res_10",
    "--workers",
    "1",
    "--load-workers",
    "128",
    "--no-use-initial-job",
]

# Map from modality to the commands to run.
MODALITY_COMMANDS = {
    "sentinel2_l2a": PLANETARY_COMPUTER_COMMAND + ["--group", "res_10"],
    "sentinel1": PLANETARY_COMPUTER_COMMAND + ["--group", "res_10"],
    "naip": PLANETARY_COMPUTER_COMMAND + ["--group", "res_0.625"],
    "openstreetmap": OPENSTREETMAP_COMMAND,
}

BEAKER_BUDGET = "ai2/d5"
BEAKER_WORKSPACE = "ai2/earth-systems"


def launch_job(
    image: str,
    ds_path: str,
    modality: str,
    clusters: list[str] | None = None,
    hostname: str | None = None,
) -> None:
    """Launch a Beaker job that materializes the specified modality.

    Args:
        image: the name of the Beaker image to use.
        ds_path: the dataset path.
        modality: the modality to materialize.
        clusters: optional list of Beaker clusters to target. One of hostname or
            clusters must be set.
        hostname: optional Beaker host to constrain to.
    """
    beaker = Beaker.from_env(default_workspace=BEAKER_WORKSPACE)
    with beaker.session():
        # Add random string since experiment names must be unique.
        task_uuid = str(uuid.uuid4())[0:16]
        experiment_name = f"helios-{modality}-{task_uuid}"

        command = [arg.format(ds_path=ds_path) for arg in MODALITY_COMMANDS[modality]]
        weka_mount = DataMount(
            source=DataSource(weka="dfive-default"),
            mount_path="/weka/dfive-default",
        )

        # Set one GPU if not targeting a specific host, otherwise we might have
        # hundreds of jobs scheduled on the same host.
        # Also we can only set cluster constraint if we do not specify hostname.
        resources: dict | None
        constraints: Constraints
        if hostname is None:
            if modality == "openstreetmap":
                resources = {"gpuCount": 8}
            else:
                resources = {"gpuCount": 1}
            constraints = Constraints(
                cluster=clusters,
            )
        else:
            resources = None
            constraints = Constraints(
                hostname=[hostname],
            )

        experiment_spec = ExperimentSpec.new(
            budget=BEAKER_BUDGET,
            task_name=experiment_name,
            beaker_image=image,
            priority=Priority.high,
            command=command,
            datasets=[weka_mount],
            resources=resources,
            preemptible=True,
            constraints=constraints,
        )
        beaker.experiment.create(experiment_name, experiment_spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch Beaker jobs to get Sentinel-2 L1C data",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        help="Path to the rslearn dataset for dataset creation assuming /weka/dfive-default/ is mounted",
        required=True,
    )
    parser.add_argument(
        "--modality",
        type=str,
        help="The modality to materialize in the jobs",
        required=True,
    )
    parser.add_argument(
        "--image_name",
        type=str,
        help="Name of the Beaker image to use for the job",
        required=True,
    )
    parser.add_argument(
        "--clusters",
        type=str,
        help="Comma-separated list of clusters to target",
        default=None,
    )
    parser.add_argument(
        "--num_jobs",
        type=int,
        help="Number of Beaker jobs to start (one of clusters+num_jobs or hosts must be set)",
        default=None,
    )
    parser.add_argument(
        "--hosts",
        type=str,
        help="Comma-separated list of hosts to start jobs on, one job per host (one of clusters+num_jobs or hosts must be set)",
        default=None,
    )
    args = parser.parse_args()

    if args.num_jobs is not None:
        for i in tqdm.tqdm(list(range(args.num_jobs)), desc="Launching jobs"):
            launch_job(
                args.image_name,
                args.ds_path,
                args.modality,
                clusters=args.clusters.split(","),
            )
    elif args.hosts is not None:
        for host in args.hosts.split(","):
            launch_job(
                args.image_name,
                args.ds_path,
                args.modality,
                hostname=host,
            )
    else:
        raise ValueError("one of num_jobs and hosts must be set")
