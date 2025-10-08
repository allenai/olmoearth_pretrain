"""Launch Beaker jobs to parallelize data materialization."""

import argparse
import uuid

from beaker import (
    Beaker,
    Constraints,
    DataMount,
    DataSource,
    EnvVar,
    ExperimentSpec,
    Priority,
)

BEAKER_BUDGET = "ai2/d5"
BEAKER_WORKSPACE = "ai2/earth-systems"


def launch_job(
    image: str,
    hostname: str,
    src_dir: str,
    dst_dir: str,
) -> None:
    """Launch a Beaker job on the specified host.

    Args:
        image: the name of the Beaker image to use.
        hostname: the Beaker host to launch on.
        src_dir: source directory
        dst_dir: destination directory
    """
    print(f"creating experiment on {hostname}")
    beaker = Beaker.from_env(default_workspace=BEAKER_WORKSPACE)
    with beaker.session():
        # Add random string since experiment names must be unique.
        task_uuid = str(uuid.uuid4())[0:16]
        experiment_name = f"helios-{task_uuid}"

        weka_mount = DataMount(
            source=DataSource(weka="dfive-default"),
            mount_path="/weka/dfive-default",
        )
        gcp_secret_mount = DataMount(
            source=DataSource(secret="RSLEARN_GCP_CREDENTIALS"),  # nosec
            mount_path="/etc/credentials/gcp_credentials.json",  # nosec
        )

        experiment_spec = ExperimentSpec.new(
            budget=BEAKER_BUDGET,
            task_name=experiment_name,
            beaker_image=image,
            priority=Priority.high,
            command=[
                "python",
                "copy_files.py",
                src_dir,
                dst_dir,
            ],
            datasets=[weka_mount, gcp_secret_mount],
            preemptible=True,
            constraints=Constraints(
                hostname=[hostname],
            ),
            env_vars=[
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
            ],
        )
        beaker.experiment.create(experiment_name, experiment_spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch Beaker jobs to get Sentinel-2 L1C data",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Name of the Beaker image to use for the job",
        required=True,
    )
    parser.add_argument(
        "--hosts",
        type=str,
        help="Comma-separated list of hosts to start jobs on",
        required=True,
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        help="Source directory",
        required=True,
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        help="Destination directory",
        required=True,
    )
    args = parser.parse_args()

    for host in args.hosts.split(","):
        launch_job(
            image=args.image,
            hostname=host,
            src_dir=args.src_dir,
            dst_dir=args.dst_dir,
        )
