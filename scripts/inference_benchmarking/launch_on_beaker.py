"""Script to run locally to launch throughput benchmarking task in Beaker."""

import argparse
from logging import getLogger

from beaker import (
    Beaker,
    Constraints,
    DataMount,
    DataSource,
    EnvVar,
    ExperimentSpec,
    ImageSource,
    Priority,
    ResultSpec,
    TaskContext,
    TaskResources,
    TaskSpec,
)

from helios.inference_benchmarking import constants

logger = getLogger(__name__)


if __name__ == "__main__":
    b = Beaker.from_env(default_workspace=constants.BEAKER_WORKSPACE)
    account = b.account.whoami()

    parser = argparse.ArgumentParser()
    # make optional cluster key arg
    parser.add_argument(
        "--cluster_gpu_key",
        type=str,
        required=False,
        default=None,
        help=f"Cluster gpu to use for the benchmark, one of {constants.BEAKER_GPU_TO_CLUSTER_MAP.keys()}",
    )
    parser.add_argument(
        "--sweep_keys",
        type=str,
        required=False,
        default="batch,image,patch,model_size",
        help=f"Sweep keys to use for the benchmark, one of {constants.SWEEPS.keys()}",
    )
    # Parse all unknown args as overrides
    args, overrides = parser.parse_known_args()
    args = parser.parse_args()

    if args.cluster_gpu_key is None:
        raise ValueError("Cluster gpu key is required")

    ExperimentSpec(
        budget=constants.BEAKER_BUDGET,
        tasks=[
            TaskSpec(
                name="benchmark",
                replicas=1,
                context=TaskContext(
                    priority=Priority(constants.BEAKER_TASK_PRIORITY),
                    preemptible=True,
                ),
                datasets=[
                    DataMount(
                        mount_path=constants.ARTIFACTS_DIR,
                        source=DataSource(weka=constants.WEKA_BUCKET),
                    )
                ],
                image=ImageSource(beaker=constants.BEAKER_IMAGE_NAME),
                command=[
                    "python",
                    "helios/inference_benchmarking/run_throughput_benchmark.py",
                    args.sweep_keys,
                    *args.overrides,
                ],
                env_vars=[
                    EnvVar(
                        name=constants.PARAM_KEYS["project"],
                        value=constants.PROJECT_NAME,
                    ),
                    EnvVar(
                        name=constants.PARAM_KEYS["owner"],
                        value=constants.ENTITY_NAME,
                    ),
                    EnvVar(
                        name="WANDB_API_KEY", secret=f"{account.name}_WANDB_API_KEY"
                    ),
                ],
                resources=TaskResources(gpu_count=1),
                constraints=Constraints(
                    cluster=constants.BEAKER_GPU_TO_CLUSTER_MAP[args.cluster_gpu_key]
                ),
                result=ResultSpec(path="/noop-results"),
            )
        ],
    )
