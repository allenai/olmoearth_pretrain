"""Script to run locally to launch throughput benchmarking task in Beaker."""

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
from helios.inference_benchmarking.data_models import RunParams

# Change this to control your sweep
configs = [
    RunParams(
        model_size="LARGE",
        use_s1=True,
        use_s2=True,
        use_landsat=True,
        image_size=image_size,
        patch_size=1,
        num_timesteps=1,
        batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        gpu_type="A100",
        bf16=True,
    )
    for image_size in [64]
    # for image_size in [1, 8, 16, 32, 64, 128]
]


if __name__ == "__main__":
    b = Beaker.from_env(default_workspace=constants.BEAKER_WORKSPACE)
    account = b.account.whoami()

    experiment_specs = [
        ExperimentSpec(
            budget=constants.BEAKER_BUDGET,
            description=config.run_name,
            tasks=[
                TaskSpec(
                    name=config.run_name,
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
                    ],
                    env_vars=[
                        EnvVar(name=key, value=value)
                        for key, value in config.to_env_vars().items()
                    ]
                    + [
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
                        cluster=constants.BEAKER_GPU_TO_CLUSTER_MAP[config.gpu_type]
                    ),
                    result=ResultSpec(path="/noop-results"),
                )
            ],
        )
        for config in configs
    ]

    for experiment_spec in experiment_specs:
        experiment_data = b.experiment.create(
            spec=experiment_spec, workspace=constants.BEAKER_WORKSPACE
        )
        print(f"Experiment URL: https://beaker.org/ex/{experiment_data.id}")
