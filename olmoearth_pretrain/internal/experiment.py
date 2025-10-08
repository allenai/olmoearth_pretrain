"""Code for configuring and running OlmoEarth Pretrain experiments."""

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from beaker import ExperimentSpec
from olmo_core.config import Config, StrEnum
from olmo_core.distributed.utils import get_local_rank
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.train import (
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import ConfigSaverCallback, WandBCallback
from olmo_core.utils import get_default_device, prepare_cli_environment, seed_all

from olmoearth_pretrain._compat import deprecated_class_alias as _deprecated_class_alias
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.data.dataset import (
    OlmoEarthDatasetConfig,
    collate_olmoearth_pretrain,
)
from olmoearth_pretrain.data.visualize import visualize_sample
from olmoearth_pretrain.inference_benchmarking.run_throughput_benchmark import (
    ThroughputBenchmarkRunnerConfig,
)
from olmoearth_pretrain.train.train_module.train_module import (
    OlmoEarthTrainModuleConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class OlmoEarthBeakerLaunchConfig(BeakerLaunchConfig):
    """Extend BeakerLaunchConfig with hostnames option.

    This enables targeting specific Beaker hosts.
    """

    hostnames: list[str] | None = None

    def build_experiment_spec(
        self, torchrun: bool = True, entrypoint: str | None = None
    ) -> ExperimentSpec:
        """Build the experiment spec."""
        # We simply call the superclass build_experiment_spec, but just replace cluster
        # setting in the Constraints with hostname setting if user provided hostname
        # list.
        spec = super().build_experiment_spec(torchrun, entrypoint)
        if self.hostnames:
            constraints = spec.tasks[0].constraints
            constraints.cluster = None
            constraints.hostname = self.hostnames
        return spec


HeliosBeakerLaunchConfig = _deprecated_class_alias(
    OlmoEarthBeakerLaunchConfig,
    "helios.internal.experiment.HeliosBeakerLaunchConfig",
)


@dataclass
class CommonComponents(Config):
    """Any configurable items that are common to all experiments."""

    run_name: str
    save_folder: str
    launch: OlmoEarthBeakerLaunchConfig
    training_modalities: list[str]
    nccl_debug: bool = False
    # callbacks: dict[str, Callback]

    def validate(self) -> None:
        """Validate the common components."""
        if not isinstance(self.training_modalities, list):
            raise ValueError("training_modalities must be a list")
        if not all(
            modality in Modality.names() for modality in self.training_modalities
        ):
            raise ValueError(
                "training_modalities must contain only valid modality names"
            )


@dataclass
class OlmoEarthVisualizeConfig(Config):
    """Configuration for visualizing the dataset."""

    output_dir: str
    num_samples: int | None = None
    global_step: int | None = None
    std_multiplier: float = 2.0


HeliosVisualizeConfig = _deprecated_class_alias(
    OlmoEarthVisualizeConfig,
    "helios.internal.experiment.HeliosVisualizeConfig",
)


@dataclass
class OlmoEarthExperimentConfig(Config):
    """Configuration for a OlmoEarth Pretrain experiment."""

    run_name: str
    launch: OlmoEarthBeakerLaunchConfig
    model: Config
    dataset: Config  # will likely be fixed for us
    data_loader: OlmoEarthDataLoaderConfig  # will likely be fixed for us
    train_module: OlmoEarthTrainModuleConfig
    trainer: TrainerConfig
    visualize: OlmoEarthVisualizeConfig | None = None
    init_seed: int = 12536


HeliosExperimentConfig = _deprecated_class_alias(
    OlmoEarthExperimentConfig,
    "helios.internal.experiment.HeliosExperimentConfig",
)


@dataclass
class BenchmarkExperimentConfig(Config):
    """Configuration for a throughput benchmarking run."""

    launch: OlmoEarthBeakerLaunchConfig
    benchmark: ThroughputBenchmarkRunnerConfig


def split_common_overrides(overrides: list[str]) -> tuple[list[str], list[str]]:
    """Split the common overrides from the command line."""
    common_overrides = [
        dotfield.replace("common.", "")
        for dotfield in overrides
        if "common." in dotfield
    ]
    non_common_overrides = [
        dotfield for dotfield in overrides if "common." not in dotfield
    ]
    return common_overrides, non_common_overrides


def build_config(
    common: CommonComponents,
    model_config_builder: Callable[[CommonComponents], Config],
    dataset_config_builder: Callable[[CommonComponents], OlmoEarthDatasetConfig],
    dataloader_config_builder: Callable[[CommonComponents], OlmoEarthDataLoaderConfig],
    trainer_config_builder: Callable[[CommonComponents], TrainerConfig],
    train_module_config_builder: Callable[
        [CommonComponents],
        OlmoEarthTrainModuleConfig,
    ],
    overrides: list[str],
    visualize_config_builder: (
        Callable[[CommonComponents], OlmoEarthVisualizeConfig] | None
    ) = None,
) -> OlmoEarthExperimentConfig:
    """Build a OlmoEarth Pretrain experiment configuration."""
    # Overide common components
    common_overrides, overrides = split_common_overrides(overrides)
    logger.info("Common overrides: %s", common_overrides)
    common = common.merge(common_overrides)
    logger.info("Common: %s", common)
    model_config = model_config_builder(common)
    dataset_config = dataset_config_builder(common)
    dataloader_config = dataloader_config_builder(common)
    trainer_config = trainer_config_builder(common)
    train_module_config = train_module_config_builder(common)
    visualize_config = (
        visualize_config_builder(common) if visualize_config_builder else None
    )
    config = OlmoEarthExperimentConfig(
        run_name=common.run_name,
        model=model_config,
        dataset=dataset_config,
        data_loader=dataloader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        visualize=visualize_config,
        launch=common.launch,
    )
    logger.info("Overrides: %s", overrides)
    config = config.merge(overrides)
    return config


def build_benchmark_config(
    common: CommonComponents,
    inference_benchmarking_config_builder: Callable[
        [], ThroughputBenchmarkRunnerConfig
    ],
    overrides: list[str],
) -> BenchmarkExperimentConfig:
    """Build a throughput benchmarking configuration."""
    inference_benchmarking_config = inference_benchmarking_config_builder()
    config = BenchmarkExperimentConfig(
        launch=common.launch,
        benchmark=inference_benchmarking_config,
    )
    config = config.merge(overrides)
    logger.info("Benchmark config: %s", config)
    return config


def benchmark(config: BenchmarkExperimentConfig) -> None:
    """Benchmark an experiment."""
    runner = config.benchmark.build()
    runner.run()


def launch_benchmark(config: BenchmarkExperimentConfig) -> None:
    """Launch a throughput benchmarking run."""
    config.launch.launch(follow=False, torchrun=False)


def train(config: OlmoEarthExperimentConfig) -> None:
    """Train an experiment."""
    # Set RNG states on all devices. Also, done in prepare_training_environment
    seed_all(config.init_seed)

    # Build components.
    # TODO: Setup init device arg and allow the model to be inited on device of our choice rather than moved over allowing for meta
    model = config.model.build()
    device = get_default_device()
    model = model.to(device)
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    # TODO: akward harcoding of the collator here
    data_loader = config.data_loader.build(
        dataset,
        collator=collate_olmoearth_pretrain,
        dp_process_group=train_module.dp_process_group,
    )
    trainer = config.trainer.build(train_module, data_loader)

    # Record the config to W&B/Comet and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict
    trainer.fit()


def visualize(config: OlmoEarthExperimentConfig) -> None:
    """Visualize the dataset for an experiment."""
    logger.info("Visualizing the dataset")
    if config.visualize is None:
        raise ValueError("visualize_config is not set")
    global_step = config.visualize.global_step
    dataset = config.dataset.build()
    if global_step is not None:
        data_loader = config.data_loader.build(
            dataset, collator=collate_olmoearth_pretrain, dp_process_group=None
        )
        sample_indices = data_loader.fast_forward(global_step)
    else:
        sample_indices = np.random.randint(
            0, len(dataset), config.visualize.num_samples
        )
    logger.info(f"sample indices: {sample_indices}")
    for sample_index in sample_indices:
        visualize_sample(dataset, sample_index, config.visualize.output_dir)
    logger.info("Done visualizing the dataset")


def launch(config: OlmoEarthExperimentConfig) -> None:
    """Launch an experiment."""
    logger.info("Launching the experiment")
    logger.info(config)
    # Set follow=False if you don't want to stream the logs to the terminal
    config.launch.launch(follow=False)


def prep(config: OlmoEarthExperimentConfig) -> None:
    """Prepare the dataset for an experiment."""
    dataset = config.dataset.build()
    # TODO: akward harcoding of the collator here
    data_loader = config.data_loader.build(
        dataset, collator=collate_olmoearth_pretrain, dp_process_group=None
    )
    data_loader.reshuffle(epoch=1)
    # Also may want to create the first index of shuffling here for starters


def launch_prep(config: OlmoEarthExperimentConfig) -> None:
    """Launch the preparation of the dataset for an experiment."""
    assert config.launch is not None
    config.launch.num_gpus = 0
    config.launch.num_nodes = 1
    logger.info(config)
    logger.info("Launching the preparation of the dataset...")
    logger.info(
        "Follow along until the dataset is prepared and saved to Weka then stop the script"
    )
    config.launch.launch(follow=True, torchrun=False)


class SubCmd(StrEnum):
    """Subcommands for OlmoEarth Pretrain experiments.

    modeled after olmo-core experiment.py potentially olmo-core might support this directly
    """

    launch = "launch"
    train = "train"
    train_single = "train_single"
    prep = "prep"
    launch_prep = "launch_prep"
    dry_run = "dry_run"
    visualize = "visualize"
    benchmark = "benchmark"
    launch_benchmark = "launch_benchmark"

    def prepare_environment(self) -> None:
        """Prepare the environment for the given subcommand."""
        if self in (
            SubCmd.launch,
            SubCmd.dry_run,
            SubCmd.prep,
            SubCmd.launch_prep,
            SubCmd.visualize,
            SubCmd.benchmark,
            SubCmd.launch_benchmark,
        ):
            prepare_cli_environment()
        elif self == SubCmd.train:
            prepare_training_environment()
        elif self == SubCmd.train_single:
            prepare_training_environment(backend=None)
        else:
            raise NotImplementedError(self)

    def run(
        self,
        config: OlmoEarthExperimentConfig | BenchmarkExperimentConfig,
    ) -> None:
        """Run the given subcommand."""
        if get_local_rank() == 0:
            print(config)
            # TODO: Add parameter count math to config
            # print(
            #     "\n"
            #     f"[b blue]Total parameters:[/]                {config.model.num_params:,d}\n"
            #     f"[b blue]Non-embedding parameters:[/]        {config.model.num_non_embedding_params:,d}"
            # )

        if self == SubCmd.launch:
            launch(config)
        elif self == SubCmd.dry_run:
            logger.info(config)
        elif self == SubCmd.visualize:
            seed_all(config.init_seed)
            visualize(config)
        elif self == SubCmd.train:
            try:
                train(config)
            finally:
                teardown_training_environment()
        elif self == SubCmd.train_single:
            if config.train_module.dp_config is not None:
                logger.warning(
                    "'dp_config' is set to %s, but you can't use data parallelism when running on a single node. Disabling.",
                    config.train_module.dp_config,
                )
                config.train_module.dp_config = None
            try:
                train(config)
            finally:
                teardown_training_environment()
        elif self == SubCmd.prep:
            prep(config)
        elif self == SubCmd.launch_prep:
            launch_prep(config)
        elif self == SubCmd.benchmark:
            benchmark(config)
        elif self == SubCmd.launch_benchmark:
            launch_benchmark(config)
        else:
            raise NotImplementedError(self)


def main(
    *,
    common_components_builder: Callable,
    model_config_builder: Callable[[CommonComponents], Config] | None = None,
    dataset_config_builder: Callable[[CommonComponents], Config] | None = None,
    dataloader_config_builder: (
        Callable[[CommonComponents], OlmoEarthDataLoaderConfig] | None
    ) = None,
    trainer_config_builder: Callable[[CommonComponents], TrainerConfig] | None = None,
    train_module_config_builder: (
        Callable[[CommonComponents], OlmoEarthTrainModuleConfig] | None
    ) = None,
    visualize_config_builder: (
        Callable[[CommonComponents], OlmoEarthVisualizeConfig] | None
    ) = None,
    inference_benchmarking_config_builder: (
        Callable[[], ThroughputBenchmarkRunnerConfig] | None
    ) = None,
) -> None:
    """Main entry point for OlmoEarth Pretrain experiments.

    overrides:  A list of field attributes with dot notation, e.g. ``foo.bar=1``.

    """
    usage = f"""
[yellow]Usage:[/] [i blue]python[/] [i cyan]{sys.argv[0]}[/] [i b magenta]{"|".join(SubCmd)}[/] [i b]RUN_NAME CLUSTER[/] [i][OVERRIDES...][/]
If running command on a local machine ie from a session, you can use the [b]local[/b] cluster name.
[b]Subcommands[/]
[b magenta]launch:[/]     Not Implemented. Launch the script on Beaker with the [b magenta]train[/] subcommand.
[b magenta]train:[/]       Run the trainer. You usually shouldn't invoke the script with this subcommand directly.
             Instead use [b magenta]launch[/] or run it with torchrun.
[b magenta]train_single:[/]       Run the trainer on a single device (GPU, CPU, MPS). num_nodes is ignored.
[b magenta]prep:[/]        Prepare the dataset ahead of training to save  to Weka.
[b magenta]launch_prep:[/] Launch the script on Beaker with the [b magenta]prep[/] subcommand.
[b magenta]dry_run:[/]     Pretty print the config and exit.
[b magenta]visualize:[/]   Visualize the dataset.

[b]Examples[/]
    # Train on 4 GPUs across 2 nodes
    torchrun train.py train
    # Visualize the dataset
    python train.py visualize
    """.strip()
    logger.info(f"Running {sys.argv}")
    if len(sys.argv) < 4 or sys.argv[1] not in set(SubCmd):
        import rich

        rich.get_console().print(usage, highlight=False)
        sys.exit(1)

    script, cmd, run_name, cluster, *overrides = sys.argv
    # TODO: we should probably have a single common components builder that can be used for all experiments
    common = common_components_builder(script, cmd, run_name, cluster, overrides)

    cmd = SubCmd(cmd)
    cmd.prepare_environment()

    if cmd == SubCmd.benchmark or cmd == SubCmd.launch_benchmark:
        if inference_benchmarking_config_builder is None:
            raise ValueError("inference_benchmarking_config_builder is not set")
        config = build_benchmark_config(
            common=common,
            inference_benchmarking_config_builder=inference_benchmarking_config_builder,
            overrides=overrides,
        )
    else:
        assert model_config_builder is not None
        assert dataset_config_builder is not None
        assert dataloader_config_builder is not None
        assert trainer_config_builder is not None
        assert train_module_config_builder is not None
        config = build_config(
            common=common,
            model_config_builder=model_config_builder,
            dataset_config_builder=dataset_config_builder,
            dataloader_config_builder=dataloader_config_builder,
            trainer_config_builder=trainer_config_builder,
            train_module_config_builder=train_module_config_builder,
            visualize_config_builder=visualize_config_builder,
            overrides=overrides,
        )

    cmd.run(config)
