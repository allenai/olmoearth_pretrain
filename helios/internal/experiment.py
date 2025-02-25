"""Code for configuring and running Helios experiments."""

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from olmo_core.config import Config, StrEnum
from olmo_core.train import (
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import ConfigSaverCallback, WandBCallback
from olmo_core.utils import get_default_device, prepare_cli_environment, seed_all

from helios.data.constants import ModalitySpec
from helios.data.dataloader import HeliosDataLoaderConfig
from helios.data.dataset import HeliosDatasetConfig, collate_helios
from helios.data.normalize import Normalizer, Strategy
from helios.data.visualize import visualize_sample
from helios.nn.latent_mim import LatentMIMConfig
from helios.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

# TODO: Make this more agnostic to the training setup
# TODO: Add support for different model configs
# TODO: Add support for different train module configs
# TODO: Add support for overrides


@dataclass
class CommonComponents(Config):
    """Any configurable items that are common to all experiments."""

    run_name: str
    save_folder: str
    supported_modalities: list[ModalitySpec]
    # callbacks: dict[str, Callback]


@dataclass
class HeliosVisualizeConfig(Config):
    """Configuration for visualizing the dataset."""

    output_dir: str
    num_samples: int = 10
    normalize_strategy: Strategy = Strategy.PREDEFINED
    std_multiplier: float = 2.0


@dataclass
class HeliosExperimentConfig(Config):
    """Configuration for a Helios experiment."""

    run_name: str
    # launch: BeakerLaunchConfig # we should use this as well
    model: LatentMIMConfig  # TODO: make this agnostic to training setup
    dataset: HeliosDatasetConfig  # will likely be fixed for us
    data_loader: HeliosDataLoaderConfig  # will likely be fixed for us
    train_module: LatentMIMTrainModuleConfig  # we will want to support different train module model combinations
    trainer: TrainerConfig
    visualize_config: HeliosVisualizeConfig | None = None
    init_seed: int = 12536


def build_config(
    common: CommonComponents,
    model_config_builder: Callable[[CommonComponents], LatentMIMConfig],
    dataset_config_builder: Callable[[CommonComponents], HeliosDatasetConfig],
    dataloader_config_builder: Callable[[CommonComponents], HeliosDataLoaderConfig],
    trainer_config_builder: Callable[[CommonComponents], TrainerConfig],
    train_module_config_builder: Callable[
        [CommonComponents],
        LatentMIMTrainModuleConfig,
    ],
    overrides: list[str],
    visualize_config_builder: (
        Callable[[CommonComponents], HeliosVisualizeConfig] | None
    ) = None,
) -> HeliosExperimentConfig:
    """Build a Helios experiment configuration."""
    model_config = model_config_builder(common)
    dataset_config = dataset_config_builder(common)
    dataloader_config = dataloader_config_builder(common)
    trainer_config = trainer_config_builder(common)
    train_module_config = train_module_config_builder(common)
    visualize_config = (
        visualize_config_builder(common) if visualize_config_builder else None
    )
    config = HeliosExperimentConfig(
        run_name=common.run_name,
        model=model_config,
        dataset=dataset_config,
        data_loader=dataloader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        visualize_config=visualize_config,
    )
    logger.info("Config: %s", config)
    logger.info("Overrides: %s", overrides)
    if len(overrides) > 0:
        # also there are some unerlying dataclasses we don't want to be able to overide and are not actually part of config
        # OmegaConf merge does not support all types we use so we would have to do some mods to make this work https://omegaconf.readthedocs.io/en/2.2_branch/structured_config.html
        config = config.merge(overrides)
    return config


def train(config: HeliosExperimentConfig) -> None:
    """Train an experiment."""
    # Set RNG states on all devices. Also, done in prepare_training_environment
    seed_all(config.init_seed)

    # Build components.
    # TODO: Setup init device arg and allow the model to be inited on device of our choice rather than moved over
    model = config.model.build()
    device = get_default_device()
    model = model.to(device)
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    # TODO: akward harcoding of the collator here
    data_loader = config.data_loader.build(
        dataset, collator=collate_helios, dp_process_group=train_module.dp_process_group
    )
    trainer = config.trainer.build(train_module, data_loader)

    # Record the config to W&B/Comet and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    trainer.fit()


def visualize(config: HeliosExperimentConfig) -> None:
    """Visualize the dataset for an experiment."""
    logger.info("Visualizing the dataset")
    if config.visualize_config is None:
        raise ValueError("visualize_config is not set")
    dataset = config.dataset.build()
    sample_indices = np.random.randint(
        0, len(dataset), config.visualize_config.num_samples
    )
    normalizer = Normalizer(
        strategy=config.visualize_config.normalize_strategy,
        std_multiplier=config.visualize_config.std_multiplier,
    )
    for sample_index in sample_indices:
        visualize_sample(
            dataset, sample_index, normalizer, config.visualize_config.output_dir
        )
    logger.info("Done visualizing the dataset")


class SubCmd(StrEnum):
    """Subcommands for Helios experiments.

    modeled after olmo-core experiment.py potentially olmo-core might support this directly
    """

    launch = "launch"
    train = "train"
    train_single = "train_single"
    prep = "prep"
    launch_prep = "launch_prep"
    dry_run = "dry_run"
    visualize = "visualize"

    def prepare_environment(self) -> None:
        """Prepare the environment for the given subcommand."""
        if self in (
            SubCmd.launch,
            SubCmd.dry_run,
            SubCmd.prep,
            SubCmd.launch_prep,
            SubCmd.visualize,
        ):
            prepare_cli_environment()
        elif self == SubCmd.train:
            prepare_training_environment()
        elif self == SubCmd.train_single:
            prepare_training_environment(backend=None)
        else:
            raise NotImplementedError(self)

    def run(self, config: HeliosExperimentConfig) -> None:
        """Run the given subcommand."""
        # if get_local_rank() == 0:
        #     print(config)
        #     print(
        #         "\n"
        #         f"[b blue]Total parameters:[/]                {config.model.num_params:,d}\n"
        #         f"[b blue]Non-embedding parameters:[/]        {config.model.num_non_embedding_params:,d}"
        #     )

        if self == SubCmd.launch:
            raise NotImplementedError
        elif self == SubCmd.dry_run:
            pass
        elif self == SubCmd.visualize:
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
            raise NotImplementedError
        elif self == SubCmd.launch_prep:
            raise NotImplementedError
        else:
            raise NotImplementedError(self)


def main(
    *,
    common_components_builder: Callable[[], CommonComponents],
    model_config_builder: Callable[[CommonComponents], LatentMIMConfig],
    dataset_config_builder: Callable[[CommonComponents], HeliosDatasetConfig],
    dataloader_config_builder: Callable[[CommonComponents], HeliosDataLoaderConfig],
    trainer_config_builder: Callable[[CommonComponents], TrainerConfig],
    train_module_config_builder: Callable[
        [CommonComponents], LatentMIMTrainModuleConfig
    ],
    visualize_config_builder: (
        Callable[[CommonComponents], HeliosVisualizeConfig] | None
    ) = None,
) -> None:
    """Main entry point for Helios experiments.

    overrides:  A list of field attributes with dot notation, e.g. ``foo.bar=1``.

    """
    usage = f"""
[yellow]Usage:[/] [i blue]python[/] [i cyan]{sys.argv[0]}[/] [i b magenta]{'|'.join(SubCmd)}[/] [i b]RUN_NAME CLUSTER[/] [i][OVERRIDES...][/]

[b]Subcommands[/]
[b magenta]launch:[/]     Not Implemented. Launch the script on Beaker with the [b magenta]train[/] subcommand.
[b magenta]train:[/]       Run the trainer. You usually shouldn't invoke the script with this subcommand directly.
             Instead use [b magenta]launch[/] or run it with torchrun.
[b magenta]train_single:[/]       Run the trainer on a single device (GPU, CPU, MPS). num_nodes is ignored.
[b magenta]prep:[/]       Not Implemented. Prepare the dataset ahead of training to save GPU time.
[b magenta]launch_prep:[/] Not Implemented. Launch the script on Beaker with the [b magenta]prep[/] subcommand.
[b magenta]dry_run:[/]     Pretty print the config and exit.
[b magenta]visualize:[/]   Visualize the dataset.

[b]Examples[/]
    # Train on 4 GPUs across 2 nodes
    torchrun train.py train
    # Visualize the dataset
    python train.py visualize
    """.strip()

    if len(sys.argv) < 2 or sys.argv[1] not in set(SubCmd):
        import rich

        rich.get_console().print(usage, highlight=False)
        sys.exit(1)

    script_name, cmd, *overrides = sys.argv
    common = common_components_builder()

    cmd = SubCmd(cmd)
    cmd.prepare_environment()
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
