"""40K step 1/4 learning rate."""

from latent_mim_128 import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_model_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)
from olmo_core.optim.scheduler import LinearWithWarmup, SequentialScheduler

from helios.internal.experiment import CommonComponents, main
from helios.train.train_module.latent_mim import LatentMIMTrainModuleConfig


def my_build_train_module_config(
    common: CommonComponents,
) -> LatentMIMTrainModuleConfig:
    """Build the train module config for an experiment."""
    train_module_config = build_train_module_config(common)
    train_module_config.scheduler = SequentialScheduler(
        schedulers=[
            LinearWithWarmup(alpha_f=0.25, warmup_steps=2000),
            LinearWithWarmup(alpha_f=0.25, warmup_steps=0),
            LinearWithWarmup(alpha_f=0.25, warmup_steps=0),
            LinearWithWarmup(alpha_f=0.25, warmup_steps=0),
            LinearWithWarmup(alpha_f=0.25, warmup_steps=0),
        ],
        schedulers_max_steps=[40000, 40000, 40000, 40000],
    )
    return train_module_config


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=my_build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )
