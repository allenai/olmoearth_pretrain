"""Trying to prototype fitting everything into olmo core."""

from olmo_core.config import DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from shared import (
    MAX_PATCH_SIZE,
    MIN_PATCH_SIZE,
    NUM_DATA_LOADER_WORKERS,
    build_dataset_config,
    build_model_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)

from helios.data.dataloader import HeliosDataLoaderConfig
from helios.internal.common import build_common_components
from helios.internal.experiment import CommonComponents, main
from helios.train.masking import MaskingConfig
from helios.train.train_module.latent_mim import LatentMIMTrainModuleConfig


def build_dataloader_config(common: CommonComponents) -> HeliosDataLoaderConfig:
    """Build the dataloader config for an experiment."""
    # things should be set during building
    # TODO: Include collate function here

    GLOBAL_BATCH_SIZE = 512
    TOKEN_BUDGET = 6000
    SAMPLE_HW_P_LIST = list(range(5, 20))

    dataloader_config = HeliosDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=3622,
        work_dir=common.save_folder,
        num_workers=NUM_DATA_LOADER_WORKERS,
        sampled_hw_p_list=SAMPLE_HW_P_LIST,
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        token_budget=TOKEN_BUDGET,
    )
    return dataloader_config


def my_build_train_module_config(
    common: CommonComponents,
) -> LatentMIMTrainModuleConfig:
    """Get updated train module with randspace masking."""
    train_module_config = build_train_module_config(common)
    train_module_config.masking_config = MaskingConfig(
        strategy_config={
            "type": "random_space",
            "encode_ratio": 0.3,
            "decode_ratio": 0.65,
        }
    )
    train_module_config.dp_config = DataParallelConfig(
        name=DataParallelType.fsdp,
        param_dtype=DType.bfloat16,
        reduce_dtype=DType.float32,
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
