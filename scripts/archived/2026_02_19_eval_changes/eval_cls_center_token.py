"""Evaluate the base model on classification tasks: baseline vs use_center_token.

Usage:
    torchrun scripts/archived/2026_02_19_eval_changes/eval_cls_center_token.py \
        evaluate eval-center-token-comparison local \
        --trainer.load_path=CHECKPOINT_DIR

The load_path should point to the directory containing the base model checkpoint.
"""

import logging

from base import build_model_config
from olmo_core.train.callbacks import (
    BeakerCallback,
    CheckpointerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.common import Duration, LoadStrategy
from olmo_core.train.config import TrainerConfig
from script import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_visualize_config,
)

from olmoearth_pretrain.evals.datasets.normalize import NormMethod
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.flexi_vit import PoolingType
from olmoearth_pretrain.train.callbacks import (
    DownstreamEvaluatorCallbackConfig,
    OlmoEarthWandBCallback,
)
from olmoearth_pretrain.train.callbacks.evaluator_callback import DownstreamTaskConfig

logger = logging.getLogger(__name__)


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build trainer config that runs classification evals and exits."""
    checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)
    wandb_callback = OlmoEarthWandBCallback(
        name=common.run_name,
        project="eval_cls_center_token",
        entity="eai-ai2",
        enabled=True,
        upload_dataset_distribution_pre_train=False,
        upload_modality_data_band_distribution_pre_train=False,
    )

    cls_datasets = [
        ("m_eurosat", "m-eurosat", 128, 8),
        ("m_so2sat", "m-so2sat", 128, 8),
        ("m_brick_kiln", "m-brick-kiln", 128, 8),
        ("m_bigearthnet", "m-bigearthnet", 64, 8),
    ]

    eval_tasks: dict[str, DownstreamTaskConfig] = {}
    for use_center in [False, True]:
        suffix = "center" if use_center else "baseline"
        for name, dataset, batch_size, workers in cls_datasets:
            eval_tasks[f"{name}_{suffix}"] = DownstreamTaskConfig(
                dataset=dataset,
                embedding_batch_size=batch_size,
                num_workers=workers,
                pooling_type=PoolingType.MEAN,
                norm_stats_from_pretrained=True,
                norm_method=NormMethod.NORM_NO_CLIP_2_STD,
                use_center_token=use_center,
                eval_interval=Duration.epochs(1),
            )

    trainer_config = (
        TrainerConfig(
            work_dir=common.save_folder,
            load_strategy=LoadStrategy.if_available,
            save_folder=common.save_folder,
            cancel_check_interval=1,
            metrics_collect_interval=10,
            max_duration=Duration.epochs(9999),
            checkpointer=checkpointer_config,
        )
        .with_callback("wandb", wandb_callback)
        .with_callback("gpu_memory_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=eval_tasks,
                eval_on_startup=True,
                cancel_after_first_eval=True,
                run_on_test=True,
            ),
        )
        .with_callback("garbage_collector", GarbageCollectorCallback(gc_interval=1))
        .with_callback("beaker", BeakerCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=5000,
                ephemeral_save_interval=250,
            ),
        )
    )
    return trainer_config


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )
