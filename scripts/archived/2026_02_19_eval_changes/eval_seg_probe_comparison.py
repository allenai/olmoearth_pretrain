"""Evaluate the base model on segmentation tasks with LinearProbe vs InterpolateLinearProbe.

Usage:
    torchrun scripts/archived/2026_02_19_eval_changes/eval_seg_probe_comparison.py \
        evaluate eval-seg-probe-comparison local \
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

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.datasets.normalize import NormMethod
from olmoearth_pretrain.evals.linear_probe import ProbeType
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.flexi_vit import PoolingType
from olmoearth_pretrain.train.callbacks import (
    DownstreamEvaluatorCallbackConfig,
    OlmoEarthWandBCallback,
)
from olmoearth_pretrain.train.callbacks.evaluator_callback import DownstreamTaskConfig

logger = logging.getLogger(__name__)

PROBE_LR = 0.001
PROBE_EPOCHS = 50


def _seg_task(
    dataset: str,
    probe_type: ProbeType,
    embedding_batch_size: int = 128,
    probe_batch_size: int = 128,
    num_workers: int = 8,
    input_modalities: list[str] | None = None,
) -> DownstreamTaskConfig:
    """Helper to build a segmentation linear probe task config."""
    return DownstreamTaskConfig(
        dataset=dataset,
        embedding_batch_size=embedding_batch_size,
        probe_batch_size=probe_batch_size,
        num_workers=num_workers,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=PROBE_LR,
        epochs=PROBE_EPOCHS,
        eval_mode="LINEAR_PROBE",
        probe_type=probe_type,
        eval_interval=Duration.epochs(PROBE_EPOCHS),
        input_modalities=input_modalities or [],
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build trainer config that runs segmentation evals and exits."""
    checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)
    wandb_callback = OlmoEarthWandBCallback(
        name=common.run_name,
        project="eval_seg_probe_comparison",
        entity="eai-ai2",
        enabled=True,
        upload_dataset_distribution_pre_train=False,
        upload_modality_data_band_distribution_pre_train=False,
    )

    eval_tasks: dict[str, DownstreamTaskConfig] = {}
    for probe_type in [ProbeType.LINEAR, ProbeType.INTERPOLATE_LINEAR]:
        suffix = probe_type.value  # "linear" or "interpolate_linear"
        eval_tasks[f"mados_{suffix}"] = _seg_task(
            "mados",
            probe_type,
        )
        eval_tasks[f"sen1floods11_{suffix}"] = _seg_task(
            "sen1floods11",
            probe_type,
        )
        eval_tasks[f"m_cashew_plant_{suffix}"] = _seg_task(
            "m-cashew-plant",
            probe_type,
            embedding_batch_size=32,
            probe_batch_size=8,
            num_workers=2,
        )
        eval_tasks[f"m_sa_crop_type_{suffix}"] = _seg_task(
            "m-sa-crop-type",
            probe_type,
            embedding_batch_size=32,
            probe_batch_size=8,
            num_workers=2,
        )
        eval_tasks[f"pastis_s2_{suffix}"] = _seg_task(
            "pastis",
            probe_type,
            embedding_batch_size=32,
            probe_batch_size=8,
            num_workers=2,
            input_modalities=[Modality.SENTINEL2_L2A.name],
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
