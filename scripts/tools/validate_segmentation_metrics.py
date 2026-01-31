#!/usr/bin/env python3
"""Validate segmentation metrics by running LP and Finetune on PASTIS with the tiny model.

This script runs:
1. Linear Probe on pastis_sentinel2
2. Finetune on pastis_sentinel2

Both should return dict metrics with: miou, overall_acc, macro_acc, micro_f1, macro_f1

Usage:
    # Dry run (just print config):
    python scripts/tools/validate_segmentation_metrics.py dry_run validate_seg_metrics local

    # Run locally:
    python scripts/tools/validate_segmentation_metrics.py evaluate validate_seg_metrics local

    # Launch on Beaker:
    python scripts/tools/validate_segmentation_metrics.py launch validate_seg_metrics ai2/saturn-cirrascale
"""

import logging
import os

from olmo_core.train.callbacks import (
    BeakerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.common import Duration, LoadStrategy
from olmo_core.train.config import TrainerConfig

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.internal.constants import WANDB_ENTITY
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexi_vit import PoolingType
from olmoearth_pretrain.nn.flexihelios import EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.callbacks import (
    DownstreamEvaluatorCallbackConfig,
    OlmoEarthWandBCallback,
)
from olmoearth_pretrain.train.callbacks.evaluator_callback import (
    DownstreamTaskConfig,
    EvalMode,
)

logger = logging.getLogger(__name__)

# Default checkpoint path for tiny model
DEFAULT_CHECKPOINT = (
    "/weka/dfive-default/helios/checkpoints/joer/tiny_lr0.0002_wd0.02/step360000"
)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1


def build_common_components(
    script: str,
    cmd: str,
    run_name: str,
    cluster: str,
    overrides: list[str],
) -> CommonComponents:
    """Build common components."""
    from olmoearth_pretrain.internal.common import build_common_components as _build

    return _build(script, cmd, run_name, cluster, overrides)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for tiny model."""
    model_size = MODEL_SIZE_ARGS["tiny_shallow_decoder"]

    encoder_config = EncoderConfig(
        embedding_size=model_size["encoder_embedding_size"],
        num_heads=model_size["encoder_num_heads"],
        depth=model_size["encoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        supported_modality_names=common.training_modalities,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=12,
    )
    decoder_config = PredictorConfig(
        encoder_embedding_size=model_size["encoder_embedding_size"],
        decoder_embedding_size=model_size["decoder_embedding_size"],
        depth=model_size["decoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        num_heads=model_size["decoder_num_heads"],
        supported_modality_names=common.training_modalities,
        max_sequence_length=12,
    )
    return LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )


# Only PASTIS tasks for validation
PASTIS_LP_TASK = {
    "pastis_sentinel2": DownstreamTaskConfig(
        dataset="pastis",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=2,
        pooling_type=PoolingType.MAX,
        norm_stats_from_pretrained=True,
        probe_lr=0.1,
        eval_interval=Duration.epochs(10),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=20,  # Reduced for faster validation
        eval_mode=EvalMode.LINEAR_PROBE,
    ),
}

PASTIS_FT_TASK = {
    "pastis_sentinel2": DownstreamTaskConfig(
        dataset="pastis",
        ft_batch_size=16,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=10,  # Reduced for faster validation
        eval_mode=EvalMode.FINETUNE,
    ),
}


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build the trainer config for validation."""
    MAX_DURATION = Duration.epochs(300)
    LOAD_STRATEGY = LoadStrategy.if_available
    checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)

    wandb_callback = OlmoEarthWandBCallback(
        name=common.run_name,
        project="2025_01_30_validate_seg_metrics",
        entity=WANDB_ENTITY,
        enabled=True,
        upload_dataset_distribution_pre_train=False,
        upload_modality_data_band_distribution_pre_train=False,
    )
    garbage_collector_callback = GarbageCollectorCallback(gc_interval=1)

    # Choose LP or FT based on environment variable
    use_finetune = os.environ.get("FINETUNE", "0") == "1"
    tasks = PASTIS_FT_TASK if use_finetune else PASTIS_LP_TASK
    logger.info(f"Using {'FINETUNE' if use_finetune else 'LINEAR_PROBE'} mode")

    trainer_config = (
        TrainerConfig(
            work_dir=common.save_folder,
            load_path=DEFAULT_CHECKPOINT,
            load_strategy=LOAD_STRATEGY,
            save_folder=common.save_folder,
            cancel_check_interval=1,
            metrics_collect_interval=10,
            max_duration=MAX_DURATION,
            checkpointer=checkpointer_config,
        )
        .with_callback("wandb", wandb_callback)
        .with_callback("gpu_memory_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=tasks,
                eval_on_startup=True,
                cancel_after_first_eval=True,
                run_on_test=True,
            ),
        )
        .with_callback("garbage_collector", garbage_collector_callback)
        .with_callback("beaker", BeakerCallback())
    )
    return trainer_config


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        trainer_config_builder=build_trainer_config,
        train_module_config_builder=None,
    )
