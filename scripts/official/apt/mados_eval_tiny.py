"""MADOS finetuning evaluation with tiny model (no APT baseline).

This script loads a pretrained tiny model and evaluates on MADOS
using standard uniform patching (no APT).

Usage:
    # Local evaluation with the tiny checkpoint
    python scripts/official/apt/mados_eval_tiny.py evaluate \
        mados_eval_tiny local \
        --trainer.load_path=/weka/dfive-default/helios/checkpoints/joer/tiny_lr0.0002_wd0.02/step360000

    # Override patch size
    python scripts/official/apt/mados_eval_tiny.py evaluate \
        mados_eval_tiny local \
        --trainer.load_path=/path/to/checkpoint \
        --trainer.callbacks.downstream_evaluator.tasks.mados_finetune.patch_size=8
"""

import logging

from olmo_core.config import DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import CosWithWarmup
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.common import Duration, LoadStrategy
from olmo_core.train.config import TrainerConfig

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.internal.common import (
    build_common_components as build_common_components_default,
)
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
    OlmoEarthVisualizeConfig,
    SubCmd,
    main,
)
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexi_vit import (
    EncoderConfig,
    PoolingType,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.callbacks import (
    DownstreamEvaluatorCallbackConfig,
    OlmoEarthWandBCallback,
)
from olmoearth_pretrain.train.callbacks.evaluator_callback import (
    DownstreamTaskConfig,
    EvalMode,
)
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1


def build_common_components(
    script: str,
    cmd: SubCmd,
    run_name: str,
    cluster: str,
    overrides: list[str],
) -> CommonComponents:
    """Build the common components for an experiment."""
    config = build_common_components_default(script, cmd, run_name, cluster, overrides)

    config.training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.LANDSAT.name,
        Modality.WORLDCOVER.name,
        Modality.SRTM.name,
        Modality.OPENSTREETMAP_RASTER.name,
        Modality.WRI_CANOPY_HEIGHT_MAP.name,
        Modality.CDL.name,
        Modality.WORLDCEREAL.name,
    ]

    return config


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config - using TINY model size."""
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


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build the train module config."""
    return ContrastiveLatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=1e-6, weight_decay=0.02, fused=False),
        rank_microbatch_size=32,
        masking_config=MaskingConfig(
            strategy_config={
                "type": "random",
                "encode_ratio": 0.5,
                "decode_ratio": 0.5,
            }
        ),
        loss_config=LossConfig(
            loss_config={
                "type": "modality_patch_discrimination_new",
                "tau": 0.1,
            }
        ),
        contrastive_config=LossConfig(
            loss_config={
                "type": "InfoNCE",
                "weight": 0.1,
            }
        ),
        token_exit_cfg={modality: 0 for modality in common.training_modalities},
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=100),
        ema_decay=(1.0, 1.0),
        dp_config=DataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Build the dataloader config."""
    return OlmoEarthDataLoaderConfig(
        num_workers=4,
        global_batch_size=32,
        token_budget=2250,
        prefetch_factor=2,
        sampled_hw_p_list=list(range(1, 13)),
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        work_dir=common.save_folder,
        seed=3622,
    )


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Build the dataset config."""
    return OlmoEarthDatasetConfig(
        h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828",
        training_modalities=common.training_modalities,
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build the trainer config for evaluation."""
    MAX_DURATION = Duration.steps(1000000)
    METRICS_COLLECT_INTERVAL = 5
    CANCEL_CHECK_INTERVAL = 1
    LOAD_STRATEGY = LoadStrategy.always

    WANDB_USERNAME = "eai-ai2"
    WANDB_PROJECT = "01_2026_apt_investigation"

    checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)
    wandb_callback = OlmoEarthWandBCallback(
        name=common.run_name,
        project=WANDB_PROJECT,
        entity=WANDB_USERNAME,
        enabled=True,
    )
    garbage_collector_callback = GarbageCollectorCallback(gc_interval=1)

    EVAL_TASKS = {
        "mados_finetune": DownstreamTaskConfig(
            dataset="mados",
            embedding_batch_size=128,
            num_workers=0,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_mode=EvalMode.FINETUNE,
            ft_lr=1e-4,
            ft_batch_size=32,
            epochs=50,
            eval_interval=Duration.steps(1),
            patch_size=4,
            use_apt=False,
        ),
    }

    trainer_config = (
        TrainerConfig(
            work_dir=common.save_folder,
            load_strategy=LOAD_STRATEGY,
            save_folder=common.save_folder,
            cancel_check_interval=CANCEL_CHECK_INTERVAL,
            metrics_collect_interval=METRICS_COLLECT_INTERVAL,
            max_duration=MAX_DURATION,
            checkpointer=checkpointer_config,
        )
        .with_callback("wandb", wandb_callback)
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=EVAL_TASKS,
                eval_on_startup=True,
                cancel_after_first_eval=True,
            ),
        )
        .with_callback("garbage_collector", garbage_collector_callback)
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=10000,
                ephemeral_save_interval=250,
            ),
        )
    )
    return trainer_config


def build_visualize_config(common: CommonComponents) -> OlmoEarthVisualizeConfig:
    """Build the visualize config."""
    return OlmoEarthVisualizeConfig(
        num_samples=None,
        output_dir=f"{common.save_folder}/visualizations",
        std_multiplier=2.0,
    )


if __name__ == "__main__":
    logger.info("Using TINY model configuration (no APT) for MADOS")

    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )
