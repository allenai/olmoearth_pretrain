r"""ViT-H (1280-dim, 32-layer) single-bandset with masked-negatives-vec loss.

Single S2 bandset (all 12 bands) + Landsat single bandset,
random band dropout ~Uniform(0, 0.3), modality_cross_random masking,
modality_patch_discrimination_masked_negatives_vec loss.
2-node FSDP, bf16, compile off for stability.

Smoke test:
    python scripts/vnext/vit_h_single_bandset.py dry_run vit_h_sb_smoke local

Full run:
    python scripts/vnext/vit_h_single_bandset.py launch vit_h_single_bandset ai2/jupiter \
        --launch.num_nodes=2 \
        --launch.num_gpus=8 \
        --launch.priority=urgent \
        --trainer.callbacks.wandb.project=2026_05_08_vith_single_bandset_sweep
"""

import copy
import logging

from olmo_core.config import DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import CosWithWarmup
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

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.evals.datasets.normalize import NormMethod
from olmoearth_pretrain.evals.metrics import EvalMetric
from olmoearth_pretrain.internal.all_evals import EMBED_DIAG_TASKS
from olmoearth_pretrain.internal.common import (
    build_common_components as build_common_components_default,
)
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
    OlmoEarthVisualizeConfig,
    SubCmd,
    main,
)
from olmoearth_pretrain.nn.flexi_vit import PoolingType
from olmoearth_pretrain.nn.flexihelios import EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.tokenization import ModalityTokenization, TokenizationConfig
from olmoearth_pretrain.train.callbacks import (
    DownstreamEvaluatorCallbackConfig,
    OlmoEarthSpeedMonitorCallback,
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
RANDOM_BAND_DROPOUT_MAX_RATE = 0.3
PATCH_EMBED_HIDDEN_SIZES: list[int] = [64]

VIT_H_SIZE = {
    "encoder_embedding_size": 1280,
    "decoder_embedding_size": 1280,
    "encoder_depth": 32,
    "decoder_depth": 4,
    "encoder_num_heads": 16,
    "decoder_num_heads": 16,
    "mlp_ratio": 4.0,
}

TRAINING_MODALITIES = [
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

ONLY_DECODE_MODALITIES = [
    Modality.WORLDCOVER.name,
    Modality.SRTM.name,
    Modality.OPENSTREETMAP_RASTER.name,
    Modality.WRI_CANOPY_HEIGHT_MAP.name,
    Modality.CDL.name,
    Modality.WORLDCEREAL.name,
]

S2_SINGLE_BANDSET = ModalityTokenization(
    band_groups=[
        [
            "B02",
            "B03",
            "B04",
            "B08",
            "B05",
            "B06",
            "B07",
            "B8A",
            "B11",
            "B12",
            "B01",
            "B09",
        ],
    ]
)

LANDSAT_SINGLE_BANDSET = ModalityTokenization(
    band_groups=[
        ["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"],
    ]
)

TOKENIZATION_CONFIG = TokenizationConfig(
    overrides={
        "sentinel2_l2a": S2_SINGLE_BANDSET,
        "landsat": LANDSAT_SINGLE_BANDSET,
    }
)

_LOSS_CONFIG_DICT = {
    "type": "modality_patch_discrimination_masked_negatives_vec",
    "tau": 0.1,
    "same_target_threshold": 0.999,
    "mask_negatives_for_modalities": ONLY_DECODE_MODALITIES,
}

MASKING_CONFIG = MaskingConfig(
    strategy_config={
        "type": "modality_cross_random",
        "encode_ratio": 0.5,
        "decode_ratio": 0.5,
        "allow_encoding_decoding_same_bandset": True,
        "only_decode_modalities": ONLY_DECODE_MODALITIES,
    },
    tokenization_config=TOKENIZATION_CONFIG,
)


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build common components with single-bandset tokenization, 2-node default."""
    config = build_common_components_default(script, cmd, run_name, cluster, overrides)
    config.training_modalities = TRAINING_MODALITIES
    config.tokenization_config = TOKENIZATION_CONFIG
    if config.launch is not None:
        config.launch.num_nodes = 2
        config.launch.num_gpus = 8
    return config


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """ViT-H encoder (1280d, 32 layers) + shallow decoder, single bandset, band dropout."""
    encoder_config = EncoderConfig(
        embedding_size=VIT_H_SIZE["encoder_embedding_size"],
        num_heads=VIT_H_SIZE["encoder_num_heads"],
        depth=VIT_H_SIZE["encoder_depth"],
        mlp_ratio=VIT_H_SIZE["mlp_ratio"],
        supported_modality_names=common.training_modalities,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=12,
        use_flash_attn=True,
        tokenization_config=TOKENIZATION_CONFIG,
        band_dropout_rate=RANDOM_BAND_DROPOUT_MAX_RATE,
        random_band_dropout=True,
        patch_embed_hidden_sizes=PATCH_EMBED_HIDDEN_SIZES,
    )
    decoder_config = PredictorConfig(
        encoder_embedding_size=VIT_H_SIZE["encoder_embedding_size"],
        decoder_embedding_size=VIT_H_SIZE["decoder_embedding_size"],
        depth=VIT_H_SIZE["decoder_depth"],
        mlp_ratio=VIT_H_SIZE["mlp_ratio"],
        num_heads=VIT_H_SIZE["decoder_num_heads"],
        supported_modality_names=common.training_modalities,
        max_sequence_length=12,
        use_flash_attn=True,
        tokenization_config=TOKENIZATION_CONFIG,
    )
    return LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Masked-negatives-vec loss, wd=0.02, lr=1e-4, FSDP bf16, compile OFF."""
    return ContrastiveLatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02, fused=True),
        rank_microbatch_size=16,
        masking_config=MASKING_CONFIG,
        loss_config=LossConfig(loss_config=copy.deepcopy(_LOSS_CONFIG_DICT)),
        contrastive_config=LossConfig(loss_config={"type": "InfoNCE", "weight": 0.1}),
        token_exit_cfg={modality: 0 for modality in common.training_modalities},
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=8000),
        ema_decay=(1.0, 1.0),
        compile_model=False,
        dp_config=DataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Dataloader with token_budget=1800 for ViT-H memory headroom."""
    return OlmoEarthDataLoaderConfig(
        num_workers=16,
        global_batch_size=512,
        token_budget=1800,
        prefetch_factor=4,
        sampled_hw_p_list=list(range(1, 13)),
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        work_dir=common.save_folder,
        seed=3622,
        num_masked_views=2,
        masking_config=MASKING_CONFIG,
    )


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Build dataset config."""
    return OlmoEarthDatasetConfig(
        h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828",
        training_modalities=common.training_modalities,
    )


EVAL_TASKS = {
    "m-eurosat": DownstreamTaskConfig(
        dataset="m-eurosat",
        embedding_batch_size=128,
        num_workers=0,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.steps(4000),
    ),
    "m_so2sat": DownstreamTaskConfig(
        dataset="m-so2sat",
        embedding_batch_size=128,
        num_workers=8,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.steps(20000),
    ),
    "mados": DownstreamTaskConfig(
        dataset="mados",
        embedding_batch_size=128,
        probe_batch_size=128,
        num_workers=8,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=False,
        probe_lr=0.01,
        epochs=50,
        eval_interval=Duration.steps(4000),
        primary_metric=EvalMetric.OVERALL_ACC,
    ),
    "yemen_crop": DownstreamTaskConfig(
        dataset="yemen_crop",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        eval_interval=Duration.steps(20000),
        probe_lr=0.001,
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
    ),
    "pastis": DownstreamTaskConfig(
        dataset="pastis",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=8,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        probe_lr=0.1,
        eval_interval=Duration.steps(20000),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
    ),
    "m-eurosat_10pct": DownstreamTaskConfig(
        dataset="m-eurosat",
        embedding_batch_size=128,
        num_workers=0,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.steps(4000),
        max_train_samples=2000,
    ),
    "pastis_10pct": DownstreamTaskConfig(
        dataset="pastis",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=8,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        probe_lr=0.1,
        eval_interval=Duration.steps(20000),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        max_train_samples=500,
    ),
    "m_bigearthnet": DownstreamTaskConfig(
        dataset="m-bigearthnet",
        embedding_batch_size=64,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.steps(4000),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        primary_metric=EvalMetric.MACRO_F1,
    ),
    "m_bigearthnet_10pct": DownstreamTaskConfig(
        dataset="m-bigearthnet",
        embedding_batch_size=64,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.steps(4000),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        primary_metric=EvalMetric.MACRO_F1,
        max_train_samples=5000,
    ),
    **EMBED_DIAG_TASKS,
}


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Trainer with kNN/probe evals + embedding diagnostics."""
    WANDB_PROJECT = "2026_05_07_scaling_investigation"

    return (
        TrainerConfig(
            work_dir=common.save_folder,
            load_strategy=LoadStrategy.if_available,
            save_folder=common.save_folder,
            cancel_check_interval=25,
            metrics_collect_interval=10,
            max_duration=Duration.steps(150000),
            checkpointer=CheckpointerConfig(work_dir=common.save_folder),
        )
        .with_callback(
            "wandb",
            OlmoEarthWandBCallback(
                name=common.run_name,
                project=WANDB_PROJECT,
                entity="eai-ai2",
                enabled=True,
            ),
        )
        .with_callback("speed_monitor", OlmoEarthSpeedMonitorCallback())
        .with_callback("gpu_memory_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(tasks=EVAL_TASKS),
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


def build_visualize_config(common: CommonComponents) -> OlmoEarthVisualizeConfig:
    """Build visualize config."""
    return OlmoEarthVisualizeConfig(
        num_samples=None,
        output_dir=str(f"{common.save_folder}/visualizations"),
        std_multiplier=2.0,
    )


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
