r"""olmo-core entrypoint for open-set text-conditioned segmentation.

Mirrors the structure of ``scripts/vnext/single_bandset_band_dropout/base_token_masked.py``
but plugs in the open-set model + train module instead of LatentMIM.

The OlmoEarth checkpoint to use as the frozen encoder is required and is
passed in via an override:

    python scripts/open_set/script.py launch RUN_NAME CLUSTER \
        --model.checkpoint_path=/weka/dfive-default/.../step370000

For convenience the ``launch_open_set.sh`` shell script wraps this — pass
the checkpoint as the first positional argument.
"""

from __future__ import annotations

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
from olmoearth_pretrain.internal.common import (
    build_common_components as build_common_components_default,
)
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
    OlmoEarthVisualizeConfig,
    SubCmd,
    main,
)
from olmoearth_pretrain.open_set.data.modality_subsample import (
    ModalitySubsampleConfig,
)
from olmoearth_pretrain.open_set.data.sampler import RandomNegativeSamplerConfig
from olmoearth_pretrain.open_set.model.cross_attn_decoder import (
    CrossAttnDecoderConfig,
)
from olmoearth_pretrain.open_set.model.open_set_model import OpenSetModelConfig
from olmoearth_pretrain.open_set.text.embedding_cache import TextEncoderConfig
from olmoearth_pretrain.open_set.train.train_module import OpenSetTrainModuleConfig
from olmoearth_pretrain.train.callbacks import (
    OlmoEarthSpeedMonitorCallback,
    OlmoEarthWandBCallback,
)
from olmoearth_pretrain.train.masking import MaskingConfig

logger = logging.getLogger(__name__)

# Patch sizes used at training time. The flexi-vit encoder samples a patch
# size in this range per batch. Keep these matched to the encoder config of
# the loaded OlmoEarth checkpoint.
MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1

# SigLIP text encoder (~1.1B parameters; produces 1152-dim embeddings).
TEXT_ENCODER_MODEL = "google/siglip2-so400m-patch14-384"
TEXT_DIM = 1152

WANDB_PROJECT = "2026_05_15_open_set"
WANDB_USERNAME = "eai-ai2"  # nosec


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build the common components — sets the training modalities."""
    config = build_common_components_default(script, cmd, run_name, cluster, overrides)
    # The label sources (OSM, CDL, ...) need to be present on the batch so
    # the sampler can extract per-class binary masks. Keep S2/S1/Landsat as
    # input modalities; everything else is a label source or context.
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


def build_model_config(common: CommonComponents) -> OpenSetModelConfig:
    """Build the open-set model config.

    ``checkpoint_path`` is intentionally left empty — the launcher passes it
    via ``--model.checkpoint_path=...`` and ``OpenSetModelConfig.validate``
    will fail loudly if it is missing.
    """
    return OpenSetModelConfig(
        checkpoint_path="",
        decoder_config=CrossAttnDecoderConfig(
            dim=512,
            depth=4,
            num_heads=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_norm=False,
            use_flash_attn=False,
        ),
        text_dim=TEXT_DIM,
        head_dim=None,  # default to decoder dim
        trainable_encoder=False,
    )


def build_train_module_config(
    common: CommonComponents,
) -> OpenSetTrainModuleConfig:
    """Build the open-set train module config."""
    return OpenSetTrainModuleConfig(
        optim_config=AdamWConfig(lr=3e-4, weight_decay=0.01, fused=False),
        rank_microbatch_size=8,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=1000),
        autocast_precision=DType.bfloat16,
        dp_config=DataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
        text_encoder_config=TextEncoderConfig(
            model_name=TEXT_ENCODER_MODEL,
            cache_dir=f"{common.save_folder}/text_emb_cache",
        ),
        sampler_config=RandomNegativeSamplerConfig(
            k_pos=2,
            k_neg=2,
            seed=0,
        ),
        modality_subsample_config=ModalitySubsampleConfig(
            min_kept=1,
            max_kept=None,
            p_subsample=0.5,
        ),
        seed=0,
        target_size_source=Modality.OPENSTREETMAP_RASTER.name,
    )


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Build the dataloader config.

    Uses the same H5 dataset path as the recent pretraining scripts. The
    masking strategy is set to ``random`` — the train module immediately
    calls ``unmask()`` so the strategy is functionally a no-op.
    """
    return OlmoEarthDataLoaderConfig(
        num_workers=8,
        global_batch_size=64,
        token_budget=2250,
        prefetch_factor=2,
        sampled_hw_p_list=list(range(1, 13)),
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        work_dir=common.save_folder,
        seed=3622,
        masking_config=MaskingConfig(
            strategy_config={
                "type": "random",
                "encode_ratio": 1.0,
                "decode_ratio": 0.0,
            }
        ),
    )


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Build the dataset config — same H5 corpus as recent pretraining runs."""
    return OlmoEarthDatasetConfig(
        h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828",
        training_modalities=common.training_modalities,
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build the trainer config.

    No downstream evaluators here yet — open-set evals (held-out classes,
    cross-source transfer) are TODO.
    """
    MAX_DURATION = Duration.epochs(50)
    METRICS_COLLECT_INTERVAL = 10
    CANCEL_CHECK_INTERVAL = 25
    LOAD_STRATEGY = LoadStrategy.if_available
    PERMANENT_SAVE_INTERVAL = 5000
    EPHERMERAL_SAVE_INTERVAL = 250

    checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)
    wandb_callback = OlmoEarthWandBCallback(
        name=common.run_name,
        project=WANDB_PROJECT,
        entity=WANDB_USERNAME,
        enabled=True,
    )
    garbage_collector_callback = GarbageCollectorCallback(gc_interval=1)

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
        .with_callback("speed_monitor", OlmoEarthSpeedMonitorCallback())
        .with_callback("gpu_memory_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("garbage_collector", garbage_collector_callback)
        .with_callback("beaker", BeakerCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=PERMANENT_SAVE_INTERVAL,
                ephemeral_save_interval=EPHERMERAL_SAVE_INTERVAL,
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
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )
