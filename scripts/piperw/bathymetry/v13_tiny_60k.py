"""Bathymetry direct-supervision pair: a tiny v1.3-style model for 60k steps.

v1.2 tiny plus the v1.3 additions on main's API (as in favyen's
``scripts/official/v1_3/base.py`` / ``claude_findings/common.py``): a 4-deep
AttentionPooler at encoder width, location-token supervision heads on the decode-only
maps, the single-pass LatentMIM train module and the decorrelated shape sampler. No
distilled student: it is detached and would not change the teacher.

Both arms load sdb (satellite-derived depth, invalid pixels MISSING) and train an sdb
regression head on the location tokens. It is decode-only and absent from the
encoder/decoder modality lists, so only the supervision head reads it. The arms differ
in one flag:

- ``sdb_probe``: the sdb head reads detached tokens (a probe trained alongside).
- ``sdb_sup``: the sdb head's L1 loss reaches the encoder (direct supervision).

See /weka/dfive-default/agent/piperbots/2026-10-29-piperw-bathymetry-direct-supervision/.
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "official" / "v1_2"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import base  # noqa: E402
import tiny  # noqa: E402
from olmo_core.config import DType  # noqa: E402
from olmo_core.distributed.parallel import (  # noqa: E402
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.train.config import TrainerConfig  # noqa: E402
from tiny_60k import build_trainer_config as build_60k_trainer_config  # noqa: E402

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig  # noqa: E402
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.encodings import PositionEncoding  # noqa: E402
from olmoearth_pretrain.nn.flexi_vit import AttentionPoolerConfig  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from olmoearth_pretrain.nn.supervision_head import (  # noqa: E402
    SupervisionHeadConfig,
    SupervisionModalityConfig,
    SupervisionTaskType,
)
from olmoearth_pretrain.train.train_module.latent_mim import (  # noqa: E402
    LatentMIMTrainModuleConfig,
)

WANDB_PROJECT = "2026_10_29_bathymetry_direct_supervision"
SDB_GEBCO_PATCHES = "/weka/dfive-default/agent/piperbots/2026-10-26-piperw-bathymetry-target/gebco_patches.npy"
ONLY_DECODE_MODALITIES = base.ONLY_DECODE_MODALITIES + [Modality.SDB.name]

POOLER_DEPTH = 4
DECODER_POSITION_ENCODING = PositionEncoding.AXIAL_2D_ROPE.value

TASK_TYPE_WEIGHTS = {
    SupervisionTaskType.CLASSIFICATION: 0.1,
    SupervisionTaskType.BINARY_CLASSIFICATION: 0.1,
    SupervisionTaskType.REGRESSION: 1.0,
}
WORLDCOVER_CLASS_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
_CDL_CODES = [
    *range(1, 7),
    *range(10, 15),
    *range(21, 40),
    *range(41, 62),
    *range(63, 73),
    74,
    75,
    76,
    77,
    81,
    82,
    83,
    87,
    88,
    92,
    111,
    112,
    121,
    122,
    123,
    124,
    131,
    141,
    142,
    143,
    152,
    176,
    190,
    195,
    *range(204, 251),
    254,
]
CDL_CLASS_VALUES = [code / 200 for code in _CDL_CODES]

TOKEN_BUDGET = 3072
MIN_TOKENS_PER_INSTANCE = 228
TEMPORAL_BIAS = 2.75
TIME_PRIORITY_PROB = 0.5
SAMPLED_HW_P_LIST = list(range(1, 17)) + [18, 20, 24, 28, 32]
RANK_MICROBATCH_SIZE = 64


def build_common_components(*args, **kwargs) -> CommonComponents:
    """v1.2 modalities plus sdb."""
    config = base.build_common_components(*args, **kwargs)
    config.training_modalities = config.training_modalities + [Modality.SDB.name]
    return config


def build_supervision_head_config(sdb_detach: bool) -> SupervisionHeadConfig:
    """v1.3's map heads plus an sdb depth head."""

    def _weight(task_type: SupervisionTaskType) -> float:
        return TASK_TYPE_WEIGHTS[task_type]

    regression = SupervisionTaskType.REGRESSION
    return SupervisionHeadConfig(
        modality_configs={
            "worldcover": SupervisionModalityConfig(
                task_type=SupervisionTaskType.CLASSIFICATION,
                num_output_channels=len(WORLDCOVER_CLASS_VALUES),
                weight=_weight(SupervisionTaskType.CLASSIFICATION),
                class_values=WORLDCOVER_CLASS_VALUES,
            ),
            "srtm": SupervisionModalityConfig(
                task_type=regression,
                num_output_channels=Modality.SRTM.num_bands,
                weight=_weight(regression),
                regression_loss_type="l1",
            ),
            "openstreetmap_raster": SupervisionModalityConfig(
                task_type=SupervisionTaskType.BINARY_CLASSIFICATION,
                num_output_channels=30,
                weight=_weight(SupervisionTaskType.BINARY_CLASSIFICATION),
                pos_weight=True,
            ),
            "wri_canopy_height_map": SupervisionModalityConfig(
                task_type=regression,
                num_output_channels=1,
                weight=_weight(regression),
                regression_loss_type="l1",
            ),
            "cdl": SupervisionModalityConfig(
                task_type=SupervisionTaskType.CLASSIFICATION,
                num_output_channels=len(CDL_CLASS_VALUES),
                weight=_weight(SupervisionTaskType.CLASSIFICATION),
                class_values=CDL_CLASS_VALUES,
            ),
            "worldcereal": SupervisionModalityConfig(
                task_type=SupervisionTaskType.BINARY_CLASSIFICATION,
                num_output_channels=8,
                weight=_weight(SupervisionTaskType.BINARY_CLASSIFICATION),
                pos_weight=True,
            ),
            Modality.SDB.name: SupervisionModalityConfig(
                task_type=regression,
                num_output_channels=1,
                weight=_weight(regression),
                regression_loss_type="l1",
                detach_input=sdb_detach,
            ),
        }
    )


def build_model_config(common: CommonComponents, sdb_detach: bool) -> LatentMIMConfig:
    """v1.2 tiny + AttentionPooler + supervision heads; sdb never enters the encoder."""
    model_common = replace(
        common,
        training_modalities=[
            m for m in common.training_modalities if m != Modality.SDB.name
        ],
    )
    config = tiny.build_model_config(model_common)
    encoder_config = config.encoder_config
    decoder_config = config.decoder_config
    width = encoder_config.embedding_size
    encoder_config.attention_pooler_config = AttentionPoolerConfig(
        pooler_dim=width,
        pooler_depth=POOLER_DEPTH,
        per_depth_read_proj=True,
        attn_dim=width,
    )
    decoder_config.use_attention_pooler = True
    decoder_config.pooler_dim = width
    decoder_config.position_encoding = DECODER_POSITION_ENCODING
    config.projection_only_target = True
    config.supervision_head_config = build_supervision_head_config(sdb_detach)
    return config


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """Tiny's optimizer in v1.3's single-pass LatentMIM train module."""
    t = tiny.build_train_module_config(common)
    config = LatentMIMTrainModuleConfig(
        optim_config=t.optim_config,
        rank_microbatch_size=RANK_MICROBATCH_SIZE,
        transform_config=t.transform_config,
        masking_config=t.masking_config,
        loss_config=t.loss_config,
        mae_loss_config=t.mae_loss_config,
        token_exit_cfg=t.token_exit_cfg,
        max_grad_norm=t.max_grad_norm,
        scheduler=t.scheduler,
        ema_decay=t.ema_decay,
        dp_config=DataParallelConfig(name=DataParallelType.ddp),
        regularizer_config=t.regularizer_config,
        autocast_precision=DType.bfloat16,
        compile_model=t.compile_model,
        compile_loss=t.compile_loss,
        find_unused_parameters=t.find_unused_parameters,
        state_dict_save_opts=t.state_dict_save_opts,
        state_dict_load_opts=t.state_dict_load_opts,
    )
    config.optim_config.fused = True
    config.masking_config.strategy_config["only_decode_modalities"] = (
        ONLY_DECODE_MODALITIES
    )
    return config


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """v1.2 dataloader with v1.3's single view and decorrelated shape sampler."""
    config = base.build_dataloader_config(common)
    config.num_masked_views = 1
    config.token_budget = TOKEN_BUDGET
    config.min_tokens_per_instance = MIN_TOKENS_PER_INSTANCE
    config.temporal_bias = TEMPORAL_BIAS
    config.time_priority_prob = TIME_PRIORITY_PROB
    config.sampled_hw_p_list = list(SAMPLED_HW_P_LIST)
    config.masking_config.strategy_config["only_decode_modalities"] = (
        ONLY_DECODE_MODALITIES
    )
    return config


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Same H5 set; sdb with the GEBCO deep-water mask and MISSING invalid pixels."""
    config = base.build_dataset_config(common)
    config.sdb_gebco_patches_file = SDB_GEBCO_PATCHES
    config.sdb_invalid_as_missing = True
    return config


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """tiny_60k's trainer, logged to this pair's project."""
    config = build_60k_trainer_config(common)
    config.callbacks["wandb"].project = WANDB_PROJECT
    return config


def run(sdb_detach: bool) -> None:
    """Run one arm."""
    main(
        common_components_builder=build_common_components,
        model_config_builder=lambda common: build_model_config(common, sdb_detach),
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=base.build_visualize_config,
    )
