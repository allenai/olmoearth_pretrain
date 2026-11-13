"""Baseline for the BlueTopo experiment: a tiny v1.3-style model for 60k steps.

v1.2 tiny plus the v1.3 additions on main's API (as in favyen's
``scripts/official/v1_3/base.py`` / ``claude_findings/common.py``): a 4-deep
AttentionPooler at encoder width, location-token supervision heads on the decode-only
maps, the single-pass LatentMIM train module and the decorrelated shape sampler. No
distilled student: it is detached and would not change the teacher.

Run directly, this trains the baseline, with no depth signal. ``run(bluetopo=...)`` adds
NOAA BlueTopo survey depth (US waters) in one of two ways; it is never an encoder input:

- ``SUPERVISION`` (v13_tiny_60k_bluetopo_sup.py): an L1 regression head on the location
  tokens whose loss reaches the encoder. Only that head reads BlueTopo.
- ``DECODE`` (v13_tiny_60k_bluetopo_decode.py): a decode-only latent-MIM target, like
  v1.2's maps. No supervision head; unlabelled pixels get BLUETOPO_DECODE_FILL.

The labels exclude the 1-degree cells held out for the BlueTopo depth eval.
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "official" / "v1_2"))

import base  # noqa: E402
import tiny  # noqa: E402
from olmo_core.config import DType  # noqa: E402
from olmo_core.distributed.parallel import (  # noqa: E402
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.train.common import Duration  # noqa: E402
from olmo_core.train.config import TrainerConfig  # noqa: E402

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
MAX_STEPS = 60_000
SUPERVISION = "supervision"
DECODE = "decode"
# Below the 0-40 m depth range, so unlabelled pixels are distinct from shallow water.
BLUETOPO_DECODE_FILL = -5.0
BLUETOPO_DIR = "/weka/dfive-default/agent/piperbots/2026-10-29-piperw-bathymetry-direct-supervision/bluetopo_train"

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


def build_common_components(
    *args, bluetopo: str | None = None, **kwargs
) -> CommonComponents:
    """v1.2 modalities, plus bluetopo if used."""
    config = base.build_common_components(*args, **kwargs)
    if bluetopo is not None:
        config.training_modalities = config.training_modalities + [
            Modality.BLUETOPO.name
        ]
    return config


def only_decode_modalities(common: CommonComponents) -> list[str]:
    """v1.2's decode-only modalities, plus bluetopo if used."""
    depth = [m for m in [Modality.BLUETOPO.name] if m in common.training_modalities]
    return base.ONLY_DECODE_MODALITIES + depth


def build_supervision_head_config(bluetopo: bool = False) -> SupervisionHeadConfig:
    """v1.3's map heads, plus a bluetopo depth head if used."""

    def _weight(task_type: SupervisionTaskType) -> float:
        return TASK_TYPE_WEIGHTS[task_type]

    regression = SupervisionTaskType.REGRESSION
    config = SupervisionHeadConfig(
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
        }
    )
    if bluetopo:
        config.modality_configs[Modality.BLUETOPO.name] = SupervisionModalityConfig(
            task_type=regression,
            num_output_channels=1,
            weight=_weight(regression),
            regression_loss_type="l1",
        )
    return config


def build_model_config(
    common: CommonComponents, bluetopo: str | None = None
) -> LatentMIMConfig:
    """v1.2 tiny + AttentionPooler + supervision heads.

    As a supervision target, bluetopo is left out of the encoder/decoder modalities so
    only its head reads it. As a decode target it stays in them, and the masking config
    keeps it decode-only.
    """
    model_common = common
    if bluetopo == SUPERVISION:
        model_common = replace(
            common,
            training_modalities=[
                m for m in common.training_modalities if m != Modality.BLUETOPO.name
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
    config.supervision_head_config = build_supervision_head_config(
        bluetopo == SUPERVISION
    )
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
        only_decode_modalities(common)
    )
    config.loss_config.loss_config["mask_negatives_for_modalities"] = (
        only_decode_modalities(common)
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
        only_decode_modalities(common)
    )
    return config


def build_dataset_config(
    common: CommonComponents, bluetopo: str | None = None
) -> OlmoEarthDatasetConfig:
    """Same H5 set, plus the BlueTopo labels if used."""
    config = base.build_dataset_config(common)
    if bluetopo is not None:
        config.bluetopo_dir = BLUETOPO_DIR
    if bluetopo == DECODE:
        config.bluetopo_fill_value = BLUETOPO_DECODE_FILL
    return config


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """v1.2 trainer, shortened to MAX_STEPS and logged to this project."""
    config = base.build_trainer_config(common)
    config.max_duration = Duration.steps(MAX_STEPS)
    config.callbacks["wandb"].project = WANDB_PROJECT
    return config


def run(bluetopo: str | None = None) -> None:
    """Run one arm: the baseline, or bluetopo as SUPERVISION or DECODE."""
    assert bluetopo in (None, SUPERVISION, DECODE), bluetopo
    main(
        common_components_builder=lambda *a, **k: build_common_components(
            *a, bluetopo=bluetopo, **k
        ),
        model_config_builder=lambda common: build_model_config(common, bluetopo),
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=lambda common: build_dataset_config(common, bluetopo),
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=base.build_visualize_config,
    )


if __name__ == "__main__":
    run()
