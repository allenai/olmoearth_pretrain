r"""The v1.3 release recipe on current code: the baseline arm for the text-pretraining runs.

v1.3 (``regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform_stunorm_mlpgram1``)
was trained on ``gabi/perceiver`` with the pre-rename register-bottleneck config, and its
script (``scripts/official/v1_3/base.py`` on ``favyen/20260917-pixreg-v1_3``) still uses
those names. This is the same recipe under the current names, following
``scripts/official/v1_3/convert_legacy_checkpoint.py``: the register bottleneck is the
AttentionPooler, the registers are location tokens, and the student's distillation heads
are the LocationTokenDistillationHead. ``tests/unit/scripts/test_text_pretrain_config.py``
checks the model against the release config run through that converter.

What it is: the v1.2 base (hidden patch-embed projection, mixed 3D RoPE, decode-only
maps) with a 768-d AttentionPooler read in four interleaved [read -> self-attn] steps,
per-modality supervision heads on the location tokens, a detached [128, 64] student
distilled by cosine + Gram, the single-pass LatentMIM train module with a
projection-only target, replicated DP + bf16 autocast, and the decorrelated shape
sampler. Training matches the release run's config.json field for field (see
``build_trainer_config`` for the two trainer differences, neither of which touches
training).

    python scripts/vnext/text_pretrain/v1_3.py launch <run_name> ai2/jupiter \
        --launch.num_gpus=8 --launch.priority=urgent
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "official"))

from olmo_core.config import DType  # noqa: E402
from olmo_core.distributed.parallel import (  # noqa: E402
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.train.config import TrainerConfig  # noqa: E402
from v1_2.base import (  # noqa: E402
    PATCH_EMBED_HIDDEN_SIZES,
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from v1_2.base import (  # noqa: E402
    build_dataloader_config as _v1_2_build_dataloader_config,
)
from v1_2.base import (  # noqa: E402
    build_size_model_config as _v1_2_build_size_model_config,
)
from v1_2.base import (  # noqa: E402
    build_train_module_config as _v1_2_build_train_module_config,
)
from v1_2.base import build_trainer_config as _v1_2_build_trainer_config  # noqa: E402

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.encodings import PositionEncoding  # noqa: E402
from olmoearth_pretrain.nn.flexi_vit import AttentionPoolerConfig  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from olmoearth_pretrain.nn.location_token_distillation_head import (  # noqa: E402
    LocationTokenDistillationHeadConfig,
)
from olmoearth_pretrain.nn.supervision_head import (  # noqa: E402
    SupervisionHeadConfig,
    SupervisionModalityConfig,
    SupervisionTaskType,
)
from olmoearth_pretrain.train.train_module.latent_mim import (  # noqa: E402
    LatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

ENCODER_SIZE_NAME = "base_shallow_decoder"
POOLER_DIM = 768
POOLER_DEPTH = 4
STUDENT_DIMS = [128, 64]
BACK_PROJECTION_HIDDEN = 256
# The release run's v1.2 base (on gabi/perceiver) had 10000; main's v1.2 has 10.
ROPE_MIXED_BASE = 10000.0

SUPERVISION_BASE_WEIGHT = 1.0
# Classification/BCE losses run ~10x larger than the L1 regressions.
TASK_TYPE_WEIGHTS = {
    SupervisionTaskType.CLASSIFICATION: 0.1,
    SupervisionTaskType.BINARY_CLASSIFICATION: 0.1,
    SupervisionTaskType.REGRESSION: 1.0,
}
WORLDCOVER_CLASS_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
# USDA NASS CDL codes, normalized as raw/200 by the CDL norm config. 81/88 are pipeline
# sentinels kept for head-dim compatibility with the release checkpoint.
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
WANDB_PROJECT = "2026_11_09_text_pretrain"


def build_supervision_head_config() -> SupervisionHeadConfig:
    """Location-token supervision heads over the decode-only map modalities."""

    def _weight(task_type: SupervisionTaskType) -> float:
        return SUPERVISION_BASE_WEIGHT * TASK_TYPE_WEIGHTS[task_type]

    classification = SupervisionTaskType.CLASSIFICATION
    binary = SupervisionTaskType.BINARY_CLASSIFICATION
    regression = SupervisionTaskType.REGRESSION
    return SupervisionHeadConfig(
        modality_configs={
            "worldcover": SupervisionModalityConfig(
                task_type=classification,
                num_output_channels=len(WORLDCOVER_CLASS_VALUES),
                weight=_weight(classification),
                class_values=WORLDCOVER_CLASS_VALUES,
            ),
            "srtm": SupervisionModalityConfig(
                task_type=regression,
                num_output_channels=Modality.SRTM.num_bands,
                weight=_weight(regression),
                regression_loss_type="l1",
            ),
            "openstreetmap_raster": SupervisionModalityConfig(
                task_type=binary,
                num_output_channels=30,
                weight=_weight(binary),
                pos_weight=True,
            ),
            "wri_canopy_height_map": SupervisionModalityConfig(
                task_type=regression,
                num_output_channels=1,
                weight=_weight(regression),
                regression_loss_type="l1",
            ),
            "cdl": SupervisionModalityConfig(
                task_type=classification,
                num_output_channels=len(CDL_CLASS_VALUES),
                weight=_weight(classification),
                class_values=CDL_CLASS_VALUES,
            ),
            "worldcereal": SupervisionModalityConfig(
                task_type=binary,
                num_output_channels=8,
                weight=_weight(binary),
                pos_weight=True,
            ),
        }
    )


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """v1.2 base + AttentionPooler, supervision heads and the distilled student."""
    config = _v1_2_build_size_model_config(
        common, ENCODER_SIZE_NAME, PATCH_EMBED_HIDDEN_SIZES
    )
    encoder = config.encoder_config
    decoder = config.decoder_config
    encoder.rope_mixed_base = decoder.rope_mixed_base = ROPE_MIXED_BASE
    encoder.attention_pooler_config = AttentionPoolerConfig(
        pooler_dim=POOLER_DIM,
        pooler_depth=POOLER_DEPTH,
        per_depth_read_proj=True,
        # Attention at encoder width: pooler_dim is purely the storage width.
        attn_dim=encoder.embedding_size,
        student_dims=list(STUDENT_DIMS),
        student_output_norm=True,
    )
    decoder.use_attention_pooler = True
    decoder.pooler_dim = POOLER_DIM
    # The decoder cross-attends the spatial (2D) location-token grid; the encoder stays 3D.
    decoder.position_encoding = PositionEncoding.AXIAL_2D_ROPE.value
    config.supervision_head_config = build_supervision_head_config()
    config.location_token_distillation_head_config = (
        LocationTokenDistillationHeadConfig(
            back_projection_hidden=BACK_PROJECTION_HIDDEN
        )
    )
    config.projection_only_target = True
    return config


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """v1.2 dataloader, one masked view, the decorrelated shape sampler."""
    config = _v1_2_build_dataloader_config(common)
    config.num_masked_views = 1
    config.token_budget = TOKEN_BUDGET
    config.min_tokens_per_instance = MIN_TOKENS_PER_INSTANCE
    config.temporal_bias = TEMPORAL_BIAS
    config.time_priority_prob = TIME_PRIORITY_PROB
    config.sampled_hw_p_list = list(SAMPLED_HW_P_LIST)
    return config


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """v1.2's settings in the single-pass train module, replicated DP and bf16."""
    base = _v1_2_build_train_module_config(common)
    return LatentMIMTrainModuleConfig(
        optim_config=base.optim_config,
        rank_microbatch_size=RANK_MICROBATCH_SIZE,
        transform_config=base.transform_config,
        masking_config=base.masking_config,
        loss_config=base.loss_config,
        mae_loss_config=base.mae_loss_config,
        token_exit_cfg=base.token_exit_cfg,
        max_grad_norm=base.max_grad_norm,
        scheduler=base.scheduler,
        ema_decay=base.ema_decay,
        regularizer_config=base.regularizer_config,
        find_unused_parameters=base.find_unused_parameters,
        state_dict_save_opts=base.state_dict_save_opts,
        state_dict_load_opts=base.state_dict_load_opts,
        # torch.compile stays off: it degraded training quality on both DP stacks.
        dp_config=DataParallelConfig(name=DataParallelType.ddp),
        autocast_precision=DType.bfloat16,
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """v1.2's trainer, keeping every checkpoint and running no in-process evals.

    Both text-pretraining arms log to ``WANDB_PROJECT``.

    The release run kept all its 5000-step checkpoints (newer olmo-core keeps 3 by
    default) and ran its evals as separate Beaker jobs, never in the training process.
    Its eval sets (AEF year-aligned kNN twins, year-aligned PASTIS) are not on main, so
    checkpoints are scored afterwards with the release protocol instead.
    """
    config = _v1_2_build_trainer_config(common)
    config.callbacks["checkpointer"].max_checkpoints = None
    config.callbacks["downstream_evaluator"].enabled = False
    config.callbacks["wandb"].project = WANDB_PROJECT
    return config


def run() -> None:
    """Run the experiment."""
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )


if __name__ == "__main__":
    run()
