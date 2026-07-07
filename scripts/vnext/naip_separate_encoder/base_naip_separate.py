"""v1.2 + a separate NAIP encoder whose features the decoder reconstructs.

This is the v1.2 (mixed 3D RoPE) baseline with an added auxiliary modality, NAIP
(``naip_10``), handled by a SEPARATE encoder (its own parameters, ViT-Small width 384).
The main encoder still processes only Sentinel-2 / Landsat / Sentinel-1 (+ static
decode-only modalities) and remains the transferable backbone. The NAIP encoder sees
only the unmasked (25%) NAIP patches; a reconstructor then predicts the masked (75%)
NAIP pixels by cross-attending to BOTH the main-encoder features and the NAIP-encoder
features.

Key differences from ``scripts/official/v1_2/base.py``:
- ``naip_10`` added to training modalities (loaded from a NAIP-containing dataset).
- Separate ViT-Small NAIP encoder + a reconstructor for NAIP pixels.
- Masking strategy ``random_time_with_decode_separate_encoder`` masks 75% of NAIP.
- Model is a ``DualEncoderLatentMIMConfig`` instead of ``LatentMIMConfig``.
"""

import logging
import sys
from pathlib import Path

from olmo_core.config import DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import CosWithWarmup

# Reuse the unchanged v1.2 builders / constants.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "official" / "v1_2"))
from base import (  # noqa: E402
    BAND_DROPOUT_MODALITIES,
    MAX_PATCH_SIZE,
    MIN_PATCH_SIZE,
    ONLY_DECODE_MODALITIES,
    PATCH_EMBED_HIDDEN_SIZES,
    RANDOM_BAND_DROPOUT_MAX_RATE,
    ROPE_MIXED_BASE,
    ROPE_TEMPORAL_COORDINATE_SCALE,
    SPATIAL_POS_ENCODING,
    build_trainer_config,
    build_visualize_config,
)
from base import (  # noqa: E402
    build_common_components as build_common_components_v1_2,
)

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig  # noqa: E402
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import (  # noqa: E402
    CommonComponents,
    SubCmd,
    main,
)
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS  # noqa: E402
from olmoearth_pretrain.nn.dual_encoder_latent_mim import (  # noqa: E402
    DualEncoderLatentMIMConfig,
)
from olmoearth_pretrain.nn.flexi_vit import (  # noqa: E402
    EncoderConfig,
    PredictorConfig,
    ReconstructorConfig,
)
from olmoearth_pretrain.train.loss import LossConfig  # noqa: E402
from olmoearth_pretrain.train.masking import MaskingConfig  # noqa: E402
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (  # noqa: E402
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

# NAIP encoder is ViT-Small width; the main model is `base` width (768).
NAIP_ENCODER_EMBEDDING_SIZE = 384
NAIP_ENCODER_DEPTH = 12
NAIP_ENCODER_NUM_HEADS = 6
NAIP_ENCODER_MLP_RATIO = 4.0

# NAIP masking: keep 25% of NAIP patches visible to the NAIP encoder, mask 75% for the
# decoder to reconstruct.
NAIP_ENCODE_RATIO = 0.25
NAIP_DECODE_RATIO = 0.75

# NAIP-containing dataset (adds naip_10 to the v1.2 modality mix).
H5PY_DIR = (
    "/weka/dfive-default/helios/dataset/osm_sampling/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_landsat_naip_10_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_"
    "worldcereal_worldcover_wri_canopy_height_map/1138828"
)


def _naip_masking_config(common: CommonComponents) -> MaskingConfig:
    """Masking config: v1.2 masking plus an independent 25/75 split for NAIP."""
    return MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode_separate_encoder",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "random_ratio": 0.5,
            "only_decode_modalities": ONLY_DECODE_MODALITIES,
            "separate_encoder_modalities": [Modality.NAIP_10.name],
            "separate_encode_ratio": NAIP_ENCODE_RATIO,
            "separate_decode_ratio": NAIP_DECODE_RATIO,
        },
        tokenization_config=common.tokenization_config,
    )


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build common components, adding naip_10 to the v1.2 modality mix."""
    config = build_common_components_v1_2(script, cmd, run_name, cluster, overrides)
    if Modality.NAIP_10.name not in config.training_modalities:
        config.training_modalities = config.training_modalities + [
            Modality.NAIP_10.name
        ]
    # tokenization_config already set by the v1.2 builder (S2/Landsat collapsed; NAIP uses
    # the default single-bandset tokenization).
    return config


def _main_modalities(common: CommonComponents) -> list[str]:
    """Training modalities handled by the MAIN encoder (everything except NAIP)."""
    return [m for m in common.training_modalities if m != Modality.NAIP_10.name]


def build_model_config(common: CommonComponents) -> DualEncoderLatentMIMConfig:
    """Build the dual-encoder (main + NAIP) model config."""
    model_size = MODEL_SIZE_ARGS["base_shallow_decoder"]
    main_modalities = _main_modalities(common)
    main_out = model_size["encoder_embedding_size"]

    # Main encoder: v1.2 base encoder over the non-NAIP modalities.
    encoder_config = EncoderConfig(
        embedding_size=main_out,
        num_heads=model_size["encoder_num_heads"],
        depth=model_size["encoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        supported_modality_names=main_modalities,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=12,
        tokenization_config=common.tokenization_config,
        band_dropout_rate=RANDOM_BAND_DROPOUT_MAX_RATE,
        random_band_dropout=True,
        band_dropout_modalities=BAND_DROPOUT_MODALITIES,
        patch_embed_hidden_sizes=PATCH_EMBED_HIDDEN_SIZES,
        position_encoding=SPATIAL_POS_ENCODING,
        rope_mixed_base=ROPE_MIXED_BASE,
        rope_temporal_coordinate_scale=ROPE_TEMPORAL_COORDINATE_SCALE,
    )

    # Separate NAIP encoder (ViT-Small width). Projects its 384-d tokens up to the main
    # encoder width so the merged tokens are consumed uniformly by the decoder.
    naip_encoder_config = EncoderConfig(
        embedding_size=NAIP_ENCODER_EMBEDDING_SIZE,
        num_heads=NAIP_ENCODER_NUM_HEADS,
        depth=NAIP_ENCODER_DEPTH,
        mlp_ratio=NAIP_ENCODER_MLP_RATIO,
        supported_modality_names=[Modality.NAIP_10.name],
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=12,
        output_embedding_size=main_out,
        tokenization_config=common.tokenization_config,
        position_encoding=SPATIAL_POS_ENCODING,
        rope_mixed_base=ROPE_MIXED_BASE,
        rope_temporal_coordinate_scale=ROPE_TEMPORAL_COORDINATE_SCALE,
    )

    # Latent-MIM decoder: main modalities only (no NAIP context).
    decoder_config = PredictorConfig(
        encoder_embedding_size=main_out,
        decoder_embedding_size=model_size["decoder_embedding_size"],
        depth=model_size["decoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        num_heads=model_size["decoder_num_heads"],
        supported_modality_names=main_modalities,
        max_sequence_length=12,
        tokenization_config=common.tokenization_config,
        position_encoding=SPATIAL_POS_ENCODING,
        rope_mixed_base=ROPE_MIXED_BASE,
        rope_temporal_coordinate_scale=ROPE_TEMPORAL_COORDINATE_SCALE,
    )

    # Reconstructor: its internal predictor cross-attends to BOTH main and NAIP context
    # (supported = main + NAIP), but only NAIP pixels are reconstructed.
    reconstructor_predictor_config = PredictorConfig(
        encoder_embedding_size=main_out,
        decoder_embedding_size=model_size["decoder_embedding_size"],
        depth=model_size["decoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        num_heads=model_size["decoder_num_heads"],
        supported_modality_names=main_modalities + [Modality.NAIP_10.name],
        max_sequence_length=12,
        tokenization_config=common.tokenization_config,
        position_encoding=SPATIAL_POS_ENCODING,
        rope_mixed_base=ROPE_MIXED_BASE,
        rope_temporal_coordinate_scale=ROPE_TEMPORAL_COORDINATE_SCALE,
    )
    reconstructor_config = ReconstructorConfig(
        decoder_config=reconstructor_predictor_config,
        supported_modality_names=[Modality.NAIP_10.name],
        max_patch_size=MAX_PATCH_SIZE,
        tokenization_config=common.tokenization_config,
    )

    return DualEncoderLatentMIMConfig(
        encoder_config=encoder_config,
        naip_encoder_config=naip_encoder_config,
        decoder_config=decoder_config,
        reconstructor_config=reconstructor_config,
        naip_modality_name=Modality.NAIP_10.name,
    )


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """v1.2 train module + NAIP masking + MAE reconstruction loss for NAIP."""
    return ContrastiveLatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02, fused=False),
        rank_microbatch_size=64,
        masking_config=_naip_masking_config(common),
        loss_config=LossConfig(
            loss_config={
                "type": "modality_patch_discrimination_masked_negatives_vec",
                "tau": 0.1,
                "same_target_threshold": 0.999,
                "mask_negatives_for_modalities": ONLY_DECODE_MODALITIES,
            }
        ),
        contrastive_config=LossConfig(
            loss_config={
                "type": "InfoNCE",
                "weight": 0.05,
            }
        ),
        # MAE pixel reconstruction loss; only NAIP is reconstructed, so this trains the
        # NAIP encoder + reconstructor to predict masked NAIP.
        mae_loss_config=LossConfig(
            loss_config={
                "type": "mae",
                "loss_function": "SmoothL1Loss",
                "beta": 0.1,
            }
        ),
        token_exit_cfg={modality: 0 for modality in common.training_modalities},
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=8000),
        ema_decay=(1.0, 1.0),
        dp_config=DataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Build the dataset config (NAIP-containing dataset)."""
    return OlmoEarthDatasetConfig(
        h5py_dir=H5PY_DIR,
        training_modalities=common.training_modalities,
    )


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Build the dataloader config with the NAIP masking strategy."""
    return OlmoEarthDataLoaderConfig(
        num_workers=16,
        global_batch_size=512,
        token_budget=2250,
        prefetch_factor=4,
        sampled_hw_p_list=list(range(1, 13)),
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        work_dir=common.save_folder,
        seed=3622,
        num_masked_views=2,
        masking_config=_naip_masking_config(common),
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
