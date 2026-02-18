"""Base model with training speed optimizations enabled.

Inherits all data, masking, and trainer config from scripts/official/script.py.
Only overrides what's needed to turn on the speed improvements.

Launch example:
    python scripts/vnext/speedups/base_speedup.py launch base_speedup_flash ai2/jupiter \
        --launch.num_gpus=8 \
        --launch.clusters=[ai2/ceres,ai2/jupiter,ai2/titan] \
        --trainer.callbacks.wandb.project=YYYY_MM_DD_speed_optimizations
"""

import logging
import sys
from pathlib import Path

# Allow importing from scripts/official/
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "official"))

from script import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_trainer_config,
    build_visualize_config,
)

from olmo_core.config import DType  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS  # noqa: E402
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, PredictorConfig  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (  # noqa: E402
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Base model config with speed optimizations: flash attention + linear patch embed."""
    model_size = MODEL_SIZE_ARGS["base_shallow_decoder"]

    encoder_config = EncoderConfig(
        embedding_size=model_size["encoder_embedding_size"],
        num_heads=model_size["encoder_num_heads"],
        depth=model_size["encoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        supported_modality_names=common.training_modalities,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=12,
        use_flash_attn=True,
        use_linear_patch_embed=True,
    )
    decoder_config = PredictorConfig(
        encoder_embedding_size=model_size["encoder_embedding_size"],
        decoder_embedding_size=model_size["decoder_embedding_size"],
        depth=model_size["decoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        num_heads=model_size["decoder_num_heads"],
        supported_modality_names=common.training_modalities,
        max_sequence_length=12,
        use_flash_attn=True,
    )
    return LatentMIMConfig(encoder_config=encoder_config, decoder_config=decoder_config)


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Train module config with torch.compile and bf16 autocast enabled."""
    from script import build_train_module_config as _base

    config = _base(common)
    config.compile_model = True
    return config


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
