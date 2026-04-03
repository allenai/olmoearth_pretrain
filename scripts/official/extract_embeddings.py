"""Extract encoder embeddings from a pretrained checkpoint and save to disk."""

import logging

from script import (
    build_common_components,
    build_dataset_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexihelios import (
    EncoderConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config -- must match the checkpoint's architecture."""
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
        use_linear_patch_embed=False,
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


def build_extract_config(common: CommonComponents) -> dict:
    """Build the extraction-specific config fields."""
    return {
        "checkpoint_path": "/weka/dfive-default/olmoearth_pretrain/checkpoints/PLACEHOLDER/stepXXXXXX",
        "output_dir": f"{common.save_folder}/embeddings",
        "patch_size": MAX_PATCH_SIZE,
        "batch_size": 64,
        "num_workers": 8,
        "sampled_hw_p": 4,
    }


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        dataset_config_builder=build_dataset_config,
        extract_config_builder=build_extract_config,
    )
