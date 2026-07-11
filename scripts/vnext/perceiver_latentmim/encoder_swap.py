"""Mainline LatentMIM-Lite recipe with the Perceiver (set-latent) encoder.

Exact reproduction of the reference run ``trope_mixed_tscale_months``
(wandb eai-ai2/2026_04_22_add_hidden_layer_to_initial_projection/nd3xh7py:
scripts/official/v1_1/base.py + mixed 3D RoPE overrides) with ONE change:
``encoder_config`` is a ``PerceiverEncoderConfig`` — the FlexiViT
self-attention trunk replaced by a set-latent bottleneck (read-in ->
latent self-attention -> read-out). Tokenization, masking, the frozen
random target encoder, decoder/predictor, losses, schedule, and the eval
suite are all the mainline standards, untouched.
"""

import dataclasses
import logging
import sys
from pathlib import Path

# The v1.1 official baseline is the config base (same trick as
# scripts/official/v1_1/temporal_rope_mixed.py, which lives in that dir).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "official" / "v1_1"))
from base import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_visualize_config,
)
from base import (
    build_model_config as build_model_config_base,
)
from base import (
    build_trainer_config as build_trainer_config_base,
)
from olmo_core.train.config import TrainerConfig  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from olmoearth_pretrain.nn.perceiver_encoder import PerceiverEncoderConfig  # noqa: E402

logger = logging.getLogger(__name__)

# Mixed 3D RoPE settings, verbatim from the reference run.
SPATIAL_POS_ENCODING = "rope_3d_mixed"
ROPE_MIXED_BASE = 10.0
ROPE_TEMPORAL_COORDINATE_SCALE = 1.0 / 30.0

# Set-latent trunk size (ViT-B scale latents; reads/self-depth per SLP spec).
NUM_LATENTS = 1024
NUM_INPUT_READS = 2

WANDB_PROJECT = "perceiver_slp"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """v1.1 base model + mixed 3D RoPE, with the perceiver encoder swap."""
    config = build_model_config_base(common)

    for cfg in (config.encoder_config, config.decoder_config):
        cfg.position_encoding = SPATIAL_POS_ENCODING
        cfg.rope_mixed_base = ROPE_MIXED_BASE
        cfg.rope_temporal_coordinate_scale = ROPE_TEMPORAL_COORDINATE_SCALE

    # The ONE change vs the reference: copy every field of the baseline
    # EncoderConfig onto a PerceiverEncoderConfig.
    base_fields = {
        f.name: getattr(config.encoder_config, f.name)
        for f in dataclasses.fields(config.encoder_config)
    }
    config.encoder_config = PerceiverEncoderConfig(
        **base_fields,
        num_latents=NUM_LATENTS,
        num_input_reads=NUM_INPUT_READS,
    )
    return config


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """v1.1 trainer/eval suite, logged to the perceiver W&B project."""
    config = build_trainer_config_base(common)
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
