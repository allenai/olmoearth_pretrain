"""Approach 1: wide (1024) encoder bottlenecked to a 128-d deliverable embedding.

Same layer counts as the base variant (12 encoder / 4 decoder) but widened to
1024. The encoder self-attention runs at 1024 and emits a 128-d embedding for
downstream evals, while the decoder and patch-discrimination loss (and its frozen
target) operate at 1024. Because the encoder patch-embedding is natively 1024,
the frozen target is the raw 1024-d patch-embedding (no expander) -- so the loss
dimension carries genuine 1024-d signal rather than a lifted 768.

The instance-contrastive (InfoNCE) loss is disabled here, and the rank microbatch
is halved (64 -> 32) so the wider model fits on 8 jupiter GPUs.

On top of the shared 4 in-loop evals this variant also runs the two fifty_cities
segmentation probes (S2 and S1+S2), on a step interval that is a multiple of the
checkpointer save_interval (5000) so a permanent checkpoint exists at each eval
step for the beaker eval job to load.
"""

import logging
from dataclasses import replace

from base import (
    PATCH_EMBED_HIDDEN_SIZES,
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_smaller_embedding_model_config,
    build_visualize_config,
)
from base_faster import build_train_module_config as _faster_build_train_module_config
from base_faster import make_build_trainer_config
from olmo_core.train.common import Duration
from olmo_core.train.config import TrainerConfig

from olmoearth_pretrain.internal.all_evals import EVAL_TASKS as ALL_EVAL_TASKS
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/vnext/2026_07_20_smaller_embedding/base_dim128.py"
DELIVERABLE_EMBEDDING_SIZE = 128
# Widen to 1024 while keeping base's depth (12 enc / 4 dec). 1024 / 64 = 16 heads.
ENCODER_WIDTH = 1024
NUM_HEADS = 16
# Loss/target dimension. Equal to the encoder width so the frozen target is the
# raw 1024-d patch-embedding (no expander).
LOSS_EMBEDDING_SIZE = 1024
# Halved from base (64) so the wider model fits on 8 jupiter GPUs.
RANK_MICROBATCH_SIZE = 32

# fifty_cities segmentation probes, pulled from the canonical catalog so the names
# match the beaker eval job's tasks_to_run filter, on a step interval that is a
# multiple of PERMANENT_SAVE_INTERVAL (5000).
_FIFTY_CITIES_EVAL_NAMES = (
    "fifty_cities_sentinel2",
    "fifty_cities_sentinel1_sentinel2",
)
FIFTY_CITIES_EVAL_TASKS = {
    name: replace(ALL_EVAL_TASKS[name], eval_interval=Duration.steps(20000))
    for name in _FIFTY_CITIES_EVAL_NAMES
}


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Wide (1024) encoder, 128-d deliverable embedding, 1024-d loss/target."""
    config = build_smaller_embedding_model_config(
        common,
        "base_shallow_decoder",
        PATCH_EMBED_HIDDEN_SIZES,
        output_embedding_size=DELIVERABLE_EMBEDDING_SIZE,
        target_embedding_size=LOSS_EMBEDDING_SIZE,
    )
    # Keep base's depth (12 enc / 4 dec) but widen everything except the 128-d
    # deliverable to 1024. Decoder still consumes the 128-d deliverable
    # (encoder_embedding_size) and outputs 1024 (== target).
    encoder = config.encoder_config
    decoder = config.decoder_config
    encoder.embedding_size = ENCODER_WIDTH
    encoder.num_heads = NUM_HEADS
    decoder.decoder_embedding_size = ENCODER_WIDTH
    decoder.num_heads = NUM_HEADS
    return config


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Faster train module, no instance-contrastive loss, half microbatch."""
    config = _faster_build_train_module_config(common)
    config.contrastive_config = None
    config.rank_microbatch_size = RANK_MICROBATCH_SIZE
    return config


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Shared 4 evals + the two fifty_cities segmentation probes."""
    trainer_config = make_build_trainer_config(MODULE_PATH)(common)
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    evaluator.tasks = {**evaluator.tasks, **FIFTY_CITIES_EVAL_TASKS}
    return trainer_config


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
