"""``v1_2_base_linear`` on the ``base_faster`` stack.

The same two-variable arm as ``scripts/vnext/temporal_rope/v1_2_base_linear.py``
-- band dropout OFF and a LINEAR patch projection -- but built on
``base_faster.py`` instead of the v1.1 baseline, so it picks up the validated
July 2026 speedups (projection-only target encoder + replicated DP with bf16
autocast, on top of base.py's fused AdamW) and runs its in-loop evals as
separate Beaker jobs.

The model itself is unchanged from that arm: ``base.py`` here is the v1.2
baseline, i.e. ``scripts/official/v1_1/base.py`` plus the ``rope_3d_mixed``
positional encoding, which is exactly what ``temporal_rope_mixed.py`` applied
on top of v1.1. The one knob that differs is ``ROPE_MIXED_BASE`` -- 10000.0
here vs 10.0 in ``temporal_rope_mixed.py`` -- so the launch must pass
``--model.{encoder,decoder}_config.rope_mixed_base=10`` to match. The original
launch passes it explicitly too.

* **No band dropout.** ``base.py`` sets ``band_dropout_rate=0.2`` with
  ``random_band_dropout=True`` on S2 and Landsat, so each forward samples a
  rate from ``Uniform(0, 0.2)`` and zeroes that fraction of band channels.
  Train-only, off at eval: a pure train/inference mismatch on the two
  modalities the backbone leans on hardest.
* **Linear patch projection.** ``PATCH_EMBED_HIDDEN_SIZES = [64]`` puts a
  per-pixel ``Linear(in_chans, 64) -> ReLU`` MLP BEFORE patchification.
  ``None`` restores the original single ``nn.Linear`` stem
  (``flexi_patch_embed.py`` branches on the list being truthy).
"""

import logging

from base import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_visualize_config,
)
from base_faster import build_model_config as _faster_build_model_config
from base_faster import build_train_module_config
from base_faster import build_trainer_config as _faster_build_trainer_config
from olmo_core.train.config import TrainerConfig

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_2/v1_2_base_linear_faster.py"

# The single ``nn.Linear`` stem: no per-pixel MLP before patchification.
PATCH_EMBED_HIDDEN_SIZES: list[int] | None = None
BAND_DROPOUT_RATE = 0.0


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The base_faster model with no band dropout and a linear patch stem."""
    config = _faster_build_model_config(common)
    encoder = config.encoder_config
    # Band dropout off. All three fields are cleared rather than just the rate:
    # with ``random_band_dropout=True`` the rate is resampled per forward, and
    # an empty modality list means no modality is eligible either way.
    encoder.band_dropout_rate = BAND_DROPOUT_RATE
    encoder.random_band_dropout = False
    encoder.band_dropout_modalities = []
    # Linear patch projection: drop the per-pixel MLP before patchification.
    encoder.patch_embed_hidden_sizes = PATCH_EMBED_HIDDEN_SIZES
    # ``post_proj_hidden_sizes`` is the other stem nonlinearity; base.py never
    # sets it, so assert rather than clear it -- if that changes this arm is no
    # longer a two-variable diff and should be revisited, not silently widened.
    assert encoder.post_proj_hidden_sizes is None, (
        "post_proj_hidden_sizes is expected to be unset; this run targets the "
        "patch-embed MLP and band dropout only"
    )
    return config


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """base_faster trainer config, with loop evals pointed at this module."""
    trainer_config = _faster_build_trainer_config(common)
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    evaluator.beaker_eval_module_path = MODULE_PATH
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
