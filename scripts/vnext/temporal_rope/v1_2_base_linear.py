"""The v1.2 backbone baseline with band dropout OFF and a LINEAR patch projection.

``temporal_rope_mixed`` -- the v1.1 hidden-projection baseline with mixed 3D RoPE
(t, row, col) at ``rope_temporal_coordinate_scale`` in months, i.e. the run that
became the v1.2 backbone -- with two flags cut from the pixel -> token stem:

* **No band dropout.** ``base.py`` sets ``band_dropout_rate=0.2`` with
  ``random_band_dropout=True`` on S2 and Landsat, so each forward samples a rate from
  ``Uniform(0, 0.2)`` and zeroes that fraction of band channels. Train-only, off at
  eval: a pure train/inference mismatch on the two modalities the backbone leans on
  hardest.
* **Linear patch projection.** ``PATCH_EMBED_HIDDEN_SIZES = [64]`` puts a per-pixel
  ``Linear(in_chans, 64) -> ReLU`` MLP BEFORE patchification, and the per-patch
  ``Linear`` then maps ``64 * p_h * p_w -> embedding_size``. ``None`` restores the
  original single ``nn.Linear`` stem (``flexi_patch_embed.py`` branches on the list
  being truthy) -- a linear layer instead of an MLP. Note this undoes the very change
  the W&B project below is named for, which is the point: it was never ablated
  against the rest of the v1.2 recipe.

Everything else -- data, masking, loss, optimizer, schedule, and the eval set -- is
``scripts/official/v1_1/base.py`` untouched, and the run keeps that base's W&B project
(``2026_04_22_add_hidden_layer_to_initial_projection``) so
``trope_mixed_tscale_months`` sits beside it as the control.
"""

import logging

from base import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)
from temporal_rope_mixed import build_model_config as _base_build_model_config

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

# The single ``nn.Linear`` stem: no per-pixel MLP before patchification.
PATCH_EMBED_HIDDEN_SIZES: list[int] | None = None
BAND_DROPOUT_RATE = 0.0


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The mixed-3D-RoPE baseline with no band dropout and a linear patch stem."""
    config = _base_build_model_config(common)
    encoder = config.encoder_config
    # Band dropout off. All three fields are cleared rather than just the rate: with
    # ``random_band_dropout=True`` the rate is resampled per forward, and an empty
    # modality list means no modality is eligible either way.
    encoder.band_dropout_rate = BAND_DROPOUT_RATE
    encoder.random_band_dropout = False
    encoder.band_dropout_modalities = []
    # Linear patch projection: drop the per-pixel MLP before patchification.
    encoder.patch_embed_hidden_sizes = PATCH_EMBED_HIDDEN_SIZES
    # ``post_proj_hidden_sizes`` is the other stem nonlinearity; base.py never sets it,
    # so assert rather than clear it -- if that changes this arm is no longer a
    # two-variable diff and should be revisited, not silently widened.
    assert encoder.post_proj_hidden_sizes is None, (
        "post_proj_hidden_sizes is expected to be unset; this run targets the "
        "patch-embed MLP and band dropout only"
    )
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
