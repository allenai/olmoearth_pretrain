"""The two-flag input-stem ablation shared by the ``nobdlinpe`` arms.

Both flags sit in the pixel -> token stem and have been carried unmeasured since
v1.1, so they are turned off together:

* **No band dropout** (``nobd``). ``base.py`` sets ``band_dropout_rate=0.2`` with
  ``random_band_dropout=True`` on S2 and Landsat, i.e. each forward samples a rate
  from ``Uniform(0, 0.2)`` and zeroes that fraction of band channels. It is train-only
  and off at eval, so it is a pure train/inference mismatch on the two modalities the
  shipped embedding leans on hardest.
* **Linear patch projection** (``linpe``). ``PATCH_EMBED_HIDDEN_SIZES = [64]`` puts a
  per-pixel ``Linear(in_chans, 64) -> ReLU`` MLP BEFORE patchification; the per-patch
  ``Linear`` then maps ``64 * p_h * p_w -> embedding_size``. Setting it to ``None``
  restores the original single ``nn.Linear`` stem (``flexi_patch_embed.py`` branches on
  the list being truthy), i.e. a linear layer instead of an MLP.

Nothing else moves: the mirrored counterpart of each arm is the run of the same name
without ``nobdlinpe``, so the existing curves are the control.
"""

import logging

from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

# The single ``nn.Linear`` stem: no per-pixel MLP before patchification.
PATCH_EMBED_HIDDEN_SIZES: list[int] | None = None
BAND_DROPOUT_RATE = 0.0


def apply_nobd_linpe(config: LatentMIMConfig) -> LatentMIMConfig:
    """Disable band dropout and the nonlinear patch projection, in place."""
    encoder = config.encoder_config
    # Band dropout off. All three fields are cleared (rather than just the rate) so a
    # later default change to ``random_band_dropout`` cannot resurrect it: with
    # ``random_band_dropout=True`` the rate is resampled per forward from the field
    # below, and an empty modality list means no modality is eligible either way.
    encoder.band_dropout_rate = BAND_DROPOUT_RATE
    encoder.random_band_dropout = False
    encoder.band_dropout_modalities = []
    # Linear patch projection: drop the per-pixel MLP before patchification.
    encoder.patch_embed_hidden_sizes = PATCH_EMBED_HIDDEN_SIZES
    # ``post_proj_hidden_sizes`` is the other stem nonlinearity; base.py never sets it,
    # so assert rather than clear it -- if that changes this arm is no longer a
    # two-variable diff and should be revisited, not silently widened.
    assert encoder.post_proj_hidden_sizes is None, (
        "post_proj_hidden_sizes is expected to be unset; this ablation targets the "
        "patch-embed MLP and band dropout only"
    )
    return config
