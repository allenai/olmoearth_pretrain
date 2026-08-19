"""Shared pieces for the pixel-resolution register (``pixreg``) runs.

These runs put the ``SpatialRegisterBottleneck``'s dynamic grid at PIXEL resolution
(``register_pixel_grid``): one 128-dim register per pixel of the finest spatial
modality, whatever patch size the encoder trunk runs at. All arms build on the d128
wideread regsup+NDVI tanchor newsampling psuniform frontier
(``regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform``)
and change, on top of it:

* ``ps14``: patch sizes sampled uniformly over 1..4 instead of 1..8. At the 3072
  token budget the worst-case pixel-register count is ``hw_p=32 * ps=4 -> 128x128 =
  16,384`` registers; at ps=8 it would be 65k, whose quadratic latent self-attention
  is not affordable. Every pixreg arm AND the control use this sampler, so the ps
  distribution is never a confound within the group.
* pixel registers: ``register_pixel_grid=True`` with TRUE latent self-attention kept,
  but run at width 128 (``register_latent_attn_dim = register_dim``) with 2 x 64-dim
  heads -- wideread LSA (768) over 16k registers is ~6x the FLOPs for the same mixing
  role, while the READS keep the wideread 12 x 64 shape (that is where the
  narrow-width pathologies were measured). Affine-free LayerNorms inside the
  bottleneck blocks (``register_norm_affine=False``): the gamma/beta gradient
  reduction over every pixel-register row is a measured cost at this scale, and the
  affine is redundant before the blocks' own projections.
* ``spatial_unfold=1`` on the supervision heads: register cells already sit at pixel
  resolution, so the default ``max_patch_size**2 = 64``-fold sub-cell unfold would
  predict a 64x-oversized map and immediately downsample it back.
* ``rank_microbatch_size = 32`` (vs 64): the pixel register grid multiplies the
  bottleneck/decoder context token count by up to ``ps**2 = 16``, and the worst-case
  16k-register LSA roughly doubles the step cost. Grad-accumulation change only; the
  loss is unchanged.

The pixel-BRANCH arms (runs 2 and 3) additionally attach the convolutional pixel
branch ported from the dual-res encoder program (``nn/pixel_branch.py``): dense
128-dim ConvNeXt-style steps on the pixel grid whose final ONLINE-pooled per-pixel
features initialize the register grid through a zero-init projection, so both arms
equal run 1 exactly at initialization. ``"conv"`` interleaves the steps with the
coarse trunk (FiLM + zero-init fusion, every 4th block -- the old program's top
pick); ``"thinconv"`` runs a standalone 4-step stack with no coarse interaction,
isolating the value of an independent high-resolution register init.
"""

import logging

from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

# P(patch_size = k) for k in 1..8: uniform over 1..4, zero above. Caps the worst-case
# pixel-register count at 128x128 = 16,384 (the ps=8 worst case would be 65k, whose
# quadratic latent self-attention is unaffordable).
PS14_PATCH_SIZE_PROBS = [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0]
# Latent self-attention head shape at the decoupled width 128: 2 x 64-dim heads,
# following the width-sweep convention of holding head_dim at 64 (head_dim < 64
# degraded the spatial evals; the head COUNT mattered for throughput, which the
# narrow LSA width buys back many times over).
REGISTER_LATENT_NUM_HEADS = 2
# Halved from the newsampling 64: the pixel register grid multiplies the bottleneck
# context by up to ps**2 = 16 and the 16k-register LSA roughly doubles worst-case
# step cost.
PIXREG_RANK_MICROBATCH_SIZE = 32

# Pixel-branch settings (runs 2 and 3) -- the old conv top pick, at Dp=128.
PIXEL_EMBEDDING_SIZE = 128
PIXEL_EVERY_K_BLOCKS = 4
PIXEL_CONV_KERNEL = 3
PIXEL_MLP_RATIO = 4.0
PIXEL_THIN_DEPTH = 4


def apply_ps14(config: OlmoEarthDataLoaderConfig) -> OlmoEarthDataLoaderConfig:
    """Restrict patch-size sampling to uniform over 1..4, in place.

    Apply AFTER ``apply_new_sampling`` / ``apply_uniform_patch_sizes``; this
    overwrites just the ``patch_size_probs`` field.
    """
    config.patch_size_probs = list(PS14_PATCH_SIZE_PROBS)
    return config


def apply_pixreg_microbatch(
    config: LatentMIMTrainModuleConfig,
) -> LatentMIMTrainModuleConfig:
    """Set the pixreg rank microbatch size (32) in place."""
    config.rank_microbatch_size = PIXREG_RANK_MICROBATCH_SIZE
    return config


def apply_pixel_registers(config: LatentMIMConfig) -> LatentMIMConfig:
    """Switch a wideread regbtl model config to pixel-resolution registers, in place.

    The reads keep the wideread shape (attention at encoder width, 12 x 64 heads at
    base); only the latent self-attention narrows to the register width. Apply AFTER
    ``add_register_supervision`` so the supervision-head unfold override lands.
    """
    encoder_config = config.encoder_config
    encoder_config.register_pixel_grid = True
    register_dim = encoder_config.register_dim
    assert register_dim is not None
    encoder_config.register_latent_attn_dim = register_dim
    encoder_config.register_latent_num_heads = REGISTER_LATENT_NUM_HEADS
    encoder_config.register_norm_affine = False
    if config.supervision_head_config is not None:
        # Register cells are already at pixel resolution: predict one value per cell
        # instead of a max_patch_size**2 sub-grid that would be downsampled right back.
        config.supervision_head_config.spatial_unfold = 1
    return config


def apply_pixel_branch(config: LatentMIMConfig, branch_type: str) -> LatentMIMConfig:
    """Attach the conv pixel branch (``"conv"`` or ``"thinconv"``), in place.

    Requires :func:`apply_pixel_registers` first (the branch's only consumer is the
    pixel-register initialization). Zero-init fusion/handoff make the model equal the
    branch-free run 1 at initialization.
    """
    encoder_config = config.encoder_config
    encoder_config.pixel_branch_type = branch_type
    encoder_config.pixel_embedding_size = PIXEL_EMBEDDING_SIZE
    encoder_config.pixel_every_k_blocks = PIXEL_EVERY_K_BLOCKS
    encoder_config.pixel_conv_kernel = PIXEL_CONV_KERNEL
    encoder_config.pixel_mlp_ratio = PIXEL_MLP_RATIO
    encoder_config.pixel_thin_depth = PIXEL_THIN_DEPTH
    return config
