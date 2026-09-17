"""Pixel-resolution registers + per-pixel raw-band reconstruction on the v1.3 recipe.

The ``pixreg_pixrecon`` arm of the pixel-branch program, re-based onto the v1.3
training stack (``../base.py``: single-pass LatentMIM, fused AdamW, DDP + bf16,
decorrelated shape sampler, projection-only target encoder). The register width is
the query-token-compaction shape (``../ablations/query_token_compaction.py``: native
``register_dim=128``, attention at encoder width, no distillation student), so the
register grid IS the served embedding. On top of that:

* **Pixel registers** (``register_pixel_grid``): the single cloned latent is laid at
  PIXEL resolution -- one 128-d register per pixel of the finest spatial modality,
  whatever patch size the trunk runs at, placed at pixel-center RoPE coordinates in
  the patch frame. The reads keep the v1.3 wideread shape (attention at 768, 12 x 64
  heads over the coarse patch tokens); only the latent self-attention, quadratic in
  the register count, narrows to width 128 with 2 x 64-dim heads, and the
  bottleneck blocks' LayerNorms drop their affine (a measured cost at this register
  count, redundant before the blocks' own projections).
* **Patch sizes 1..4 and grids up to hw_p=24**: at the 3072 token budget the
  worst-case pixel-register count is ``24 * 4 = 96 x 96 = 9,216``. The v1.3 sampler
  goes to ps=8 and hw_p=32 (``128 x 128 = 16k`` registers at ps=4, 65k at ps=8),
  whose quadratic latent self-attention is not affordable.
* **Map supervision at one value per cell** (``spatial_unfold=1``): the cells already
  sit at pixel resolution, so the default ``max_patch_size**2`` sub-cell unfold would
  predict a 64x-oversized map and immediately downsample it back. Base weight 0.1
  (the ``w0p1`` of the original arm; v1.3 trains at 1.0).
* **Per-pixel raw-band reconstruction** (``pixrecon``): two extra TIME-CONDITIONED
  supervision heads on the register grid -- a small MLP on ``[register_cell ;
  phi(day_of_year)]`` evaluated at every observed timestep -- pointed at the
  normalized Sentinel-2 L2A (12 bands) and Sentinel-1 (2 bands) inputs themselves,
  MSE at weight 0.05 each, MISSING_VALUE-masked. The targets are the raw inputs
  already in every batch, so nothing else changes.

WHY: the pixel-register MIM targets sit at the sampled patch size and the map
supervision is largely static, so nothing directly demands that a pixel register
store its own pixel's temporal signature. Reconstructing the raw bands per (pixel,
timestep) is the strongest cheap detail-forcing signal available: the head is tiny
(the register must store the trajectory, the head just decodes it given time), and
the low-order day-of-year basis limits the demand to the seasonal component --
clouds and one-off events stay unpredictable and are ignored.

The original arm
(``regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_ps14_pixreg_pixrecon``,
W&B project ``2026_08_19_pixel_branch``) additionally had the temporally anchored
register read (``tanchor``) and an NDVI head; neither is carried over.

IN-LOOP EVALS: the AEF trials + PASTIS scored on the register grid itself (no
student). At the eval window (``ws16_ps1``) the pixel grid equals the patch grid.
"""

import logging
import sys
from pathlib import Path

# The experiments import the release recipe from the directory above.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    aeftrial_loop_eval_tasks,
    build_common_components,
    build_dataset_config,
    build_register_bottleneck_model_config,
    build_supervision_head_config,
    build_visualize_config,
    route_loop_evals_through_beaker,
)
from base import build_dataloader_config as _base_build_dataloader_config  # noqa: E402
from base import (
    build_train_module_config as _base_build_train_module_config,  # noqa: E402
)
from base import build_trainer_config as _base_build_trainer_config  # noqa: E402

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from olmoearth_pretrain.nn.supervision_head import (  # noqa: E402
    SupervisionModalityConfig,
    SupervisionTaskType,
)
from olmoearth_pretrain.train.train_module.latent_mim import (  # noqa: E402
    LatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/experiments/pixreg_pixrecon.py"
WANDB_PROJECT = "2026_08_19_pixel_branch"

# --- registers ---------------------------------------------------------------------------
# Native register width; the grid is the served embedding (no student).
REGISTER_DIM = 128
# Latent self-attention at the register width with 2 x 64-dim heads, following the
# width-sweep convention of holding head_dim at 64 (head_dim < 64 degraded the spatial
# evals; the head COUNT mattered for throughput, which the narrow LSA width buys back
# many times over). The reads keep the wideread 12 x 64 shape from the v1.3 base.
REGISTER_LATENT_ATTN_DIM = REGISTER_DIM
REGISTER_LATENT_NUM_HEADS = 2

# --- supervision -------------------------------------------------------------------------
# The original pixreg arm's map-supervision weight (v1.3 trains at 1.0).
SUPERVISION_BASE_WEIGHT = 0.1
# Raw-band reconstruction: the time-series input modalities the time-conditioned heads
# reconstruct per (pixel, timestep), and the per-modality loss weight. 0.05 x 2 = 0.1
# total: an auxiliary detail-forcing nudge, not a competing objective.
PIXEL_RECON_MODALITIES = (
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
)
PIXEL_RECON_WEIGHT = 0.05
# Annual-harmonic day-of-year basis (K=4 spans phenology-scale structure).
PIXEL_RECON_TIME_HARMONICS = 4

# --- shape sampler -----------------------------------------------------------------------
# Patch sizes sampled uniformly over 1..MAX_SAMPLED_PATCH_SIZE (the dataloader draws
# np.arange(min, max + 1)); the MODEL keeps max_patch_size=8 from the v1.2 base.
MAX_SAMPLED_PATCH_SIZE = 4
# v1.3's grid list minus 28 and 32: caps the pixel register grid at 24 * 4 = 96 x 96.
SAMPLED_HW_P_LIST = list(range(1, 17)) + [18, 20, 24]
# Halved from the v1.3 64: the pixel register grid multiplies the bottleneck/decoder
# context by up to ps**2 = 16. Grad-accumulation change only; the loss is unchanged.
RANK_MICROBATCH_SIZE = 32

# --- in-loop evals -----------------------------------------------------------------------
# Nine single-width tasks fit comfortably in 40k steps (as the qtc ablation).
LOOP_EVAL_INTERVAL_STEPS = 40000


def apply_pixel_registers(config: LatentMIMConfig) -> LatentMIMConfig:
    """Switch a register-bottleneck model config to pixel-resolution registers, in place.

    The reads keep the v1.3 shape (attention at encoder width); only the latent
    self-attention narrows to the register width. Apply AFTER attaching the
    supervision heads so the ``spatial_unfold`` override lands.
    """
    encoder_config = config.encoder_config
    encoder_config.register_pixel_grid = True
    encoder_config.register_latent_attn_dim = REGISTER_LATENT_ATTN_DIM
    encoder_config.register_latent_num_heads = REGISTER_LATENT_NUM_HEADS
    encoder_config.register_norm_affine = False
    if config.supervision_head_config is not None:
        # Register cells are already at pixel resolution: predict one value per cell
        # instead of a max_patch_size**2 sub-grid that would be downsampled right back.
        config.supervision_head_config.spatial_unfold = 1
    return config


def apply_pixel_reconstruction(config: LatentMIMConfig) -> LatentMIMConfig:
    """Add per-pixel raw-band reconstruction heads (the ``pixrecon`` part), in place.

    One time-conditioned supervision head per time-series input modality (S2 L2A,
    S1): a small MLP on ``[register_cell ; phi(day_of_year)]`` predicts the cell's
    normalized band values at every observed timestep (MSE, MISSING_VALUE-masked).
    Apply AFTER attaching the supervision heads (it extends the existing config).
    """
    assert config.supervision_head_config is not None, (
        "apply_pixel_reconstruction requires a supervision_head_config"
    )
    for name in PIXEL_RECON_MODALITIES:
        config.supervision_head_config.modality_configs[name] = (
            SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=Modality.get(name).num_bands,
                weight=PIXEL_RECON_WEIGHT,
                regression_loss_type="mse",
                time_conditioned=True,
                time_harmonics=PIXEL_RECON_TIME_HARMONICS,
            )
        )
    return config


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d128 pixel registers + map supervision (w0.1) + S2 L2A / S1 reconstruction."""
    config = build_register_bottleneck_model_config(common, register_dim=REGISTER_DIM)
    config.supervision_head_config = build_supervision_head_config(
        base_weight=SUPERVISION_BASE_WEIGHT
    )
    return apply_pixel_reconstruction(apply_pixel_registers(config))


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """v1.3 sampler with patch sizes 1..4 and grids capped at hw_p=24."""
    config = _base_build_dataloader_config(common)
    config.max_patch_size = MAX_SAMPLED_PATCH_SIZE
    config.sampled_hw_p_list = list(SAMPLED_HW_P_LIST)
    return config


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """v1.3 train module at the pixel-register microbatch (32)."""
    config = _base_build_train_module_config(common)
    config.rank_microbatch_size = RANK_MICROBATCH_SIZE
    return config


def build_trainer_config(common: CommonComponents):
    """AEF trials + PASTIS on the register grid via Beaker; pixel-branch W&B project."""
    trainer_config = route_loop_evals_through_beaker(
        _base_build_trainer_config(common),
        MODULE_PATH,
        aeftrial_loop_eval_tasks(LOOP_EVAL_INTERVAL_STEPS),
    )
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
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
