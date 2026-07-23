"""Shared config overrides for the v1.2 register-bottleneck new-sampling runs.

These wrap the committed d128 wideread regsup / regsup_latlon runs (which use the
decorrelated grid/timestep shape sampler) and change only:

* supervision weight -> w0p1 (``base_weight`` 0.1), set in each run's model builder;
* dataloader shape sampling: the timestep axis is sampled independently of the grid
  (``time_priority_prob``) and biased toward the full sequence (``temporal_bias``); a
  token floor (``min_tokens_per_instance``) drops degenerate tiny shapes; ps=1 is
  oversampled for the ps=1 deployment; larger grids are reachable; and the decode-only
  map modalities no longer consume the encoder budget;
* ``rank_microbatch_size`` 64 -> 32 to absorb the larger token budget in memory
  (grad-accumulation change only; loss is unchanged).

Everything else is inherited from the committed runs, so these stay comparable to the
``_w0p1`` baselines except for the sampling change under study.
"""

from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

# w0p1: 10x the committed SUPERVISION_WEIGHT (0.01) -> effective 0.1 on regression/
# latlon arms, 0.01 on classification/BCE (see regbtl_v1_2_regsup_common).
SUPERVISION_BASE_WEIGHT = 0.1

# --- shape-sampler hyperparameters (see the shape-sampler distribution analysis) ---
# 6144 ~= the requested ~6090; a multiple of 256 and, with maps excluded, enough for
# the full 12 months at grids up to hw=13.
TOKEN_BUDGET = 6144
# Minimum tokens a shape must cost. With maps excluded this is 3*hw^2*t, so 228 drops
# hw<=2 entirely and forces small grids onto long sequences (hw=3 -> t>=9, hw=4 -> t>=5).
MIN_TOKENS_PER_INSTANCE = 228
# Skews the timestep draw toward the maximum of its feasible window (weight t**bias).
TEMPORAL_BIAS = 2.75
# Half of batches sample timesteps first (then a grid that fits); half sample grid first.
TIME_PRIORITY_PROB = 0.5
# P(patch_size = k) for k in 1..8 (min/max patch size are 1/8). Oversamples the ps=1
# deployment resolution while keeping flexi-ViT coverage of coarser patches.
PATCH_SIZE_PROBS = [0.40, 0.15, 0.13, 0.10, 0.08, 0.06, 0.045, 0.035]
# Base grids 1..16 plus a coarse incremental tail; the token floor drops hw<=2 and
# large grids naturally carry few timesteps. Nothing special about the exact values.
SAMPLED_HW_P_LIST = list(range(1, 17)) + [18, 20, 24, 28, 32]
# Halved from 64 so the ~2.7x token budget fits memory (baseline ran attention un-flashed).
RANK_MICROBATCH_SIZE = 32


def apply_new_sampling(config: OlmoEarthDataLoaderConfig) -> OlmoEarthDataLoaderConfig:
    """Set the decorrelated shape-sampling knobs on a dataloader config in place."""
    config.token_budget = TOKEN_BUDGET
    config.exclude_only_decode_from_budget = True
    config.min_tokens_per_instance = MIN_TOKENS_PER_INSTANCE
    config.temporal_bias = TEMPORAL_BIAS
    config.time_priority_prob = TIME_PRIORITY_PROB
    config.patch_size_probs = PATCH_SIZE_PROBS
    config.sampled_hw_p_list = SAMPLED_HW_P_LIST
    return config


def apply_microbatch(config: LatentMIMTrainModuleConfig) -> LatentMIMTrainModuleConfig:
    """Halve the rank microbatch size in place so the larger budget fits memory."""
    config.rank_microbatch_size = RANK_MICROBATCH_SIZE
    return config
