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
# A multiple of 256. With maps excluded this fits the full 12 months at grids up to
# hw=9 (3*9^2*12=2916). 6144 (full year to hw=13) was ~2.4x slower to train for
# marginal extra coverage; the decorrelated sampler already gets full-year-at-large-grid
# here, which the old anti-correlated sampler never did.
TOKEN_BUDGET = 3072
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
# Uniform over ps=1..8 -- the dataloader default (what patch_size_probs=None does).
# The ``_psuniform`` arms use this to isolate the ps=1 oversampling above from the rest of
# the newsampling recipe. The 4-point P(ps=1) sweep (0.125 / 0.40 / 0.70 / 1.00) showed the
# ps=1 gain on the frozen ps=1 PASTIS probes is FLAT from 0.125 to 0.70 and drops at 1.00,
# while the ps=4 evals degrade as P(ps=1) rises -- i.e. the oversampling is not what buys
# the newsampling gain, so uniform is the better default. See the directory README.
UNIFORM_PATCH_SIZE_PROBS = [1.0 / 8] * 8
# Base grids 1..16 plus a coarse incremental tail; the token floor drops hw<=2 and
# large grids naturally carry few timesteps. Nothing special about the exact values.
SAMPLED_HW_P_LIST = list(range(1, 17)) + [18, 20, 24, 28, 32]
# Back to the base 64 (matches old runs = 1 microbatch/step). At budget 3072 with the
# broadcastable key mask (nn/attention.py: no dense (B,H,N,Nk) mask, no O(N^2) score
# materialization) + expandable_segments, this should fit. NOTE: micro 64 @ 3072 has
# the same per-forward token load as the micro 32 @ 6144 config that OOM'd *before* the
# mask fix, so this relaunch is the real test of that fix -- if it OOMs, drop to 32
# (throughput is ~unchanged; micro only affects memory, not tokens/step).
RANK_MICROBATCH_SIZE = 64


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


def apply_uniform_patch_sizes(
    config: OlmoEarthDataLoaderConfig,
) -> OlmoEarthDataLoaderConfig:
    """Revert patch-size sampling to uniform, keeping every other newsampling knob.

    Apply AFTER :func:`apply_new_sampling`, which sets the oversampled
    :data:`PATCH_SIZE_PROBS`; this overwrites just that one field.
    """
    config.patch_size_probs = list(UNIFORM_PATCH_SIZE_PROBS)
    return config


def apply_shape_sweep(
    config: OlmoEarthDataLoaderConfig,
    *,
    token_budget: int,
    temporal_bias: float,
) -> OlmoEarthDataLoaderConfig:
    """Override the two swept shape-sampler axes, keeping every other knob fixed.

    Apply AFTER :func:`apply_new_sampling` (and :func:`apply_uniform_patch_sizes`);
    this overwrites only ``token_budget`` and ``temporal_bias``.

    The two axes are NOT independent, which is the point of sweeping them together.
    ``token_budget`` sets how wide the feasible timestep window is at each grid size;
    ``temporal_bias`` only skews the draw *within* that window. It cannot create tokens,
    so at a fixed budget a higher bias trades spatial extent for temporal extent. With
    the token floor at 228 and decode-only maps excluded (cost = 3*hw^2*t), full-year
    (t=12) shapes are reachable only up to hw=6 at budget 1536, hw=9 at 3072, and hw=13
    at 6144 -- and the ws16 ps=1 eval shape (hw=16) tops out at t=2 / t=4 / t=8
    respectively. So a cheap budget with an aggressive bias trains almost entirely on
    small-grid-long-sequence and large-grid-single-timestep shapes, and never gets close
    to the large-grid-AND-long-sequence regime the frozen ps=1 probes evaluate at.

    Duration is deliberately NOT touched. ``base.MAX_DURATION`` is ``epochs(300)``, and
    an epoch is a fixed number of *instances* (``total_batches = instances /
    global_batch_size``), independent of ``token_budget`` -- the sampler changes each
    instance's shape, not how many there are. So every arm runs the same 662,700 steps
    on an identical LR schedule and differs only in tokens per step, which keeps these
    runs comparable to every existing 300-epoch run. Do not switch to a steps- or
    tokens-based duration to "compute-match": step counts would diverge from the
    committed runs, and ``Duration.tokens`` is unusable here because
    ``Trainer.tokens_per_batch`` returns ``global_batch_size`` (512 *instances*).
    """
    config.token_budget = token_budget
    config.temporal_bias = temporal_bias
    return config


def apply_microbatch(config: LatentMIMTrainModuleConfig) -> LatentMIMTrainModuleConfig:
    """Halve the rank microbatch size in place so the larger budget fits memory."""
    config.rank_microbatch_size = RANK_MICROBATCH_SIZE
    return config
