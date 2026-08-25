"""Shared pieces for the EARLY-READ register-bottleneck arms.

Base and A/B partner for every arm here:
``regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform``.
These arms are that run with ONE thing changed -- the encoder/bottleneck depth split --
and everything else inherited by importing its builders rather than re-deriving them:
``register_dim=128``, wideread (``register_attn_dim`` = encoder width), regsup with the
time-conditioned NDVI head at w0p1, ``register_temporal_anchor="year_start"``, the
decorrelated sampler at uniform patch sizes, and the NDVI extra-decode dataset /
dataloader / train-module path.

THE CHANGE: the base spends 12 encoder layers on the full patch-token set
(``n_h*n_w * T * channel_groups``) and 4 ``[read -> latent self-attend]`` blocks on the
register grid. These arms invert that split -- a SHALLOW trunk (3-6 layers) and a DEEP
bottleneck (8-12 blocks) -- so almost all depth runs on the compressed ``n_h*n_w`` grid.

WHY (what licenses this):

* ``2026_05_18_v2_knn_lp_evals``, capacity-matched: ``il`` (4 reads, all from encoder
  layer 12) vs ``mdr3`` (4 reads, from layers 3/6/9/12) is +0.11 pts over 54 tasks --
  a null. The K/V source depth was never the active ingredient; the BLOCK COUNT was
  (both 4-block arms beat both 3-block arms by ~1 pt). Layer 3 is already a viable read
  source, so the deep trunk is not supplying something the read depends on. Caveat: that
  evidence is d768 and single-seed, which is why the 6+8 hedge exists.
* Every mechanism that lets training re-weight toward LATE depths is neutral-to-harmful
  on the frozen evals in that same file (``lrw``: +0.01 on noic, -1.03 on ictok;
  learned vs uniform ``fsum``: -0.06). Training's preference for the final layer is a
  pretext-side artifact, not an eval-side gain -- which is why the learned-read-gate and
  fused-read knobs were removed from the encoder rather than swept further.
* Precedent: TokenLearner (arXiv:2106.11297) reduces tokens partway through a ViT and
  spends the savings on more layers over the reduced set -- ViT-B/16 baseline 55.6
  GFLOPs / 84.73%, vs 21-layer TokenLearner 47.1 GFLOPs / 85.21%.

WHAT STAYS THE SAME INSIDE THE BOTTLENECK: single-source re-reads
so every read re-queries the same shallow trunk output
through its own norm. Under ``interleave`` the reads are still distinct from one another
-- read 1 queries blank cloned latents, read N queries registers refined N-1 times
(Perceiver iterative attention). The depth diversity moves from the KEY side to the QUERY
side, which is the half that does not require a deep trunk.

WHY THE ANCHOR NEEDS NO CONTROL RUN: with a 3-layer trunk the anchored reads are the
model's temporal machinery -- axial 3D RoPE with registers at t=0 and K/V patches carrying
anchor-relative days is the only path by which time enters after the trunk. The base
already sets ``register_temporal_anchor="year_start"``, so the arms inherit it and the
depth split is the sole difference. No anchor-vs-no-anchor confound to subtract.

NOTE ON WIDTH: ``register_dim=128`` is the STORAGE width. Under wideread the reads and
latent blocks still attend at encoder width (768), and a read block's dominant term is the
K/V projection ``2*N*d_enc*d_attn``, which ``register_dim`` does not touch -- so the
compute arithmetic in the arm docstrings is the same as it would be at d768.
"""

import logging
from dataclasses import replace

from olmo_core.train.common import Duration
from regbtl_v1_2_common import LOOP_EVAL_CLUSTERS
from regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform import (
    build_dataloader_config as _base_build_dataloader_config,
)
from regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform import (
    build_model_config as _base_build_model_config,
)
from regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform import (
    build_train_module_config as _base_build_train_module_config,
)
from regbtl_v1_2_newsampling_common import TEMPORAL_BIAS, apply_shape_sweep

from olmoearth_pretrain.internal.all_evals import (
    EMBEDDING_EVAL_TASKS as _EMBEDDING_EVAL_TASKS,
)
from olmoearth_pretrain.internal.experiment import CommonComponents
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

# In-loop eval cadence, matching the other loop-eval dicts in regbtl_v1_2_common: a
# multiple of the checkpointer's save_interval (5000) so a permanent checkpoint exists at
# each eval step for the (separate, non-blocking) eval Beaker job to load.
_LOOP_EVAL_INTERVAL = Duration.steps(20000)

# The three embedding-product datasets these arms are judged on, at the everything-config
# sensor stack: Sentinel-1 + Sentinel-2 + Landsat. That is the sensor-fair match to AEF
# (which fuses Landsat internally), and it is the stack the year-aligned re-export exists
# to serve -- one calendar-year window set that OlmoEarth and AEF/Tessera both read.
#
# pastis is LP-only by convention (it keeps the pastis 128x128 / tile_samples / mIoU
# conventions rather than the AEF center-pixel ones); ethiopia and descals carry both a
# linear-probe and a kNN variant, because the AEF paper scores every dataset as
# best-of-{kNN-1, kNN-3, linear} and reporting LP alone would compare our single number
# against their best-of-three.
#
# The two _knn tasks ALSO carry AEF's balanced-trial protocol -- _aef_ps1_task attaches a
# BalancedTrialConfig whenever eval_mode is KNN, and the cap is prefix-matched so the
# _year_aligned re-exports inherit their parent's value (ethiopia 49, descals 200). Keep
# the plain kNN/LP as the primary readout for the depth-split A/B and treat the trial
# number as the AEF-comparability check: on ethiopia the trial arm has disagreed with the
# plain arm before (the gram sweep), and choosing between them after the fact would be
# post-hoc selection.
#
# THE BRIDGE TASK: ``pastis_ws16_ps1_sentinel1_sentinel2_pretrain_export`` is not part of
# the year-aligned S1+S2+Landsat set -- it is here solely because the A/B partner already
# runs it in-loop (it is in regbtl_v1_2_common.PASTIS_EMBEDDING_LOOP_EVAL_TASKS, at this
# same 20000-step interval, from the same registry entry). Without it these arms and the
# partner would share ZERO metrics and the comparison could not be read off the in-loop
# curves at all. Being one task, it is the only thing that can be compared without
# re-running the partner or sweeping its checkpoints -- so treat it as the tripwire, not
# the verdict: a single task sits inside the eval noise floor (LP 0.88 pts mean, 5.71 max)
# and cannot settle the depth split on its own.
#
# Looked up by name out of the canonical registry (not reconstructed here) so the names
# match what the eval job's ``tasks_to_run`` filter expects, and so a typo raises KeyError
# at import instead of silently dropping a task.
_EARLYREAD_LOOP_EVAL_NAMES = (
    "pastis_ws16_ps1_sentinel1_sentinel2_pretrain_export",
    "pastis_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat",
    "ethiopia_crops_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat",
    "ethiopia_crops_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat_knn",
    "descals_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat",
    "descals_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat_knn",
)
EARLYREAD_LOOP_EVAL_TASKS = {
    name: replace(_EMBEDDING_EVAL_TASKS[name], eval_interval=_LOOP_EVAL_INTERVAL)
    for name in _EARLYREAD_LOOP_EVAL_NAMES
}


# The year-aligned AEF pair ONLY -- ethiopia and descals, linear probe and kNN. Used by
# the checkpoint-sweep shims that score already-trained runs (cand_ndvi, sphere_unif0p1)
# as baselines for this sweep. Deliberately narrower than EARLYREAD_LOOP_EVAL_TASKS: a
# standalone checkpoint sweep otherwise runs the whole shared catalog (eurosat, mados,
# fifty_cities, ...), which costs hours per checkpoint to answer nothing being asked.
#
# The two _knn entries carry AEF's balanced-trial protocol automatically (_aef_ps1_task
# attaches it whenever eval_mode is KNN), which is where the aeftrial_* metrics come from.
_YEAR_ALIGNED_ONLY_NAMES = (
    "ethiopia_crops_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat",
    "ethiopia_crops_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat_knn",
    "descals_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat",
    "descals_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat_knn",
)
YEAR_ALIGNED_ONLY_LOOP_EVAL_TASKS = {
    name: replace(_EMBEDDING_EVAL_TASKS[name], eval_interval=_LOOP_EVAL_INTERVAL)
    for name in _YEAR_ALIGNED_ONLY_NAMES
}


def set_year_aligned_only_loop_evals(trainer_config, module_path: str):
    """REPLACE the eval set with the year-aligned ethiopia + descals probes only.

    For the checkpoint-sweep baselines. ``checkpoint_sweep_evals`` only reads a module's
    own tasks when ``OE_LOOP_EVAL_FROM_TRAIN_CONFIG`` is set in the environment -- without
    it a standalone sweep silently uses the shared catalog regardless of what this returns.
    Set that env var when launching, or this has no effect.
    """
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    evaluator.tasks = dict(YEAR_ALIGNED_ONLY_LOOP_EVAL_TASKS)
    evaluator.run_as_beaker_job = False
    trainer_config.callbacks["downstream_evaluator"] = evaluator
    return trainer_config


def build_earlyread_model_config(
    common: CommonComponents,
    *,
    trunk_depth: int,
    latent_depth: int,
    register_dim: int | None = None,
    output_dim: int | None = None,
) -> LatentMIMConfig:
    """The d128 NDVI tanchor base, with the encoder/bottleneck depth split reallocated.

    Everything except the two depths comes from the base script's own
    ``build_model_config``, so this cannot drift from the A/B partner: if the base changes
    its register width, supervision heads, anchor, or bottleneck flags, these arms follow.

    Args:
        common: The shared components (modalities, tokenization) for this run.
        trunk_depth: Encoder self-attention layers over the FULL patch-token set, before
            the bottleneck reads it. 12 is the base; 3 is the primary early-read arm; 6
            is the hedge.
        latent_depth: ``[read -> latent self-attend]`` blocks over the register grid.
            Under ``register_interleave`` this is BOTH the read count and the
            latent-block count.
        register_dim: Internal width of the read/latent stack. ``None`` keeps the base's
            128. Setting 768 makes every bottleneck GEMM square and tensor-core friendly
            (at 768 the ``attn_dim`` decoupling becomes a no-op, i.e. plain tied-width
            blocks) at the cost of 6x the projection FLOPs and 36x the MLP FLOPs. That is
            a bet that the stack is launch-bound, where extra FLOPs at an unchanged kernel
            count are close to free -- it is NOT a speedup, and the arm's s/step is itself
            the measurement of which regime we are in.
        output_dim: If set, a single ``Linear(register_dim, output_dim)`` on the
            bottleneck's output, so the decoder, supervision heads and evals all consume
            ``output_dim`` while the stack runs at ``register_dim``. Use with
            ``register_dim=768, output_dim=128`` to keep the shipped embedding at 128.
            In the gradient path, unlike the detached ``register_projection_dims``
            student.

    Returns:
        The base model config with ``depth`` and ``register_latent_depth`` overridden.
    """
    config = _base_build_model_config(common)
    encoder_config = config.encoder_config

    encoder_config.depth = trunk_depth
    encoder_config.register_latent_depth = latent_depth

    if register_dim is not None:
        # Only the encoder's INTERNAL width moves. The decoder keeps the base's 128,
        # which is what the output projection delivers to it.
        encoder_config.register_dim = register_dim
    if output_dim is not None:
        encoder_config.register_output_dim = output_dim
        if config.decoder_config.register_dim != output_dim:
            raise ValueError(
                "the decoder cross-attends the SHIPPED register grid, so its "
                f"register_dim ({config.decoder_config.register_dim}) must equal "
                f"output_dim ({output_dim})"
            )

    # Single-source re-reads: every read re-queries the trunk's final layer through its
    # own norm (per_depth_read_proj). Asserted rather than assigned so that a change to
    # the BASE script fails loudly here instead of silently switching these arms to a
    # different bottleneck; the anchor is what makes a shallow trunk viable at all.
    assert encoder_config.register_interleave, (
        "early-read arms need the interleaved [read -> self] schedule"
    )
    assert encoder_config.register_per_depth_read_proj, (
        "early-read arms give each read its own lens on the shared source"
    )
    assert encoder_config.register_temporal_anchor is not None, (
        "a shallow trunk leaves the anchored read as the only path for temporal "
        "geometry; the base is expected to set register_temporal_anchor"
    )

    return config


def build_budget_dataloader_config(common: CommonComponents, token_budget: int):
    """The base's newsampling/psuniform dataloader at a different token budget.

    Wraps the base script's own ``build_dataloader_config`` (ndvi extra-decode +
    ``apply_new_sampling`` + ``apply_uniform_patch_sizes``) and then applies
    ``apply_shape_sweep``, which is documented to run last and to overwrite only
    ``token_budget`` and ``temporal_bias``. ``temporal_bias`` is held at the recipe's
    ``TEMPORAL_BIAS`` so the budget is the single axis that moves.

    WHAT A HIGHER BUDGET BUYS: ``max_sequence_length=12`` caps t, and at budget 3072 that
    cap is already saturated for hw<=9 (3*81*12 = 2916). So the budget does not lift the
    ceiling -- it raises the grid size at which a full-year sequence is still reachable
    (hw<=9 at 3072, hw<=13 at 6144), which is where most of the sampled grid distribution
    currently sits truncated (at 3072: hw=12 -> t<=7, hw=16 -> t<=4, hw>=24 -> t=1).

    NOT scaled with the budget: ``MIN_TOKENS_PER_INSTANCE`` (228) is an absolute floor on
    the low end. Raising the budget therefore widens the spread of per-instance cost
    rather than doubling its mean, so measured MACs at the LARGEST shape a budget allows
    are an upper bound on the average step, not the expected value.

    Duration is untouched by design (see ``apply_shape_sweep``): ``epochs(300)`` is a
    fixed number of instances, so every arm runs the same 662,700 steps on the same LR
    schedule and differs only in tokens per step.
    """
    return apply_shape_sweep(
        _base_build_dataloader_config(common),
        token_budget=token_budget,
        temporal_bias=TEMPORAL_BIAS,
    )


def build_budget_train_module_config(
    common: CommonComponents, rank_microbatch_size: int
):
    """The base's train module with the rank microbatch size overridden.

    ``rank_microbatch_size`` affects ONLY memory -- not tokens/step, not the loss, not the
    LR schedule -- so lowering it for a larger token budget costs nothing but a little
    throughput. The recipe's own note records that micro 32 @ budget 6144 OOM'd *before*
    the broadcastable key-mask fix, so 32 is the conservative setting to pair with 6144
    even though the early-read arms carry less activation memory than the base.
    """
    config = _base_build_train_module_config(common)
    config.rank_microbatch_size = rank_microbatch_size
    return config


def set_earlyread_loop_evals(trainer_config, module_path: str):
    """REPLACE the in-loop eval set with the three S1+S2+Landsat embedding probes.

    Unlike ``add_loop_eval_beaker_job`` (which the base script uses, merging fifty_cities
    into the shared catalog), this DISCARDS the catalog: these arms are judged on the
    embedding product -- the frozen ps=1 register grid, probed per-pixel -- and the
    catalog evals would inflate the eval job's runtime measuring axes this experiment does
    not turn on.

    That is a deliberate divergence from the A/B partner's eval set, with ONE task held in
    common on purpose: ``pastis_ws16_ps1_sentinel1_sentinel2_pretrain_export`` is carried
    over from the partner's own in-loop set, same registry entry and same interval, so the
    two runs share a live metric. Everything else the partner emits (fifty_cities, the
    S2-only PASTIS export) has no counterpart here. For the real comparison, sweep the
    partner's saved checkpoints under this task set rather than re-launching it -- that
    also gets you matched steps, which the in-loop path does not guarantee.

    The in-loop eval job reads this task dict back out of the training script's
    ``build_trainer_config`` (``checkpoint_sweep_evals.get_train_run_eval_tasks``), so
    replacing it here is what actually restricts what the eval jobs run -- there is no
    separate CLI or env-var filter to set.

    Requires the ``landsat_moNN`` layers on weka (``setup_extra_layers.py``, layer set
    ``landsat``) for the year-aligned windows. The Landsat input is optional in every
    model.yaml, so windows the Landsat materialize has not reached run WITHOUT it rather
    than failing -- which means a partially-materialized dataset degrades silently into
    an S1+S2 eval. Check coverage before reading a Landsat delta off these numbers.
    """
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    evaluator.tasks = dict(EARLYREAD_LOOP_EVAL_TASKS)
    evaluator.run_as_beaker_job = True
    evaluator.beaker_eval_module_path = module_path
    evaluator.beaker_eval_clusters = list(LOOP_EVAL_CLUSTERS)
    return trainer_config
