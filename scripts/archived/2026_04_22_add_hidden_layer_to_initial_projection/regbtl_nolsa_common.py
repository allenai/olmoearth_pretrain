"""Shared pieces for the "no latent self-attention" (nolsa) register-bottleneck runs.

These runs A/B the spatial register bottleneck WITHOUT its latent self-attention blocks
(``register_latent_self_attn=False``): the registers are produced by the cross-attention
read(s) alone, with no register-to-register mixing. They build on the
``hidden1_supervision_register_bottleneck`` base (2D RoPE + bottleneck + low-weight
register supervision) and differ only in the read schedule (il / il_pdproj / mdr3).

Two things are baked in here rather than passed as CLI overrides, because the in-loop evals
run as SEPARATE Beaker jobs (``run_as_beaker_job=True``) that reconstruct the model from the
launching script's ``build_model_config``:

* the full architecture (dynamic grid, d768, RoPE coordinate scale 0.25, and
  ``register_latent_self_attn=False``) -- so the rebuilt eval-job model matches the trained
  checkpoint exactly, with no fragile nested-list overrides on the eval launch command; and
* the extra fifty-cities in-loop evals + the Beaker-job wiring (see
  :func:`add_loop_eval_beaker_job`), which each nolsa script applies in its
  ``build_trainer_config``.
"""

import logging
from dataclasses import replace

from hidden1_supervision_register_bottleneck import (
    build_model_config as _base_build_model_config,
)
from olmo_core.train.common import Duration

from olmoearth_pretrain.internal.all_evals import EVAL_TASKS as _ALL_EVAL_TASKS
from olmoearth_pretrain.internal.experiment import CommonComponents
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

# The regbtl_base10k_scale0.25 sweep passes --rope_coordinate_scale=0.25 on the CLI (the
# base script defaults to 1.0). Baked in so the eval-job model reconstruction matches the
# checkpoint without needing that override.
ROPE_COORDINATE_SCALE = 0.25
# d768: the register bottleneck width (gdyn_d768 frontier).
REGISTER_DIM = 768

# Clusters the in-loop eval Beaker jobs may run on (the first is used as the launch cluster).
LOOP_EVAL_CLUSTERS = ["ai2/jupiter", "ai2/ceres"]

# fifty_cities random-split S2 and S1+S2 segmentation probes, added to the in-loop evals now
# that they run in a separate (non-blocking) Beaker job. Reuse the canonical configs from
# all_evals -- so the names match the eval job's tasks_to_run filter -- but switch to a
# step-based interval that is a multiple of the checkpointer save_interval (5000), so a
# permanent checkpoint exists at each eval step for the eval job to load.
_FIFTY_CITIES_LOOP_EVAL_NAMES = (
    "fifty_cities_sentinel2",
    "fifty_cities_sentinel1_sentinel2",
)
FIFTY_CITIES_LOOP_EVAL_TASKS = {
    name: replace(_ALL_EVAL_TASKS[name], eval_interval=Duration.steps(20000))
    for name in _FIFTY_CITIES_LOOP_EVAL_NAMES
}


def build_nolsa_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Common nolsa model config: dynamic grid, d768, RoPE scale 0.25, latent self-attn OFF.

    The per-run read schedule (interleave / per-depth proj / multi-depth read layers /
    contrastive source) is layered on top by each individual nolsa script.
    """
    config = _base_build_model_config(common)
    for sub_config in (config.encoder_config, config.decoder_config):
        sub_config.rope_coordinate_scale = ROPE_COORDINATE_SCALE
        sub_config.register_dim = REGISTER_DIM
    encoder_config = config.encoder_config
    encoder_config.register_grid_size = 0  # dynamic single-latent grid (gdyn)
    encoder_config.register_latent_self_attn = False
    return config


def add_loop_eval_beaker_job(trainer_config, module_path: str):
    """Add the fifty-cities in-loop evals and route the in-loop evals through a Beaker job.

    ``run_as_beaker_job=True`` makes each due evaluator launch a Beaker job that evaluates
    the just-saved checkpoint (instead of blocking training); ``beaker_eval_module_path``
    points at the launching script so the eval job rebuilds the matching architecture.
    """
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    evaluator.tasks = {**evaluator.tasks, **FIFTY_CITIES_LOOP_EVAL_TASKS}
    evaluator.run_as_beaker_job = True
    evaluator.beaker_eval_module_path = module_path
    evaluator.beaker_eval_clusters = list(LOOP_EVAL_CLUSTERS)
    return trainer_config
