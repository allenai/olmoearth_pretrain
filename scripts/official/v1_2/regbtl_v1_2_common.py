"""Shared pieces for the v1.2 register-bottleneck (``_il``) runs.

These build the Perceiver-style spatial register bottleneck on top of the v1.2 base
(``base.py``: hidden patch-embed projection + mixed 3D RoPE). The register grid is a
purely spatial summary with no temporal axis, so the RoPE is deliberately asymmetric:

* the ENCODER self-attention keeps v1.2's 3D mixed RoPE (``rope_3d_mixed``) over
  ``(t, row, col)`` -- the patch encoder is unchanged from v1.2;
* the register bottleneck reads / mixes the patch tokens with 2D ``(row, col)`` RoPE
  (the temporal coordinate is sliced off inside ``Encoder.forward``); and
* the DECODER cross-attends the mask tokens to the 2D register grid, so it runs with 2D
  RoPE (``"rope"``). The mask tokens recover their timestep from the additive slot-index
  + month encodings, which turn back on automatically outside 3D RoPE (see
  ``CompositeEncodings``). This is the same temporal-APE path the original 2D ``_il``
  sweeps used, and it is required because the registers have no time coordinate to
  rotate the decoder's keys by.

Fixed across the sweep (the ``gdyn_d768_il_pdproj`` frontier):

* ``gdyn`` -- dynamic single-latent grid (``register_grid_size=0``) cloned to match the
  patch grid at forward time;
* ``il``   -- interleaved reads (``[read -> self] x register_latent_depth``);
* ``pdproj`` -- per-depth read projections (each read block gets its own norm + K/V proj);
* ``register_dim=768`` and ``register_latent_depth=4``.

The sweep varies two axes per script: the instance contrastive loss (``ic`` / ``noic``,
in the train-module builder) and the bottleneck's latent self-attention (``lsa`` /
``nolsa``, via ``register_latent_self_attn``).

The architecture is baked into :func:`build_regbtl_model_config` rather than passed as CLI
overrides, because the in-loop evals run as SEPARATE Beaker jobs (``run_as_beaker_job=
True``) that reconstruct the model from the launching script's ``build_model_config`` -- so
the rebuilt eval-job model must match the trained checkpoint exactly.
"""

import logging
from dataclasses import replace

from base import build_model_config as _base_build_model_config
from olmo_core.train.common import Duration

from olmoearth_pretrain.internal.all_evals import (
    EMBEDDING_EVAL_TASKS as _EMBEDDING_EVAL_TASKS,
)
from olmoearth_pretrain.internal.all_evals import EVAL_TASKS as _ALL_EVAL_TASKS
from olmoearth_pretrain.internal.experiment import CommonComponents
from olmoearth_pretrain.nn.encodings import PositionEncoding
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

# The register bottleneck width (the gdyn_d768 frontier).
REGISTER_DIM = 768
# Latent-transformer depth over the register grid. With interleave this is also the number
# of cross-attention reads, giving the ``[read -> self] x4`` schedule (nolsa keeps the 4
# reads and drops only the self-attention blocks).
REGISTER_LATENT_DEPTH = 4
# The decoder cross-attends the (spatial) register grid, so it runs with 2D RoPE while the
# encoder keeps 3D. "rope" == 2D axial RoPE.
DECODER_POSITION_ENCODING = PositionEncoding.AXIAL_2D_ROPE.value

# Clusters the in-loop eval Beaker jobs may run on.
LOOP_EVAL_CLUSTERS = ["ai2/jupiter", "ai2/ceres"]

# fifty_cities random-split S2 and S1+S2 segmentation probes, added to the in-loop evals now
# that they run in a separate (non-blocking) Beaker job. Reuse the canonical configs from
# all_evals -- so the names match the eval job's tasks_to_run filter -- but on a step-based
# interval that is a multiple of the checkpointer save_interval (5000), so a permanent
# checkpoint exists at each eval step for the eval job to load.
_FIFTY_CITIES_LOOP_EVAL_NAMES = (
    "fifty_cities_sentinel2",
    "fifty_cities_sentinel1_sentinel2",
)
FIFTY_CITIES_LOOP_EVAL_TASKS = {
    name: replace(_ALL_EVAL_TASKS[name], eval_interval=Duration.steps(20000))
    for name in _FIFTY_CITIES_LOOP_EVAL_NAMES
}

# PASTIS per-pixel (patch_size=1, window_size=16) embedding evals on the pastis_rslearn
# pretraining-mirror export -- the deployment scenario for these models (frozen ps=1
# embeddings probed for segmentation, the way Tessera/AEF are compared). The in-loop eval
# job reconstructs these task CONFIGS from this module's build_trainer_config (see
# checkpoint_sweep_evals.get_train_run_eval_tasks), so they need no EMBEDDING_EVALS gate --
# only the pastis_rslearn dataset must be materialized where the eval job can read it.
_PASTIS_EMBEDDING_LOOP_EVAL_NAMES = (
    "pastis_ws16_ps1_sentinel2_pretrain_export",
    "pastis_ws16_ps1_sentinel1_sentinel2_pretrain_export",
)
PASTIS_EMBEDDING_LOOP_EVAL_TASKS = {
    name: replace(_EMBEDDING_EVAL_TASKS[name], eval_interval=Duration.steps(20000))
    for name in _PASTIS_EMBEDDING_LOOP_EVAL_NAMES
}


def build_regbtl_model_config(
    common: CommonComponents,
    *,
    latent_self_attn: bool,
    register_dim: int = REGISTER_DIM,
) -> LatentMIMConfig:
    """v1.2 base + spatial register bottleneck: ``gdyn`` + ``il`` + ``pdproj``.

    The encoder keeps v1.2's 3D mixed RoPE; the bottleneck reads spatially (2D). The
    decoder is switched to 2D RoPE so its mask tokens cross-attend the register grid.
    ``latent_self_attn`` toggles the bottleneck's latent self-attention (lsa / nolsa);
    ``register_dim`` sets the bottleneck width (d768 is the original frontier).
    """
    config = _base_build_model_config(common)
    encoder_config = config.encoder_config
    decoder_config = config.decoder_config

    for sub_config in (encoder_config, decoder_config):
        sub_config.use_register_bottleneck = True
        sub_config.register_dim = register_dim

    # gdyn: dynamic single-latent grid that matches the patch grid at forward time.
    encoder_config.register_grid_size = 0
    # il: interleave reads with the latent transformer ([read -> self] per layer).
    encoder_config.register_interleave = True
    # pdproj: each read block gets its own input norm + K/V projection.
    encoder_config.register_per_depth_read_proj = True
    encoder_config.register_latent_depth = REGISTER_LATENT_DEPTH
    encoder_config.register_latent_self_attn = latent_self_attn

    # The decoder cross-attends the spatial (2D) register grid; the encoder stays 3D.
    decoder_config.position_encoding = DECODER_POSITION_ENCODING

    return config


def add_loop_eval_beaker_job(trainer_config, module_path: str):
    """Add the fifty_cities + PASTIS embedding in-loop evals and route them through Beaker.

    ``run_as_beaker_job=True`` makes each due evaluator launch a Beaker job that evaluates
    the just-saved checkpoint (instead of blocking training); ``beaker_eval_module_path``
    points at the launching script so the eval job rebuilds the matching architecture.
    """
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    evaluator.tasks = {
        **evaluator.tasks,
        **FIFTY_CITIES_LOOP_EVAL_TASKS,
        **PASTIS_EMBEDDING_LOOP_EVAL_TASKS,
    }
    evaluator.run_as_beaker_job = True
    evaluator.beaker_eval_module_path = module_path
    evaluator.beaker_eval_clusters = list(LOOP_EVAL_CLUSTERS)
    return trainer_config
