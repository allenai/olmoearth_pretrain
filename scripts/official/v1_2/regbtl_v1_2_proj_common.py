"""Shared pieces for the detached 768->128 register-projection ("student") runs.

Motivation (2026-07-30 embedding-evals synthesis): the d768 register runs beat every
d128 arm by ~6 mIoU on the frozen ps=1 PASTIS probes (0.6447 vs 0.5800 at ws16
S1+S2; Tessera v2 large = 0.5938) while d128 matches d768 on the center-pixel
classification tasks -- i.e. training the narrow width NATIVELY under the pretext
loss is the bottleneck, not the 128-dim budget itself. Tessera v2's playbook (and
SEED/CompRess) says the fix is distillation from a wide teacher. These runs amortize
that distillation into pretraining: the d768 bottleneck trains exactly as before
(the teacher), and a DETACHED low-dim student is trained alongside it against the
improving teacher. Stop-gradient means the encoder run is unchanged -- variant
``sup768`` is encoder-identical to ``regbtl_v1_2_gdyn_d768_regsup_w1_newsampling_
psuniform`` -- and every checkpoint ships a 768d head plus a 128d student whose
first 64 dims are a self-sufficient Matryoshka prefix, all evaluated side by side
in-loop.

Two student architectures (``register_projection_type``):

* ``lin`` -- per-cell ``Linear(768, 128)`` on the detached registers: is the
  teacher's information linearly readable per cell at the low width?
* ``pcv`` -- a second wideread Perceiver bottleneck at d128 re-reading the detached
  final-layer tokens (the deployed d128 architecture, trained by distillation
  instead of the pretext loss): was the d128 architecture ever the problem, or only
  its native training signal?

Three supervision placements (``supervision_source``), crossing the student axis
for 6 runs total:

* ``sup768`` (``"registers"``) -- heads on the d768 registers only (the current
  regsup recipe); the student trains purely by distillation.
* ``supboth`` (``"both"``)     -- separate heads on both widths.
* ``sup128`` (``"projection"``) -- heads on the detached student only: the encoder
  gets NO supervision gradient, isolating whether regsup helps by shaping the
  encoder or by shaping the output space.

The student is always trained with the distillation terms (cosine via a learned
back-projection + Gram/relational matching, ``LatentMIMTrainModule`` defaults).

Recipe otherwise matches the d768 w1 newsampling psuniform run: wideread regbtl at
register_dim 768, regsup base_weight 1.0, decorrelated sampler at uniform patch
sizes. The architecture is baked into each run script (the in-loop Beaker eval jobs
rebuild the model from the launching module's ``build_model_config``).
"""

import logging
from dataclasses import replace

from regbtl_v1_2_common import (
    FIFTY_CITIES_LOOP_EVAL_TASKS,
    LOOP_EVAL_CLUSTERS,
    PASTIS_EMBEDDING_LOOP_EVAL_TASKS,
)
from regbtl_v1_2_faster_common import build_wideread_regbtl_model_config
from regbtl_v1_2_regsup_common import add_register_supervision

from olmoearth_pretrain.internal.experiment import CommonComponents
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

# The teacher (primary bottleneck) width and the student (shipped embedding) widths:
# the student runs at 128 and its first 64 dims are trained as a self-sufficient
# Matryoshka prefix (own back-projection / Gram term / supervision head), so one
# stored artifact serves both widths by truncation (Tessera-v2 per-prefix heads).
REGISTER_DIM = 768
PROJECTION_DIMS = [128, 64]
# w1 everywhere heads attach, matching the best d768 PASTIS run (regsup w1) so the
# sup768 variant is encoder-identical to it.
SUPERVISION_BASE_WEIGHT_W1 = 1.0

# The PASTIS ws16/ps1 embedding evals duplicated onto the projected head at each
# Matryoshka width: the ``_proj{d}`` tasks probe (a prefix of)
# ``projected_registers`` instead of the register grid, so every in-loop eval step
# scores the same checkpoint at all three widths (768 / 128 / 64).
PASTIS_PROJ_EMBEDDING_LOOP_EVAL_TASKS = {
    f"{name}_proj{dim}": replace(
        task, eval_on_projected_registers=True, eval_projection_dim=dim
    )
    for name, task in PASTIS_EMBEDDING_LOOP_EVAL_TASKS.items()
    for dim in PROJECTION_DIMS
}


def build_proj_model_config(
    common: CommonComponents,
    *,
    projection_type: str,
    supervision_source: str,
) -> LatentMIMConfig:
    """d768 wideread regbtl + regsup(w1) + a detached [128, 64] Matryoshka student.

    Args:
        common: The common experiment components.
        projection_type: ``"linear"`` or ``"perceiver"`` (the student architecture).
        supervision_source: ``"registers"`` (sup768), ``"both"`` (supboth) or
            ``"projection"`` (sup128) -- where the supervision heads attach.
    """
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    config = add_register_supervision(
        config, include_latlon=False, base_weight=SUPERVISION_BASE_WEIGHT_W1
    )
    config.encoder_config.register_projection_dims = list(PROJECTION_DIMS)
    config.encoder_config.register_projection_type = projection_type
    config.supervision_source = supervision_source
    return config


def add_proj_loop_eval_beaker_job(trainer_config, module_path: str):
    """In-loop evals on BOTH heads, routed through Beaker.

    The standard fifty_cities + PASTIS embedding evals probe the d768 registers as
    usual; the ``_proj128`` / ``_proj64`` duplicates probe the detached student (full
    width / 64d Matryoshka prefix), so the distillation quality is tracked per
    checkpoint at every shipped width without a separate eval launch.
    """
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    evaluator.tasks = {
        **evaluator.tasks,
        **FIFTY_CITIES_LOOP_EVAL_TASKS,
        **PASTIS_EMBEDDING_LOOP_EVAL_TASKS,
        **PASTIS_PROJ_EMBEDDING_LOOP_EVAL_TASKS,
    }
    evaluator.run_as_beaker_job = True
    evaluator.beaker_eval_module_path = module_path
    evaluator.beaker_eval_clusters = list(LOOP_EVAL_CLUSTERS)
    return trainer_config
