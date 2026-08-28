"""d768 teacher + lin student with STUDENT supervision at 0.1x (supstu0p1), w1.

The dose-response test motivated by the 2026-08-19 supboth backfill:
``lin_supboth_w1`` (student supervision at FULL weight) is the only arm ever to
flip Descals kNN against AEF under the balanced-trial protocol (0.727-0.737 vs
0.712-0.722, both inputs, both k) -- but at full dose it overpays, tying
``lin_sup768_w1`` on trial wins (27/42) at a worse mean (-0.64 pts: canada
coarse kNN/ridge, us_trees ridge, ~1 pt lcmap/glance, -2 pts PASTIS LP). This
arm scales the student's supervision heads to 0.1x of the register head's w1
(``supervision_source="both"``, ``projection_supervision_weight_scale=0.1``) to
test whether the Descals gain is dose-independent while the broad tax scales
with dose -- the pattern the pcv supstu arms hinted at. If so, this keeps
lin_sup768's cells and adds Descals. The supstu grid cell was only ever run for
the pcv student (the gram sweep); this is the missing lin cell.

Teacher and distillation objective are byte-identical to
``regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform`` (w1,
newsampling psuniform, default cosine+Gram student losses); the ONLY change is
the student-side supervision heads at 0.1x. One of four supstu arms launched
2026-08-20 (this, +ndvi, +ndvi+cloudmask0p5, +ndvi+cloudmask0p5+stuunif0p1).

In-loop evals: the year-aligned early-read set on both heads (student tasks
first), including aeftrial metrics -- ``set_proj_earlyread_loop_evals``.

THE CHANGE, AND ONLY IT: ``register_projection_output_norm=True`` puts a
``LayerNorm`` on the linear student's output. Everything else -- teacher, sampler,
supervision, distillation terms, data -- is byte-identical to ``regbtl_v1_2_gdyn_d768_proj128lin_supstu0p1_w1_newsampling_psuniform``, so the
pair is a one-flag contrast.

WHY. The primary bottleneck ends in a LayerNorm and so does the perceiver student
(its ``SpatialRegisterBottleneck`` does), which leaves the bare ``Linear`` as the
only served representation with no norm at its own width. Nothing in the loss pins
its scale either: the cosine term is taken after a learned back-projection, which
absorbs any rescaling. Measured on the shipped d128 arm, the consequence is mild --
a d64 prefix of the (export-normalized) vector has norm 0.86 +/- 0.03, so it
under-fills the int8 quantizer rather than clipping, and every cosine consumer is
invariant to it -- so this is a test of whether normalizing during TRAINING changes
what the student learns, not a fix for a measured failure. The priors are against a
score change: eval-side normalization measured a null for kNN twice, and the
hard-pinned-radius sphere arm was dropped at break-even.

THE NORM IS AT THE FULL STUDENT WIDTH, NOT PER PREFIX. ``LN(z)[:64]`` is not
``LN(z[:64])`` -- the statistics are taken over different dimension sets -- so a
per-width norm would make the served d64 something other than a slice of the served
d128, and the eval's ``eval_projection_dim`` (a plain ``grid[..., :64]``) would then
score a vector no truncating consumer ever sees. Slicing a full-width norm is what
deployment actually reads, so that is what is trained and what is measured.

IN-LOOP EVALS: the AEF balanced trials (eight year-aligned datasets, kNN twins, so
the ``aeftrial_{ridge,knn5,knn20}`` metrics come for free) plus year-aligned
PASTIS, all on unmasked S1+S2+Landsat, at BOTH student widths (d128 and d64) --
``set_proj_aeftrial_loop_evals``. The d768 teacher is not scored: these runs are
judged on the shipped embedding. 18 tasks, so the interval is 80k rather than the
proj chain's 40k.

EVAL-ARM NAMING, for whoever sweeps these later: do NOT name the eval arm
``lin_sup768_w1_..._norm`` or anything else with an existing arm as a PREFIX -- the
CSV export merges bin runs to arms by longest-prefix match and would silently fold
this into its unnormalized sibling. Use something mutually non-prefixing, e.g.
``linnorm_sup768_w1``.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_faster_common import build_faster_train_module_config
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_dataloader_config as _base_build_dataloader_config,
)
from regbtl_v1_2_newsampling_common import (
    apply_microbatch,
    apply_new_sampling,
    apply_uniform_patch_sizes,
)
from regbtl_v1_2_proj_common import (
    SUPERVISION_BASE_WEIGHT_W1,
    build_proj_model_config,
    set_proj_aeftrial_loop_evals,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

# The dose under test: student supervision heads at 0.1x the register head's w1.
PROJECTION_SUPERVISION_SCALE = 0.1
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d768_proj128lin_supstu0p1_w1_newsampling_psuniform_stunorm.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d768 teacher + lin student, supboth at 0.1x, LayerNorm on the student."""
    config = build_proj_model_config(
        common,
        base_weight=SUPERVISION_BASE_WEIGHT_W1,
        projection_type="linear",
        supervision_source="both",
        projection_supervision_weight_scale=PROJECTION_SUPERVISION_SCALE,
    )
    config.encoder_config.register_projection_output_norm = True
    return config


def build_dataloader_config(common: CommonComponents):
    """Newsampling dataloader at uniform patch sizes."""
    return apply_uniform_patch_sizes(
        apply_new_sampling(_base_build_dataloader_config(common))
    )


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """1fwd + fused AdamW train module at the newsampling microbatch size."""
    return apply_microbatch(build_faster_train_module_config(common))


def build_trainer_config(common: CommonComponents):
    """Base trainer + AEF trials and PASTIS on both student widths."""
    return set_proj_aeftrial_loop_evals(_base_build_trainer_config(common), MODULE_PATH)


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
