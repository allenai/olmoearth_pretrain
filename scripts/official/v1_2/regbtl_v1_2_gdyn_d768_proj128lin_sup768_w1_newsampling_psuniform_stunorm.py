"""d768 teacher + detached linear [128, 64] student (lin), supervision: sup768, w1, newsampling psuniform.

One of the 6 detached-student ("proj128") runs -- see ``regbtl_v1_2_proj_common``
for the full motivation and the variant matrix. The student is trained at 128d with
its first 64 dims as a self-sufficient Matryoshka prefix (per-prefix back-projection,
Gram term and, when enabled, supervision head), so one artifact ships both widths.
W1 arm: under the new sampler the w1 teacher leads w0p1 by ~1.4-1.9 mIoU on the
ps=1 PASTIS exports at 520k+, and the teacher's ceiling is what the student
inherits. Embedding evals run every 40k steps with the PASTIS tasks first --
the 20k cadence made eval jobs overlap on the shared W&B run and drop the
tail (S1+S2 projected) metrics. This arm:

* student architecture: **per-cell Linear(768, 128) on the detached registers**
* supervision heads: **d768 registers only (the current regsup recipe)**

Encoder-identical to ``regbtl_v1_2_gdyn_d768_regsup_w1_newsampling_psuniform``
(the student is invisible to the encoder), so its d768 register evals double as a
sanity anchor against that run.

Recipe: d768 wideread regbtl, regsup base_weight 1.0 (w1), decorrelated sampler at
UNIFORM patch sizes (the newsampling psuniform recipe). The student is trained by
the distillation terms (cosine + Gram) alone; its input is detached, so the
encoder and the primary bottleneck train exactly as regsup_w1 does.

THE CHANGE, AND ONLY IT: ``register_projection_output_norm=True`` puts a
``LayerNorm`` on the linear student's output. Everything else -- teacher, sampler,
supervision, distillation terms, data -- is byte-identical to ``regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform``, so the
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

MODULE_PATH = "scripts/official/v1_2/regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_stunorm.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d768 teacher + linear [128, 64] student, sup768, LayerNorm on the student."""
    config = build_proj_model_config(
        common,
        base_weight=SUPERVISION_BASE_WEIGHT_W1,
        projection_type="linear",
        supervision_source="registers",
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
