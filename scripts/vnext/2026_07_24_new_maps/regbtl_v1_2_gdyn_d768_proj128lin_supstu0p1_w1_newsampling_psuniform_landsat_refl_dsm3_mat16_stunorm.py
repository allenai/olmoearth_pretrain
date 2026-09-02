"""d768 teacher + linear [128,64,32,16] student at 0.1x student supervision, new maps.

``scripts/official/v1_2/regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_
psuniform.py`` rebuilt on the new-maps base, with the full GLO30 DSM target. The
student wiring is unchanged from the official run:

* student architecture: **per-cell Linear(768, 128) on the detached registers**, with
  the first 64 dims a self-sufficient Matryoshka prefix, so one artifact ships both
  widths. Its input is detached, so the encoder and the primary bottleneck train
  exactly as the plain regsup_w1 arm does.
* supervision heads: **both widths** (``supervision_source="both"``), with the
  student's heads scaled to **0.1x** the register head's w1
  (``projection_supervision_weight_scale=0.1``). This is the one knob that separates
  this arm from its ``_sup768_`` sibling: there the student learns only from the
  distillation terms, here it also gets a weak direct supervision signal.
* regsup at ``base_weight=1.0`` (w1) -- the weight the proj program settled on,
  since the w1 teacher's ceiling is what the student inherits.

Three things move relative to the official run:

* maps: ``srtm`` -> ``glo30`` and ``wri_canopy_height_map`` -> ``meta_canopy_height``.
* radiometry: Landsat as TOA reflectance / brightness temperature with the matching
  reflectance-scale norm stats.
* the GLO30 target: elevation + slope (bands 0, 1) as a 2-channel L1 regression, plus
  aspect as a separate 2-channel L1 regression on the derived ``glo30_aspect``
  modality (``[sin, cos]`` of the bearing, flat pixels written out as MISSING_VALUE).
  Raw aspect degrees cannot be regressed directly -- circular target, and the -1 flat
  sentinel z-scores to almost exactly due north. See the d128 ``_dsm3`` sibling for
  the measured numbers and the DSM-not-DTM caveat.

No NDVI arm and no temporal anchor here: this run's point is the student, and its
comparison partners on the proj axis do not have them.

WHY, AND THE CAVEAT. ``supstu0p1`` on the OLD maps is the strongest student measured
so far -- at step 400k its d128 head reads africa trial kNN-5 0.9224 and plain KNN
0.9468, above both shipped arms and above its own teacher, which no other ladder
does. This arm ports that knob onto the new-maps lineage. Note that the three
ingredients it inherits from the sibling all measured as NULLS in the 2026-08-24
sweep: Landsat reflectance vs DN is +0.26 pts mean over six matched cells with signs
flipping inside each dataset, the GLO30 slope+aspect target is a wash-to-negative
against the elevation-only arm, and the new map set is indistinguishable from the
old. So the expected gain here is whatever student supervision itself buys, not a
sum of four effects.

WHY THIS ONE MATTERS: it ships 128 dims. The in-flight d768 new-maps reflectance arms
are diagnostic only; this is the shippable-width member of that family, and its
in-loop evals score the checkpoint at 768 / 128 / 64 every 40k steps.

MATRYOSHKA WIDTHS. ``register_projection_dims = [128, 64, 32, 16]`` instead of the
family default ``[128, 64]``: the student still runs at 128 (the model builds one
``Linear(768, max(dims))``) and 64/32/16 are self-sufficient prefixes of it, each
carrying its own distillation terms. The two narrow widths are the point of this
arm -- 32 and 16 dims are a 4x and 8x storage saving over the shipped 128, and
nothing in this program has ever measured whether the representation survives that
far down. The d64 evidence is not encouraging (its cosine distillation loss runs
~1.5x the d128 head's and the gap widens through training), so a plausible outcome
is that 32/16 degrade sharply and the answer is "128 is the floor".

The in-loop evals are deliberately LEFT at the family's 128/64 probes rather than
extended to four widths: identical in-loop config keeps this arm's curves directly
comparable to its sibling's, and a wider task list is what previously overflowed the
eval window and silently dropped tail metrics. 32 and 16 get measured by offline
checkpoint sweeps, which is where every number in this program is settled anyway.

Run name: ``regbtl_v1_2_gdyn_d768_proj128lin_supstu0p1_w1_psuniform_newmaps_refl_dsm3_mat16``
(79 chars, inside the in-loop eval callback's 94-char budget).

THE CHANGE, AND ONLY IT: ``register_projection_output_norm=True`` puts a
``LayerNorm`` on the linear student's output. Teacher, sampler, supervision,
distillation terms, maps, radiometry and the DSM targets are byte-identical to
``..._landsat_refl_dsm3_mat16``, so the pair is a one-flag contrast.

WHY. The primary bottleneck ends in a LayerNorm and so does the perceiver student,
which leaves the bare ``Linear`` as the only served representation with no norm at
its own width -- and nothing in the loss pins its scale, since the cosine term is
taken after a learned back-projection that absorbs any rescaling. On the shipped
d128 arm the measured consequence is mild (a d64 prefix of the export-normalized
vector has norm 0.86 +/- 0.03, so it under-fills the int8 quantizer rather than
clipping, and cosine consumers are invariant), so this tests whether normalizing
during TRAINING changes what the student learns. Priors are against a score change.

THE NORM IS AT THE FULL STUDENT WIDTH (128), NOT PER PREFIX, and on THIS run that
choice is load-bearing: the Matryoshka ladder here is [128, 64, 32, 16], and
``LN(z)[:d]`` is not ``LN(z[:d])`` for any d < 128, because the statistics are taken
over different dimension sets. A per-width norm would make each served width
something other than a slice of the widest one, and the eval's
``eval_projection_dim`` (a plain ``grid[..., :d]``) would score a vector no
truncating consumer ever sees. Slicing one full-width norm is what deployment reads.
Consequence to watch on this arm specifically: the norm of the d16 prefix is the
least constrained of the four, so if the flag hurts anywhere it should hurt there
first.

IN-LOOP EVALS: the AEF balanced trials (eight year-aligned datasets, kNN twins, so
the ``aeftrial_*`` metrics come for free) plus year-aligned PASTIS, all on unmasked
S1+S2+Landsat, at d128 and d64 -- ``set_proj_aeftrial_loop_evals``. The d768 teacher
and the d32/d16 rungs are not scored in-loop: 18 tasks is already double the proj
chain's job, hence the 80k interval. Score d32/d16 in a checkpoint sweep instead.

EVAL-ARM NAMING: do not give the eval arm a name with an existing arm as a PREFIX --
the CSV export merges by longest-prefix match and would fold it into its
unnormalized sibling.
"""

import logging

from base import build_common_components, build_visualize_config
from base import build_trainer_config as _base_build_trainer_config
from perceiver_common import (
    GLO30_ELEV_SLOPE_BAND_INDICES,
    SUPERVISION_BASE_WEIGHT_W1,
    apply_landsat_reflectance,
    apply_microbatch,
    apply_new_sampling,
    apply_uniform_patch_sizes,
    build_extra_decode_dataloader_config,
    build_extra_decode_dataset_config,
    build_extra_decode_train_module_config,
    build_proj_model_config,
    set_proj_aeftrial_loop_evals,
)

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

REGISTER_DIM = 768
# The dose under test: student supervision heads at 0.1x the register head's w1.
PROJECTION_SUPERVISION_SCALE = 0.1
# Overridden locally rather than in perceiver_common: PROJECTION_DIMS there is shared
# by every proj arm in this directory, and widening it would silently re-shape them.
MATRYOSHKA_DIMS = [128, 64, 32, 16]
# The aspect sin/cos target is derived in the dataset from the raw glo30 aspect band.
EXTRA_DECODE_MODALITIES = [Modality.GLO30_ASPECT.name]
# 40k, not the 20k default: this run's 8-task eval jobs take longer than 20k training
# steps, so consecutive jobs would overlap on one resumed W&B run and the overlapping
# writer's rows get silently dropped.
EMBEDDING_EVAL_INTERVAL_STEPS = 40000
MODULE_PATH = (
    "scripts/vnext/2026_07_24_new_maps/"
    "regbtl_v1_2_gdyn_d768_proj128lin_supstu0p1_w1_newsampling_psuniform"
    "_landsat_refl_dsm3_mat16_stunorm.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d768 teacher + linear student at 0.1x sup, Matryoshka [128, 64, 32, 16]."""
    config = build_proj_model_config(
        common,
        projection_type="linear",
        supervision_source="both",
        projection_supervision_weight_scale=PROJECTION_SUPERVISION_SCALE,
        register_dim=REGISTER_DIM,
        base_weight=SUPERVISION_BASE_WEIGHT_W1,
        glo30_bands=GLO30_ELEV_SLOPE_BAND_INDICES,
        include_glo30_aspect=True,
    )
    config.encoder_config.register_projection_dims = list(MATRYOSHKA_DIMS)
    config.encoder_config.register_projection_output_norm = True
    return config


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Reflectance H5 + norms, deriving the glo30 aspect sin/cos target."""
    return apply_landsat_reflectance(
        build_extra_decode_dataset_config(common, EXTRA_DECODE_MODALITIES)
    )


def build_dataloader_config(common: CommonComponents):
    """Extra-decode-aware newsampling dataloader at uniform patch sizes."""
    return apply_uniform_patch_sizes(
        apply_new_sampling(
            build_extra_decode_dataloader_config(common, EXTRA_DECODE_MODALITIES)
        )
    )


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """Extra-decode-aware 1fwd + fused AdamW train module at the newsampling micro."""
    return apply_microbatch(
        build_extra_decode_train_module_config(common, EXTRA_DECODE_MODALITIES)
    )


def build_trainer_config(common: CommonComponents):
    """New-maps base trainer + AEF trials and PASTIS on both student widths."""
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
