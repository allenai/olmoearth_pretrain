"""d768 teacher + detached pcv student: within-scene Gram + student supervision at 0.01x (ndvi + tanchor) (arm: gramonly_supstu0p01_flatlr).

Why combine. The single-knob arms test two independent routes to one deficiency:
the distillation objective applies almost no pressure to discriminate CELLS within
a scene. The cosine term is per-cell but satisfiable by emitting the scene-mean
direction (register cells are spatially smooth and highly correlated), and ~98% of
the flat Gram's pairs are cross-scene -- easy near-orthogonal negatives the student
matches for free. That matters because a dense ps=1 probe measures exactly the
within-scene discrimination neither term supplies, and the perceiver student, which
must build its own spatial summary rather than linearly re-reading the teacher's,
is the one that degrades: it peaks at 240k and loses 2.5 mIoU by 440k while its
teacher stays flat, and the narrower d64 prefix loses more than d128.

``gramonly`` supplies that pressure RELATIONALLY: the flat Gram is switched off and
the whole relational budget goes to the block-diagonal term, where every pair
relates two cells of the same scene. Note the cosine term is untouched at 1.0, so
this is not "gram only" in the sense of dropping the pointwise term.

``supstu0p1`` supplies the same pressure DIRECTLY: supervision heads on the student
predicting per-pixel maps (worldcover, CDL, canopy height, worldcereal), so cell
(i, j) must encode what is at (i, j). A spatially collapsed student cannot produce
a varying map at all. Evidence it works: ``supboth_w1`` (the same heads at full
weight) never turns over through 400k where ``sup768`` peaks at 240k and falls --
though it pays for that stability, peaking 1.3 mIoU BELOW sup768's peak. 0.01x is a tenth of that again: it asks whether the
protection survives at a dose small enough to cost nothing early, or whether it
is simply too weak to act. The 0.1x sibling of this arm is the comparison.

Whether the two compose, or one subsumes the other, is not answerable from the
single-knob arms, which is what this run is for.

On the ndvi + tanchor recipe, which stacks a THIRD source of the same pressure: the
time-conditioned NDVI head puts a small MLP on ``[register_cell ; phi(day_of_year)]``
and asks each cell to predict its own NDVI at each observed timestep, so every cell
must store its own decodable temporal trajectory -- per-cell supervision on the
phenology axis that PASTIS crop segmentation actually turns on. ``tanchor``
anchors the register read at Jan 1 of the sample's first observation year, making
the read's temporal coordinate ~day-of-year (season-selective, year-invariant).
Note tanchor re-anchors the perceiver STUDENT's read as well as the teacher's,
since the student mirrors the primary's ``register_temporal_anchor``; that is fine
here because the whole run is anchored, but it is why anchored and unanchored arms
must not be mixed inside one run.


This is the FLAT-LR half of the pair. The student gets its own param group
on ConstantWithWarmup -- the encoder's warmup mirrored, then no decay -- while
the sibling without the ``_flatlr`` suffix inherits the encoder's 10x cosine
decay, as every other arm and the baseline do. Running both keeps the schedule
attributable instead of folded into the combination: the baseline student peaks
at 240k and falls as its LR is cut, so whether that decline is staleness (flat
LR should reverse it) or continued movement in a bad direction (flat LR should
worsen it) is exactly what the pair separates.

Baseline for comparison is the in-flight ``proj128pcv_sup768_w1_newsamp_psuniform``
(no student supervision, flat Gram only), with the single-knob ``gramonly`` and
``supstu0p1`` runs isolating each half.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_visualize_config,
)
from regbtl_v1_2_newsampling_common import (
    apply_microbatch,
    apply_new_sampling,
    apply_uniform_patch_sizes,
)
from regbtl_v1_2_proj_common import (
    STUDENT_ARMS,
    add_proj_loop_eval_beaker_job,
    apply_arm,
    build_arm_model_config,
)
from regbtl_v1_2_regsup_common import (
    build_extra_decode_dataloader_config,
    build_extra_decode_dataset_config,
    build_extra_decode_train_module_config,
)

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

ARM = STUDENT_ARMS["gramonly_supstu0p01_flatlr"]
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d768_proj128pcv_gramonly_supstu0p01_flatlr_ndvi_w1_tanchor_newsampling_psuniform.py"
)

# Anchor the register read at Jan 1 of the sample's first observation year.
REGISTER_TEMPORAL_ANCHOR = "year_start"
# NDVI is supervision-only: derived in the dataset from raw S2 B04/B08 and marked
# decode-only, so it reaches the supervision head without ever being a model input.
EXTRA_DECODE_MODALITIES = [Modality.NDVI.name]


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d768 teacher + detached perceiver student, supervision on both widths."""
    return build_arm_model_config(
        common,
        ARM,
        temporal_anchor=REGISTER_TEMPORAL_ANCHOR,
        include_ndvi=True,
    )


def build_dataset_config(common: CommonComponents):
    """Base dataset config, additionally deriving ndvi from the raw S2 bands."""
    return build_extra_decode_dataset_config(common, EXTRA_DECODE_MODALITIES)


def build_dataloader_config(common: CommonComponents):
    """Newsampling dataloader at uniform patch sizes."""
    return apply_uniform_patch_sizes(
        apply_new_sampling(
            build_extra_decode_dataloader_config(common, EXTRA_DECODE_MODALITIES)
        )
    )


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """1fwd + fused AdamW at the newsampling microbatch, plus this arm's knobs."""
    return apply_arm(
        apply_microbatch(
            build_extra_decode_train_module_config(common, EXTRA_DECODE_MODALITIES)
        ),
        ARM,
    )


def build_trainer_config(common: CommonComponents):
    """Base trainer + in-loop evals on the d768, 128d and 64d heads."""
    return add_proj_loop_eval_beaker_job(
        _base_build_trainer_config(common),
        MODULE_PATH,
        embedding_eval_interval_steps=40000,
    )


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
