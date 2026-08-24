"""supstu0p1 with Matryoshka [128, 64, 32, 16]: the narrow-width probe on the old maps.

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
first), including aeftrial metrics -- ``set_proj_earlyread_loop_evals``, left at the
128 student probe so the curves stay comparable to the arm this forks.

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
    set_proj_earlyread_loop_evals,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

# The dose under test: student supervision heads at 0.1x the register head's w1.
PROJECTION_SUPERVISION_SCALE = 0.1
# Local override: regbtl_v1_2_proj_common.PROJECTION_DIMS is shared by every proj
# arm in v1_2, and widening it there would silently re-shape all of them.
MATRYOSHKA_DIMS = [128, 64, 32, 16]
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d768_proj128lin_supstu0p1_w1_newsampling_psuniform_mat16.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d768 teacher + lin student, supervision on both widths, student at 0.1x."""
    config = build_proj_model_config(
        common,
        base_weight=SUPERVISION_BASE_WEIGHT_W1,
        projection_type="linear",
        supervision_source="both",
        projection_supervision_weight_scale=PROJECTION_SUPERVISION_SCALE,
    )
    config.encoder_config.register_projection_dims = list(MATRYOSHKA_DIMS)
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
    """Base trainer + the year-aligned early-read evals on both heads."""
    return set_proj_earlyread_loop_evals(
        _base_build_trainer_config(common), MODULE_PATH
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
