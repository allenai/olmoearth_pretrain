"""EARLY READ + SHARED K/V + DOUBLED BUDGET: 3-layer trunk, 35 blocks, tb 6144.

``regbtl_v1_2_earlyread_e3_l35_d128_ndvi_tanchor_newsampling_psuniform_sharedkv`` at
``token_budget=6144`` (``apply_shape_sweep``, ``temporal_bias`` held at the recipe's 2.75
so the budget is the only axis that moves). Its control is that 3072 arm, so the budget
effect is isolated; the depth axis is controlled by its l-partner at the same budget.

WHY THE BUDGET MOVES AT ALL: ``max_sequence_length=12`` caps t, and at budget 3072 that
cap is already saturated for hw<=9. Doubling does not lift the ceiling -- it raises the
grid size at which a full-year sequence is reachable, hw<=9 -> hw<=13, which is where most
of the sampled grid distribution currently sits truncated (at 3072: hw=12 -> t<=7,
hw=16 -> t<=4, hw>=24 -> t=1). The gain concentrates in the hw=10..16 band, where the
frozen ws16 ps=1 eval shape lives. hw=32 stays near single-shot either way (t=1 -> t=2).

WHY IT IS AFFORDABLE HERE: the shared K/V removes ~73% of each read block's FLOPs, so the
whole family sits at 0.28-0.53x the cand_ndvi frontier at budget 3072. The doubled budget
spends part of that headroom on data rather than banking it.

MICROBATCH: 32, the established setting for this budget (the unshared ``e3_l12`` tb6144
arm trains at it). Not lowered further -- shared K/V stores ONE copy of the context for
the whole read stack instead of one per read, which is precisely the term that made the
deep unshared arms OOM. 64 may well fit and would be worth trying once one of these has
shown a steady-state memory figure; microbatch affects only memory, not tokens/step, the
loss, or the LR schedule.

Duration is untouched: ``epochs(300)`` is a fixed instance count, so this runs the same
662,700 steps as every other arm and differs only in tokens per step.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_earlyread_common import (
    build_budget_dataloader_config,
    build_budget_train_module_config,
    build_earlyread_model_config,
    set_earlyread_loop_evals,
)
from regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform import (
    build_dataset_config,
)
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_visualize_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

TRUNK_DEPTH = 3
LATENT_DEPTH = 35
TOKEN_BUDGET = 6144
RANK_MICROBATCH_SIZE = 32
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_earlyread_e3_l35_d128_ndvi_tanchor_newsampling_psuniform_sharedkv_tb6144.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """3-layer trunk + 35 interleaved read/latent blocks over one shared K/V."""
    return build_earlyread_model_config(
        common,
        trunk_depth=TRUNK_DEPTH,
        latent_depth=LATENT_DEPTH,
        shared_read_kv=True,
    )


def build_dataloader_config(common: CommonComponents):
    """The ndvi newsampling/psuniform dataloader at token_budget 6144."""
    return build_budget_dataloader_config(common, TOKEN_BUDGET)


def build_train_module_config(common: CommonComponents):
    """ndvi-aware faster train module at the tb6144 family's microbatch (32)."""
    return build_budget_train_module_config(common, RANK_MICROBATCH_SIZE)


def build_trainer_config(common: CommonComponents):
    """Base trainer + ONLY the S1+S2+Landsat embedding evals (pastis/ethiopia/descals)."""
    return set_earlyread_loop_evals(_base_build_trainer_config(common), MODULE_PATH)


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
