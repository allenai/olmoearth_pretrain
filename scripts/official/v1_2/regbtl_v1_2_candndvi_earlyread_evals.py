"""cand_ndvi evaluated on the EARLY-READ six-task embedding set.

An eval-only shim. ``build_model_config`` and every data builder are imported unchanged
from ``regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform``, so the architecture matches
its checkpoints exactly and they load; the ONLY difference is
``build_trainer_config``, which swaps the run's own eval set for
``set_earlyread_loop_evals``.

WHY IT EXISTS. The early-read sweep replaced the shared catalog with six
embedding-product tasks (pastis/ethiopia/descals year-aligned at S1+S2+Landsat, plus the
pastis_ws16_ps1 bridge task), while cand_ndvi runs the catalog. Only the
bridge task overlaps, so the sweep currently has no baseline row on the tasks it is
actually judged by -- every comparison is arm-vs-arm. Pointing
``checkpoint_sweep_evals`` at this module (via ``full_eval_sweep --module_path``) puts
cand_ndvi on those six tasks at whatever steps are swept, which also
gives matched-step comparisons that the in-loop path cannot guarantee.

Not for training. ``run()`` is intentionally absent: this module exists to be read by
``get_train_run_eval_tasks``, which calls ``build_trainer_config`` and takes the
evaluator's ``tasks``.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_earlyread_common import set_year_aligned_only_loop_evals
from regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform import (  # noqa: F401
    build_dataloader_config,
    build_dataset_config,
    build_model_config,
    build_train_module_config,
)
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (  # noqa: F401
    build_common_components,
    build_visualize_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_2/regbtl_v1_2_candndvi_earlyread_evals.py"


def build_trainer_config(common: CommonComponents):
    """The run's trainer config with ONLY the year-aligned ethiopia + descals evals."""
    return set_year_aligned_only_loop_evals(
        _base_build_trainer_config(common), MODULE_PATH
    )
