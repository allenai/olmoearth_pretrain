"""lin_sup768_w1 evaluated on the year-aligned early-read set, BOTH heads.

An eval-only shim (the sphere_unif0p1 shim pattern). ``build_model_config`` and every
data builder are imported unchanged from
``regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform``, so the
architecture matches its checkpoints exactly and they load; the ONLY difference is
``build_trainer_config``, which swaps the run's own eval set (the PASTIS /
fifty_cities proj tasks) for ``set_proj_earlyread_loop_evals`` -- the 6 year-aligned
S1+S2+Landsat probes on the d768 teacher plus their ``_proj128`` duplicates on the
shipped student, including the aeftrial_* balanced-trial metrics on the
ethiopia/descals kNN tasks.

WHY IT EXISTS. The ndvi x student-uniformity 2x2 (launched 2026-08-18) reports this
12-task set in-loop, and its (off, off) cell is the completed lin_sup768_w1 run --
deliberately not retrained. That run's own in-loop set shares only the PASTIS S1+S2
bridge task with the new arms, so without this sweep the 2x2 has no baseline row on
the tasks it is judged by. Pointing ``checkpoint_sweep_evals`` at this module (with
``OE_LOOP_EVAL_FROM_TRAIN_CONFIG`` set) scores the saved checkpoints on those 12
tasks at matched steps.

Not for training. ``run()`` is intentionally absent: this module exists to be read
by ``get_train_run_eval_tasks``, which calls ``build_trainer_config`` and takes the
evaluator's ``tasks`` (nothing else -- the beaker-job routing flags are ignored).
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (  # noqa: F401
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform import (  # noqa: F401
    build_dataloader_config,
    build_model_config,
    build_train_module_config,
)
from regbtl_v1_2_proj_common import set_proj_earlyread_loop_evals

from olmoearth_pretrain.internal.experiment import CommonComponents

logger = logging.getLogger(__name__)

MODULE_PATH = (
    "scripts/official/v1_2/regbtl_v1_2_proj128lin_sup768_w1_proj_earlyread_evals.py"
)


def build_trainer_config(common: CommonComponents):
    """The run's trainer config with ONLY the 12 early-read/proj128 eval tasks."""
    return set_proj_earlyread_loop_evals(
        _base_build_trainer_config(common), MODULE_PATH
    )
