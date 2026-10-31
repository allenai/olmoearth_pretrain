"""Eval-only module: nisar_osm LP tasks (NISAR through the S1 slot) on the v1.3 control (card t-grl7q6).

Same model / common components as ``control_pixtgt.py`` (``v1_3_rc_ld1_pixtgt_pix512``) so its checkpoint loads;
``build_trainer_config`` carries only the S1-slot nisar_osm_evals tasks (the control has no NISAR slot).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nisar"))

from control_pixtgt import (  # noqa: E402, F401
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_model_config,
    build_train_module_config,
    build_visualize_config,
)
from control_pixtgt import build_trainer_config as _build_trainer_config  # noqa: E402
from nisar_osm_evals import nisar_osm_eval_tasks  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents  # noqa: E402


def build_trainer_config(common: CommonComponents):
    """The control's trainer config with only the S1-slot nisar_osm tasks."""
    trainer_config = _build_trainer_config(common)
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    evaluator.tasks = nisar_osm_eval_tasks(native_slot=False)
    return trainer_config
