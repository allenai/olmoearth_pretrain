"""Eval-only module: NISAR land-cover LP tasks on a no_nisar checkpoint (card t-7xgg82).

Same model / common components as ``no_nisar.py`` (so the checkpoint loads), but
``build_trainer_config`` carries only the nisar_land_evals tasks. Launched by
checkpoint_sweep_evals.py with OE_LOOP_EVAL_FROM_TRAIN_CONFIG=1, which reads the tasks
from this config (see launch_nisar_land_evals.sh in the session folder).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from no_nisar import (  # noqa: E402, F401
    WITH_NISAR,
    build_model_config,
    build_train_module_config,
)
from no_nisar import build_trainer_config as _build_trainer_config  # noqa: E402
from nisar_common import build_common_components_fn  # noqa: E402
from nisar_land_evals import nisar_land_eval_tasks  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents  # noqa: E402

assert WITH_NISAR is False
build_common_components = build_common_components_fn(WITH_NISAR)


def build_trainer_config(common: CommonComponents):
    """The arm's trainer config with only the NISAR land-cover tasks."""
    trainer_config = _build_trainer_config(common)
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    evaluator.tasks = nisar_land_eval_tasks(WITH_NISAR)
    return trainer_config
