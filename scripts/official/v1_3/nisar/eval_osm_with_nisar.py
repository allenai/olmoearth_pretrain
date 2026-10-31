"""Eval-only module: nisar_osm LP tasks (NISAR slot + S1 slot) on a with_nisar checkpoint (card t-grl7q6).

Same model / common components as ``with_nisar.py`` so the checkpoint loads; ``build_trainer_config`` carries only
the nisar_osm_evals tasks. Launched by checkpoint_sweep_evals.py with OE_LOOP_EVAL_FROM_TRAIN_CONFIG=1.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from with_nisar import (  # noqa: E402, F401
    WITH_NISAR,
    build_model_config,
    build_train_module_config,
)
from with_nisar import build_trainer_config as _build_trainer_config  # noqa: E402
from nisar_common import build_common_components_fn  # noqa: E402
from nisar_osm_evals import nisar_osm_eval_tasks  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents  # noqa: E402

assert WITH_NISAR is True
build_common_components = build_common_components_fn(WITH_NISAR)


def build_trainer_config(common: CommonComponents):
    """The arm's trainer config with only the nisar_osm tasks."""
    trainer_config = _build_trainer_config(common)
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    evaluator.tasks = nisar_osm_eval_tasks(native_slot=True)
    return trainer_config
