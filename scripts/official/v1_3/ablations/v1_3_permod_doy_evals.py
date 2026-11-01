"""Eval entry point for the v1.3 per-modality-DOY run with the v1.3 control's 32 in-loop eval tasks.

``v1_3_permod_doy.py`` (run ``agent-hollis-v13-permod-doy-20261026``) trained without
in-loop evals (E5 pops ``downstream_evaluator``). Same construction as
``v1_4_areal_doy_evals.py`` (the slot-DOY run's re-eval, branch
favyen/20261031-v14-slot-doy-evals), so the two runs are scored on identical tasks: the
model is built exactly as the training script did (E5 common components: S2 + S1 + maps,
no Landsat; ``v1_4_areal_doy.build_model_config``, DOY time code) and the trainer config,
i.e. the eval task list and hyperparameters, comes from the control's training script
``rc_ld1_pixtgt_pix512.py``, with Landsat dropped from each task's ``input_modalities``
(the model has no Landsat; the evaluator would otherwise skip 30 of 32 tasks).

Per-modality dates: the run was pretrained with each modality token stamped with its own
capture date (packed ``(T, 12)`` timestamps). Importing this module sets
``OE_EVAL_PER_MODALITY_TIMESTAMPS=1``, so the rslearn eval datasets (the 30 *_year_aligned
tasks) pack each modality's own acquisition dates (S1 and S2 are separate items per
monthly layer) next to the stock shared stamp. ``pastis`` and ``m-eurosat`` are S2-only
non-rslearn datasets and keep (T, 3) stamps (S2 = shared stamp there, so nothing changes).

Use with ``olmoearth_pretrain/internal/checkpoint_sweep_evals.py`` and
``OE_LOOP_EVAL_FROM_TRAIN_CONFIG=1`` (optionally ``OE_LOOP_EVAL_TASK_SHARD=i/n``).
"""

import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import e5_allcap_common as e5  # noqa: E402
from pure_perceiver_mix import build_mix_trainer_config  # noqa: E402
from v1_4_areal_doy import (  # noqa: E402, F401
    build_model_config,
    build_train_module_config,
)

from olmoearth_pretrain.evals.datasets.rslearn_dataset import (  # noqa: E402
    PER_MODALITY_TIMESTAMPS_ENV,
)
from olmoearth_pretrain.internal.experiment import CommonComponents  # noqa: E402

os.environ[PER_MODALITY_TIMESTAMPS_ENV] = "1"

MODULE_PATH = "scripts/official/v1_3/ablations/v1_3_permod_doy_evals.py"

build_common_components = e5.build_common_components


def build_trainer_config(common: CommonComponents):  # noqa: ANN201
    """The control's (``rc_ld1_pixtgt_pix512``) in-loop evals, minus inputs this model lacks."""
    config = build_mix_trainer_config(
        common, MODULE_PATH, ps4_student_evals=True, ps2_student_evals=True
    )
    evaluator = config.callbacks["downstream_evaluator"]
    supported = set(common.training_modalities)
    evaluator.tasks = {
        name: replace(
            task,
            input_modalities=[m for m in task.input_modalities if m in supported],
        )
        for name, task in evaluator.tasks.items()
    }
    return config
