"""Eval entry point for the v1.4 slot-DOY run with the v1.3 control's 32 in-loop eval tasks.

``v1_4_areal_doy.py`` (run ``agent-hollis-v14-areal-doy-667k-20261025``) trained without
in-loop evals (E5 pops ``downstream_evaluator``). For a like-for-like comparison with the
v1.3 control ``v1_3_rc_ld1_pixtgt_pix512`` re-evaluated after the eval-mask fix, this module
builds the model exactly as the DOY training script did (E5 common components: S2 + S1 +
maps, no Landsat; DOY time code) and takes the trainer config, i.e. the eval task list and
hyperparameters, from the control's training script ``rc_ld1_pixtgt_pix512.py``.

Use with ``olmoearth_pretrain/internal/checkpoint_sweep_evals.py`` and
``OE_LOOP_EVAL_FROM_TRAIN_CONFIG=1`` (optionally ``OE_LOOP_EVAL_TASK_SHARD=i/n``).

One deviation from the control: 30 of the 32 tasks feed S1 + S2 + Landsat, and this model
has no Landsat modality. The evaluator skips any task whose inputs the model does not all
support, so Landsat is dropped from each task's ``input_modalities`` (task names and every
other field unchanged): the model is scored on S1 + S2, the sensors it was trained on.
"""

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

from olmoearth_pretrain.internal.experiment import CommonComponents  # noqa: E402

MODULE_PATH = "scripts/official/v1_3/ablations/v1_4_areal_doy_evals.py"

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
