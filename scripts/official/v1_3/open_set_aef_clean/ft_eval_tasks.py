"""Fine-tuning evals for the AEF-clean open-set comparison (AEF supplemental + PASTIS).

The in-loop evals probe the frozen 128-d student. These tasks fine-tune the whole
encoder end to end (``EvalMode.FINETUNE``) with a linear head on the d768 registers:
the student reads a *detached* copy of the registers, so fine-tuning through it would
leave the encoder frozen.

Datasets are the in-loop ones, on the same S1 + S2 + Landsat year-aligned imagery:

* the eight ``AEF_SUPPLEMENTAL_YEAR_ALIGNED`` datasets, center-cropped to a 16 px window
  around the labeled pixel (``label_at_center_pixel``) and scored on balanced accuracy
  from the register at that pixel (``use_center_token``);
* ``pastis_year_aligned``, tiled into ``PASTIS_WINDOW_SIZE`` windows, scored on mIoU.

Tokens are 4 px patches (``patch_size=4``, the fine-tune default) with one Perceiver
latent per pixel (``latent_patch_size=1``, the pix512 eval convention), so the head
decodes one register per pixel.

Each task appears once per learning rate in ``FT_LRS`` (the stock fine-tune sweep's
LRs, one seed), keyed ``<task>_ft_<tag>`` (``lr1e4`` = 1e-4; no dots, since the
key is a path segment in CLI overrides); a Beaker job selects one key with
``--trainer.callbacks.downstream_evaluator.tasks_to_run``. The ``ft_eval_<model>.py``
modules next to this file are ``TRAIN_SCRIPT_PATH`` wrappers for
``olmoearth_pretrain/internal/checkpoint_sweep_evals.py`` with
``OE_LOOP_EVAL_FROM_TRAIN_CONFIG=1``.
"""

from dataclasses import replace
from typing import Any

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.datasets.normalize import NormMethod
from olmoearth_pretrain.evals.metrics import EvalMetric
from olmoearth_pretrain.internal.all_evals import AEF_SUPPLEMENTAL_YEAR_ALIGNED
from olmoearth_pretrain.nn.pooling import PoolingType
from olmoearth_pretrain.train.callbacks.evaluator_callback import (
    DownstreamTaskConfig,
    EvalMode,
)

# full_eval_sweep_finetune.FT_LRS, keyed by a dot-free tag.
FT_LRS = {"lr1e4": 1e-4, "lr5e4": 5e-4, "lr1e3": 1e-3}
FT_EPOCHS = 50
PATCH_SIZE = 4
LATENT_PATCH_SIZE = 1
PASTIS_WINDOW_SIZE = 32
PASTIS_BATCH_SIZE = 16
# Loading is I/O-latency bound (one worker reads ~1 window/s off weka): 24 workers
# trained us_trees at ~26 windows/s vs ~9 with 8, at ~7% host CPU. A 1-GPU Beaker
# job is allotted 22.5 CPUs.
NUM_WORKERS = 24

_MODALITIES = [
    Modality.SENTINEL1.name,
    Modality.SENTINEL2_L2A.name,
    Modality.LANDSAT.name,
]


def _aef_ft_task(dataset: str) -> DownstreamTaskConfig:
    """One AEF supplemental dataset as center-pixel fine-tuned classification."""
    return DownstreamTaskConfig(
        dataset=dataset,
        ft_batch_size=32,
        num_workers=NUM_WORKERS,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        input_modalities=_MODALITIES,
        epochs=FT_EPOCHS,
        eval_mode=EvalMode.FINETUNE,
        primary_metric=EvalMetric.BALANCED_ACCURACY,
        window_size=16,
        patch_size=PATCH_SIZE,
        latent_patch_size=LATENT_PATCH_SIZE,
        use_center_token=True,
        label_at_center_pixel=True,
    )


def _pastis_ft_task() -> DownstreamTaskConfig:
    """Year-aligned PASTIS as fine-tuned per-pixel segmentation."""
    return DownstreamTaskConfig(
        dataset="pastis_year_aligned",
        ft_batch_size=PASTIS_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        input_modalities=_MODALITIES,
        epochs=FT_EPOCHS,
        eval_mode=EvalMode.FINETUNE,
        primary_metric=EvalMetric.MIOU,
        window_size=PASTIS_WINDOW_SIZE,
        tile_samples=True,
        patch_size=PATCH_SIZE,
        latent_patch_size=LATENT_PATCH_SIZE,
    )


_BASE_TASKS = {
    "pastis_year_aligned": _pastis_ft_task(),
    **{
        name.removesuffix("_year_aligned"): _aef_ft_task(name)
        for name in AEF_SUPPLEMENTAL_YEAR_ALIGNED
    },
}

FT_TASKS = {
    f"{name}_ft_{tag}": replace(task, ft_lr=lr)
    for name, task in _BASE_TASKS.items()
    for tag, lr in FT_LRS.items()
}


def make_wrapper(train_mod: Any) -> dict:
    """The builders a ``ft_eval_<model>.py`` wrapper re-exports from its training script.

    The model, common components and train module come from the training script, so
    the checkpoint loads into the architecture it was trained with; only the
    evaluator's task set is replaced.
    """

    def build_trainer_config(common):
        config = train_mod.build_trainer_config(common)
        config.callbacks["downstream_evaluator"].tasks = dict(FT_TASKS)
        return config

    return {
        "build_common_components": train_mod.build_common_components,
        "build_model_config": train_mod.build_model_config,
        "build_train_module_config": train_mod.build_train_module_config,
        "build_trainer_config": build_trainer_config,
    }
