"""Every frozen-embedding eval, on the v1.3 Base checkpoints, read from the 128-d student.

Module for ``olmoearth_pretrain/internal/checkpoint_sweep_evals.py`` (via
``TRAIN_SCRIPT_PATH``) that evaluates the v1.3 Base training run
(``gabrielt/regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform_stunorm_mlpgram1``,
shipped as OlmoEarth-v1_3-Base) at a grid of checkpoints, to study how every eval
evolves over training: which are noisy, and which correlate.

The model, train module and launch config are the release recipe
(``scripts/official/v1_3/base.py``). ``build_trainer_config`` exists only so the
harness can read the task set from it (``OE_LOOP_EVAL_FROM_TRAIN_CONFIG=1``); each
Beaker job then picks its bundle with ``downstream_evaluator.tasks_to_run``.

The task set is the union of every frozen-embedding catalog (``EVAL_TASKS``,
``EMBEDDING_EVAL_TASKS``, ``EMBED_DIAG_TASKS``), the catalog tasks that are commented
out or were removed (``REVIVED_TASKS``), and evals ported from unmerged branches
(``PORTED_TASKS``). Every task probes the full 128-d student (the shipped
embedding) and keeps its own hyperparameters; ``EVAL_TASKS`` additionally get
``norm_stats_from_pretrained=True``, as ``full_eval_sweep.py`` passes for OlmoEarth
checkpoint sweeps. Fine-tuning tasks are excluded: they train the encoder, so there
is no fixed embedding to read.
"""

import sys
from dataclasses import replace
from pathlib import Path

# The v1.3 recipe imports the v1.2 config from scripts/official.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "official"))

from olmo_core.train.common import Duration  # noqa: E402
from olmo_core.train.config import TrainerConfig  # noqa: E402
from v1_3.base import (  # noqa: E402, F401
    build_common_components,
    build_model_config,
    build_train_module_config,
)

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.evals.datasets.normalize import NormMethod  # noqa: E402
from olmoearth_pretrain.evals.metrics import EvalMetric  # noqa: E402
from olmoearth_pretrain.internal.all_evals import (  # noqa: E402
    EMBED_DIAG_TASKS,
    EMBEDDING_EVAL_TASKS,
    EVAL_TASKS,
    _pastis_ps1_task,
)
from olmoearth_pretrain.internal.experiment import CommonComponents  # noqa: E402
from olmoearth_pretrain.nn.pooling import PoolingType  # noqa: E402
from olmoearth_pretrain.train.callbacks import (  # noqa: E402
    DownstreamEvaluatorCallbackConfig,
)
from olmoearth_pretrain.train.callbacks.evaluator_callback import (  # noqa: E402
    DownstreamTaskConfig,
    EvalMode,
)

# Catalog tasks disabled on main, restored with their last committed configs.
REVIVED_TASKS = {
    # Commented out in all_evals.py ("Remove failing eval", a77e4d2f9).
    "burnrisk_8d_nbac": DownstreamTaskConfig(
        dataset="burnrisk_8d_nbac",
        embedding_batch_size=32,
        probe_batch_size=16,
        patch_size=5,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.0001,
        eval_interval=Duration.epochs(10),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        use_dice_loss=True,
        primary_metric=EvalMetric.CLASS_F1,
        primary_metric_class=1,
    ),
    # Commented out in all_evals.py for OOMs (4bdaeb598).
    "oil_spill_detection": DownstreamTaskConfig(
        dataset="oil_spill_detection",
        embedding_batch_size=128,
        probe_batch_size=8,
        num_workers=8,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.01,
        eval_interval=Duration.epochs(10),
        input_modalities=[Modality.SENTINEL1.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
    ),
    # Commented out in all_evals.py next to mapbiomas_3k_dense.
    "mapbiomas_3k_sparse": DownstreamTaskConfig(
        dataset="mapbiomas_3k_sparse",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=8,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.0001,
        eval_interval=Duration.epochs(10),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MACRO_F1,
    ),
    # Removed in 63e0d1e71 in favor of burnrisk_8d_nbac; still in the registry.
    "canada_wildfire_sat_eval_split": DownstreamTaskConfig(
        dataset="canada_wildfire_sat_eval_split",
        embedding_batch_size=32,
        probe_batch_size=16,
        patch_size=5,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.1,
        eval_interval=Duration.epochs(10),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        use_dice_loss=True,
        primary_metric=EvalMetric.CLASS_F1,
        primary_metric_class=1,
    ),
    # Removed in c2430a53c; the dataset config is still on main.
    "breizhcrops": DownstreamTaskConfig(
        dataset="breizhcrops",
        embedding_batch_size=128,
        probe_batch_size=128,
        num_workers=0,
        pooling_type=PoolingType.MAX,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.epochs(50),
        patch_size=1,
        eval_mode=EvalMode.LINEAR_PROBE,
        probe_lr=0.1,
        epochs=50,
    ),
}

# Evals that live only on unmerged branches.
PORTED_TASKS = {
    # SwissCrop25 LOYO fold S5, 64x64 S2 monthly mosaics (joer/swisscrop25-eval, 407d67a3).
    "swisscrop_sentinel2": DownstreamTaskConfig(
        dataset="swisscrop",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=8,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        probe_lr=0.1,
        eval_interval=Duration.epochs(50),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MIOU,
    ),
    # PLANTEUR (PASTIS2 on the French overseas territories), calendar-2019 S2 probe:
    # the v1.3 report's Table 4 task (piperw/bg8void-eval-v2, 20e949f4).
    "planteur_2019_probe_sentinel2": replace(
        _pastis_ps1_task([Modality.SENTINEL2_L2A.name], window_size=16),
        dataset="pastis2_drom_bg8void_2019_s2",
    ),
}


def _student128(tasks: dict, pretrained_norm: bool) -> dict:
    """Read every task from the full 128-d student, optionally with pretrain norm."""
    out = {}
    for name, task in tasks.items():
        task = replace(task, eval_on_student_registers=True, eval_student_dim=None)
        if pretrained_norm:
            task = replace(task, norm_stats_from_pretrained=True)
        out[name] = task
    return out


_CATALOGS = [
    _student128(EVAL_TASKS, pretrained_norm=True),
    _student128(EMBEDDING_EVAL_TASKS, pretrained_norm=False),
    _student128(EMBED_DIAG_TASKS, pretrained_norm=False),
    _student128(REVIVED_TASKS, pretrained_norm=False),
    _student128(PORTED_TASKS, pretrained_norm=False),
]
SWEEP_TASKS: dict[str, DownstreamTaskConfig] = {}
for _catalog in _CATALOGS:
    _clash = SWEEP_TASKS.keys() & _catalog.keys()
    assert not _clash, f"task names defined twice: {sorted(_clash)}"
    SWEEP_TASKS.update(_catalog)


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Carry SWEEP_TASKS for checkpoint_sweep_evals.py to read; nothing else is used."""
    return TrainerConfig(save_folder=common.save_folder).with_callback(
        "downstream_evaluator", DownstreamEvaluatorCallbackConfig(tasks=SWEEP_TASKS)
    )
