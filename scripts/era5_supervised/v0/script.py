"""ERA5 daily encoder pretraining — objective A (multi-task supervised).

This script is the v0 launcher for objective A of the ERA5 daily encoder
pretraining plan. It wires:

* the dedicated `Era5DailyEncoder` (no FlexiViT involved on this path),
* the multi-task ERA5 dataloader (`MultiTaskEra5DataLoader`),
* the `MultiObjectiveEra5TrainModule` configured with a single
  `SupervisedObjective` (objective A).

Tasks are pulled from the eval-dataset registry populated by
`olmoearth_pretrain.evals.studio_ingest.cli ingest`. Override
`common.tasks=[lfmc,burn_risk,...]` on the command line to change the
basket of tasks at run time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from olmo_core.config import DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import CosWithWarmup
from olmo_core.train.callbacks import (
    BeakerCallback,
    CheckpointerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.common import Duration, LoadStrategy
from olmo_core.train.config import TrainerConfig

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import (
    MAX_ERA5L_DAY_10_SEQUENCE_LENGTH,
    Modality,
)
from olmoearth_pretrain.data.multi_task_era5_dataset import (
    Era5TaskSpec,
    MultiTaskEra5DataLoaderConfig,
    MultiTaskEra5DatasetConfig,
)
from olmoearth_pretrain.evals.datasets.rslearn_builder import (
    get_task_info,
    parse_model_config,
)
from olmoearth_pretrain.evals.studio_ingest.registry import Registry
from olmoearth_pretrain.evals.task_types import TaskType
from olmoearth_pretrain.internal.common import (
    build_common_components as build_common_components_default,
)
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
    OlmoEarthVisualizeConfig,
    SubCmd,
)
from olmoearth_pretrain.nn.era5_encoder import Era5DailyEncoderConfig, Era5Pooling
from olmoearth_pretrain.train.callbacks import (
    OlmoEarthSpeedMonitorCallback,
    OlmoEarthWandBCallback,
)
from olmoearth_pretrain.train.train_module.era5_multiobjective import (
    Era5MultiObjectiveModelConfig,
    MultiObjectiveEra5TrainModuleConfig,
    SupervisedObjectiveConfig,
    SupervisedTaskConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Common components — task basket
# ---------------------------------------------------------------------------


@dataclass
class Era5SupervisedCommonComponents(CommonComponents):
    """Common components extended with the per-experiment task basket.

    The basket is just a list of task names registered in the eval registry
    (`evals/studio_ingest/registry.json`). All other per-task knobs
    (`task_type`, `num_classes`, sampling weight, ...) are looked up from
    the registry at build time so this script stays declarative.
    """

    tasks: list[str] = field(default_factory=list)
    task_weights: dict[str, float] = field(default_factory=dict)
    registry_path: str | None = None
    # ------------------------------------------------------------------
    # Direct-load (bypass the eval registry).
    #
    # When ``direct_weka_path`` is set, the registry is skipped entirely and a
    # single task is loaded straight from an rslearn dataset on Weka using
    # ``direct_model_yaml_path``. This lets us train objective A on a full
    # original rslearn dataset without first ingesting/copying it.
    #
    # ``direct_modality_layer_name`` MUST match the ERA5 *input key* in the
    # model.yaml ``data.init_args.inputs`` block (e.g. ``era5_daily``), not the
    # on-disk rslearn layer name (``era5_365dhistory``).
    # ------------------------------------------------------------------
    direct_task_name: str | None = None
    direct_weka_path: str | None = None
    direct_model_yaml_path: str | None = None
    direct_task_type: str = "classification"
    direct_num_classes: int | None = 2
    direct_is_multilabel: bool = False
    direct_modality_layer_name: str = "era5_daily"
    direct_train_groups: list[str] = field(default_factory=lambda: ["train"])
    direct_max_samples: int | None = None
    encoder_embedding_size: int = 384
    encoder_depth: int = 8
    encoder_num_heads: int = 6
    encoder_pooling: str = Era5Pooling.MEAN.value
    global_batch_size: int = 64
    rank_microbatch_size: int = 16
    num_workers: int = 4
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.02
    warmup_steps: int = 500
    max_epochs: int = 50
    save_interval: int = 1000
    eval_interval: int = 1000


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> Era5SupervisedCommonComponents:
    """Build the common components for the ERA5 supervised pretraining run."""
    base = build_common_components_default(script, cmd, run_name, cluster, overrides)
    return Era5SupervisedCommonComponents(
        run_name=base.run_name,
        save_folder=base.save_folder,
        launch=base.launch,
        nccl_debug=base.nccl_debug,
        tokenization_config=base.tokenization_config,
        # Objective A only consumes the daily ERA5 modality at the encoder.
        training_modalities=[Modality.ERA5L_DAY_10.name],
        tasks=[],
    )


# ---------------------------------------------------------------------------
# Helpers — resolve task specs from the registry
# ---------------------------------------------------------------------------


def _task_type_from_str(value: str) -> TaskType:
    """Coerce a string to `TaskType`, accepting both enum value and name."""
    try:
        return TaskType(value)
    except ValueError:
        return TaskType[value.upper()]


def _resolve_direct_task_spec(
    common: Era5SupervisedCommonComponents,
) -> Era5TaskSpec:
    """Build a single direct-load task spec that bypasses the eval registry."""
    if common.direct_model_yaml_path is None:
        raise ValueError(
            "common.direct_weka_path is set but common.direct_model_yaml_path "
            "is not — both are required for direct-load."
        )
    task_type = _task_type_from_str(common.direct_task_type)
    if task_type not in (TaskType.CLASSIFICATION, TaskType.REGRESSION):
        raise ValueError(
            f"direct_task_type={common.direct_task_type!r} is unsupported "
            "(only classification / regression are wired)."
        )
    # Only reduce a segmentation target to a scalar when the rslearn dataset is
    # genuinely a segmentation task *and* we want classification out of it.
    # Otherwise fall back to the spec's default extractor (which expects a real
    # classification / regression target dict). We pass the extractor *by name*
    # so the spec stays serializable inside the olmo_core `Config` tree.
    label_extractor_name = None
    if task_type == TaskType.CLASSIFICATION:
        rslearn_task_type = get_task_info(
            parse_model_config(common.direct_model_yaml_path)
        )["task_type"]
        if rslearn_task_type == "segmentation":
            label_extractor_name = "segmentation_to_scalar"
        else:
            logger.info(
                "Direct-load classification task %r: rslearn task_type=%r "
                "(not segmentation) — using the default classification label "
                "extractor.",
                common.direct_task_name,
                rslearn_task_type,
            )
    return Era5TaskSpec(
        name=common.direct_task_name or "direct_task",
        weight=1.0,
        task_type=task_type,
        is_multilabel=common.direct_is_multilabel,
        num_classes=common.direct_num_classes,
        modality_layer_name=common.direct_modality_layer_name,
        weka_path=common.direct_weka_path,
        model_yaml_path=common.direct_model_yaml_path,
        groups_override=common.direct_train_groups,
        norm_stats_from_pretrained=True,
        label_extractor_name=label_extractor_name,
        max_samples=common.direct_max_samples,
    )


def _resolve_task_specs(
    common: Era5SupervisedCommonComponents,
) -> list[Era5TaskSpec]:
    """Look up registry entries for each requested task and build specs."""
    if common.direct_weka_path is not None:
        return [_resolve_direct_task_spec(common)]
    if not common.tasks:
        raise ValueError(
            "common.tasks is empty — set it to a list of registered task "
            "names (e.g. common.tasks=[lfmc,burn_risk])."
        )
    registry = Registry.load(common.registry_path)
    specs: list[Era5TaskSpec] = []
    for name in common.tasks:
        entry = registry.get(name)
        task_type = _task_type_from_str(entry.task_type)
        if task_type not in (TaskType.CLASSIFICATION, TaskType.REGRESSION):
            raise ValueError(
                f"Task {name!r} has unsupported task_type={entry.task_type!r} "
                "for ERA5 supervised pretraining v0 (only classification / "
                "regression are wired)."
            )
        specs.append(
            Era5TaskSpec(
                name=name,
                weight=common.task_weights.get(name, 1.0),
                task_type=task_type,
                is_multilabel=entry.is_multilabel,
                num_classes=entry.num_classes,
                norm_stats_from_pretrained=entry.use_pretrain_norm,
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


def build_model_config(
    common: Era5SupervisedCommonComponents,
) -> Era5MultiObjectiveModelConfig:
    """Build the ERA5 encoder + supervised head registry."""
    specs = _resolve_task_specs(common)
    supervised_objective = SupervisedObjectiveConfig(
        name="supervised",
        weight=1.0,
        tasks=[
            SupervisedTaskConfig(
                name=spec.name,
                task_type=TaskType(spec.task_type).value,
                num_classes=spec.num_classes,
                is_multilabel=spec.is_multilabel,
                weight=spec.weight,
            )
            for spec in specs
        ],
    )
    encoder_config = Era5DailyEncoderConfig(
        embedding_size=common.encoder_embedding_size,
        depth=common.encoder_depth,
        num_heads=common.encoder_num_heads,
        max_sequence_length=MAX_ERA5L_DAY_10_SEQUENCE_LENGTH,
        modality_name=Modality.ERA5L_DAY_10.name.lower(),
        pooling=common.encoder_pooling,
    )
    return Era5MultiObjectiveModelConfig(
        encoder_config=encoder_config,
        supervised_objective=supervised_objective,
    )


def build_train_module_config(
    common: Era5SupervisedCommonComponents,
) -> MultiObjectiveEra5TrainModuleConfig:
    """Build the multi-objective train module config."""
    return MultiObjectiveEra5TrainModuleConfig(
        optim_config=AdamWConfig(
            lr=common.learning_rate, weight_decay=common.weight_decay, fused=False
        ),
        rank_microbatch_size=common.rank_microbatch_size,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=common.warmup_steps),
        dp_config=DataParallelConfig(
            name=DataParallelType.ddp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )


def build_dataset_config(
    common: Era5SupervisedCommonComponents,
) -> MultiTaskEra5DatasetConfig:
    """Build the multi-task ERA5 dataset config."""
    return MultiTaskEra5DatasetConfig(
        tasks=_resolve_task_specs(common),
        max_sequence_length=MAX_ERA5L_DAY_10_SEQUENCE_LENGTH,
        registry_path=common.registry_path,
    )


def build_dataloader_config(
    common: Era5SupervisedCommonComponents,
) -> MultiTaskEra5DataLoaderConfig:
    """Build the multi-task ERA5 dataloader config."""
    return MultiTaskEra5DataLoaderConfig(
        global_batch_size=common.global_batch_size,
        num_workers=common.num_workers,
        prefetch_factor=2,
        seed=3622,
        drop_last=True,
        work_dir=common.save_folder,
        persistent_workers=common.num_workers > 0,
    )


def build_trainer_config(common: Era5SupervisedCommonComponents) -> TrainerConfig:
    """Build the trainer config (no downstream eval callbacks in v0)."""
    checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)
    wandb_callback = OlmoEarthWandBCallback(
        name=common.run_name,
        project="2026_05_21_era5_supervised",
        entity="eai-ai2",  # nosec
        enabled=True,
    )
    trainer_config = (
        TrainerConfig(
            work_dir=common.save_folder,
            load_strategy=LoadStrategy.if_available,
            save_folder=common.save_folder,
            cancel_check_interval=25,
            metrics_collect_interval=10,
            max_duration=Duration.epochs(common.max_epochs),
            checkpointer=checkpointer_config,
        )
        .with_callback("wandb", wandb_callback)
        .with_callback("speed_monitor", OlmoEarthSpeedMonitorCallback())
        .with_callback("gpu_memory_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("garbage_collector", GarbageCollectorCallback(gc_interval=1))
        .with_callback("beaker", BeakerCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=common.save_interval,
                ephemeral_save_interval=common.save_interval // 4 or 100,
            ),
        )
    )
    return trainer_config


def build_visualize_config(
    common: Era5SupervisedCommonComponents,
) -> OlmoEarthVisualizeConfig:
    """Visualize config (unused for objective A; included for parity)."""
    return OlmoEarthVisualizeConfig(
        num_samples=None,
        output_dir=f"{common.save_folder}/visualizations",
        std_multiplier=2.0,
    )


# Re-export so ``base.py`` can find them without per-attribute imports.
__all__ = [
    "Config",
    "build_common_components",
    "build_dataloader_config",
    "build_dataset_config",
    "build_model_config",
    "build_train_module_config",
    "build_trainer_config",
    "build_visualize_config",
]
