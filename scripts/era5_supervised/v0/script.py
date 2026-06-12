"""ERA5 daily encoder pretraining — objectives A and/or B.

This script is the v0 launcher for the ERA5 daily encoder pretraining
plan.  It supports three modes via ``enable_supervised`` /
``enable_reconstruction``:

* **A-only** (default): multi-task supervised objective.
* **B-only**: ERA5 reconstruction (corrupt → encode → decode → loss).
* **A+B**: both objectives run on every batch.

Tasks are referenced by nickname via `common.tasks=[burnrisk_canada_nbac,...]`.
By default nicknames resolve against the *direct rslearn* registry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from direct_registry import DirectRslearnRegistry, DirectRslearnTaskEntry
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
from olmoearth_pretrain.nn.era5_decoder import Era5TimeQueryDecoderConfig
from olmoearth_pretrain.nn.era5_encoder import Era5DailyEncoderConfig, Era5Pooling
from olmoearth_pretrain.nn.transforms.era5_corruption import CorruptionConfig
from olmoearth_pretrain.train.callbacks import (
    OlmoEarthWandBCallback,
)
from olmoearth_pretrain.train.callbacks.era5_evaluator_callback import (
    Era5DownstreamEvaluatorCallbackConfig,
    Era5LinearProbeTaskConfig,
)
from olmoearth_pretrain.train.train_module.era5_multiobjective import (
    Era5MultiObjectiveModelConfig,
    MultiObjectiveEra5TrainModuleConfig,
    ReconstructionObjectiveConfig,
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
    # Per-run overrides of the registry entry's split selection, keyed by
    # nickname (mirrors ``task_weights``). Use these to restrict a task to a
    # subset of windows at launch time without editing the registry, e.g.
    task_groups: dict[str, list[str]] = field(default_factory=dict)
    task_tags: dict[str, dict[str, str]] = field(default_factory=dict)
    registry_path: str | None = None
    # ------------------------------------------------------------------
    # Task source selection.
    #
    # By default ``common.tasks=[nickname, ...]`` is resolved against the
    # *direct rslearn* registry (``direct_registry.json`` shipped next to this
    # script), which points straight at rslearn datasets on Weka — no eval
    # ingestion required. Set ``use_eval_registry=True`` to instead resolve
    # those nicknames against the olmoearth_pretrain eval-dataset registry.
    # ``direct_registry_path`` overrides the direct registry location.
    # ------------------------------------------------------------------
    use_eval_registry: bool = False
    direct_registry_path: str | None = None
    # ------------------------------------------------------------------
    # Single-task direct-load (bypass *both* registries).
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
    direct_train_tags: dict[str, str] = field(default_factory=dict)
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
    # ------------------------------------------------------------------
    # Objective selection.  Both default to True/False respectively so
    # existing A-only launches are unaffected.
    # ------------------------------------------------------------------
    enable_supervised: bool = True
    enable_reconstruction: bool = False
    # ------------------------------------------------------------------
    # Reconstruction objective (B) knobs.
    # ------------------------------------------------------------------
    recon_weight: float = 1.0
    recon_decoder_depth: int = 1
    recon_decoder_num_heads: int = 6
    recon_decoder_dropout: float = 0.0
    recon_huber_delta: float = 1.0
    recon_raw_loss_on_masked_only: bool = True
    recon_swt_lambda: float = 0.1
    recon_swt_levels: list[int] = field(default_factory=lambda: [0, 1, 2])
    recon_swt_wavelet: str = "db2"
    # Corruption knobs
    recon_num_time_masks: int = 3
    recon_time_mask_min_len: int = 7
    recon_time_mask_max_len: int = 30
    # ------------------------------------------------------------------
    # Downstream evaluation (linear probe).  Runs for A-only, B-only,
    # and A+B — this is the primary encoder-quality signal for B-only.
    # ------------------------------------------------------------------
    enable_downstream_eval: bool = True
    eval_probe_lr: float = 1e-3
    eval_probe_epochs: int = 50
    eval_probe_batch_size: int = 256
    eval_embedding_batch_size: int = 128
    eval_run_on_test: bool = False
    eval_on_startup: bool = False
    eval_tasks: list[str] = field(default_factory=list)
    eval_max_samples: int | None = None


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


def _require_supervised_task_type(task_type: TaskType, who: str) -> None:
    """Reject task types the supervised objective cannot consume."""
    if task_type not in (TaskType.CLASSIFICATION, TaskType.REGRESSION):
        raise ValueError(
            f"{who}: task_type={task_type!r} is unsupported for ERA5 "
            "supervised pretraining v0 (only classification / regression "
            "are wired)."
        )


def _auto_label_extractor(
    task_type: TaskType, model_yaml_path: str, task_name: str | None
) -> str | None:
    """Pick a `LabelExtractor` name based on the rslearn task in model.yaml.

    Only reduce a segmentation target to a scalar when the rslearn dataset is
    genuinely a segmentation task *and* we want classification out of it.
    Otherwise return ``None`` so the spec falls back to its default extractor
    (which expects a real classification / regression target dict). The name is
    passed (rather than a callable) so the spec stays serializable inside the
    olmo_core `Config` tree.
    """
    if task_type != TaskType.CLASSIFICATION:
        return None
    rslearn_task_type = get_task_info(parse_model_config(model_yaml_path))["task_type"]
    if rslearn_task_type == "segmentation":
        return "segmentation_to_scalar"
    logger.info(
        "Classification task %r: rslearn task_type=%r (not segmentation) — "
        "using the default classification label extractor.",
        task_name,
        rslearn_task_type,
    )
    return None


def _resolve_direct_task_spec(
    common: Era5SupervisedCommonComponents,
) -> Era5TaskSpec:
    """Build a single direct-load task spec from the scalar overrides."""
    if common.direct_model_yaml_path is None:
        raise ValueError(
            "common.direct_weka_path is set but common.direct_model_yaml_path "
            "is not — both are required for direct-load."
        )
    task_type = _task_type_from_str(common.direct_task_type)
    _require_supervised_task_type(task_type, f"direct task {common.direct_task_name!r}")
    label_extractor_name = _auto_label_extractor(
        task_type, common.direct_model_yaml_path, common.direct_task_name
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
        tags_override=common.direct_train_tags or None,
        norm_stats_from_pretrained=True,
        label_extractor_name=label_extractor_name,
        max_samples=common.direct_max_samples,
    )


def _spec_from_direct_entry(
    entry: DirectRslearnTaskEntry,
    weight_override: float | None,
    groups_override: list[str] | None,
    tags_override: dict[str, str] | None,
) -> Era5TaskSpec:
    """Build a fully-specified `Era5TaskSpec` from a direct-registry entry.

    ``groups_override`` / ``tags_override`` are per-run launch-time overrides
    (``None`` means "fall back to the registry entry"). They compose downstream:
    rslearn applies both the group AND the tag filter when both are set.
    """
    task_type = _task_type_from_str(entry.task_type)
    _require_supervised_task_type(task_type, f"task {entry.name!r}")
    label_extractor_name = entry.label_extractor_name or _auto_label_extractor(
        task_type, entry.model_yaml_path, entry.name
    )
    groups = groups_override if groups_override is not None else entry.groups
    tags = tags_override if tags_override is not None else entry.tags
    return Era5TaskSpec(
        name=entry.name,
        weight=weight_override if weight_override is not None else entry.weight,
        task_type=task_type,
        is_multilabel=entry.is_multilabel,
        num_classes=entry.num_classes,
        modality_layer_name=entry.modality_layer_name,
        weka_path=entry.weka_path,
        model_yaml_path=entry.model_yaml_path,
        groups_override=groups or None,
        tags_override=tags or None,
        norm_stats_from_pretrained=entry.norm_stats_from_pretrained,
        label_extractor_name=label_extractor_name,
        max_samples=entry.max_samples,
    )


def _resolve_direct_registry_specs(
    common: Era5SupervisedCommonComponents,
) -> list[Era5TaskSpec]:
    """Resolve ``common.tasks`` nicknames against the direct rslearn registry.

    Each entry is expanded into a fully-specified `Era5TaskSpec` (weka path +
    model.yaml + task metadata), so multiple direct tasks compose into one
    supervised objective and nothing downstream needs the eval registry.
    Per-run ``common.task_groups`` / ``common.task_tags`` (keyed by nickname)
    override the entry's split selection at launch time.
    """
    registry = DirectRslearnRegistry.load(common.direct_registry_path)
    return [
        _spec_from_direct_entry(
            registry.get(name),
            common.task_weights.get(name),
            common.task_groups.get(name),
            common.task_tags.get(name),
        )
        for name in common.tasks
    ]


def _resolve_eval_registry_specs(
    common: Era5SupervisedCommonComponents,
) -> list[Era5TaskSpec]:
    """Resolve ``common.tasks`` against the olmoearth_pretrain eval registry."""
    registry = Registry.load(common.registry_path)
    specs: list[Era5TaskSpec] = []
    for name in common.tasks:
        entry = registry.get(name)
        task_type = _task_type_from_str(entry.task_type)
        _require_supervised_task_type(task_type, f"task {name!r}")
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


def _resolve_task_specs(
    common: Era5SupervisedCommonComponents,
) -> list[Era5TaskSpec]:
    """Resolve the task basket into a list of `Era5TaskSpec`s.

    Resolution order:
      1. Scalar single-task direct overrides (``common.direct_weka_path``).
      2. ``common.tasks`` against the eval registry (``use_eval_registry``).
      3. ``common.tasks`` against the direct rslearn registry (default).

    ``common.tasks`` is always required — even B-only runs need it for
    data sourcing (labels are simply ignored by objective B).
    """
    if common.direct_weka_path is not None:
        return [_resolve_direct_task_spec(common)]
    if not common.tasks:
        raise ValueError(
            "common.tasks is empty — set it to a list of registered nicknames "
            "(e.g. common.tasks=[burnrisk_canada_nbac]). Nicknames resolve "
            "against direct_registry.json unless common.use_eval_registry=True. "
            "Even reconstruction-only (B) runs need tasks for data sourcing."
        )
    if common.use_eval_registry:
        return _resolve_eval_registry_specs(common)
    return _resolve_direct_registry_specs(common)


# ---------------------------------------------------------------------------
# Helpers — resolve eval task configs from the direct registry
# ---------------------------------------------------------------------------


def _resolve_eval_task_configs(
    common: Era5SupervisedCommonComponents,
) -> list[Era5LinearProbeTaskConfig]:
    """Resolve eval tasks against the direct registry into probe configs.

    Uses ``common.eval_tasks`` if set, otherwise falls back to
    ``common.tasks``. Per-task val/test split selection comes from the
    registry's ``val_groups``/``val_tags``/``test_groups``/``test_tags``.
    """
    task_names = common.eval_tasks if common.eval_tasks else common.tasks
    if not task_names:
        return []

    registry = DirectRslearnRegistry.load(common.direct_registry_path)
    configs: list[Era5LinearProbeTaskConfig] = []
    for name in task_names:
        entry = registry.get(name)
        task_type = _task_type_from_str(entry.task_type)
        label_extractor_name = entry.label_extractor_name or _auto_label_extractor(
            task_type, entry.model_yaml_path, entry.name
        )
        configs.append(
            Era5LinearProbeTaskConfig(
                name=entry.name,
                weka_path=entry.weka_path,
                model_yaml_path=entry.model_yaml_path,
                task_type=TaskType(task_type).value,
                num_classes=entry.num_classes,
                is_multilabel=entry.is_multilabel,
                modality_layer_name=entry.modality_layer_name,
                label_extractor_name=label_extractor_name,
                norm_stats_from_pretrained=entry.norm_stats_from_pretrained,
                train_groups=entry.groups,
                train_tags=entry.tags or None,
                val_groups=entry.val_groups,
                val_tags=entry.val_tags or None,
                test_groups=entry.test_groups,
                test_tags=entry.test_tags or None,
                probe_lr=common.eval_probe_lr,
                probe_epochs=common.eval_probe_epochs,
                probe_batch_size=common.eval_probe_batch_size,
                embedding_batch_size=common.eval_embedding_batch_size,
                eval_interval=Duration.steps(common.eval_interval),
                max_eval_samples=common.eval_max_samples,
            )
        )
    return configs


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


def build_model_config(
    common: Era5SupervisedCommonComponents,
) -> Era5MultiObjectiveModelConfig:
    """Build the ERA5 encoder + objective head(s)."""
    specs = _resolve_task_specs(common)

    # -- Objective A (supervised) --
    supervised_objective: SupervisedObjectiveConfig | None = None
    if common.enable_supervised:
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

    # -- Objective B (reconstruction) --
    reconstruction_objective: ReconstructionObjectiveConfig | None = None
    if common.enable_reconstruction:
        decoder_config = Era5TimeQueryDecoderConfig(
            embedding_size=common.encoder_embedding_size,
            depth=common.recon_decoder_depth,
            num_heads=common.recon_decoder_num_heads,
            max_sequence_length=MAX_ERA5L_DAY_10_SEQUENCE_LENGTH,
            num_output_channels=Modality.ERA5L_DAY_10.num_bands,
            add_day_of_year_features=True,
            dropout=common.recon_decoder_dropout,
        )
        corruption_config = CorruptionConfig(
            num_time_masks=common.recon_num_time_masks,
            time_mask_min_len=common.recon_time_mask_min_len,
            time_mask_max_len=common.recon_time_mask_max_len,
        )
        reconstruction_objective = ReconstructionObjectiveConfig(
            name="reconstruction",
            weight=common.recon_weight,
            decoder=decoder_config,
            corruption=corruption_config,
            huber_delta=common.recon_huber_delta,
            raw_loss_on_masked_only=common.recon_raw_loss_on_masked_only,
            swt_lambda=common.recon_swt_lambda,
            swt_levels=common.recon_swt_levels,
            swt_wavelet=common.recon_swt_wavelet,
        )

    if not common.enable_supervised and not common.enable_reconstruction:
        raise ValueError(
            "At least one objective must be enabled: set "
            "common.enable_supervised=True and/or "
            "common.enable_reconstruction=True"
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
        reconstruction_objective=reconstruction_objective,
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
    """Build the trainer config with optional downstream linear-probe eval."""
    checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)
    wandb_callback = OlmoEarthWandBCallback(
        name=common.run_name,
        project="2026_05_21_era5_supervised",
        entity="eai-ai2",  # nosec
        enabled=True,
        upload_dataset_distribution_pre_train=False,
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

    eval_task_configs = _resolve_eval_task_configs(common)
    if eval_task_configs:
        trainer_config = trainer_config.with_callback(
            "era5_downstream_eval",
            Era5DownstreamEvaluatorCallbackConfig(
                tasks=eval_task_configs,
                enabled=common.enable_downstream_eval,
                eval_on_startup=common.eval_on_startup,
                run_on_test=common.eval_run_on_test,
                num_workers=common.num_workers,
                max_sequence_length=MAX_ERA5L_DAY_10_SEQUENCE_LENGTH,
            ),
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
