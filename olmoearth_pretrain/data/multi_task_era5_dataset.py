"""Multi-task ERA5-daily dataloader for supervised pretraining (objective A).

The pretraining target for objective A is to learn a single ERA5-daily encoder
on a basket of supervised tasks that have *already been ingested* through
`olmoearth_pretrain.evals.studio_ingest.cli`. Each task lives as its own
rslearn dataset on Weka and is registered in the eval-dataset registry
(`evals/studio_ingest/registry.json`).

This module exposes:

* `Era5TaskDataset` — a thin per-task wrapper around an rslearn `ModelDataset`
  that extracts just the ERA5-daily modality + per-step timestamps + the
  task label, padded to a fixed sequence length and returned as a
  `Era5SupervisedSample`.
* `MultiTaskEra5DataLoader` — an olmo-core `DataLoaderBase`-compatible loader
  that round-robins between N `Era5TaskDataset`s using per-task weights,
  shards each task's index list across DP ranks, and yields one task's batch
  per global step (so that the per-task supervised head + loss is well
  defined within the batch).

Objectives B / C will add additional `_Objective` implementations on the
train-module side; this dataloader is reused for A only — they will plug in
their own data streams alongside (the train module supports it).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
import torch
from einops import rearrange
from olmo_core.data.data_loader import DataLoaderBase
from rslearn.train.dataset import ModelDataset as RsModelDataset
from rslearn.train.model_context import RasterImage
from torch import Tensor
from torch.utils.data import Dataset

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import (
    MAX_ERA5L_DAY_10_SEQUENCE_LENGTH,
    Modality,
)
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.evals.constants import RSLEARN_TO_OLMOEARTH
from olmoearth_pretrain.evals.datasets.rslearn_builder import (
    build_model_dataset,
    parse_model_config,
)
from olmoearth_pretrain.evals.studio_ingest.registry import Registry
from olmoearth_pretrain.evals.task_types import TaskType

logger = logging.getLogger(__name__)


_ERA5_MODALITY = Modality.ERA5L_DAY_10


class Era5SupervisedSample(NamedTuple):
    """A single ERA5 supervised training sample.

    All tensors are CPU tensors; the train module moves them to the device.

    Shapes:
        era5         : ``[T, C]``       float32
        timestamps   : ``[T, 3]``       int64    ``[day-of-year, month0, year]``
        ignore_mask  : ``[T]``          bool     (always all-False; kept for interface compat)
        label        : task-specific (scalar or vector)
        task_name    : str
    """

    era5: Tensor
    timestamps: Tensor
    ignore_mask: Tensor
    label: Tensor
    task_name: str


class Era5SupervisedBatch(NamedTuple):
    """A collated batch of `Era5SupervisedSample`s (all from one task)."""

    era5: Tensor  # [B, T_max, C]
    timestamps: Tensor  # [B, T_max, 3]
    ignore_mask: Tensor  # [B, T_max]
    labels: Tensor  # task-specific stacking of labels
    task_name: str


# A LabelExtractor takes the rslearn target dict and returns a torch Tensor.
LabelExtractor = Callable[[Any], Tensor]


def default_classification_label(target: Any) -> Tensor:
    """Extract a scalar class label from an rslearn ClassificationTask target.

    rslearn's `ClassificationTask` typically stores its target as a dict
    ``{"class": Tensor, "valid": Tensor}``. Plain integer labels are also
    accepted for forward compatibility.
    """
    if isinstance(target, dict):
        if "class" in target:
            cls = target["class"]
            valid = target.get("valid")
            if valid is not None and not bool(valid):
                # Mark the row as missing; downstream heads use the ignore index.
                from olmoearth_pretrain.nn.era5_heads import SEGMENTATION_IGNORE_LABEL

                return torch.tensor(SEGMENTATION_IGNORE_LABEL, dtype=torch.long)
            return torch.as_tensor(cls, dtype=torch.long).reshape(())
    return torch.as_tensor(target, dtype=torch.long).reshape(())


def default_regression_label(target: Any, key: str = "value") -> Tensor:
    """Extract a scalar regression label from an rslearn target dict."""
    if isinstance(target, dict):
        if key in target:
            return torch.as_tensor(target[key], dtype=torch.float32).reshape(())
        if "regress" in target and isinstance(target["regress"], dict):
            return default_regression_label(target["regress"], key=key)
    if isinstance(target, Tensor):
        return target.to(dtype=torch.float32).reshape(())
    return torch.as_tensor(target, dtype=torch.float32).reshape(())


def make_regression_extractor(key: str = "value") -> LabelExtractor:
    """Return a regression `LabelExtractor` for a given target dict key."""

    def _fn(target: Any) -> Tensor:
        return default_regression_label(target, key=key)

    return _fn


def segmentation_to_scalar_label(target: Any) -> Tensor:
    """Reduce a (1x1) rslearn segmentation target to a single class label.

    Some direct-load classification tasks (e.g. burn-risk) are defined in
    rslearn as a ``MultiTask`` -> ``SegmentationTask`` whose per-pixel target is
    pooled to a single 1x1 pixel by an ``AdaptivePooling`` transform. rslearn
    returns that target as
    ``{"segmentation": {"classes": RasterImage, "valid": RasterImage}}`` where
    the ``RasterImage`` holds a CTHW tensor. We collapse it to a scalar class id
    for the ERA5 classification head; pixels whose ``valid`` mask is all-zero are
    mapped to the ignore label.
    """
    from olmoearth_pretrain.nn.era5_heads import SEGMENTATION_IGNORE_LABEL

    seg = target
    if isinstance(target, dict):
        seg = target.get("segmentation", target)
    if not isinstance(seg, dict) or "classes" not in seg:
        raise KeyError(
            "Expected a segmentation target dict with a 'classes' entry, got "
            f"keys={sorted(seg.keys()) if isinstance(seg, dict) else type(seg)}"
        )

    def _to_hw(value: Any) -> Tensor:
        if hasattr(value, "get_hw_tensor"):
            return value.get_hw_tensor()
        if hasattr(value, "image"):
            return value.image.reshape(value.image.shape[-2:])
        return torch.as_tensor(value)

    classes = _to_hw(seg["classes"]).reshape(-1).long()
    valid = seg.get("valid")
    if valid is not None:
        valid_t = _to_hw(valid).reshape(-1)
        if not bool((valid_t > 0).any()):
            return torch.tensor(SEGMENTATION_IGNORE_LABEL, dtype=torch.long)
    # Already 1x1 after pooling; max() keeps the positive class if any pixel is
    # positive (robust even if pooling is changed upstream).
    return classes.max().reshape(())


# Named label extractors that take no constructor args. Referenced by name from
# ``Era5TaskSpec.label_extractor_name`` so that the (non-serializable) callable
# never has to live on a field that is part of an olmo_core ``Config`` tree
# (OmegaConf cannot build a schema for a ``Callable`` annotation).
LABEL_EXTRACTORS: dict[str, LabelExtractor] = {
    "default_classification": default_classification_label,
    "segmentation_to_scalar": segmentation_to_scalar_label,
}


@dataclass
class Era5TaskSpec:
    """How to load + interpret a single task's ERA5 supervised data.

    Args:
        name: Task identifier (matches the eval-registry entry name).
        weight: Sampling weight relative to the other tasks (defaults to 1.0).
        task_type: Supervised task type (`classification` / `regression`).
        is_multilabel: Multi-label classification flag.
        num_classes: Number of output classes (regression: number of outputs).
        regression_label_key: Key used to read regression labels from the
            rslearn target dict (only used when `task_type=regression`).
        label_extractor_name: Optional name of a registered `LabelExtractor`
            (see `LABEL_EXTRACTORS`) that overrides the default per-`task_type`
            extraction logic. A plain string is used (rather than a callable)
            so this spec stays serializable inside an olmo_core `Config`.
        modality_layer_name: rslearn input-dict key that holds the ERA5 daily
            tensor. Defaults to ``"era5l_day_10"`` which matches
            ``Modality.ERA5L_DAY_10.name`` and the mapping in
            `evals/constants.py`.
        groups_override / tags_override / source_path / model_yaml_path /
            weka_path: Optional overrides for non-registry usage.
        norm_stats_from_pretrained: Use the pretrain `computed.json` stats.
            When False, the registry entry's per-task stats are used.
    """

    name: str
    weight: float = 1.0
    task_type: TaskType | str = TaskType.REGRESSION
    is_multilabel: bool = False
    num_classes: int | None = None
    regression_label_key: str = "value"
    label_extractor_name: str | None = None
    modality_layer_name: str = "era5l_day_10"
    groups_override: list[str] | None = None
    tags_override: dict[str, str] | None = None
    norm_stats_from_pretrained: bool = True
    split: str = "train"
    weka_path: str | None = None
    model_yaml_path: str | None = None
    max_samples: int | None = None

    def get_label_extractor(self) -> LabelExtractor:
        """Resolve the `LabelExtractor` to use for this task."""
        if self.label_extractor_name is not None:
            try:
                return LABEL_EXTRACTORS[self.label_extractor_name]
            except KeyError:
                raise ValueError(
                    f"Unknown label_extractor_name={self.label_extractor_name!r}; "
                    f"registered extractors: {sorted(LABEL_EXTRACTORS)}"
                ) from None
        task_type = TaskType(self.task_type)
        if task_type == TaskType.CLASSIFICATION:
            return default_classification_label
        if task_type == TaskType.REGRESSION:
            return make_regression_extractor(self.regression_label_key)
        raise ValueError(
            f"Unsupported task_type for ERA5 supervised pretraining: {task_type}"
        )


class Era5TaskDataset(Dataset):
    """Wrap one rslearn `ModelDataset` and emit `Era5SupervisedSample`s.

    Only the ERA5-daily input modality is read from each sample; image
    modalities (S1, S2, ...) are skipped. This keeps objective A's input
    strictly ERA5-only as agreed.
    """

    def __init__(
        self,
        spec: Era5TaskSpec,
        model_dataset: RsModelDataset,
        max_sequence_length: int = MAX_ERA5L_DAY_10_SEQUENCE_LENGTH,
        normalizer: Normalizer | None = None,
    ) -> None:
        """Initialize from a task spec and its backing rslearn dataset."""
        self.task_spec = spec
        self.dataset = model_dataset
        self.max_sequence_length = max_sequence_length
        self.normalizer = normalizer or Normalizer(Strategy.COMPUTED)
        self._label_extractor = spec.get_label_extractor()
        self._num_bands = _ERA5_MODALITY.num_bands

    def __len__(self) -> int:
        """Return number of samples in the underlying dataset."""
        return len(self.dataset)

    def _extract_era5(self, inputs: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(era5[T, C], timestamps[T, 3], ignore_mask[T])``."""
        layer_name = self.task_spec.modality_layer_name
        if layer_name not in inputs:
            raise KeyError(
                f"Task {self.task_spec.name!r}: rslearn input dict is missing the "
                f"ERA5-daily layer {layer_name!r}. Available keys: "
                f"{sorted(inputs.keys())}"
            )
        raster = inputs[layer_name]
        if isinstance(raster, RasterImage):
            img = raster.image
            per_step_timestamps = raster.timestamps
        else:
            img = raster
            per_step_timestamps = None
        if isinstance(img, np.ndarray):
            img = torch.as_tensor(img)
        if img.ndim != 4:
            raise ValueError(
                f"Expected (C, T, H, W) ERA5 tensor, got shape {tuple(img.shape)}"
            )
        c, t, h, w = img.shape
        if h != 1 or w != 1:
            # Daily ERA5 is non-spatial; average any residual spatial dim so we
            # don't silently drop data.
            img = img.mean(dim=(-1, -2), keepdim=True)
            h = w = 1
        # (C, T, 1, 1) -> (T, C)
        era5 = rearrange(img[..., 0, 0], "c t -> t c").float()
        if era5.shape[-1] != self._num_bands:
            raise ValueError(
                f"Task {self.task_spec.name!r}: ERA5 sample has {era5.shape[-1]} bands "
                f"but ERA5L_DAY_10 expects {self._num_bands}."
            )
        # Normalize per band using the configured normalizer.
        np_era5 = self.normalizer.normalize(_ERA5_MODALITY, era5.numpy())
        era5 = torch.as_tensor(np_era5, dtype=torch.float32)
        timestamps = self._build_timestamps(t, per_step_timestamps)

        if t != self.max_sequence_length:
            raise ValueError(
                f"Task {self.task_spec.name!r}: expected sequence length "
                f"{self.max_sequence_length}, got {t}. ERA5 is a dense "
                f"reanalysis product — all samples must have uniform length."
            )
        ignore_mask = torch.zeros(t, dtype=torch.bool)
        return era5, timestamps, ignore_mask

    @staticmethod
    def _build_timestamps(
        num_steps: int,
        per_step_timestamps: list[tuple[Any, Any]] | None,
    ) -> Tensor:
        """Build a ``(T, 3)`` int64 tensor of ``[day-of-year, month0, year]``."""
        ts = torch.zeros(num_steps, 3, dtype=torch.long)
        if per_step_timestamps is not None and len(per_step_timestamps) == num_steps:
            for i, (start, _end) in enumerate(per_step_timestamps):
                ts[i, 0] = int(start.timetuple().tm_yday)
                ts[i, 1] = int(start.month) - 1
                ts[i, 2] = int(start.year)
            return ts
        # Synthesize daily timestamps starting Jan-1 of a placeholder year.
        # This still gives the encoder usable day-of-year + month signals.
        for i in range(num_steps):
            day_of_year = (i % 365) + 1
            # Month-0 is the approximate calendar month for that day-of-year.
            month0 = min(11, max(0, ((day_of_year - 1) * 12) // 365))
            ts[i, 0] = day_of_year
            ts[i, 1] = month0
            ts[i, 2] = 1970
        return ts

    def __getitem__(self, idx: int) -> Era5SupervisedSample:
        """Load and return the ERA5 supervised sample at *idx*."""
        sample = self.dataset[idx]
        # rslearn ModelDataset yields (inputs, target, metadata).
        if isinstance(sample, tuple) and len(sample) == 3:
            inputs, target, _meta = sample
        elif isinstance(sample, tuple) and len(sample) == 2:
            inputs, target = sample
        else:
            raise TypeError(
                f"Unexpected rslearn sample type for task {self.task_spec.name!r}: "
                f"{type(sample).__name__}"
            )
        era5, timestamps, ignore_mask = self._extract_era5(inputs)
        label = self._label_extractor(target)
        return Era5SupervisedSample(
            era5=era5,
            timestamps=timestamps,
            ignore_mask=ignore_mask,
            label=label,
            task_name=self.task_spec.name,
        )


def _collate_samples(
    samples: list[Era5SupervisedSample], task_name: str
) -> Era5SupervisedBatch:
    """Stack a list of `Era5SupervisedSample`s from one task into a batch."""
    era5 = torch.stack([s.era5 for s in samples], dim=0)
    timestamps = torch.stack([s.timestamps for s in samples], dim=0)
    ignore_mask = torch.stack([s.ignore_mask for s in samples], dim=0)
    labels = torch.stack([s.label for s in samples], dim=0)
    return Era5SupervisedBatch(
        era5=era5,
        timestamps=timestamps,
        ignore_mask=ignore_mask,
        labels=labels,
        task_name=task_name,
    )


def _build_task_dataset(
    spec: Era5TaskSpec,
    registry: Registry | None,
    max_sequence_length: int,
) -> Era5TaskDataset:
    """Build the per-task `Era5TaskDataset` from the spec + registry."""
    weka_path = spec.weka_path
    model_yaml_path = spec.model_yaml_path
    if weka_path is None or model_yaml_path is None:
        if registry is None:
            raise ValueError(
                f"Task {spec.name!r}: either weka_path+model_yaml_path or a "
                "registry must be provided."
            )
        entry = registry.get(spec.name)
        weka_path = weka_path or entry.weka_path
        model_yaml_path = model_yaml_path or entry.model_yaml_path
    model_config = parse_model_config(str(model_yaml_path))
    model_dataset = build_model_dataset(
        model_config=model_config,
        source_path=str(weka_path),
        split=spec.split,
        groups_override=spec.groups_override,
        tags_override=spec.tags_override,
        max_samples=spec.max_samples,
    )
    normalizer = (
        Normalizer(Strategy.COMPUTED)
        if spec.norm_stats_from_pretrained
        else Normalizer(Strategy.PREDEFINED)
    )
    return Era5TaskDataset(
        spec=spec,
        model_dataset=model_dataset,
        max_sequence_length=max_sequence_length,
        normalizer=normalizer,
    )


@dataclass
class MultiTaskEra5DatasetConfig(Config):
    """Dataset config consumed by the `OlmoEarthExperimentConfig` plumbing.

    Args:
        tasks: List of `Era5TaskSpec`s. Each entry corresponds to one
            ingested rslearn dataset.
        max_sequence_length: Expected sequence length (validated, not padded).
        registry_path: Optional custom registry path.
    """

    tasks: list[Era5TaskSpec] = field(default_factory=list)
    max_sequence_length: int = MAX_ERA5L_DAY_10_SEQUENCE_LENGTH
    registry_path: str | None = None

    def validate(self) -> None:
        """Validate the configuration."""
        if not self.tasks:
            raise ValueError("MultiTaskEra5DatasetConfig requires at least one task")
        for spec in self.tasks:
            if spec.weight <= 0:
                raise ValueError(
                    f"Task {spec.name!r}: weight must be strictly positive"
                )
            # Verify the layer name maps to ERA5L_DAY_10 (or aliases).
            layer = spec.modality_layer_name
            if layer not in RSLEARN_TO_OLMOEARTH:
                logger.warning(
                    "Task %r: modality_layer_name=%s is not in RSLEARN_TO_OLMOEARTH",
                    spec.name,
                    layer,
                )

    def build(self) -> MultiTaskEra5Dataset:
        """Materialize the task datasets."""
        self.validate()
        registry = Registry.load(self.registry_path) if self._needs_registry() else None
        task_datasets: dict[str, Era5TaskDataset] = {}
        for spec in self.tasks:
            task_datasets[spec.name] = _build_task_dataset(
                spec, registry, self.max_sequence_length
            )
        return MultiTaskEra5Dataset(
            task_datasets=task_datasets,
            task_specs={spec.name: spec for spec in self.tasks},
        )

    def _needs_registry(self) -> bool:
        return any(
            spec.weka_path is None or spec.model_yaml_path is None
            for spec in self.tasks
        )


class MultiTaskEra5Dataset:
    """In-memory bundle of per-task `Era5TaskDataset`s."""

    def __init__(
        self,
        task_datasets: dict[str, Era5TaskDataset],
        task_specs: dict[str, Era5TaskSpec],
    ) -> None:
        """Initialize with matching task datasets and specs."""
        if set(task_datasets) != set(task_specs):
            raise ValueError(
                "task_datasets and task_specs must cover the same set of tasks"
            )
        self.task_datasets = task_datasets
        self.task_specs = task_specs

    @property
    def task_names(self) -> list[str]:
        """Sorted list of task names tracked by the dataset."""
        return sorted(self.task_datasets.keys())

    def task_lengths(self) -> dict[str, int]:
        """Number of samples per task."""
        return {name: len(ds) for name, ds in self.task_datasets.items()}

    def total_samples(self) -> int:
        """Aggregate sample count across tasks (used by the loader's epoch)."""
        return sum(len(ds) for ds in self.task_datasets.values())


@dataclass
class MultiTaskEra5DataLoaderConfig(Config):
    """Config for `MultiTaskEra5DataLoader`."""

    global_batch_size: int = 64
    num_workers: int = 4
    prefetch_factor: int = 2
    seed: int = 1337
    drop_last: bool = True
    work_dir: str | None = None
    persistent_workers: bool = True

    def build(
        self,
        dataset: MultiTaskEra5Dataset,
        dp_process_group: Any | None = None,
    ) -> MultiTaskEra5DataLoader:
        """Build a `MultiTaskEra5DataLoader`."""
        if self.work_dir is None:
            raise ValueError(
                "MultiTaskEra5DataLoaderConfig.work_dir must be set before build()."
            )
        dp_world_size = 1
        dp_rank = 0
        if dp_process_group is not None:
            dp_world_size = dp_process_group.size()
            dp_rank = dp_process_group.rank()
        return MultiTaskEra5DataLoader(
            dataset=dataset,
            work_dir=self.work_dir,
            global_batch_size=self.global_batch_size,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            seed=self.seed,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
        )


class MultiTaskEra5DataLoader(DataLoaderBase):
    """olmo-core compatible loader yielding one task's batch per global step.

    Each global step:
    1. Samples a task name proportional to per-task weights (rngd by seed +
       epoch + step), so the *expected* fraction of steps spent on task ``t``
       is ``w_t / sum(w_*)``.
    2. Pulls ``global_batch_size`` sample indices from that task, sharded
       across DP ranks (so the local batch is `rank_batch_size`).
    3. Yields a single `Era5SupervisedBatch` for the local rank.
    """

    _epoch: int | None
    batches_processed: int

    def __init__(
        self,
        dataset: MultiTaskEra5Dataset,
        work_dir: Any,
        global_batch_size: int,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        seed: int = 1337,
        drop_last: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        """Initialize the multi-task ERA5 data loader."""
        super().__init__(
            work_dir=work_dir,
            global_batch_size=global_batch_size,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
        )
        self.dataset = dataset
        self.seed = seed
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.token_budget: int | None = (
            None  # Required by OlmoEarthSpeedMonitorCallback.pre_train(); not used
        )

        self._task_names = sorted(dataset.task_specs.keys())
        weights = np.array(
            [dataset.task_specs[name].weight for name in self._task_names],
            dtype=np.float64,
        )
        self._task_weights = weights / weights.sum()
        # Pre-computed per-epoch schedule, populated in `reshuffle`. Stored as
        # arrays so it round-trips cheaply to PyTorch worker processes.
        # `_schedule_task_ids[s]` is the task index for global step `s` and
        # `_schedule_indices[s, :]` are the dataset indices for that step.
        self._schedule_task_ids: np.ndarray | None = None
        self._schedule_indices: np.ndarray | None = None

    @property
    def total_batches(self) -> int:
        """Number of global batches per epoch (`drop_last` always honored)."""
        return self.dataset.total_samples() // self.global_batch_size

    def reshuffle(self, epoch: int | None = None, **kwargs: Any) -> None:
        """Pre-compute the deterministic per-step schedule for a new epoch.

        Schedules every global step's ``(task, indices)`` upfront so that
        PyTorch worker processes can simply look up their step rather than
        racing on shared cursors.
        """
        if epoch is None:
            epoch = 1 if self._epoch is None else self._epoch + 1
        if epoch <= 0:
            raise ValueError(f"'epoch' must be >= 1, got {epoch}")
        self._epoch = epoch
        total = self.total_batches
        if total == 0:
            raise ValueError(
                "MultiTaskEra5DataLoader has 0 batches per epoch — increase "
                "the per-task sample count or decrease global_batch_size."
            )
        rng = np.random.default_rng(self.seed + epoch)

        # Per-task permutations + cursors used while building the schedule.
        task_perms: dict[str, np.ndarray] = {}
        task_cursors: dict[str, int] = {}
        for name, ds in self.dataset.task_datasets.items():
            n = len(ds)
            if n == 0:
                raise ValueError(f"Task {name!r} has 0 samples in this split")
            task_perms[name] = rng.permutation(n)
            task_cursors[name] = 0

        task_id_choices = rng.choice(
            len(self._task_names),
            size=total,
            p=self._task_weights,
        )
        schedule_indices = np.empty((total, self.global_batch_size), dtype=np.int64)
        for step, task_id in enumerate(task_id_choices):
            name = self._task_names[int(task_id)]
            perm = task_perms[name]
            cur = task_cursors[name]
            if cur + self.global_batch_size > len(perm):
                # Wrap-around: reshuffle this task with a stable seed.
                sub_rng = np.random.default_rng(
                    self.seed + epoch * 10_000 + (hash(name) & 0xFFFF) + step
                )
                task_perms[name] = sub_rng.permutation(len(perm))
                perm = task_perms[name]
                cur = 0
            schedule_indices[step] = perm[cur : cur + self.global_batch_size]
            task_cursors[name] = cur + self.global_batch_size
        self._schedule_task_ids = task_id_choices.astype(np.int64)
        self._schedule_indices = schedule_indices

    def _iter_batches(self) -> Iterable[Era5SupervisedBatch]:
        if self._schedule_task_ids is None or self._schedule_indices is None:
            raise RuntimeError(
                "Call reshuffle() before iterating MultiTaskEra5DataLoader"
            )
        total = self.total_batches
        start_step = self.batches_processed
        worker_iter = _ScheduleDispatcher(
            task_datasets=self.dataset.task_datasets,
            task_names=self._task_names,
            schedule_task_ids=self._schedule_task_ids,
            schedule_indices=self._schedule_indices,
            start_step=start_step,
            total=total,
            dp_rank=self.dp_rank,
            dp_world_size=self.dp_world_size,
        )
        loader = torch.utils.data.DataLoader(
            worker_iter,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=(
                self.persistent_workers if self.num_workers > 0 else False
            ),
        )
        yield from loader

    def get_mock_batch(self) -> Era5SupervisedBatch:
        """Return a synthetic batch (used by the trainer's dry-run)."""
        if not self._task_names:
            raise RuntimeError("Dataloader has no tasks configured")
        task_name = self._task_names[0]
        spec = self.dataset.task_specs[task_name]
        bsz = self.rank_batch_size
        t_max = MAX_ERA5L_DAY_10_SEQUENCE_LENGTH
        c = _ERA5_MODALITY.num_bands
        era5 = torch.randn(bsz, t_max, c)
        timestamps = torch.zeros(bsz, t_max, 3, dtype=torch.long)
        timestamps[..., 0] = torch.arange(1, t_max + 1).unsqueeze(0)
        timestamps[..., 1] = (timestamps[..., 0] - 1) * 12 // 365
        timestamps[..., 2] = 1970
        ignore_mask = torch.zeros(bsz, t_max, dtype=torch.bool)
        if TaskType(spec.task_type) == TaskType.CLASSIFICATION:
            labels = torch.zeros(bsz, dtype=torch.long)
        else:
            labels = torch.zeros(bsz, dtype=torch.float32)
        return Era5SupervisedBatch(
            era5=era5,
            timestamps=timestamps,
            ignore_mask=ignore_mask,
            labels=labels,
            task_name=task_name,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return a checkpointable state dict."""
        return {
            "seed": self.seed,
            "epoch": self._epoch,
            "batches_processed": self.batches_processed,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore from a previous `state_dict`."""
        self.seed = state_dict.get("seed", self.seed)
        self._epoch = state_dict.get("epoch", self._epoch)
        self.batches_processed = state_dict.get("batches_processed", 0)


class _ScheduleDispatcher(torch.utils.data.IterableDataset):
    """Pure-lookup IterableDataset over a pre-computed schedule.

    Each PyTorch worker process owns a different slice of the global-step
    sequence (``step % num_workers == worker_id``); the schedule itself is
    read-only and identical across workers/ranks, so no per-worker mutable
    state is needed (which is what made the previous cursor-based approach
    unsafe under ``num_workers > 0``).
    """

    def __init__(
        self,
        task_datasets: dict[str, Era5TaskDataset],
        task_names: list[str],
        schedule_task_ids: np.ndarray,
        schedule_indices: np.ndarray,
        start_step: int,
        total: int,
        dp_rank: int,
        dp_world_size: int,
    ) -> None:
        super().__init__()
        self.task_datasets = task_datasets
        self.task_names = task_names
        self.schedule_task_ids = schedule_task_ids
        self.schedule_indices = schedule_indices
        self.start_step = start_step
        self.total = total
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size

    def __iter__(self) -> Iterator[Era5SupervisedBatch]:
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        for step in range(self.start_step, self.total):
            if (step % num_workers) != worker_id:
                continue
            task_id = int(self.schedule_task_ids[step])
            task_name = self.task_names[task_id]
            global_indices = self.schedule_indices[step]
            local_indices = global_indices[self.dp_rank :: self.dp_world_size]
            task_ds = self.task_datasets[task_name]
            samples = [task_ds[int(i)] for i in local_indices]
            yield _collate_samples(samples, task_name)
