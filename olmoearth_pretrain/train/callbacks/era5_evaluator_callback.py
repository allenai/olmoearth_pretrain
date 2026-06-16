"""Linear-probe evaluator callback for the ERA5 daily encoder.

This callback is **objective-agnostic**: it works for A-only (supervised),
B-only (reconstruction), and A+B combined runs. It evaluates encoder quality
by training a lightweight linear probe on frozen embeddings.

Flow per eval step:
  1. Load train/val (optionally test) splits via `Era5TaskDataset` from the
     direct-rslearn registry.
  2. Extract mean-pooled embeddings from `Era5DailyEncoder` (clean inputs).
  3. Train a linear probe on train embeddings → evaluate on val/test.
  4. Log metrics to wandb and the trainer metric store.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from olmo_core.train.callbacks.callback import Callback, CallbackConfig
from olmo_core.train.common import Duration
from olmo_core.train.trainer import Trainer
from torch.utils.data import DataLoader

from olmoearth_pretrain.data.constants import (
    MAX_ERA5L_DAY_10_SEQUENCE_LENGTH,
    Modality,
)
from olmoearth_pretrain.data.multi_task_era5_dataset import (
    Era5TaskDataset,
    Era5TaskSpec,
    _build_task_dataset,
    _collate_samples,
)
from olmoearth_pretrain.evals.datasets.configs import EvalDatasetConfig, TaskType
from olmoearth_pretrain.evals.linear_probe import ProbeType, train_and_eval_probe
from olmoearth_pretrain.evals.metrics import EvalResult, EvalTaskResult
from olmoearth_pretrain.train.callbacks.evaluator_callback import (
    _record_eval_result,
    eval_result_log_dict,
)
from olmoearth_pretrain.train.callbacks.wandb import OlmoEarthWandBCallback

logger = logging.getLogger(__name__)


@dataclass
class Era5LinearProbeTaskConfig:
    """Configuration for one linear-probe eval task sourced from the direct registry."""

    name: str
    weka_path: str
    model_yaml_path: str
    task_type: str
    num_classes: int | None
    is_multilabel: bool = False
    modality_layer_name: str = "era5_daily"
    label_extractor_name: str | None = None
    norm_stats_from_pretrained: bool = True
    train_groups: list[str] | None = None
    train_tags: dict[str, str] | None = None
    val_groups: list[str] | None = None
    val_tags: dict[str, str] | None = None
    test_groups: list[str] | None = None
    test_tags: dict[str, str] | None = None
    probe_lr: float = 1e-3
    probe_epochs: int = 50
    probe_batch_size: int = 256
    embedding_batch_size: int = 128
    eval_interval: Duration = field(default_factory=lambda: Duration.epochs(1))
    max_eval_samples: int | None = None


def _build_eval_dataset(
    task_config: Era5LinearProbeTaskConfig,
    split: str,
    max_sequence_length: int,
) -> Era5TaskDataset:
    """Build an `Era5TaskDataset` for the given split using registry metadata."""
    if split == "train":
        groups = task_config.train_groups
        tags = task_config.train_tags
    elif split in ("val", "valid"):
        groups = task_config.val_groups
        tags = task_config.val_tags
    else:
        groups = task_config.test_groups
        tags = task_config.test_tags

    spec = Era5TaskSpec(
        name=task_config.name,
        weight=1.0,
        task_type=task_config.task_type,
        is_multilabel=task_config.is_multilabel,
        num_classes=task_config.num_classes,
        modality_layer_name=task_config.modality_layer_name,
        label_extractor_name=task_config.label_extractor_name,
        norm_stats_from_pretrained=task_config.norm_stats_from_pretrained,
        weka_path=task_config.weka_path,
        model_yaml_path=task_config.model_yaml_path,
        groups_override=groups,
        tags_override=tags if tags else None,
        split=split if split != "valid" else "val",
        max_samples=task_config.max_eval_samples,
    )
    return _build_task_dataset(
        spec, registry=None, max_sequence_length=max_sequence_length
    )


def _collate_fn(samples: list) -> Any:
    """Collate wrapper that passes a dummy task_name (unused by probe)."""
    if not samples:
        return None
    return _collate_samples(samples, samples[0].task_name)


def _extract_embeddings(
    dataset: Era5TaskDataset,
    encoder: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract frozen mean-pooled embeddings from the encoder for all samples."""
    if len(dataset) == 0:
        raise ValueError(
            "Cannot extract embeddings from an empty dataset "
            f"(task={dataset.task_spec.name!r}, split={dataset.task_spec.split!r})."
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    all_embeddings: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    was_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            era5 = batch.era5.to(device)
            timestamps = batch.timestamps.to(device)
            ignore_mask = batch.ignore_mask.to(device)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                out = encoder(era5=era5, timestamps=timestamps, ignore_mask=ignore_mask)

            all_embeddings.append(out["pooled"].float().cpu())
            all_labels.append(batch.labels)

    if was_training:
        encoder.train()

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return embeddings, labels


def _get_encoder(trainer: Trainer) -> torch.nn.Module:
    """Resolve the encoder from the trainer's model (DDP-safe)."""
    model = trainer.train_module.model
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "encoder"):
        return model.encoder
    return model


def _get_wandb_callback(trainer: Trainer) -> OlmoEarthWandBCallback | None:
    """Return the (enabled) wandb callback attached to the trainer, if any."""
    for callback in trainer._iter_callbacks():
        if isinstance(callback, OlmoEarthWandBCallback) and callback.enabled:
            return callback
    return None


def _log_to_wandb(
    trainer: Trainer,
    prefix: str,
    name: str,
    result: EvalResult,
    step: int,
) -> None:
    """Log an EvalResult to wandb at an explicit trainer step.

    The step is passed explicitly so eval metrics land on the same step axis as
    the training metrics. Without it, ``wandb.log`` falls back to its internal
    auto-incrementing counter, which races with the trainer's step-keyed logging
    and causes wandb to drop out-of-order points ("Steps must be monotonically
    increasing, so this data will be ignored").
    """
    wandb_callback = _get_wandb_callback(trainer)
    if wandb_callback is not None:
        wandb_callback.wandb.log(eval_result_log_dict(prefix, name, result), step=step)


class Era5DownstreamEvaluatorCallback(Callback):
    """Runs linear-probe evaluations on the ERA5 encoder at a configured cadence."""

    def __init__(
        self,
        tasks: list[Era5LinearProbeTaskConfig],
        max_sequence_length: int = MAX_ERA5L_DAY_10_SEQUENCE_LENGTH,
        eval_on_startup: bool = False,
        run_on_test: bool = False,
        num_workers: int = 4,
    ) -> None:
        """Initialize the ERA5 downstream evaluator callback."""
        super().__init__()
        self.tasks = tasks
        self.max_sequence_length = max_sequence_length
        self.eval_on_startup = eval_on_startup
        self.run_on_test = run_on_test
        self.num_workers = num_workers
        # Step at which we last evaluated, so the end-of-training eval doesn't
        # redundantly re-run when training happens to stop on an interval.
        self._last_eval_step = -1

    def pre_train(self) -> None:
        """Run an evaluation on startup if configured."""
        if self.eval_on_startup:
            logger.info("Running ERA5 linear-probe eval on startup.")
            self._run_all_evals()

    def post_step(self) -> None:
        """Run evaluations for tasks whose interval has elapsed."""
        for task in self.tasks:
            eval_interval_steps = self.trainer.convert_duration_to_steps(
                task.eval_interval
            )
            if self.step <= 1 or self.step % eval_interval_steps != 0:
                continue
            self._run_eval(task)

    def post_train(self) -> None:
        """Always run a final eval when training finishes.

        Without this, a run whose final step never lands on an ``eval_interval``
        multiple (e.g. short runs where total steps < ``eval_interval``) would
        finish without ever logging a single eval result.
        """
        if self.step == self._last_eval_step:
            return
        logger.info("Running final ERA5 linear-probe eval at step %d.", self.step)
        self._run_all_evals()

    def _run_all_evals(self) -> None:
        for task in self.tasks:
            self._run_eval(task)

    def _run_eval(self, task: Era5LinearProbeTaskConfig) -> None:
        logger.info("ERA5 linear-probe eval: %s (step %d)", task.name, self.step)
        start_time = time.monotonic()

        device = self.trainer.device
        encoder = _get_encoder(self.trainer)

        try:
            train_ds = _build_eval_dataset(task, "train", self.max_sequence_length)
            val_ds = _build_eval_dataset(task, "val", self.max_sequence_length)
        except Exception:
            logger.exception(
                "Failed to build eval datasets for %s, skipping.", task.name
            )
            return

        if len(train_ds) == 0 or len(val_ds) == 0:
            logger.warning(
                "Skipping eval %s: empty split(s) (train=%d, val=%d). Check the "
                "task's group/tag filters in the direct registry.",
                task.name,
                len(train_ds),
                len(val_ds),
            )
            return

        train_embeddings, train_labels = _extract_embeddings(
            train_ds, encoder, device, task.embedding_batch_size, self.num_workers
        )
        val_embeddings, val_labels = _extract_embeddings(
            val_ds, encoder, device, task.embedding_batch_size, self.num_workers
        )

        test_embeddings: torch.Tensor | None = None
        test_labels: torch.Tensor | None = None
        if self.run_on_test:
            try:
                test_ds = _build_eval_dataset(task, "test", self.max_sequence_length)
                test_embeddings, test_labels = _extract_embeddings(
                    test_ds,
                    encoder,
                    device,
                    task.embedding_batch_size,
                    self.num_workers,
                )
            except Exception:
                logger.warning(
                    "Test split unavailable for %s, skipping test eval.", task.name
                )

        task_type = TaskType(task.task_type)
        eval_config = EvalDatasetConfig(
            task_type=task_type,
            num_classes=task.num_classes or 2,
            is_multilabel=task.is_multilabel,
            imputes=[],
            supported_modalities=[Modality.ERA5L_DAY_10.name],
        )

        logger.info(
            "Running linear probe for %s: train=%d, val=%d, test=%s",
            task.name,
            train_embeddings.shape[0],
            val_embeddings.shape[0],
            test_embeddings.shape[0] if test_embeddings is not None else "N/A",
        )

        result: EvalTaskResult = train_and_eval_probe(
            config=eval_config,
            lr=task.probe_lr,
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            val_embeddings=val_embeddings,
            val_labels=val_labels,
            test_embeddings=test_embeddings,
            test_labels=test_labels,
            device=device,
            batch_size=task.probe_batch_size,
            epochs=task.probe_epochs,
            eval_interval=task.probe_epochs,
            probe_type=ProbeType.LINEAR,
        )

        eval_time = time.monotonic() - start_time
        step = self.step
        self._last_eval_step = step
        self.trainer.record_metric(f"eval_time/{task.name}", eval_time)

        if result.val_result is not None:
            _record_eval_result(self.trainer, "eval", task.name, result.val_result)
            _log_to_wandb(self.trainer, "eval", task.name, result.val_result, step)
            logger.info(
                "ERA5 probe %s val: %.4f (%.1fs)",
                task.name,
                result.val_result.primary,
                eval_time,
            )

        if self.run_on_test and result.test_result is not None:
            _record_eval_result(
                self.trainer, "eval/test", task.name, result.test_result
            )
            _log_to_wandb(
                self.trainer, "eval/test", task.name, result.test_result, step
            )
            logger.info(
                "ERA5 probe %s test: %.4f",
                task.name,
                result.test_result.primary,
            )

        # Log eval time to wandb at the same explicit step as the eval metrics.
        wandb_callback = _get_wandb_callback(self.trainer)
        if wandb_callback is not None:
            wandb_callback.wandb.log({f"eval_time/{task.name}": eval_time}, step=step)

        del train_embeddings, train_labels, val_embeddings, val_labels
        del test_embeddings, test_labels
        torch.cuda.empty_cache()
        gc.collect()


@dataclass
class Era5DownstreamEvaluatorCallbackConfig(CallbackConfig):
    """Config for the ERA5 linear-probe evaluator callback."""

    tasks: list[Era5LinearProbeTaskConfig] = field(default_factory=list)
    enabled: bool = True
    eval_on_startup: bool = False
    run_on_test: bool = False
    num_workers: int = 4
    max_sequence_length: int = MAX_ERA5L_DAY_10_SEQUENCE_LENGTH

    def build(self, trainer: Trainer) -> Callback | None:
        """Build the callback, or return None if disabled."""
        if not self.enabled or not self.tasks:
            return None

        return Era5DownstreamEvaluatorCallback(
            tasks=self.tasks,
            max_sequence_length=self.max_sequence_length,
            eval_on_startup=self.eval_on_startup,
            run_on_test=self.run_on_test,
            num_workers=self.num_workers,
        )
