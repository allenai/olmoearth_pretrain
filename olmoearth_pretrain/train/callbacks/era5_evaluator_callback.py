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

import contextlib
import gc
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from olmo_core.train.callbacks.callback import Callback, CallbackConfig
from olmo_core.train.common import Duration
from olmo_core.train.trainer import Trainer
from torch.utils.data import DataLoader

from olmoearth_pretrain.data.constants import (
    ERA5_INPUT_SEQUENCE_LENGTH,
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
    # When set, the probe's weight init AND the training DataLoader's shuffle
    # order are pinned to this seed (see `_run_eval`), making probe results
    # comparable across runs / probe-LR values. None keeps legacy behavior
    # (both drawn from the ambient global RNG state).
    probe_seed: int | None = None
    probe_epochs: int = 50
    probe_batch_size: int = 256
    embedding_batch_size: int = 128
    eval_interval: Duration = field(default_factory=lambda: Duration.epochs(1))
    max_eval_samples: int | None = None
    height_width: int | None = None


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
    """Collate wrapper for the downstream eval DataLoader."""
    if not samples:
        return None
    return _collate_samples(samples)


def _materialize_batches(
    dataset: Era5TaskDataset,
    batch_size: int,
    num_workers: int = 4,
) -> list[Any]:
    """Load every sample of *dataset* into a list of collated CPU batches.

    Materializing lets callers (e.g. the checkpoint sweep) pay the dataset
    read cost once and re-run the encoder over the same batches many times.
    The full eval datasets are small (thousands of [T, C] daily sequences),
    so holding them in RAM is cheap.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    return [batch for batch in loader if batch is not None]


def _extract_embeddings(
    batches: list[Any],
    encoder: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract frozen mean-pooled embeddings from the encoder for all batches."""
    all_embeddings: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    was_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        for batch in batches:
            era5 = batch.era5.to(device)
            timestamps = batch.timestamps.to(device)
            valid_mask = getattr(batch, "valid_mask", None)
            if valid_mask is not None:
                valid_mask = valid_mask.to(device)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                out = encoder(era5=era5, timestamps=timestamps, valid_mask=valid_mask)

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
        max_sequence_length: int = ERA5_INPUT_SEQUENCE_LENGTH,
        eval_on_startup: bool = False,
        run_on_test: bool = False,
        num_workers: int = 4,
        checkpoint_sweep_dir: str | None = None,
        checkpoint_sweep_interval: int = 0,
    ) -> None:
        """Initialize the ERA5 downstream evaluator callback.

        Args:
            tasks: Linear-probe eval task configs (from the direct registry).
            max_sequence_length: Maximum daily-timestep sequence length fed to
                the eval datasets/encoder.
            eval_on_startup: Run a full evaluation in ``pre_train`` (ignored
                when ``checkpoint_sweep_dir`` is set).
            run_on_test: Also evaluate each task's test split.
            num_workers: DataLoader workers for eval batch materialization.
            checkpoint_sweep_dir: When set (eval-only runs), instead of a
                single startup eval, iterate over the ``step*`` checkpoints
                saved under this run folder: load each checkpoint's weights
                and run all probe evals, logging metrics at that checkpoint's
                step. This reproduces the eval-vs-step curves of a training
                run against frozen, saved weights.
            checkpoint_sweep_interval: Only evaluate checkpoints whose step is
                a multiple of this value (e.g. 5000 -> step0, step5000, ...).
                The last (highest-step) checkpoint is always included. 0 means
                every saved checkpoint.
        """
        super().__init__()
        self.tasks = tasks
        self.max_sequence_length = max_sequence_length
        self.eval_on_startup = eval_on_startup
        self.run_on_test = run_on_test
        self.num_workers = num_workers
        self.checkpoint_sweep_dir = checkpoint_sweep_dir
        self.checkpoint_sweep_interval = checkpoint_sweep_interval
        # Step at which we last evaluated, so the end-of-training eval doesn't
        # redundantly re-run when training happens to stop on an interval.
        self._last_eval_step = -1
        # Per-task cache of materialized eval batches, so a checkpoint sweep
        # pays the dataset read cost once and reuses the raw batches for every
        # checkpoint (embeddings still get recomputed per checkpoint).
        self._batch_cache: dict[str, tuple[list[Any], list[Any], list[Any] | None]] = {}

    def pre_train(self) -> None:
        """Run the checkpoint sweep or a startup evaluation if configured."""
        if self.checkpoint_sweep_dir is not None:
            self._run_checkpoint_sweep()
            return
        if self.eval_on_startup:
            logger.info("Running ERA5 linear-probe eval on startup.")
            self._run_all_evals()

    def _discover_checkpoints(self) -> list[tuple[int, Path]]:
        """List (step, path) for saved checkpoints, filtered by the sweep interval."""
        assert self.checkpoint_sweep_dir is not None
        root = Path(self.checkpoint_sweep_dir)
        checkpoints: list[tuple[int, Path]] = []
        for child in root.iterdir():
            match = re.fullmatch(r"step(\d+)", child.name)
            if match and child.is_dir():
                checkpoints.append((int(match.group(1)), child))
        checkpoints.sort()
        if not checkpoints:
            raise ValueError(
                f"No step* checkpoints found under {self.checkpoint_sweep_dir!r}"
            )
        if self.checkpoint_sweep_interval > 0:
            last_step = checkpoints[-1][0]
            checkpoints = [
                (step, path)
                for step, path in checkpoints
                if step % self.checkpoint_sweep_interval == 0 or step == last_step
            ]
        return checkpoints

    def _run_checkpoint_sweep(self) -> None:
        """Evaluate every selected checkpoint, logging at its original step."""
        checkpoints = self._discover_checkpoints()
        logger.info(
            "ERA5 checkpoint sweep over %d checkpoints (interval=%d): %s",
            len(checkpoints),
            self.checkpoint_sweep_interval,
            [step for step, _ in checkpoints],
        )
        for step, path in checkpoints:
            logger.info("Checkpoint sweep: loading step %d from %s", step, path)
            self.trainer.checkpointer.load(
                str(path),
                self.trainer.train_module,
                load_trainer_state=False,
                load_optim_state=False,
            )
            for task in self.tasks:
                self._run_eval(task, log_step=step)

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
        if self.checkpoint_sweep_dir is not None:
            # The sweep already evaluated every selected checkpoint; the
            # trainer's own step (0) is meaningless here.
            return
        if self.step == self._last_eval_step:
            return
        logger.info("Running final ERA5 linear-probe eval at step %d.", self.step)
        self._run_all_evals()

    def _run_all_evals(self) -> None:
        for task in self.tasks:
            self._run_eval(task)

    def _get_task_batches(
        self, task: Era5LinearProbeTaskConfig
    ) -> tuple[list[Any], list[Any], list[Any] | None] | None:
        """Materialize (and cache) the raw eval batches for *task*."""
        if task.name in self._batch_cache:
            return self._batch_cache[task.name]

        try:
            train_ds = _build_eval_dataset(task, "train", self.max_sequence_length)
            val_ds = _build_eval_dataset(task, "val", self.max_sequence_length)
        except Exception:
            logger.exception(
                "Failed to build eval datasets for %s, skipping.", task.name
            )
            return None

        if len(train_ds) == 0 or len(val_ds) == 0:
            logger.warning(
                "Skipping eval %s: empty split(s) (train=%d, val=%d). Check the "
                "task's group/tag filters in the direct registry.",
                task.name,
                len(train_ds),
                len(val_ds),
            )
            return None

        train_batches = _materialize_batches(
            train_ds, task.embedding_batch_size, self.num_workers
        )
        val_batches = _materialize_batches(
            val_ds, task.embedding_batch_size, self.num_workers
        )

        test_batches: list[Any] | None = None
        if self.run_on_test:
            try:
                test_ds = _build_eval_dataset(task, "test", self.max_sequence_length)
                test_batches = _materialize_batches(
                    test_ds, task.embedding_batch_size, self.num_workers
                )
            except Exception:
                logger.warning(
                    "Test split unavailable for %s, skipping test eval.", task.name
                )

        batches = (train_batches, val_batches, test_batches)
        self._batch_cache[task.name] = batches
        return batches

    def _run_eval(
        self, task: Era5LinearProbeTaskConfig, log_step: int | None = None
    ) -> None:
        step = self.step if log_step is None else log_step
        logger.info("ERA5 linear-probe eval: %s (step %d)", task.name, step)
        start_time = time.monotonic()

        device = self.trainer.device
        encoder = _get_encoder(self.trainer)

        batches = self._get_task_batches(task)
        if batches is None:
            return
        train_batches, val_batches, test_batches = batches

        train_embeddings, train_labels = _extract_embeddings(
            train_batches, encoder, device
        )
        val_embeddings, val_labels = _extract_embeddings(val_batches, encoder, device)

        test_embeddings: torch.Tensor | None = None
        test_labels: torch.Tensor | None = None
        if test_batches is not None:
            test_embeddings, test_labels = _extract_embeddings(
                test_batches, encoder, device
            )

        task_type = TaskType(task.task_type)

        # Spatial probes (segmentation / regression) expect (N, H, W, D)
        # embeddings and (N, H, W) labels.  Classification uses a flat
        # LinearProbe that needs (N, D) — don't unsqueeze for it.
        _spatial = task_type in (TaskType.SEGMENTATION, TaskType.REGRESSION)
        if _spatial and train_embeddings.ndim == 2:
            train_embeddings = train_embeddings.unsqueeze(1).unsqueeze(1)
            val_embeddings = val_embeddings.unsqueeze(1).unsqueeze(1)
            if test_embeddings is not None:
                test_embeddings = test_embeddings.unsqueeze(1).unsqueeze(1)
        if _spatial and train_labels.ndim == 1:
            train_labels = train_labels.unsqueeze(-1).unsqueeze(-1)
            val_labels = val_labels.unsqueeze(-1).unsqueeze(-1)
            if test_labels is not None:
                test_labels = test_labels.unsqueeze(-1).unsqueeze(-1)
        eval_config = EvalDatasetConfig(
            task_type=task_type,
            num_classes=task.num_classes or 2,
            is_multilabel=task.is_multilabel,
            imputes=[],
            supported_modalities=[Modality.ERA5L_DAY_10.name],
            height_width=task.height_width,
        )

        logger.info(
            "Running linear probe for %s: train=%d, val=%d, test=%s",
            task.name,
            train_embeddings.shape[0],
            val_embeddings.shape[0],
            test_embeddings.shape[0] if test_embeddings is not None else "N/A",
        )

        # When probe_seed is set, pin the probe's init and batch ordering:
        # both the nn.Linear/Conv init and the shuffling DataLoader's
        # RandomSampler draw from the process-global torch RNG, so seeding it
        # here makes the probe fully deterministic in those two respects.
        # fork_rng restores the global RNG afterwards so a mid-training eval
        # doesn't perturb the pretraining RNG stream.
        if task.probe_seed is not None:
            rng_context = torch.random.fork_rng(
                devices=[device] if device.type == "cuda" else []
            )
        else:
            rng_context = contextlib.nullcontext()
        with rng_context:
            if task.probe_seed is not None:
                torch.manual_seed(task.probe_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(task.probe_seed)
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
        self._last_eval_step = self.step
        # During a checkpoint sweep the trainer's own step is frozen at 0, so
        # recording into the trainer metric store would flush everything at
        # step 0 (out of order vs. our explicit per-checkpoint wandb steps).
        # Log to the trainer store only in the normal in-training path.
        record_to_trainer = log_step is None
        if record_to_trainer:
            self.trainer.record_metric(f"eval_time/{task.name}", eval_time)

        if result.val_result is not None:
            if record_to_trainer:
                _record_eval_result(self.trainer, "eval", task.name, result.val_result)
            _log_to_wandb(self.trainer, "eval", task.name, result.val_result, step)
            logger.info(
                "ERA5 probe %s val: %.4f (%.1fs)",
                task.name,
                result.val_result.primary,
                eval_time,
            )

        if self.run_on_test and result.test_result is not None:
            if record_to_trainer:
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
    max_sequence_length: int = ERA5_INPUT_SEQUENCE_LENGTH
    checkpoint_sweep_dir: str | None = None
    checkpoint_sweep_interval: int = 0

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
            checkpoint_sweep_dir=self.checkpoint_sweep_dir,
            checkpoint_sweep_interval=self.checkpoint_sweep_interval,
        )
