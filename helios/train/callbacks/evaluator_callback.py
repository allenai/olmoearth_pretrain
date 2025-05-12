"""Downstream evaluator callback."""

import gc
import logging
import time
from dataclasses import dataclass, field

import torch
from olmo_core.train.callbacks.callback import Callback, CallbackConfig
from olmo_core.train.common import Duration
from olmo_core.train.trainer import Trainer
from torch.utils.data import DataLoader

from helios.evals.datasets import get_eval_dataset
from helios.evals.datasets.configs import DATASET_TO_CONFIG, TaskType
from helios.evals.datasets.utils import eval_collate_fn
from helios.evals.embeddings import get_embeddings
from helios.evals.knn import run_knn
from helios.evals.linear_probe import train_and_eval_probe
from helios.nn.flexihelios import PoolingType

logger = logging.getLogger(__name__)


class DownstreamEvaluator:
    """Evaluator for downstream tasks."""

    def __init__(
        self,
        evaluation_name: str,
        dataset: str,
        trainer: Trainer,
        eval_interval: Duration,
        batch_size: int = 128,
        num_workers: int = 8,
        patch_size: int = 4,
        pooling_type: PoolingType = PoolingType.MEAN,
        norm_stats_from_pretrained: bool = True,
        device: torch.device | None = None,
        probe_lr: float | None = None,
        input_modalities: list[str] = field(default_factory=list),
    ) -> None:
        """Initialize the downstream evaluator.

        Args:
            evaluation_name: Name of the evaluation.
            dataset: Dataset to evaluate on.
            trainer: Trainer object.
            eval_interval: Interval to evaluate on.
            batch_size: Batch size.
            num_workers: Number of workers.
            patch_size: Patch size.
            pooling_type: Pooling type.
            norm_stats_from_pretrained: Whether to use normalized stats from pretrained model.
            device: Device to evaluate on.
            probe_lr: Learning rate for probe.
            input_modalities: Input modalities, only used for multimodal tasks.
        """
        self.evaluation_name = evaluation_name
        self.dataset = dataset
        self.config = DATASET_TO_CONFIG[dataset]
        self.eval_interval = eval_interval
        self.trainer = trainer
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pooling_type = pooling_type
        self.norm_stats_from_pretrained = norm_stats_from_pretrained
        self.probe_lr = probe_lr
        self.patch_size = patch_size
        self.input_modalities = input_modalities

    def _get_data_loader(self, split: str) -> DataLoader:
        """Get the data loader for the given split."""
        return DataLoader(
            get_eval_dataset(
                eval_dataset=self.dataset,
                split=split,
                partition="default",
                norm_stats_from_pretrained=self.norm_stats_from_pretrained,
                input_modalities=self.input_modalities,
            ),
            collate_fn=eval_collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def _get_embeddings(
        self, data_loader: DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the embeddings for the given data loader."""
        return get_embeddings(
            data_loader=data_loader,
            task_type=self.config.task_type,
            model=self.trainer.train_module.model.encoder,
            patch_size=self.patch_size,
            pooling_type=self.pooling_type,
        )

    def val(self) -> float:
        """Validate the model on the downstream task."""
        train_loader = self._get_data_loader("train")
        val_loader = self._get_data_loader("valid")

        train_embeddings, train_labels = self._get_embeddings(train_loader)
        test_embeddings, test_labels = self._get_embeddings(val_loader)

        logger.info(
            f"train embeddings shape for {self.dataset}: {train_embeddings.shape}"
        )
        logger.info(
            f"test embeddings shape for {self.dataset}: {test_embeddings.shape}"
        )
        logger.info(f"train labels shape for {self.dataset}: {train_labels.shape}")
        logger.info(f"test labels shape for {self.dataset}: {test_labels.shape}")

        if self.config.task_type == TaskType.CLASSIFICATION:
            val_result = run_knn(
                config=self.config,
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                test_embeddings=test_embeddings,
                test_labels=test_labels,
                device=self.device,
            )
        elif self.config.task_type == TaskType.SEGMENTATION:
            if self.probe_lr is None:
                raise ValueError("probe_lr cannot be none for segmentation tasks.")
            if self.config.height_width is None:
                raise ValueError(
                    "config.height_width cannot be none for segmentation tasks."
                )
            if self.config.height_width % self.patch_size != 0:
                raise ValueError("Image height / width indivisable by patch size.")
            val_result = train_and_eval_probe(
                config=self.config,
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                test_embeddings=test_embeddings,
                test_labels=test_labels,
                device=self.device,
                batch_size=self.batch_size,
                lr=self.probe_lr,
                grid_size=int(self.config.height_width / self.patch_size),
            )
        else:
            raise ValueError(f"Unrecognized task type: {self.config.task_type}")
        logger.info(f"Downstream evaluator {self.evaluation_name} score: {val_result}")
        # free memory
        del train_embeddings, train_labels, test_embeddings, test_labels
        torch.cuda.empty_cache()
        gc.collect()

        return val_result


@dataclass
class DownstreamEvaluatorCallback(Callback):
    """Runs in-loop evaluations periodically during training."""

    evaluators: list[DownstreamEvaluator] = field(default_factory=list)

    def post_step(self) -> None:
        """Run the evaluators."""
        for evaluator in self.evaluators:
            eval_interval_steps = self.trainer.convert_duration_to_steps(
                evaluator.eval_interval
            )
            if self.step <= 1 or self.step % eval_interval_steps != 0:
                continue
            logger.info(f"Running {evaluator.evaluation_name} evaluations...")
            start_time = time.monotonic()
            val_result = evaluator.val()
            self.trainer.record_metric(f"eval/{evaluator.evaluation_name}", val_result)
            logger.info(
                f"Finished {evaluator.evaluation_name} evaluations in {time.monotonic() - start_time:.1f} seconds."
            )


@dataclass
class DownstreamTaskConfig:
    """Config for a downstream task."""

    dataset: str
    batch_size: int = 128
    num_workers: int = 8
    pooling_type: PoolingType = PoolingType.MEAN
    norm_stats_from_pretrained: bool = True
    input_modalities: list[str] = field(default_factory=list)
    # for MADOS and a default partition, the following lrs
    # did best for Galileo:
    # ViT-nano = 0.8 or 0.5
    # ViT-tiny = 0.1
    # ViT-base = 0.01
    probe_lr: float | None = None
    patch_size: int = 4
    eval_interval: Duration = field(default_factory=lambda: Duration.epochs(1))


@dataclass
class DownstreamEvaluatorCallbackConfig(CallbackConfig):
    """Config for the downstream evaluator callback."""

    tasks: dict[str, DownstreamTaskConfig]

    enabled: bool = True

    def build(self, trainer: Trainer) -> Callback | None:
        """Build the downstream evaluator callback."""
        if not self.enabled:
            return None

        evaluators: list[DownstreamEvaluator] = []
        # Check that probe_lr is set for segmentation tasks
        for evaluation_name, task in self.tasks.items():
            config = DATASET_TO_CONFIG[task.dataset]
            if config.task_type == TaskType.SEGMENTATION:
                if task.probe_lr is None:
                    raise ValueError(f"probe_lr cannot be None for {task.dataset}")

            # Check that input_modalities is only set for multimodal tasks
            if (
                task.dataset not in ["pastis", "sickle"]
                and len(task.input_modalities) > 0
            ):
                raise ValueError(
                    f"input_modalities must be set for multimodal tasks, got {task.dataset}"
                )
            # Make sure input_modalities contains only unique modalities
            if len(task.input_modalities) != len(set(task.input_modalities)):
                raise ValueError(
                    f"input_modalities must contain unique modalities, got {task.input_modalities}"
                )
            # Sort to ensure consistent order
            task.input_modalities.sort()

            evaluators.append(
                DownstreamEvaluator(
                    evaluation_name=evaluation_name,
                    dataset=task.dataset,
                    trainer=trainer,
                    batch_size=task.batch_size,
                    num_workers=task.num_workers,
                    pooling_type=task.pooling_type,
                    norm_stats_from_pretrained=task.norm_stats_from_pretrained,
                    input_modalities=task.input_modalities,
                    device=trainer.device,
                    probe_lr=task.probe_lr,
                    patch_size=task.patch_size,
                    eval_interval=task.eval_interval,
                )
            )
        return DownstreamEvaluatorCallback(
            evaluators=evaluators,
        )
