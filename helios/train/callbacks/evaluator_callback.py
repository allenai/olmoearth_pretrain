"""Downstream evaluator callback."""

import logging
import time
from dataclasses import dataclass, field

import torch
from olmo_core.eval.evaluator import Evaluator
from olmo_core.train.callbacks.callback import Callback, CallbackConfig
from olmo_core.train.common import Duration
from olmo_core.train.trainer import Trainer
from torch.utils.data import DataLoader
from upath import UPath

from helios.evals.datasets import GeobenchDataset
from helios.evals.embeddings import get_embeddings
from helios.evals.knn import run_knn
from helios.nn.flexihelios import PoolingType

logger = logging.getLogger(__name__)


# Geobench classification
METRIC_NAME = "accuracy"
NAME_PREFIX = "Geobench"
GEOBENCH_DIR = UPath("/weka/dfive-default/presto-geobench/dataset/geobench")


class DownstreamEvaluator:
    """Evaluator for downstream tasks."""

    def __init__(
        self,
        name: str,
        task: str,
        trainer: Trainer,
        batch_size: int = 128,
        num_workers: int = 8,
        pooling_type: PoolingType = PoolingType.MEAN,
        norm_stats_from_pretrained: bool = True,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the downstream evaluator."""
        self.name = name
        self.task = task
        self.trainer = trainer
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pooling_type = pooling_type
        self.norm_stats_from_pretrained = norm_stats_from_pretrained

    def _get_data_loader(self, split: str) -> DataLoader:
        """Get the data loader for the given split."""
        return DataLoader(
            GeobenchDataset(
                GEOBENCH_DIR,
                self.task,
                split,
                "default",
                norm_stats_from_pretrained=self.norm_stats_from_pretrained,
            ),
            collate_fn=GeobenchDataset.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def _get_embeddings(self, data_loader: DataLoader) -> tuple:
        """Get the embeddings for the given data loader."""
        return get_embeddings(
            data_loader=data_loader,
            model=self.trainer.train_module.model.encoder,
            patch_size=self.trainer.train_module.model.encoder.max_patch_size,
            pooling_type=self.pooling_type,
        )

    def val(self) -> float:
        """Validate the model on the downstream task."""
        try:
            train_loader = self._get_data_loader("train")
            val_loader = self._get_data_loader("valid")

            train_embeddings, train_labels = self._get_embeddings(train_loader)
            test_embeddings, test_labels = self._get_embeddings(val_loader)

            val_result = run_knn(
                eval_type="KNN-20",
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                test_embeddings=test_embeddings,
                test_labels=test_labels,
                num_classes=train_loader.dataset.num_classes,
                is_multilabel=train_loader.dataset.is_multilabel,
                device=self.device,
            )
            logger.info(
                f"Downstream evaluator {self.name} {METRIC_NAME} score: {val_result}"
            )
            return val_result
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return 0


@dataclass
class DownstreamEvaluatorCallback(Callback):
    """Runs in-loop evaluations periodically during training."""

    evaluators: list[DownstreamEvaluator] = field(default_factory=list)
    eval_duration: Duration = field(default_factory=lambda: Duration.epochs(1))

    def post_step(self) -> None:
        """Run the evaluators."""
        # Compute the evaluation interval in steps.
        eval_interval_steps = self.trainer.convert_duration_to_steps(self.eval_duration)
        if self.step <= 1 or self.step % eval_interval_steps != 0:
            return

        for evaluator in self.evaluators:
            logger.info(f"Running {evaluator.name} evaluations...")
            start_time = time.monotonic()
            val_result = evaluator.val()
            self.trainer.record_metric(
                f"eval/{evaluator.name}/{METRIC_NAME}", val_result
            )
            logger.info(
                f"Finished {evaluator.name} evaluations in {time.monotonic() - start_time:.1f} seconds."
            )
            logger.info(f"Metric {METRIC_NAME}: {val_result}")


@dataclass
class DownstreamTaskConfig:
    """Config for a downstream task."""

    name: str
    batch_size: int = 128
    num_workers: int = 8
    pooling_type: PoolingType = PoolingType.MEAN
    norm_stats_from_pretrained: bool = True


@dataclass
class DownstreamEvaluatorCallbackConfig(CallbackConfig):
    """Config for the downstream evaluator callback."""

    tasks: list[DownstreamTaskConfig]
    eval_duration: Duration = field(default_factory=lambda: Duration.epochs(1))
    enabled: bool = True

    def build(self, trainer: Trainer) -> Callback | None:
        """Build the downstream evaluator callback."""
        if not self.enabled:
            return None

        evaluators: list[Evaluator] = [
            DownstreamEvaluator(
                name=f"{NAME_PREFIX}-{task.name}",
                task=task.name,
                trainer=trainer,
                batch_size=task.batch_size,
                num_workers=task.num_workers,
                pooling_type=task.pooling_type,
                norm_stats_from_pretrained=task.norm_stats_from_pretrained,
                device=trainer.device,
            )
            for task in self.tasks
        ]

        return DownstreamEvaluatorCallback(
            evaluators=evaluators,
            eval_duration=self.eval_duration,
        )
