"""Downstream evaluator callback."""

import gc
import logging
import time
from dataclasses import dataclass, field
from functools import partial

import torch
from olmo_core.train.callbacks.callback import Callback, CallbackConfig
from olmo_core.train.common import Duration
from olmo_core.train.trainer import Trainer
from torch.utils.data import DataLoader

from helios.evals.datasets import EvalDatasetPartition, get_eval_dataset
from helios.evals.datasets.configs import TaskType, dataset_to_config
from helios.evals.datasets.normalize import NormMethod
from helios.evals.datasets.utils import eval_collate_fn
from helios.evals.embeddings import get_embeddings
from helios.evals.eval_wrapper import get_eval_wrapper
from helios.evals.knn import run_knn
from helios.evals.linear_probe import train_and_eval_probe
from helios.nn.flexihelios import PoolingType
from helios.train.callbacks.wandb import HeliosWandBCallback

logger = logging.getLogger(__name__)


@dataclass
class DownstreamTaskConfig:
    """Config for a downstream task."""

    dataset: str
    embedding_batch_size: int = 128
    num_workers: int = 8
    pooling_type: PoolingType = PoolingType.MEAN
    norm_stats_from_pretrained: bool = True
    input_modalities: list[str] = field(default_factory=list)
    # Sweep across lrs for segmentation tasks
    probe_lr: float | None = None
    patch_size: int = 4
    probe_batch_size: int = 32
    epochs: int = 50  # Number of training epochs for linear probing task
    eval_interval: Duration = field(default_factory=lambda: Duration.epochs(1))
    eval_mode: str | None = None
    probe_type: str = "linear"
    partition: str = field(default_factory=lambda: EvalDatasetPartition.TRAIN1X)
    norm_method: str = field(default_factory=lambda: NormMethod.NORM_NO_CLIP)


class DownstreamEvaluator:
    """Evaluator for downstream tasks."""

    def __init__(
        self,
        evaluation_name: str,
        task: DownstreamTaskConfig,
        trainer: Trainer,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the downstream evaluator.

        Args:
            evaluation_name: Name of the evaluation.
            task: Task configuration.
            trainer: Trainer object.
            device: Device to evaluate on.
        """
        self.evaluation_name = evaluation_name
        self.config = dataset_to_config(task.dataset)
        self.trainer = trainer
        self.device = device
        # Add all task attributes to self
        self.dataset = task.dataset
        self.embedding_batch_size = task.embedding_batch_size
        self.num_workers = task.num_workers
        self.pooling_type = task.pooling_type
        self.norm_stats_from_pretrained = task.norm_stats_from_pretrained
        self.input_modalities = task.input_modalities
        self.probe_lr = task.probe_lr
        self.patch_size = task.patch_size
        self.probe_batch_size = task.probe_batch_size
        self.epochs = task.epochs
        self.eval_interval = task.eval_interval
        self.eval_mode = task.eval_mode
        self.probe_type = task.probe_type
        self.partition = task.partition
        self.norm_method = task.norm_method
        if self.eval_mode is None:
            self.eval_mode = (
                "knn"
                if self.config.task_type == TaskType.CLASSIFICATION
                else "linear_probe"
            )

        assert self.eval_mode in [
            "knn",
            "linear_probe",
        ], f"Unexpected eval mode {self.eval_mode}"
        if self.eval_mode == "linear_probe":
            if self.probe_lr is None:
                raise ValueError("probe_lr cannot be none for segmentation tasks.")
            if self.config.task_type == TaskType.SEGMENTATION:
                if self.config.height_width is None:
                    raise ValueError(
                        "config.height_width cannot be none for segmentation tasks."
                    )
                if self.config.height_width % self.patch_size != 0:
                    raise ValueError("Image height / width indivisable by patch size.")
        self.eval_function = (
            run_knn
            if self.eval_mode == "knn"
            else partial(
                train_and_eval_probe,
                batch_size=self.probe_batch_size,
                epochs=self.epochs,
                eval_interval=self.eval_interval.value,
                probe_type=self.probe_type,
                lr=self.probe_lr,
                patch_size=self.patch_size,
            )
        )

    def _get_data_loader(self, split: str) -> DataLoader:
        """Get the data loader for the given split."""
        return DataLoader(
            get_eval_dataset(
                eval_dataset=self.dataset,
                split=split,
                partition=self.partition,
                norm_stats_from_pretrained=self.norm_stats_from_pretrained,
                input_modalities=self.input_modalities,
            ),
            collate_fn=eval_collate_fn,
            batch_size=self.embedding_batch_size,
            num_workers=self.num_workers,
        )

    def _get_embeddings(
        self, data_loader: DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the embeddings for the given data loader."""
        print(
            f"Getting embeddings for {self.dataset} with norm method {self.norm_method}"
        )
        if hasattr(self.trainer.train_module.model, "encoder"):
            model = self.trainer.train_module.model.encoder
        else:
            model = self.trainer.train_module.model

        # Superset of the kwargs the wrapper may need
        wrapper_kwargs = {
            "task_type": self.config.task_type,
            "patch_size": self.patch_size,
            "pooling_type": self.pooling_type,
            "concat_features": (self.probe_type == "attn_pool"),
        }
        model = get_eval_wrapper(model, **wrapper_kwargs)
        return get_embeddings(
            data_loader=data_loader,
            model=model,
        )

    def val(self) -> float:
        """Validate the model on the downstream task."""
        train_loader = self._get_data_loader("train")
        val_loader = self._get_data_loader("valid")

        start_time = time.time()
        logger.info(f"Getting train embeddings for {self.dataset}...")
        train_embeddings, train_labels = self._get_embeddings(train_loader)
        logger.info(f"Getting test embeddings for {self.dataset}...")
        test_embeddings, test_labels = self._get_embeddings(val_loader)
        logger.info(
            f"Time to get embeddings for {self.dataset}: {time.time() - start_time:.2f}s"
        )

        logger.info(
            f"train embeddings shape for {self.dataset}: {train_embeddings.shape}"
        )
        logger.info(
            f"test embeddings shape for {self.dataset}: {test_embeddings.shape}"
        )
        logger.info(f"train labels shape for {self.dataset}: {train_labels.shape}")
        logger.info(f"test labels shape for {self.dataset}: {test_labels.shape}")

        val_result = self.eval_function(  # type: ignore
            config=self.config,
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            test_embeddings=test_embeddings,
            test_labels=test_labels,
            device=self.device,
        )
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
    eval_on_startup: bool = False
    cancel_after_first_eval: bool = False

    def pre_train(self) -> None:
        """Run the evaluators on startup."""
        if self.eval_on_startup:
            logger.info(f"Running {len(self.evaluators)} evaluators on startup.")

            # self.trainer.record_metric() is not logging to wandb at this point
            # therefore we log to wandb manually
            wandb_callback = next(
                callback
                for callback in self.trainer._iter_callbacks()
                if isinstance(callback, HeliosWandBCallback)
            )
            for evaluator in self.evaluators:
                val_result, eval_time = self._perform_eval(evaluator)
                wandb_callback.wandb.log(
                    {"eval/" + evaluator.evaluation_name: val_result}
                )
                wandb_callback.wandb.log(
                    {"eval_time/" + evaluator.evaluation_name: eval_time}
                )

        if self.cancel_after_first_eval:
            self.trainer.cancel_run(
                "Cancelled from evaluator callback since 'cancel_after_first_eval' is set",
                no_sync=True,  # 'no_sync' because we're calling this from all ranks at the same time.
            )

    def post_step(self) -> None:
        """Run the evaluators in-loop."""
        for evaluator in self.evaluators:
            eval_interval_steps = self.trainer.convert_duration_to_steps(
                evaluator.eval_interval
            )
            if self.step <= 1 or self.step % eval_interval_steps != 0:
                continue
            self._perform_eval(evaluator)

    def _perform_eval(self, evaluator: DownstreamEvaluator) -> tuple[float, float]:
        """Run the evaluator."""
        logger.info(f"Running {evaluator.evaluation_name} evaluations...")
        start_time = time.monotonic()
        val_result = evaluator.val()
        self.trainer.record_metric(f"eval/{evaluator.evaluation_name}", val_result)
        eval_time = time.monotonic() - start_time
        self.trainer.record_metric(f"eval_time/{evaluator.evaluation_name}", eval_time)
        logger.info(
            f"Finished {evaluator.evaluation_name} evaluations in {eval_time:.1f} seconds."
        )
        return val_result, eval_time


@dataclass
class DownstreamEvaluatorCallbackConfig(CallbackConfig):
    """Config for the downstream evaluator callback."""

    tasks: dict[str, DownstreamTaskConfig]
    enabled: bool = True
    # Whether to run the evaluators on startup
    eval_on_startup: bool = False
    # Whether to cancel the training after the first evaluation
    # This combined with ``eval_on_startup=True`` is useful if you just want to run in-loop evals
    # without training any longer.
    cancel_after_first_eval: bool = False

    def build(self, trainer: Trainer) -> Callback | None:
        """Build the downstream evaluator callback."""
        if not self.enabled:
            return None

        evaluators: list[DownstreamEvaluator] = []
        # Check that probe_lr is set for segmentation tasks
        for evaluation_name, task in self.tasks.items():
            config = dataset_to_config(task.dataset)
            if config.task_type == TaskType.SEGMENTATION:
                if task.probe_lr is None:
                    raise ValueError(f"probe_lr cannot be None for {task.dataset}")

            # Check that input_modalities is only set for multimodal tasks
            if (
                not (
                    (task.dataset in ["pastis", "sickle"])
                    or task.dataset.startswith("cropharvest")
                )
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
                    task=task,
                    trainer=trainer,
                    device=trainer.device,
                )
            )
        return DownstreamEvaluatorCallback(
            evaluators=evaluators,
            eval_on_startup=self.eval_on_startup,
            cancel_after_first_eval=self.cancel_after_first_eval,
        )
