"""Downstream evaluator callback."""

import gc
import logging
import os
import random
import time
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial

import numpy as np
import torch
from olmo_core.train.callbacks.callback import Callback, CallbackConfig
from olmo_core.train.common import Duration
from olmo_core.train.trainer import Trainer
from torch.utils.data import DataLoader

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.datasets import (
    EvalDatasetPartition,
    get_eval_dataset,
)
from olmoearth_pretrain.evals.datasets.configs import (
    EvalDatasetConfig,
    TaskType,
    dataset_to_config,
    get_eval_mode,
)
from olmoearth_pretrain.evals.datasets.normalize import NormMethod
from olmoearth_pretrain.evals.datasets.utils import eval_collate_fn
from olmoearth_pretrain.evals.embeddings import get_embeddings
from olmoearth_pretrain.evals.eval_wrapper import get_eval_wrapper
from olmoearth_pretrain.evals.finetune import run_finetune_eval
from olmoearth_pretrain.evals.knn import run_knn
from olmoearth_pretrain.evals.linear_probe import ProbeType, train_and_eval_probe
from olmoearth_pretrain.nn.flexi_vit import PoolingType
from olmoearth_pretrain.train.callbacks.wandb import OlmoEarthWandBCallback

logger = logging.getLogger(__name__)


def _seed_worker(worker_id: int, base_seed: int) -> None:
    """Seed DataLoader worker RNGs deterministically."""
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class EvalMode(StrEnum):
    """Eval mode."""

    KNN = "knn"
    LINEAR_PROBE = "linear_probe"
    FINETUNE = "finetune"


@dataclass
class DownstreamTaskConfig:
    """Config for a downstream task."""

    dataset: str
    num_workers: int = 8
    pooling_type: str = PoolingType.MEAN
    norm_stats_from_pretrained: bool = True
    # Only for multimodal tasks, e.g. pastis, sickle, nandi, awf, cropharvest
    input_modalities: list[str] = field(default_factory=list)
    # Only for rslearn datasets, e.g. nandi, awf
    input_layers: list[str] = field(default_factory=list)
    # LP / KNN (embedding-based)
    embedding_batch_size: int = 128
    # LP
    probe_lr: float | None = None
    probe_batch_size: int = 32
    linear_probe_eval_interval: int = 50  # calculate val results every N epochs
    # FT
    ft_lr: float | None = None
    ft_batch_size: int = 32
    finetune_seed: int = 42
    # LP / FT
    epochs: int = 50
    # LP / KNN / FT
    patch_size: int = 4
    eval_interval: Duration = field(default_factory=lambda: Duration.epochs(1))
    eval_mode: EvalMode | None = None
    probe_type: ProbeType = ProbeType.LINEAR
    use_pooled_tokens: bool = False
    partition: str = field(default_factory=lambda: EvalDatasetPartition.TRAIN1X)
    norm_method: NormMethod = field(default_factory=lambda: NormMethod.NORM_NO_CLIP)
    select_final_test_miou_based_on_epoch_of_max_val_miou: bool = False


class DownstreamEvaluator:
    """Evaluator for downstream tasks."""

    def __init__(
        self,
        evaluation_name: str,
        task: DownstreamTaskConfig,
        trainer: Trainer,
        device: torch.device | None = None,
        run_on_test: bool = False,
    ) -> None:
        """Initialize the downstream evaluator.

        Args:
            evaluation_name: Name of the evaluation.
            task: Task configuration.
            trainer: Trainer object.
            device: Device to evaluate on.
            run_on_test: Whether to run the evaluators on the val set
                only (=False) or on the test and val set (=True)
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
        self.input_layers = task.input_layers
        self.probe_lr = task.probe_lr
        self.probe_batch_size = task.probe_batch_size
        self.ft_lr = task.ft_lr
        self.ft_batch_size = task.ft_batch_size
        self.finetune_seed = task.finetune_seed
        self.epochs = task.epochs
        self.linear_probe_eval_interval = task.linear_probe_eval_interval
        self.patch_size = task.patch_size
        self.eval_interval = task.eval_interval
        self.eval_mode = task.eval_mode
        self.probe_type = task.probe_type
        self.partition = task.partition
        self.norm_method = task.norm_method
        self.use_pooled_tokens = task.use_pooled_tokens
        self.select_final_test_miou_based_on_epoch_of_max_val_miou = (
            task.select_final_test_miou_based_on_epoch_of_max_val_miou
        )
        self.run_on_test = run_on_test
        if self.select_final_test_miou_based_on_epoch_of_max_val_miou:
            assert self.run_on_test, (
                "if select_final_test_miou_based_on_epoch_of_max_val_miou is True, "
                "run_on_test must be True"
            )
        if self.eval_mode is None:
            self.eval_mode = get_eval_mode(self.config.task_type)  # type: ignore
        if isinstance(self.eval_mode, str) and self.eval_mode is not None:
            # This will check if the eval mode is valid
            self.eval_mode = EvalMode(self.eval_mode)

        assert self.eval_mode in EvalMode, f"Unexpected eval mode {self.eval_mode}"

        if self.eval_mode == EvalMode.LINEAR_PROBE:
            if self.probe_lr is None:
                raise ValueError("probe_lr cannot be none for segmentation tasks.")
            if self.config.task_type == TaskType.SEGMENTATION:
                if self.config.height_width is None:
                    raise ValueError(
                        "config.height_width cannot be none for segmentation tasks."
                    )
                if self.config.height_width % self.patch_size != 0:
                    raise ValueError("Image height / width indivisable by patch size.")

        if self.eval_mode == EvalMode.FINETUNE:
            if self.ft_lr is None:
                raise ValueError("ft_lr cannot be none for finetune tasks.")
            if self.config.task_type == TaskType.SEGMENTATION:
                if self.config.height_width is None:
                    raise ValueError(
                        "config.height_width cannot be none for segmentation tasks."
                    )
                if self.config.height_width % self.patch_size != 0:
                    raise ValueError("Image height / width indivisable by patch size.")

        self.eval_function = (
            run_knn
            if self.eval_mode == EvalMode.KNN
            else partial(
                # TODO: THis is updated dynamically in the get_embeddings function
                train_and_eval_probe,
                batch_size=self.probe_batch_size,
                epochs=self.epochs,
                eval_interval=self.linear_probe_eval_interval,
                probe_type=self.probe_type,
                lr=self.probe_lr,
                select_final_test_miou_based_on_epoch_of_max_val_miou=self.select_final_test_miou_based_on_epoch_of_max_val_miou,
            )
            if self.eval_mode == EvalMode.LINEAR_PROBE
            else None  # "finetune" handled explictly below in .val()
        )

    def _get_data_loader(
        self, split: str, batch_size: int, seed: int | None = None
    ) -> DataLoader:
        """Get the data loader for the given split."""
        logger.info(
            f"Getting data loader for {self.dataset} with norm method {self.norm_method} and norm stats from pretrained {self.norm_stats_from_pretrained}"
        )

        generator = None
        worker_init_fn = None
        if seed is not None:
            split_offsets = {"train": 0, "valid": 1, "test": 2}
            split_seed = seed + split_offsets.get(split, 0)
            generator = torch.Generator()
            generator.manual_seed(split_seed)
            worker_init_fn = partial(_seed_worker, base_seed=split_seed)

        return DataLoader(
            get_eval_dataset(
                eval_dataset=self.dataset,
                split=split,
                partition=self.partition,
                norm_stats_from_pretrained=self.norm_stats_from_pretrained,
                input_modalities=self.input_modalities,
                input_layers=self.input_layers,
                norm_method=self.norm_method,
            ),
            collate_fn=eval_collate_fn,
            batch_size=batch_size,
            num_workers=self.num_workers,
            generator=generator,
            worker_init_fn=worker_init_fn,
            shuffle=(split == "train"),  # Only shuffle train data
        )

    def _get_embeddings(
        self, data_loader: DataLoader, is_train: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the embeddings for the given data loader."""
        print(
            f"Getting embeddings for {self.dataset} with norm method {self.norm_method}"
        )
        if hasattr(self.trainer.train_module.model, "encoder"):
            model = self.trainer.train_module.model.encoder
        else:
            model = self.trainer.train_module.model

        if hasattr(model, "patch_size"):
            # For non-helios models we override the task patch size with the model patch size
            self.patch_size = model.patch_size
            logger.info(
                f"Using patch size {self.patch_size} for {self.dataset} with model patch size {model.patch_size} and task patch size {self.patch_size}"
            )
        else:
            logger.info(
                f"No patch size found from model for {self.dataset}, using task patch size {self.patch_size}"
            )

        # Superset of the kwargs the wrapper may need
        wrapper_kwargs = {
            "task_type": self.config.task_type,
            "patch_size": self.patch_size,
            "pooling_type": self.pooling_type,
            "concat_features": (self.probe_type == "attn_pool"),
            "use_pooled_tokens": self.use_pooled_tokens,
        }
        model = get_eval_wrapper(model, **wrapper_kwargs)
        return get_embeddings(data_loader=data_loader, model=model, is_train=is_train)

    def _val_embed_probe(self) -> tuple[float, float]:
        """Validate the model using embeddings and probe (knn or linear probe)."""
        logger.info(f"Validating {self.dataset} with {self.eval_mode}")
        train_loader = self._get_data_loader("train", self.embedding_batch_size)
        val_loader = self._get_data_loader("valid", self.embedding_batch_size)
        test_loader = self._get_data_loader("test", self.embedding_batch_size)

        start_time = time.time()
        logger.info(f"Getting train embeddings for {self.dataset}...")
        train_embeddings, train_labels = self._get_embeddings(
            train_loader, is_train=True
        )
        logger.info(f"Getting val embeddings for {self.dataset}...")
        val_embeddings, val_labels = self._get_embeddings(val_loader, is_train=False)
        if self.run_on_test:
            logger.info(f"Getting test embeddings for {self.dataset}...")
            test_embeddings, test_labels = self._get_embeddings(
                test_loader, is_train=False
            )
        else:
            test_embeddings, test_labels = None, None
        logger.info(
            f"Time to get embeddings for {self.dataset}: {time.time() - start_time:.2f}s"
        )

        logger.info(
            f"train embeddings shape for {self.dataset}: {train_embeddings.shape}"
        )
        logger.info(f"val embeddings shape for {self.dataset}: {val_embeddings.shape}")
        if test_embeddings is not None:
            logger.info(
                f"test embeddings shape for {self.dataset}: {test_embeddings.shape}"
            )
        logger.info(f"train labels shape for {self.dataset}: {train_labels.shape}")
        logger.info(f"val labels shape for {self.dataset}: {val_labels.shape}")
        if test_labels is not None:
            logger.info(f"test labels shape for {self.dataset}: {test_labels.shape}")

        kwargs = {
            "config": self.config,
            "train_embeddings": train_embeddings,
            "train_labels": train_labels,
            "val_embeddings": val_embeddings,
            "val_labels": val_labels,
            "test_embeddings": test_embeddings,
            "test_labels": test_labels,
            "device": self.device,
        }
        val_result, test_result = self.eval_function(**kwargs)  # type: ignore
        logger.info(f"Downstream evaluator {self.evaluation_name} score: {val_result}")

        # Free memory aggressively between evals
        del train_embeddings, train_labels, test_embeddings, test_labels
        torch.cuda.empty_cache()
        gc.collect()

        return val_result, test_result

    def _get_best_checkpoint_path(self) -> str:
        """Get the best checkpoint path."""
        best_checkpoint_path = os.path.join(
            self.trainer.save_folder,
            self.evaluation_name,
            f"lr{self.ft_lr}",
            "best.ckpt",
        )
        return best_checkpoint_path

    def _val_finetune(self) -> tuple[float, float]:
        """Validate the model using finetuning."""
        logger.info(f"Validating {self.dataset} with finetune")

        train_loader = self._get_data_loader(
            "train", self.ft_batch_size, seed=self.finetune_seed
        )
        val_loader = self._get_data_loader("valid", self.ft_batch_size)

        if self.run_on_test:
            test_loader = self._get_data_loader("test", self.ft_batch_size)
        else:
            test_loader = None

        # Use encoder if present
        if hasattr(self.trainer.train_module.model, "encoder"):
            model = self.trainer.train_module.model.encoder
        else:
            model = self.trainer.train_module.model

        original_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

        # Resolve patch size if model exposes it
        if hasattr(model, "patch_size"):
            logger.info(
                f"Using patch size {max(self.patch_size, model.patch_size)} for {self.dataset}\
                with model patch size {model.patch_size} and task patch size {self.patch_size}\
                (max of {self.patch_size} and {model.patch_size})"
            )
            # Use the max patch size of the model and the task
            self.patch_size = max(self.patch_size, model.patch_size)
        else:
            logger.info(
                f"No patch size found for {self.dataset}, using patch size {self.patch_size}"
            )

        # Skip task if best checkpoint already exists
        best_checkpoint_path = self._get_best_checkpoint_path()
        if os.path.exists(best_checkpoint_path):
            logger.info("Best checkpoint already exists, skipping finetuning")
            return 0.0, 0.0
        else:
            logger.info("Best checkpoint does not exist, running finetuning")
            val_result, test_result = run_finetune_eval(
                task_name=self.evaluation_name,
                task_config=self.config,
                trainer=self.trainer,
                model=model,
                device=self.device or self.trainer.device,
                lr=self.ft_lr,  # type: ignore
                epochs=self.epochs,
                patch_size=self.patch_size,
                pooling_type=self.pooling_type,
                use_pooled_tokens=self.use_pooled_tokens,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                seed=self.finetune_seed,
                best_checkpoint_path=best_checkpoint_path,
            )
            logger.info(
                f"Downstream evaluator {self.evaluation_name} val score: {val_result}, test score: {test_result}"
            )
            model.load_state_dict(original_state)

            torch.cuda.empty_cache()
            gc.collect()
            return val_result, test_result

    def val(self) -> tuple[float, float]:
        """Validate the model on the downstream task."""
        if self.eval_mode in (EvalMode.KNN, EvalMode.LINEAR_PROBE):
            return self._val_embed_probe()
        elif self.eval_mode == EvalMode.FINETUNE:
            return self._val_finetune()
        else:
            raise ValueError(f"Unsupported eval_mode: {self.eval_mode}")


@dataclass
class DownstreamEvaluatorCallback(Callback):
    """Runs in-loop evaluations periodically during training."""

    evaluators: list[DownstreamEvaluator] = field(default_factory=list)
    eval_on_startup: bool = False
    cancel_after_first_eval: bool = False
    run_on_test: bool = False

    def _check_supported_modalities(self, evaluator: DownstreamEvaluator) -> bool:
        """Check if the evaluator is supported by the model."""
        task_supported_modalities = evaluator.config.supported_modalities
        logger.info(f"Task supported modalities: {task_supported_modalities}")
        task_instance_used_modalities = evaluator.input_modalities
        logger.info(f"Task instance used modalities: {task_instance_used_modalities}")
        if len(task_instance_used_modalities) == 0:
            task_instance_used_modalities = task_supported_modalities

        if not self.trainer.train_module.model.supports_multiple_modalities_at_once:
            if len(task_instance_used_modalities) > 1:
                return False

        does_model_support_all_task_instance_used_modalities = set(
            task_instance_used_modalities
        ).issubset(set(self.model_supported_modalities))
        return does_model_support_all_task_instance_used_modalities

    @property
    def model_supported_modalities(self) -> list[str]:
        """Get the supported modalities for the model."""
        if hasattr(self.trainer.train_module.model, "supported_modalities"):
            return self.trainer.train_module.model.supported_modalities
        elif hasattr(self.trainer.train_module.model, "encoder"):
            if hasattr(
                self.trainer.train_module.model.encoder, "supported_modality_names"
            ):
                return self.trainer.train_module.model.encoder.supported_modality_names
        else:
            logger.info(
                "Can't find a supported_modalities attribute; defaulting to all modalities."
            )
        return Modality.names()

    def _check_input_requirements(self, evaluator: DownstreamEvaluator) -> bool:
        """Check if the evaluator is supported by the model."""
        model = self.trainer.train_module.model

        # Check required modalities
        required_modalities_present = True
        if hasattr(model, "required_modalities"):
            required_modalities_present = set(model.required_modalities).issubset(
                set(evaluator.input_modalities)
            )

        # Check timeseries requirement
        has_timeseries = True
        if hasattr(model, "requires_timeseries") and model.requires_timeseries:
            has_timeseries = evaluator.config.timeseries

        return required_modalities_present and has_timeseries

    def pre_train(self) -> None:
        """Run the evaluators on startup."""
        if self.eval_on_startup:
            logger.info(f"Running {len(self.evaluators)} evaluators on startup.")

            # self.trainer.record_metric() is not logging to wandb at this point
            # therefore we log to wandb manually
            wandb_callback = next(
                callback
                for callback in self.trainer._iter_callbacks()
                if isinstance(callback, OlmoEarthWandBCallback)
            )

            for evaluator in self.evaluators:
                if not self._check_supported_modalities(evaluator):
                    logger.info(
                        f"Skipping {evaluator.evaluation_name} because it requires a modality that is not supported by the model"
                    )
                    continue
                if not self._check_input_requirements(evaluator):
                    logger.info(
                        f"Skipping {evaluator.evaluation_name} because it doesn't match input requirements of the model"
                    )
                    continue

                # Separate finetune step metric per task
                if evaluator.eval_mode == EvalMode.FINETUNE:
                    if wandb_callback.enabled:
                        wandb_callback.wandb.define_metric(
                            f"{evaluator.evaluation_name}/*",
                            step_metric=f"{evaluator.evaluation_name}_step",
                        )
                        wandb_callback.wandb.log(
                            {f"{evaluator.evaluation_name}_step": 0}
                        )

                val_result, test_result, eval_time = self._perform_eval(evaluator)
                # Only logging valid results to wandb
                if val_result > 0 and test_result > 0:
                    if wandb_callback.enabled:
                        wandb_callback.wandb.log(
                            {"eval/" + evaluator.evaluation_name: val_result}
                        )
                        wandb_callback.wandb.log(
                            {"eval_time/" + evaluator.evaluation_name: eval_time}
                        )
                        if self.run_on_test:
                            wandb_callback.wandb.log(
                                {"eval/test/" + evaluator.evaluation_name: test_result}
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
            if not self._check_supported_modalities(evaluator):
                logger.info(
                    f"Skipping {evaluator.evaluation_name} because it requires a modality that is not supported by the model"
                )
                continue
            self._perform_eval(evaluator)

    def _perform_eval(
        self, evaluator: DownstreamEvaluator
    ) -> tuple[float, float, float]:
        """Run the evaluator."""
        logger.info(f"Running {evaluator.evaluation_name} evaluations...")
        start_time = time.monotonic()
        val_result, test_result = evaluator.val()
        self.trainer.record_metric(f"eval/{evaluator.evaluation_name}", val_result)
        if self.run_on_test:
            self.trainer.record_metric(
                f"eval/test/{evaluator.evaluation_name}", test_result
            )
        eval_time = time.monotonic() - start_time
        self.trainer.record_metric(f"eval_time/{evaluator.evaluation_name}", eval_time)
        logger.info(
            f"Finished {evaluator.evaluation_name} evaluations in {eval_time:.1f} seconds."
        )
        return val_result, test_result, eval_time


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
    tasks_to_run: list[str] | None = None
    # whether to run the evaluators on the val set only (=False) or on the test and val set (=True)
    run_on_test: bool = False
    filter_for_eval_mode: EvalMode | None = None

    def verify_input_modalities(
        self, task: DownstreamTaskConfig, config: EvalDatasetConfig
    ) -> None:
        """Verify the input modality configuration for a task."""
        # Check that input_modalities is only set for multimodal tasks
        if (
            not (
                (task.dataset in ["pastis", "pastis128", "sickle", "nandi", "awf"])
                or task.dataset.startswith("cropharvest")
            )
            and len(task.input_modalities) > 0
        ):
            raise ValueError(
                f"input_modalities is only supported for multimodal tasks, got {task.dataset}"
            )
        # Make sure input_modalities contains only unique modalities
        if len(task.input_modalities) != len(set(task.input_modalities)):
            raise ValueError(
                f"input_modalities must contain unique modalities, got {task.input_modalities}"
            )
        if not set(task.input_modalities).issubset(set(config.supported_modalities)):
            raise ValueError(
                f"input_modalities must be a subset of supported_modalities, got {task.input_modalities} and {config.supported_modalities}"
            )

    def verify_input_layers(self, task: DownstreamTaskConfig) -> None:
        """Check input_layers config."""
        rslearn_datasets = {"nandi", "awf"}
        layers = task.input_layers or []

        # input_layers not allowed on non-rslearn datasets
        if task.dataset not in rslearn_datasets and layers:
            raise ValueError(
                f"`input_layers` not supported for dataset '{task.dataset}'."
            )

        # input_layers must be unique
        if len(layers) != len(set(layers)):
            raise ValueError(f"`input_layers` must be unique, got {layers}")

    def build(self, trainer: Trainer) -> Callback | None:
        """Build the downstream evaluator callback."""
        if not self.enabled:
            return None

        evaluators: list[DownstreamEvaluator] = []
        # Check that probe_lr is set for segmentation tasks
        for evaluation_name, task in self.tasks.items():
            if (
                self.tasks_to_run is not None
                and evaluation_name not in self.tasks_to_run
            ):
                logger.info(
                    f"Skipping {evaluation_name} because it is not in the tasks_to_run list"
                )
                continue
            if (
                self.filter_for_eval_mode is not None
                and task.eval_mode != self.filter_for_eval_mode
            ):
                logger.info(
                    f"Skipping {evaluation_name} because it is not in the filter_for_eval_mode list"
                )
                continue

            config = dataset_to_config(task.dataset)
            if config.task_type == TaskType.SEGMENTATION:
                if task.probe_lr is None and task.ft_lr is None:
                    raise ValueError(
                        f"probe_lr and ft_lr cannot both be None for {task.dataset}"
                    )

            self.verify_input_modalities(task, config)
            # Sort to ensure consistent order
            task.input_modalities.sort()
            self.verify_input_layers(task)
            logger.info(f"Adding {evaluation_name} with eval mode {task.eval_mode}")
            evaluators.append(
                DownstreamEvaluator(
                    evaluation_name=evaluation_name,
                    task=task,
                    trainer=trainer,
                    device=trainer.device,
                    run_on_test=self.run_on_test,
                )
            )
        return DownstreamEvaluatorCallback(
            evaluators=evaluators,
            eval_on_startup=self.eval_on_startup,
            cancel_after_first_eval=self.cancel_after_first_eval,
            run_on_test=self.run_on_test,
        )
