"""Downstream evaluator callback."""

import gc
import logging
import os
import random
import time
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial
from typing import Any

import numpy as np
import torch
from olmo_core.distributed.utils import get_rank
from olmo_core.train.callbacks.callback import Callback, CallbackConfig
from olmo_core.train.callbacks.checkpointer import CheckpointerCallback
from olmo_core.train.common import Duration
from olmo_core.train.trainer import Trainer
from torch.utils.data import DataLoader, IterableDataset
from upath import UPath

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.datasets import get_eval_dataset
from olmoearth_pretrain.evals.datasets.configs import (
    EvalDatasetConfig,
    TaskType,
    dataset_to_config,
    get_eval_mode,
)
from olmoearth_pretrain.evals.datasets.normalize import NormMethod
from olmoearth_pretrain.evals.datasets.utils import eval_collate_fn_variable_time
from olmoearth_pretrain.evals.embedding_diagnostics import (
    compute_embedding_diagnostics,
    compute_spatial_embedding_diagnostics,
)
from olmoearth_pretrain.evals.embedding_transforms import (
    dequantize_embeddings,
    dequantize_embeddings_percentile,
    load_quantile_config,
    reduce_embedding_dim,
)
from olmoearth_pretrain.evals.embeddings import get_embeddings
from olmoearth_pretrain.evals.eval_wrapper import get_eval_wrapper
from olmoearth_pretrain.evals.finetune import run_finetune_eval
from olmoearth_pretrain.evals.knn import run_knn
from olmoearth_pretrain.evals.linear_probe import ProbeType, train_and_eval_probe
from olmoearth_pretrain.evals.metrics import EvalMetric, EvalResult, EvalTaskResult
from olmoearth_pretrain.nn.pooling import PoolingType
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
    EMBEDDING_DIAGNOSTICS = "embedding_diagnostics"


@dataclass
class DownstreamTaskConfig:
    """Config for a downstream task."""

    dataset: str
    num_workers: int = 8
    pooling_type: str = PoolingType.MEAN
    norm_stats_from_pretrained: bool = True
    # Only for multimodal tasks, e.g. pastis, nandi, awf
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
    ft_grad_accum_steps: int = 1
    finetune_seed: int = 42
    ft_head_type: str = (
        "linear"  # "linear" or "unet" (unet only valid for segmentation/regression)
    )
    # LP / FT
    epochs: int = 50
    # LP / KNN / FT
    patch_size: int = 4
    eval_interval: Duration = field(default_factory=lambda: Duration.epochs(1))
    eval_mode: EvalMode | None = None
    probe_type: ProbeType = ProbeType.LINEAR
    use_pooled_tokens: bool = False
    # Use the center spatial patch embedding instead of pooling across all patches
    # for classification tasks. Has no effect on segmentation tasks.
    use_center_token: bool = False
    # Fraction of training labels to use for low-label evals. Dataset-specific
    # code translates this into fixed partitions or deterministic subsamples.
    label_fraction: float = 1.0
    # Default to 2std no clip - this matches what our model sees in pretraining,
    # so when using dataset stats (e.g. for MADOS) consistency is important.
    norm_method: NormMethod = field(
        default_factory=lambda: NormMethod.NORM_NO_CLIP_2_STD
    )
    select_best_by_primary_metric: bool = False
    # Subsample train embeddings for faster probe training (None = use all)
    max_train_samples: int | None = None
    # Seed for the max_train_samples subsample so the subset is reproducible
    # across checkpoints in a sweep.
    max_train_samples_seed: int = 42
    # Quantize embeddings to int8 for storage efficiency evaluation
    quantize_embeddings: bool = False
    # Number of bits for percentile-based quantization (1, 2, 4, or 8)
    # If None, uses legacy power-based int8 quantization when quantize_embeddings=True
    quantize_bits: int | None = None
    # Path to HDF5 file with precomputed quantile boundaries for percentile quantization
    quantile_config_path: str | None = None
    # Reduce embedding dimensionality via PCA (None = no reduction)
    embedding_dim: int | None = None
    # Use weighted dice loss instead of cross-entropy (only for specific tasks like wildfire)
    use_dice_loss: bool = False
    # Override the default primary metric (e.g. EvalMetric.F1 instead of ACCURACY).
    # None = use the default for the task type (accuracy for classification, miou for segmentation).
    primary_metric: EvalMetric | None = None
    # Class index for CLASS_F1 primary metric. Required when primary_metric is CLASS_F1.
    primary_metric_class: int | None = None
    # For pretrain_subset dataset: path to training h5py data
    h5py_dir: str | None = None
    # For pretrain_subset: max samples to load
    pretrain_max_samples: int = 512
    # For pretrain subset auxiliary probes: target modality to predict.
    pretrain_target_modality: str | None = None
    # For multi-band regression targets (e.g. glo30), the band selected for this
    # single-channel probe (0=elevation, 1=slope, 2=aspect). None = full target.
    pretrain_target_band_index: int | None = None
    pretrain_label_seed: int = 42
    pretrain_train_samples: int = 512
    pretrain_valid_samples: int = 512
    pretrain_test_samples: int = 512
    # Geographic vs random index selection for pretrain subset auxiliary probes.
    # "random" picks indices uniformly; "geographic" buckets samples into
    # latlon-bin holdouts so train/val/test are spatially disjoint.
    pretrain_split_strategy: str = "random"
    pretrain_geographic_bin_size_deg: float = 5.0


class DownstreamEvaluator:
    """Evaluator for downstream tasks."""

    def __init__(
        self,
        evaluation_name: str,
        task: DownstreamTaskConfig,
        trainer: Trainer,
        device: torch.device | None = None,
        run_on_test: bool = False,
        n_bootstrap: int = 0,
        bootstrap_seed: int = 42,
    ) -> None:
        """Initialize the downstream evaluator.

        Args:
            evaluation_name: Name of the evaluation.
            task: Task configuration.
            trainer: Trainer object.
            device: Device to evaluate on.
            run_on_test: Whether to run the evaluators on the val set
                only (=False) or on the test and val set (=True)
            n_bootstrap: Number of bootstrap samples for uncertainty estimation (0 = no bootstrap)
            bootstrap_seed: Random seed for bootstrap sampling
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
        self.probe_batch_size = task.probe_batch_size
        self.ft_lr = task.ft_lr
        self.ft_batch_size = task.ft_batch_size
        self.ft_grad_accum_steps = task.ft_grad_accum_steps
        self.finetune_seed = task.finetune_seed
        self.ft_head_type = task.ft_head_type
        self.epochs = task.epochs
        self.linear_probe_eval_interval = task.linear_probe_eval_interval
        self.patch_size = task.patch_size
        self.max_train_samples = task.max_train_samples
        self.max_train_samples_seed = task.max_train_samples_seed
        self.eval_interval = task.eval_interval
        self.eval_mode = task.eval_mode
        self.probe_type = task.probe_type
        self.label_fraction = task.label_fraction
        self.norm_method = task.norm_method
        self.use_pooled_tokens = task.use_pooled_tokens
        self.use_center_token = task.use_center_token
        self.select_best_by_primary_metric = task.select_best_by_primary_metric
        self.quantize_embeddings = task.quantize_embeddings
        self.quantize_bits = task.quantize_bits
        self.quantile_config_path = task.quantile_config_path
        # Load quantile config if path is provided
        self.quantile_config: dict | None = None
        if self.quantile_config_path is not None:
            logger.info(f"Loading quantile config from {self.quantile_config_path}")
            self.quantile_config = load_quantile_config(self.quantile_config_path)
        self.embedding_dim = task.embedding_dim
        self.use_dice_loss = task.use_dice_loss
        self.primary_metric = task.primary_metric
        self.primary_metric_class = task.primary_metric_class
        self.h5py_dir = task.h5py_dir
        self.pretrain_max_samples = task.pretrain_max_samples
        self.pretrain_target_modality = task.pretrain_target_modality
        self.pretrain_target_band_index = task.pretrain_target_band_index
        self.pretrain_label_seed = task.pretrain_label_seed
        self.pretrain_train_samples = task.pretrain_train_samples
        self.pretrain_valid_samples = task.pretrain_valid_samples
        self.pretrain_test_samples = task.pretrain_test_samples
        self.pretrain_split_strategy = task.pretrain_split_strategy
        self.pretrain_geographic_bin_size_deg = task.pretrain_geographic_bin_size_deg
        self.run_on_test = run_on_test
        self.n_bootstrap = n_bootstrap
        self.bootstrap_seed = bootstrap_seed
        if self.select_best_by_primary_metric:
            assert self.run_on_test, (
                "if select_best_by_primary_metric is True, run_on_test must be True"
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
            if self.config.task_type in (
                TaskType.SEGMENTATION,
                TaskType.PER_PIXEL_REGRESSION,
            ):
                if self.config.height_width is None:
                    raise ValueError(
                        "config.height_width cannot be none for segmentation tasks."
                    )
                if self.config.height_width % self.patch_size != 0:
                    raise ValueError(
                        f"Image height / width indivisable by patch size. {self.config.height_width} % {self.patch_size} != 0"
                    )

        if self.eval_mode == EvalMode.FINETUNE:
            if self.ft_lr is None:
                raise ValueError("ft_lr cannot be none for finetune tasks.")
            if self.ft_grad_accum_steps < 1:
                raise ValueError(
                    f"ft_grad_accum_steps must be >= 1, got {self.ft_grad_accum_steps}"
                )
            if self.config.task_type == TaskType.SEGMENTATION:
                if self.config.height_width is None:
                    raise ValueError(
                        "config.height_width cannot be none for segmentation tasks."
                    )
                if self.config.height_width % self.patch_size != 0:
                    raise ValueError("Image height / width indivisable by patch size.")

        self.eval_function = (
            partial(
                run_knn,
                primary_metric=self.primary_metric,
                primary_metric_class=self.primary_metric_class,
            )
            if self.eval_mode == EvalMode.KNN
            else (
                partial(
                    train_and_eval_probe,
                    batch_size=self.probe_batch_size,
                    epochs=self.epochs,
                    eval_interval=self.linear_probe_eval_interval,
                    probe_type=self.probe_type,
                    lr=self.probe_lr,
                    select_best_by_primary_metric=self.select_best_by_primary_metric,
                    use_dice_loss=self.use_dice_loss,
                    primary_metric=self.primary_metric,
                    primary_metric_class=self.primary_metric_class,
                )
                if self.eval_mode == EvalMode.LINEAR_PROBE
                else None
            )  # "finetune" handled explictly below in .val()
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

        extra_kwargs: dict[str, Any] = {}
        if self.dataset.startswith("pretrain_subset") and self.h5py_dir is not None:
            extra_kwargs["h5py_dir"] = self.h5py_dir
            extra_kwargs["training_modalities"] = self.input_modalities
            extra_kwargs["max_samples"] = self.pretrain_max_samples
            extra_kwargs["target_modality"] = self.pretrain_target_modality
            extra_kwargs["target_band_index"] = self.pretrain_target_band_index
            extra_kwargs["pretrain_split"] = split
            extra_kwargs["pretrain_label_seed"] = self.pretrain_label_seed
            extra_kwargs["pretrain_train_samples"] = self.pretrain_train_samples
            extra_kwargs["pretrain_valid_samples"] = self.pretrain_valid_samples
            extra_kwargs["pretrain_test_samples"] = self.pretrain_test_samples
            extra_kwargs["pretrain_split_strategy"] = self.pretrain_split_strategy
            extra_kwargs["pretrain_geographic_bin_size_deg"] = (
                self.pretrain_geographic_bin_size_deg
            )
        eval_ds = get_eval_dataset(
            eval_dataset=self.dataset,
            split=split,
            label_fraction=self.label_fraction,
            norm_stats_from_pretrained=self.norm_stats_from_pretrained,
            input_modalities=self.input_modalities,
            norm_method=self.norm_method,
            **extra_kwargs,
        )
        is_iterable = isinstance(eval_ds, IterableDataset)
        return DataLoader(
            eval_ds,
            collate_fn=eval_collate_fn_variable_time,
            batch_size=batch_size,
            num_workers=self.num_workers,
            generator=None if is_iterable else generator,
            worker_init_fn=worker_init_fn,
            shuffle=False if is_iterable else (split == "train"),
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
            "use_center_token": self.use_center_token,
        }
        model = get_eval_wrapper(model, **wrapper_kwargs)
        return get_embeddings(
            data_loader=data_loader,
            model=model,
            is_train=is_train,
            quantize=self.quantize_embeddings,
            quantize_bits=self.quantize_bits,
            quantile_config=self.quantile_config,
        )

    def _val_embed_probe(self) -> EvalTaskResult:
        """Validate the model using embeddings and probe (knn or linear probe)."""
        logger.info(f"Validating {self.dataset} with {self.eval_mode}")
        logger.info(f"Getting train loader for {self.dataset}...")
        train_loader = self._get_data_loader("train", self.embedding_batch_size)
        logger.info(f"Getting val loader for {self.dataset}...")
        val_loader = self._get_data_loader("valid", self.embedding_batch_size)

        start_time = time.time()
        logger.info(f"Getting train embeddings for {self.dataset}...")
        train_embeddings, train_labels = self._get_embeddings(
            train_loader, is_train=True
        )
        logger.info(f"Train embeddings shape: {train_embeddings.shape}")
        logger.info(
            f"Train label counts: {torch.unique(train_labels, return_counts=True)}"
        )

        # Subsample train embeddings if configured
        if (
            self.max_train_samples
            and train_embeddings.shape[0] > self.max_train_samples
        ):
            logger.info(
                f"Subsampling train embeddings from {train_embeddings.shape[0]} "
                f"to {self.max_train_samples} (seed={self.max_train_samples_seed})"
            )
            generator = torch.Generator().manual_seed(self.max_train_samples_seed)
            indices = torch.randperm(train_embeddings.shape[0], generator=generator)[
                : self.max_train_samples
            ]
            train_embeddings = train_embeddings[indices]
            train_labels = train_labels[indices]

        logger.info(f"Getting val embeddings for {self.dataset}...")
        val_embeddings, val_labels = self._get_embeddings(val_loader, is_train=False)
        logger.info(f"Val embeddings shape: {val_embeddings.shape}")
        logger.info(f"Val label counts: {torch.unique(val_labels, return_counts=True)}")
        if self.run_on_test:
            logger.info(f"Getting test loader for {self.dataset}...")
            test_loader = self._get_data_loader("test", self.embedding_batch_size)
            logger.info(f"Getting test embeddings for {self.dataset}...")
            test_embeddings, test_labels = self._get_embeddings(
                test_loader, is_train=False
            )
            logger.info(f"Test embeddings shape: {test_embeddings.shape}")
            logger.info(
                f"Test label counts: {torch.unique(test_labels, return_counts=True)}"
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

        if self.quantize_embeddings:
            logger.info(f"Dequantizing embeddings for {self.dataset}")
            if self.quantize_bits is not None and self.quantile_config is not None:
                # Percentile-based dequantization
                key = f"{self.quantize_bits}bit"
                midpoints = self.quantile_config[key]["midpoints"]
                train_embeddings = dequantize_embeddings_percentile(
                    train_embeddings, midpoints
                )
                val_embeddings = dequantize_embeddings_percentile(
                    val_embeddings, midpoints
                )
                if test_embeddings is not None:
                    test_embeddings = dequantize_embeddings_percentile(
                        test_embeddings, midpoints
                    )
            else:
                # Legacy power-based dequantization
                train_embeddings = dequantize_embeddings(train_embeddings)
                val_embeddings = dequantize_embeddings(val_embeddings)
                if test_embeddings is not None:
                    test_embeddings = dequantize_embeddings(test_embeddings)

        # Reduce embedding dimensionality via PCA if specified
        if self.embedding_dim is not None:
            original_dim = train_embeddings.shape[-1]
            logger.info(
                f"Reducing embeddings from {original_dim} to {self.embedding_dim} dims for {self.dataset}"
            )
            train_embeddings, val_embeddings, test_embeddings, variance_retained = (
                reduce_embedding_dim(
                    train_embeddings,
                    val_embeddings,
                    test_embeddings,
                    self.embedding_dim,
                )
            )
            logger.info(f"PCA variance retained: {variance_retained:.4f}")

        kwargs = {
            "config": self.config,
            "train_embeddings": train_embeddings,
            "train_labels": train_labels,
            "val_embeddings": val_embeddings,
            "val_labels": val_labels,
            "test_embeddings": test_embeddings,
            "test_labels": test_labels,
            "device": self.device,
            "n_bootstrap": self.n_bootstrap,
            "bootstrap_seed": self.bootstrap_seed,
        }
        result = self.eval_function(**kwargs)  # type: ignore

        # Free memory aggressively between evals
        del train_embeddings, train_labels, test_embeddings, test_labels
        del val_embeddings, val_labels
        torch.cuda.empty_cache()
        gc.collect()

        return result

    def _get_best_checkpoint_path(self) -> str:
        """Get the best checkpoint path."""
        best_checkpoint_path = os.path.join(
            self.trainer.save_folder,
            self.evaluation_name,
            f"lr{self.ft_lr}",
            "best.ckpt",
        )
        return best_checkpoint_path

    def _get_resume_checkpoint_path(self) -> str:
        """Get the resume checkpoint path for resumable training."""
        resume_checkpoint_path = os.path.join(
            self.trainer.save_folder,
            self.evaluation_name,
            f"lr{self.ft_lr}",
            "last.ckpt",
        )
        return resume_checkpoint_path

    def _val_finetune(self) -> EvalTaskResult:
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
        resume_checkpoint_path = self._get_resume_checkpoint_path()
        if os.path.exists(best_checkpoint_path):
            logger.info(
                f"Best checkpoint for {self.evaluation_name} already exists, "
                f"skipping finetuning and evaluating on the best checkpoint..."
            )

        if os.path.exists(resume_checkpoint_path):
            logger.info(
                f"Found resume checkpoint at {resume_checkpoint_path}, will resume training"
            )
        else:
            logger.info("No resume checkpoint found, starting fresh")

        result = run_finetune_eval(
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
            resume_checkpoint_path=resume_checkpoint_path,
            primary_metric=self.primary_metric,
            primary_metric_class=self.primary_metric_class,
            ft_grad_accum_steps=self.ft_grad_accum_steps,
            head_type=self.ft_head_type,  # type: ignore[arg-type]
            use_dice_loss=self.use_dice_loss,
        )
        logger.info(
            f"Downstream evaluator {self.evaluation_name} val score: {result.val_result}, test score: {result.test_result}"
        )
        model.load_state_dict(original_state)
        del original_state

        torch.cuda.empty_cache()
        gc.collect()
        return result

    def _val_embedding_diagnostics(self) -> EvalTaskResult:
        """Compute embedding diagnostics only (no downstream task)."""
        logger.info(f"Computing embedding diagnostics for {self.dataset}")
        data_loader = self._get_data_loader("train", self.embedding_batch_size)
        embeddings, _ = self._get_embeddings(data_loader, is_train=False)
        logger.info(f"Embeddings shape for {self.dataset}: {embeddings.shape}")

        if embeddings.ndim >= 3:
            diagnostics = compute_spatial_embedding_diagnostics(embeddings)
        else:
            diagnostics = compute_embedding_diagnostics(embeddings)
        logger.info(f"Embedding diagnostics for {self.dataset}: {diagnostics}")

        result = EvalTaskResult(val_result=None, test_result=None)
        result.embedding_diagnostics = diagnostics
        return result

    def val(self) -> EvalTaskResult:
        """Validate the model on the downstream task."""
        if self.eval_mode == EvalMode.EMBEDDING_DIAGNOSTICS:
            return self._val_embedding_diagnostics()
        elif self.eval_mode in (EvalMode.KNN, EvalMode.LINEAR_PROBE):
            return self._val_embed_probe()
        elif self.eval_mode == EvalMode.FINETUNE:
            return self._val_finetune()
        else:
            raise ValueError(f"Unsupported eval_mode: {self.eval_mode}")


def _make_other_prefix(prefix: str) -> str:
    """Turn 'eval' -> 'eval_other', 'eval/test' -> 'eval_other/test'."""
    parts = prefix.split("/", 1)
    parts[0] = parts[0] + "_other"
    return "/".join(parts)


# Name of the wandb metric that eval curves are plotted against. It equals the
# training step at which the evaluated checkpoint was saved. Both the in-loop
# evals (this callback) and the built-in beaker-job evals log on this x-axis so
# their curves share one directly-comparable axis in wandb -- rather than the
# in-loop evals landing on wandb's native per-run ``step`` while the beaker-job
# evals (a separate run) land on ``checkpoint_step``.
CHECKPOINT_STEP_METRIC = "checkpoint_step"

# Eval metric prefixes bound to the ``checkpoint_step`` x-axis.
EVAL_METRIC_PREFIXES = (
    "eval/*",
    "eval/test/*",
    "eval_other/*",
    "eval_other/test/*",
    "eval_time/*",
    "eval_embed_diagnostics/*",
)


def define_checkpoint_step_metrics(wandb_module: Any) -> None:
    """Tell wandb to plot the eval metric prefixes against ``checkpoint_step``.

    ``wandb_module`` is the live ``wandb`` handle (e.g. ``wandb_callback.wandb``).
    Safe to call once after the run is initialized.
    """
    wandb_module.define_metric(CHECKPOINT_STEP_METRIC)
    for metric_prefix in EVAL_METRIC_PREFIXES:
        wandb_module.define_metric(metric_prefix, step_metric=CHECKPOINT_STEP_METRIC)


def eval_result_log_dict(
    prefix: str, name: str, result: EvalResult
) -> dict[str, float]:
    """Build the wandb log dict for an EvalResult.

    Primary metric goes to ``{prefix}/{name}`` (e.g. ``eval/m_eurosat``).
    Non-primary metrics go to ``{prefix}_other/.../{name}/{metric_name}``.
    """
    other_prefix = _make_other_prefix(prefix)
    log_dict: dict[str, float] = {f"{prefix}/{name}": result.primary}
    for metric_name, metric_value in result.metrics.items():
        if metric_name == result.primary_metric_key:
            continue
        log_dict[f"{other_prefix}/{name}/{metric_name}"] = metric_value
    return log_dict


def _log_eval_result_to_wandb(
    wandb_callback: Any, prefix: str, name: str, result: EvalResult
) -> None:
    """Log an EvalResult to wandb using the shared key layout."""
    wandb_callback.wandb.log(eval_result_log_dict(prefix, name, result))


def _record_eval_result(
    trainer: Trainer, prefix: str, name: str, result: EvalResult
) -> None:
    """Record an EvalResult to trainer metrics."""
    other_prefix = _make_other_prefix(prefix)
    trainer.record_metric(f"{prefix}/{name}", result.primary)
    for metric_name, metric_value in result.metrics.items():
        if metric_name == result.primary_metric_key:
            continue
        trainer.record_metric(f"{other_prefix}/{name}/{metric_name}", metric_value)


@dataclass
class DownstreamEvaluatorCallback(Callback):
    """Runs in-loop evaluations periodically during training."""

    evaluators: list[DownstreamEvaluator] = field(default_factory=list)
    eval_on_startup: bool = False
    cancel_after_first_eval: bool = False
    run_on_test: bool = False
    n_bootstrap: int = 0
    bootstrap_seed: int = 42
    run_as_beaker_job: bool = False
    beaker_eval_module_path: str | None = None
    beaker_eval_clusters: list[str] | None = None
    beaker_eval_priority: str | None = None
    beaker_eval_extra_overrides: list[str] | None = None

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

    def _log_eval_results_to_logger_pretrain(
        self, evaluator: DownstreamEvaluator, result: EvalTaskResult
    ) -> None:
        """Log the evaluation results."""
        val_result = result.val_result
        test_result = result.test_result
        bootstrap_stats = result.bootstrap_stats

        # Log bootstrap statistics if available
        if bootstrap_stats:
            logger.info(
                f"Downstream evaluator {evaluator.evaluation_name} bootstrap stats: "
                f"mean={bootstrap_stats.get('mean', 'N/A'):.4f}, "
                f"std={bootstrap_stats.get('std', 'N/A'):.4f}, "
                f"95% CI=[{bootstrap_stats.get('ci_lower', 'N/A'):.4f}, "
                f"{bootstrap_stats.get('ci_upper', 'N/A'):.4f}]"
            )

        if val_result is not None:
            logger.info(
                f"Downstream evaluator {evaluator.evaluation_name} score: {val_result.primary} (metrics: {val_result.metrics})"
            )
        if self.run_on_test and test_result is not None:
            logger.info(
                f"Downstream evaluator {evaluator.evaluation_name} test score: {test_result.primary} (metrics: {test_result.metrics})"
            )

    def _log_eval_results_to_wandb_pretrain(
        self, evaluator: DownstreamEvaluator, result: EvalTaskResult
    ) -> None:
        """Log the evaluation results to wandb."""
        wandb_callback = next(
            callback
            for callback in self.trainer._iter_callbacks()
            if isinstance(callback, OlmoEarthWandBCallback)
        )
        val_result = result.val_result
        test_result = result.test_result
        eval_time = result.eval_time
        bootstrap_stats = result.bootstrap_stats

        if wandb_callback.enabled:
            if val_result is not None:
                _log_eval_result_to_wandb(
                    wandb_callback, "eval", evaluator.evaluation_name, val_result
                )
            wandb_callback.wandb.log(
                {"eval_time/" + evaluator.evaluation_name: eval_time}
            )
            if result.embedding_diagnostics:
                wandb_callback.wandb.log(
                    {
                        f"eval_embed_diagnostics/{evaluator.evaluation_name}/{k}": v
                        for k, v in result.embedding_diagnostics.items()
                    }
                )

        # Separate finetune step metric per task
        if evaluator.eval_mode == EvalMode.FINETUNE:
            if wandb_callback.enabled:
                wandb_callback.wandb.define_metric(
                    f"{evaluator.evaluation_name}/*",
                    step_metric=f"{evaluator.evaluation_name}_step",
                )
                wandb_callback.wandb.log({f"{evaluator.evaluation_name}_step": 0})

        # Log test results and bootstrap stats independently of val validity
        test_valid = test_result is not None and test_result.primary >= 0
        if wandb_callback.enabled and test_valid:
            if bootstrap_stats:
                wandb_callback.wandb.log(
                    {
                        f"eval/test/{evaluator.evaluation_name}_bootstrap_mean": bootstrap_stats.get(
                            "mean"
                        ),
                        f"eval/test/{evaluator.evaluation_name}_bootstrap_std": bootstrap_stats.get(
                            "std"
                        ),
                        f"eval/test/{evaluator.evaluation_name}_bootstrap_ci_lower": bootstrap_stats.get(
                            "ci_lower"
                        ),
                        f"eval/test/{evaluator.evaluation_name}_bootstrap_ci_upper": bootstrap_stats.get(
                            "ci_upper"
                        ),
                    }
                )
            if self.run_on_test and test_result is not None:
                _log_eval_result_to_wandb(
                    wandb_callback, "eval/test", evaluator.evaluation_name, test_result
                )

    def pre_train(self) -> None:
        """Run the evaluators on startup."""
        # Bind eval metrics to the shared ``checkpoint_step`` x-axis so in-loop
        # eval curves line up with the built-in beaker-job evals (which log on the
        # same axis from a separate run). Only needed for in-loop (non-beaker)
        # evals; the beaker-job path defines these metrics in its own run.
        if not self.run_as_beaker_job:
            wandb_callback = self._get_wandb_callback()
            if (
                wandb_callback is not None
                and wandb_callback.enabled
                and get_rank() == 0
            ):
                define_checkpoint_step_metrics(wandb_callback.wandb)

        if self.eval_on_startup and self.run_as_beaker_job:
            supported = [
                evaluator
                for evaluator in self.evaluators
                if self._check_supported_modalities(evaluator)
            ]
            if supported:
                # Evaluates the checkpoint at the (possibly resumed) current step;
                # skipped inside the launcher if no such checkpoint exists yet.
                self._launch_beaker_eval_job(
                    self.step, [evaluator.evaluation_name for evaluator in supported]
                )
        elif self.eval_on_startup:
            logger.info(f"Running {len(self.evaluators)} evaluators on startup.")

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
                result = self._perform_eval(evaluator)
                self._log_eval_results_to_logger_pretrain(evaluator, result)
                self._log_eval_results_to_wandb_pretrain(evaluator, result)

        if self.cancel_after_first_eval:
            self.trainer.cancel_run(
                "Cancelled from evaluator callback since 'cancel_after_first_eval' is set",
                no_sync=True,  # 'no_sync' because we're calling this from all ranks at the same time.
            )

    def _is_due(self, evaluator: DownstreamEvaluator) -> bool:
        """Whether ``evaluator`` is scheduled to run at the current step."""
        eval_interval_steps = self.trainer.convert_duration_to_steps(
            evaluator.eval_interval
        )
        return self.step > 1 and self.step % eval_interval_steps == 0

    def post_step(self) -> None:
        """Run the evaluators in-loop."""
        if self.run_as_beaker_job:
            due = [
                evaluator
                for evaluator in self.evaluators
                if self._is_due(evaluator)
                and self._check_supported_modalities(evaluator)
            ]
            if due:
                self._launch_beaker_eval_job(
                    self.step, [evaluator.evaluation_name for evaluator in due]
                )
            return

        for evaluator in self.evaluators:
            if not self._is_due(evaluator):
                continue
            if not self._check_supported_modalities(evaluator):
                logger.info(
                    f"Skipping {evaluator.evaluation_name} because it requires a modality that is not supported by the model"
                )
                continue
            self._perform_eval(evaluator)

    def _get_wandb_callback(self) -> OlmoEarthWandBCallback | None:
        """Return the wandb callback if one is registered."""
        return next(
            (
                callback
                for callback in self.trainer._iter_callbacks()
                if isinstance(callback, OlmoEarthWandBCallback)
            ),
            None,
        )

    def _permanent_save_interval(self) -> int | None:
        """Permanent-checkpoint interval (in steps) of the trainer's checkpointer.

        Beaker-job evals load a *saved* checkpoint at their eval step, but only
        permanent checkpoints (saved every ``save_interval`` steps) survive -- a
        single ephemeral checkpoint is kept at a time and deleted well before a
        queued eval job runs. This lets ``_launch_beaker_eval_job`` skip steps that
        will not have a durable checkpoint. Returns ``None`` if no enabled
        checkpointer is found (caller then falls back to the launcher's on-disk
        existence check).
        """
        checkpointer = next(
            (
                callback
                for callback in self.trainer._iter_callbacks()
                if isinstance(callback, CheckpointerCallback)
            ),
            None,
        )
        if checkpointer is None or not getattr(checkpointer, "enabled", True):
            return None
        return checkpointer.save_interval

    def _launch_beaker_eval_job(self, step: int, task_names: list[str]) -> None:
        """Launch a Beaker job to evaluate the checkpoint at ``step``.

        Runs only on rank 0 and never blocks training: it spawns a launcher
        subprocess that submits the eval experiment and returns immediately. The
        eval job evaluates the saved checkpoint (so it does not touch the live
        training model) and logs metrics on a ``checkpoint_step`` wandb x-axis.

        Note: the spawned job rebuilds its evaluators from the canonical eval task
        set, then filters to ``task_names`` -- so the task names used in the in-loop
        config must match those known to the eval job for the filter to select them.
        """
        # Importing lazily keeps the (optional) beaker dependency and import cost out
        # of the module-load path.
        from olmoearth_pretrain.internal.loop_eval_launch import (
            launch_checkpoint_eval_job,
            resolve_parent_priority,
        )

        if get_rank() != 0:
            return

        # The eval job loads the saved checkpoint at ``step``, but only permanent
        # checkpoints persist long enough for a queued Beaker job to load them
        # (ephemeral checkpoints are rotated out within a few hundred steps). If
        # ``step`` will not have a permanent checkpoint, skip the launch instead of
        # spawning a job that will crash with "No step directories found" once the
        # ephemeral checkpoint is gone.
        save_interval = self._permanent_save_interval()
        if save_interval is not None and step % save_interval != 0:
            logger.warning(
                "Skipping in-loop Beaker eval launch for step %d: it is not a "
                "permanent checkpoint step (checkpointer save_interval=%d). Set the "
                "evaluator eval_interval to a multiple of %d so a durable checkpoint "
                "exists for the eval job to load.",
                step,
                save_interval,
                save_interval,
            )
            return

        module_path = self.beaker_eval_module_path or os.environ.get(
            "TRAIN_SCRIPT_PATH"
        )
        if not module_path:
            logger.warning(
                "run_as_beaker_job is set but no module path is available "
                "(set beaker_eval_module_path or the TRAIN_SCRIPT_PATH env var); "
                "skipping eval launch for step %d.",
                step,
            )
            return

        clusters = self.beaker_eval_clusters
        if not clusters:
            env_clusters = os.environ.get("OE_LOOP_EVAL_CLUSTERS")
            clusters = (
                [c.strip() for c in env_clusters.split(",") if c.strip()]
                if env_clusters
                else None
            )
        if not clusters:
            logger.warning(
                "run_as_beaker_job is set but no clusters are available "
                "(set beaker_eval_clusters or the OE_LOOP_EVAL_CLUSTERS env var); "
                "skipping eval launch for step %d.",
                step,
            )
            return

        priority = self.beaker_eval_priority or resolve_parent_priority()

        save_folder = str(self.trainer.save_folder)
        wandb_callback = self._get_wandb_callback()
        train_run_name = (
            wandb_callback.name
            if wandb_callback is not None and wandb_callback.name
            else os.path.basename(save_folder.rstrip("/"))
        )

        launch_checkpoint_eval_job(
            module_path=module_path,
            checkpoint_dir=save_folder,
            step=step,
            cluster=clusters[0],
            run_name=f"{train_run_name}_eval_step{step}",
            priority=priority,
            tasks_to_run=task_names,
            wandb_project=(
                wandb_callback.project if wandb_callback is not None else None
            ),
            wandb_entity=(
                wandb_callback.entity if wandb_callback is not None else None
            ),
            # Group all of this run's eval jobs together (and alongside training).
            wandb_group=train_run_name,
            # All eval steps log to one consolidated wandb run (resumed via the
            # shared runid file set in launch_checkpoint_eval_job), keyed on
            # checkpoint_step -- instead of a separate run per eval step.
            wandb_run_name=f"{train_run_name}_loop_evals",
            # Collect all of this run's eval experiments into one Beaker group
            # (same name as the consolidated wandb run) to declutter the console.
            beaker_group=f"{train_run_name}_loop_evals",
            extra_overrides=self.beaker_eval_extra_overrides,
            log_dir=os.path.join(save_folder, "loop_eval_launch_logs"),
        )

    def _perform_eval(self, evaluator: DownstreamEvaluator) -> EvalTaskResult:
        """Run the evaluator."""
        logger.info(f"Running {evaluator.evaluation_name} evaluations...")

        # Record the checkpoint step alongside the eval metrics so they share the
        # ``checkpoint_step`` x-axis with the built-in beaker-job evals. It is
        # numerically equal to the training step, so existing curves are unchanged.
        self.trainer.record_metric(CHECKPOINT_STEP_METRIC, self.step)

        start_time = time.monotonic()
        result = evaluator.val()

        val_result = result.val_result
        test_result = result.test_result
        bootstrap_stats = result.bootstrap_stats

        # Record validation metrics
        if val_result is not None:
            _record_eval_result(
                self.trainer, "eval", evaluator.evaluation_name, val_result
            )

        if self.run_on_test and test_result is not None:
            _record_eval_result(
                self.trainer, "eval/test", evaluator.evaluation_name, test_result
            )

        # Log bootstrap statistics if available
        if bootstrap_stats:
            self.trainer.record_metric(
                f"eval/test/{evaluator.evaluation_name}_bootstrap_mean",
                bootstrap_stats.get("mean"),
            )
            self.trainer.record_metric(
                f"eval/test/{evaluator.evaluation_name}_bootstrap_std",
                bootstrap_stats.get("std"),
            )
            self.trainer.record_metric(
                f"eval/test/{evaluator.evaluation_name}_bootstrap_ci_lower",
                bootstrap_stats.get("ci_lower"),
            )
            self.trainer.record_metric(
                f"eval/test/{evaluator.evaluation_name}_bootstrap_ci_upper",
                bootstrap_stats.get("ci_upper"),
            )
        if result.embedding_diagnostics:
            for metric_name, metric_value in result.embedding_diagnostics.items():
                self.trainer.record_metric(
                    f"eval_embed_diagnostics/{evaluator.evaluation_name}/{metric_name}",
                    metric_value,
                )

        eval_time = time.monotonic() - start_time
        self.trainer.record_metric(f"eval_time/{evaluator.evaluation_name}", eval_time)
        logger.info(
            f"Finished {evaluator.evaluation_name} evaluations in {eval_time:.1f} seconds."
        )

        result.eval_time = eval_time
        return result


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
    # Bootstrap sampling for uncertainty estimation (applies to KNN and Linear Probe)
    n_bootstrap: int = 0  # Number of bootstrap samples (0 = no bootstrap)
    bootstrap_seed: int = 42  # Random seed for bootstrap sampling

    # --- Run evals as separate Beaker jobs (instead of inline, blocking training) ---
    # When True, an eval that comes due launches a Beaker job (via
    # checkpoint_sweep_evals.py) that evaluates the just-saved checkpoint, and the
    # training loop continues without running the eval inline. Requires that a
    # checkpoint exists at the eval step, so eval_interval should be a multiple of the
    # checkpointer save_interval.
    run_as_beaker_job: bool = False
    # Module path used to rebuild the model/train-module config in the eval job
    # (passed as TRAIN_SCRIPT_PATH). Falls back to the $TRAIN_SCRIPT_PATH env var.
    beaker_eval_module_path: str | None = None
    # Cluster(s) the eval job may run on. The first is used as the launch positional.
    # Falls back to $OE_LOOP_EVAL_CLUSTERS (comma-separated) if unset.
    beaker_eval_clusters: list[str] | None = None
    # Priority for the eval job. None -> match the running training job's priority.
    beaker_eval_priority: str | None = None
    # Extra CLI overrides appended verbatim to the eval launch command.
    beaker_eval_extra_overrides: list[str] | None = None

    def verify_input_modalities(
        self, task: DownstreamTaskConfig, config: EvalDatasetConfig
    ) -> None:
        """Verify the input modality configuration for a task."""
        # Make sure input_modalities contains only unique modalities
        if len(task.input_modalities) != len(set(task.input_modalities)):
            raise ValueError(
                f"input_modalities must contain unique modalities, got {task.input_modalities}"
            )
        if not set(task.input_modalities).issubset(set(config.supported_modalities)):
            raise ValueError(
                f"input_modalities must be a subset of supported_modalities, got {task.input_modalities} and {config.supported_modalities}"
            )

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

            if task.h5py_dir is not None and not UPath(task.h5py_dir).exists():
                raise FileNotFoundError(
                    f"h5py_dir for eval task '{evaluation_name}' does not exist: "
                    f"{task.h5py_dir}. Pretrain-subset eval snapshots are created "
                    f"by scripts/tools/20260611_snapshot_pretrain_eval_subset.py "
                    f"and must match the path constants in "
                    f"olmoearth_pretrain/internal/all_evals.py."
                )

            config = dataset_to_config(task.dataset)
            if (
                config.task_type
                in (
                    TaskType.SEGMENTATION,
                    TaskType.PER_PIXEL_REGRESSION,
                    TaskType.WINDOW_REGRESSION,
                )
                and task.eval_mode != EvalMode.EMBEDDING_DIAGNOSTICS
            ):
                if task.probe_lr is None and task.ft_lr is None:
                    raise ValueError(
                        f"probe_lr and ft_lr cannot both be None for {task.dataset}"
                    )

            self.verify_input_modalities(task, config)
            # Sort to ensure consistent order
            task.input_modalities.sort()
            logger.info(f"Adding {evaluation_name} with eval mode {task.eval_mode}")
            evaluators.append(
                DownstreamEvaluator(
                    evaluation_name=evaluation_name,
                    task=task,
                    trainer=trainer,
                    device=trainer.device,
                    run_on_test=self.run_on_test,
                    n_bootstrap=self.n_bootstrap,
                    bootstrap_seed=self.bootstrap_seed,
                )
            )
        return DownstreamEvaluatorCallback(
            evaluators=evaluators,
            eval_on_startup=self.eval_on_startup,
            cancel_after_first_eval=self.cancel_after_first_eval,
            run_on_test=self.run_on_test,
            n_bootstrap=self.n_bootstrap,
            bootstrap_seed=self.bootstrap_seed,
            run_as_beaker_job=self.run_as_beaker_job,
            beaker_eval_module_path=self.beaker_eval_module_path,
            beaker_eval_clusters=self.beaker_eval_clusters,
            beaker_eval_priority=self.beaker_eval_priority,
            beaker_eval_extra_overrides=self.beaker_eval_extra_overrides,
        )
