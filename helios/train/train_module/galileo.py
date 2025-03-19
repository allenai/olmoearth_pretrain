"""Training and optimizer abstraction for Helios."""

from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

import numpy as np
import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from olmo_core.distributed.parallel import DataParallelConfig
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.float8 import Float8Config
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.common import Duration, ReduceType
from olmo_core.train.train_module.transformer import (
    TransformerActivationCheckpointingConfig,
)
import sys
import gc
from helios.data.constants import Modality
from helios.data.dataset import HeliosSample
from helios.nn.latent_mim import LatentMIM
from helios.train.loss import LossConfig
from helios.train.masking import MaskedHeliosSample, MaskingConfig
from helios.train.train_module.train_module import (
    HeliosTrainModule,
    HeliosTrainModuleConfig,
)
from helios.train.utils import split_batch

logger = getLogger(__name__)
import os

import psutil


def log_memory_usage_for_process(process):
    """Log memory usage for a given process and return memory stats."""
    try:
        memory_info = process.memory_info()
        rss = memory_info.rss
        pss = 0
        uss = 0
        shared = 0

        # # Log the process memory usage
        # logger.info(
        #     f"Process (PID {process.pid}) memory usage: RSS={rss / (1024 * 1024 * 1024):.2f} GB"
        # )

        # Iterate over memory maps
        for mmap in process.memory_maps():
            pss += mmap.pss
            uss += mmap.private_clean + mmap.private_dirty
            shared += mmap.shared_clean + mmap.shared_dirty

        return rss, pss, uss, shared

    except psutil.NoSuchProcess:
        # The process may have terminated between the time we got the list and now
        return 0, 0, 0, 0


def log_total_memory_usage():
    """Log total memory usage for the main process and its children."""
    # Get the current process (main process)
    main_process = psutil.Process(os.getpid())

    # Initialize total memory usage counters
    total_rss = 0
    total_pss = 0
    total_uss = 0
    total_shared = 0

    # Log memory usage for the main process
    logger.info("Logging memory usage for main process")
    rss, pss, uss, shared = log_memory_usage_for_process(main_process)
    total_rss += rss
    total_pss += pss
    total_uss += uss
    total_shared += shared

    # Iterate over child processes and log their memory usage
    logger.info("Logging memory usage for child processes")
    for child in main_process.children(recursive=True):
        rss, pss, uss, shared = log_memory_usage_for_process(child)
        total_rss += rss
        total_pss += pss
        total_uss += uss
        total_shared += shared

    # Log the total memory usage
    # logger.info(
    #     f"Total memory usage: RSS={total_rss / (1024 * 1024 * 1024):.2f} GB, PSS={total_pss / (1024 * 1024 * 1024):.2f} GB, USS={total_uss / (1024 * 1024 * 1024):.2f} GB, Shared={total_shared / (1024 * 1024 * 1024):.2f} GB"
    # )
    return total_pss / (1024 * 1024 * 1024)


@dataclass
class GalileoTrainModuleConfig(HeliosTrainModuleConfig):
    """A configuration class for building :class:`GalileoTrainModule` instances.

    Args:
        loss_config: The loss configuration for the model.
        masking_config: The masking configuration for the model.
        ema_decay: EMA decay rate for target encoder (default: 0.99).
    """

    loss_config_a: LossConfig = field(
        default_factory=lambda: LossConfig(loss_config={"type": "patch_discrimination"})
    )
    loss_config_b: LossConfig = field(
        default_factory=lambda: LossConfig(loss_config={"type": "patch_discrimination"})
    )
    masking_config_a: MaskingConfig = field(
        default_factory=lambda: MaskingConfig(strategy_config={"type": "random"})
    )
    masking_config_b: MaskingConfig = field(
        default_factory=lambda: MaskingConfig(strategy_config={"type": "random"})
    )
    token_exit_cfg_a: dict[str, int] = field(
        default_factory=lambda: {modality: 0 for modality in Modality.names()}
    )
    token_exit_cfg_b: dict[str, int] = field(
        default_factory=lambda: {modality: 0 for modality in Modality.names()}
    )
    warmup_duration: Duration = field(default_factory=lambda: Duration.epochs(2))
    ema_decay: tuple[float, float] = (0.996, 1.0)
    max_grad_norm: float = 1.0

    def build(
        self,
        model: LatentMIM,
        device: torch.device | None = None,
    ) -> "GalileoTrainModule":
        """Build the corresponding :class:`LatentMIMTrainModule`.

        Args:
            model: The model to train.
            device: The device to train on.
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        return GalileoTrainModule(
            model=model,
            device=device,
            **kwargs,
        )


class GalileoTrainModule(HeliosTrainModule):
    """A :class:`TrainModule`.

    Initialize the training module.

    Args:
        model: The transformer model to train.
        optim: The corresponding optimizer config.
        masking_config_a: The masking configuration for the model.
        masking_config_b: The masking configuration for the model.
        loss_config_a: The loss configuration for the model.
        loss_config_b: The loss configuration for the model.
        rank_microbatch_size: The rank microbatch size in instances.
        compile_model: Whether to compile to the model.
        float8_config: Float8 configuration for the model.
        dp_config: Data parallel configuration for the model.
        ac_config: Activation checkpointing configuration for the model.
        compile_loss: Whether to compile the loss function.
        autocast_precision: Enable AMP with this data type.
        max_grad_norm: Clip gradient norms to this value.
        scheduler: Optional learning rate scheduler.
        device: The device to train on.
        state_dict_save_opts: Override state dict options for saving.
        state_dict_load_opts: Override state dict options for loading.
        token_exit_cfg_a: The token exit configuration for the model.
        token_exit_cfg_b: The token exit configuration for the model.
        warmup_duration: The warmup duration for the model.
    """

    def __init__(
        self,
        model: LatentMIM,
        optim_config: OptimConfig,
        masking_config_a: MaskingConfig,
        masking_config_b: MaskingConfig,
        loss_config_a: LossConfig,
        loss_config_b: LossConfig,
        rank_microbatch_size: int,
        token_exit_cfg_a: dict[str, int],
        token_exit_cfg_b: dict[str, int],
        compile_model: bool = False,
        float8_config: Float8Config | None = None,
        dp_config: DataParallelConfig | None = None,
        ac_config: TransformerActivationCheckpointingConfig | None = None,
        compile_loss: bool = False,
        autocast_precision: torch.dtype | None = None,
        max_grad_norm: float | None = None,
        scheduler: Scheduler | None = None,
        device: torch.device | None = None,
        state_dict_save_opts: dist_cp_sd.StateDictOptions | None = None,
        state_dict_load_opts: dist_cp_sd.StateDictOptions | None = None,
        ema_decay: tuple[float, float] = (0.996, 1.0),
        warmup_duration: Duration = Duration.epochs(2),
    ):
        """Initialize the training module.

        Args:
            model: The transformer model to train.
            optim_config: The corresponding optimizer config.
            masking_config_a: The masking configuration for the model.
            masking_config_b: The masking configuration for the model.
            loss_config_a: The loss configuration for the model.
            loss_config_b: The loss configuration for the model.
            rank_microbatch_size: The rank microbatch size in instances.
            compile_model: Whether to compile to the model.
            float8_config: Float8 configuration for the model.
            dp_config: Data parallel configuration for the model.
            ac_config: Activation checkpointing configuration for the model.
            compile_loss: Whether to compile the loss function.
            autocast_precision: Enable AMP with this data type.
            max_grad_norm: Clip gradient norms to this value.
            scheduler: Optional learning rate scheduler.
            device: The device to train on.
            state_dict_save_opts: Override state dict options for saving.
            state_dict_load_opts: Override state dict options for loading.
            ema_decay: EMA decay rate for target encoder, as a tuple of (start_ema_decay, end_ema_decay)
            token_exit_cfg_a: The token exit configuration for the model.
            token_exit_cfg_b: The token exit configuration for the model.
            warmup_duration: The warmup duration for the model.
        """
        super().__init__(
            model=model,
            optim_config=optim_config,
            rank_microbatch_size=rank_microbatch_size,
            compile_model=compile_model,
            float8_config=float8_config,
            dp_config=dp_config,
            ac_config=ac_config,
            compile_loss=compile_loss,
            autocast_precision=autocast_precision,
            max_grad_norm=max_grad_norm,
            scheduler=scheduler,
            device=device,
            state_dict_save_opts=state_dict_save_opts,
            state_dict_load_opts=state_dict_load_opts,
            warmup_duration=warmup_duration,
        )
        self.start_ema, self.end_ema = ema_decay
        self.token_exit_cfg_a = token_exit_cfg_a
        self.base_loss_a = loss_config_a.build()
        self.masking_strategy_a = masking_config_a.build()
        self.token_exit_cfg_b = token_exit_cfg_b
        self.base_loss_b = loss_config_b.build()
        self.masking_strategy_b = masking_config_b.build()

    def loss_fn_a(self, pred: Any, targets: Any) -> torch.Tensor:
        """Compute the loss between the predicted and target tensors."""
        return self.base_loss_a.compute(pred, targets)

    def loss_fn_b(self, pred: Any, targets: Any) -> torch.Tensor:
        """Compute the loss between the predicted and target tensors."""
        return self.base_loss_b.compute(pred, targets)

    def eval_loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the loss between the predicted and target tensors."""
        raise NotImplementedError("eval loss fn not implemented")

    def update_target_encoder(self) -> None:
        """Update the target encoder."""
        # Update target encoder with EMA this should be a callback
        cur_ema_value = (
            self.start_ema
            + self.trainer.global_step
            * (self.end_ema - self.start_ema)
            / self.trainer.max_steps
        )
        with torch.no_grad():
            logger.info(f"Using ema decay {cur_ema_value}")
            for param, target_param in zip(
                self.model.encoder.parameters(), self.model.target_encoder.parameters()
            ):
                target_param.data = (
                    cur_ema_value * target_param.data + (1 - cur_ema_value) * param.data
                )

    def train_batch(self, batch: HeliosSample, dry_run: bool = False) -> None:
        """Train a batch.

        NOTE: Gradient accumulation/microbatching is not invariant for all losses across the same global batch size.

        - All Disc loss with same global batch size but different micro-batch sizes result in different gradients,
        though this matches the implementation in gallileo.
        - If the min hw is too low when subsampling, we may get micro-batches with uneven
        numbers of tokens making the loss for token averaged losses
        like l1 and l2 weight microbatches with less tokens relatively more.

        NOTE: For contrastive losses, the loss is invariant to the global batch size across GPUS as well
        """

        # # # Collect all tensors
        # tensors = [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)]

        # # Sort tensors by size (bytes)
        # tensors = sorted(tensors, key=lambda x: x.numel() * x.element_size(), reverse=True)

        # total_tensor_ram = sum([tensor.numel() * tensor.element_size() / (1024 ** 2) for tensor in tensors])
        # logger.info(f"Total tensor RAM: {total_tensor_ram:.2f} MB")
        # # Print top N largest tensors
        # # total_size = 0
        # # print(f"Found {len(tensors)} tensors in memory.")
        # # for i, tensor in enumerate(tensors[:100]):  # Show top 10
        # #     size_in_mb = tensor.numel() * tensor.element_size() / (1024 ** 2)
        # #     if i < 5:
        # #         print(f"Tensor {i}: Shape {tensor.shape}, Dtype {tensor.dtype}, Size {size_in_mb:.2f} MB")
        # #     total_size += size_in_mb
        # # logger.info(f"Total size of top 100 tensors: {total_size:.2f} MB")
        # logger.info("Logging total virtual memory usage")
        # logger.info(f"Total virtual memory usage: {psutil.virtual_memory().total / (1024 * 1024 * 1024):.2f} GB")
        # total_pss = log_total_memory_usage()

        # self.trainer.record_metric("worker_memory/total_pss", total_pss, ReduceType.mean)
        if dry_run:
            return
        self.update_target_encoder()
        # Set the model to train mode
        self.model.train()

        # Set the maximum number of tokens
        token_budget = self.model.token_budget
        total_batch_loss = torch.tensor(0.0, device=self.device)
        # Split into micro-batches.
        patch_size, batch = batch
        for modality_name in batch.modalities:
            # log all the shapes of the modalities
            modality_data = getattr(batch, modality_name)
            logger.info(f"Modality: {modality_name}, shape: {modality_data.shape}")
        microbatches = split_batch(batch, self.rank_microbatch_size)
        num_microbatches = len(microbatches)
        for microbatch_idx, microbatch in enumerate(microbatches, start=1):
            with self._train_microbatch_context(microbatch_idx, num_microbatches):
                logger.info(
                    f"Training microbatch {microbatch_idx} of {num_microbatches} with batch size {microbatch.batch_size}"
                )
                # Gallileo does this subsetting at the microbatch level so we follow that for now
                # Smallest h /w must be bigger than the smallest patch size

                # apply transforms after the subsetting
                microbatch = self.model.transform.apply(microbatch).to_device(self.device)

                # Each microbatch should have about the same number of encoded tokens if
                # we mask here
                logger.info(f"Patch size: {patch_size}")
                logger.info(f"patch size type: {type(patch_size)}")
                if microbatch_idx % 2 == 0:
                    masked_batch = self.masking_strategy_a.apply_mask(
                        microbatch, patch_size=patch_size
                    )

                    # Run Encoder and decoder on the augmented input
                    decoded, target_output = self.model_forward_a(
                        masked_batch, patch_size, self.token_exit_cfg_a
                    )
                    loss = self.loss_fn_a(decoded, target_output)
                else:
                    masked_batch = self.masking_strategy_b.apply_mask(
                        microbatch, patch_size=patch_size
                    )

                    # Run Encoder and decoder on the augmented input
                    decoded, target_output = self.model_forward_b(
                        masked_batch, patch_size, self.token_exit_cfg_b
                    )
                    loss = self.loss_fn_b(decoded, target_output)
                # Scale loss by number of microbatches
                loss = loss / num_microbatches
                loss_val = get_local_tensor(loss)
                total_batch_loss += loss_val

                # Skip bad batches# Skip bad batches
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logger.warning(
                        f"NaN or Inf detected in loss at microbatch {microbatch_idx}, stopping training for this batch."
                    )
                    del decoded, target_output
                    break

                del decoded, target_output
                loss.backward()

        self.trainer.record_metric(
            f"train/{self.base_loss_a.name}+{self.base_loss_b.name}",
            total_batch_loss,
            ReduceType.mean,
        )

        if dry_run:
            return

        # del batch  # In case this helps with memory utilization.
        # del masked_batch

    def eval_batch(
        self, batch: dict[str, Any], labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Evaluate a batch."""
        raise NotImplementedError("eval batch not implemented")

    def model_forward_a(
        self, batch: MaskedHeliosSample, patch_size: int, token_exit_cfg: dict[str, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass."""
        with self._model_forward_context():
            decoded = self.model.forward_a(batch, patch_size)
            with torch.no_grad():
                logger.info("target encoder running here")
                target_output = self.model.target_encoder.forward(
                    batch.unmask(),
                    patch_size=patch_size,
                    token_exit_cfg=token_exit_cfg,
                )
            return decoded, target_output

    def model_forward_b(
        self, batch: MaskedHeliosSample, patch_size: int, token_exit_cfg: dict[str, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass."""
        with self._model_forward_context():
            decoded = self.model.forward_b(batch, patch_size)
            with torch.no_grad():
                logger.info("target encoder running here")
                target_output = self.model.target_encoder.forward(
                    batch.unmask(),
                    patch_size=patch_size,
                    token_exit_cfg=token_exit_cfg,
                )
            return decoded, target_output
