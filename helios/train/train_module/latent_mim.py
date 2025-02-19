"""Training and optimizer abstraction for Helios."""

from dataclasses import dataclass
from logging import getLogger
from typing import Any

import numpy as np
import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from olmo_core.distributed.parallel import DataParallelConfig
from olmo_core.distributed.utils import get_world_size
from olmo_core.float8 import Float8Config
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.common import ReduceType
from olmo_core.train.train_module.transformer import (
    TransformerActivationCheckpointingConfig,
)

from helios.data.dataset import HeliosSample
from helios.train.loss import LossConfig
from helios.train.masking import MaskedHeliosSample, MaskingConfig
from helios.train.train_module.train_module import (
    HeliosTrainModule,
    HeliosTrainModuleConfig,
)

logger = getLogger(__name__)

TRAIN_PATCH_DISC_LOSS_METRIC = "train/patch_disc_loss"


@dataclass
class LatentMIMTrainModuleConfig(HeliosTrainModuleConfig):
    """A configuration class for building :class:`LatentMIMTrainModule` instances.

    Args:
        loss_config: The loss configuration for the model.
        masking_config: The masking configuration for the model.
        ema_decay: EMA decay rate for target encoder (default: 0.99).
    """

    loss_config: LossConfig = LossConfig(loss_config={"type": "patch_discrimination"})
    masking_config: MaskingConfig = MaskingConfig(strategy_config={"type": "random"})
    ema_decay: float = 0.99

    def build(
        self,
        model: Any,
        device: torch.device | None = None,
    ) -> "LatentMIMTrainModule":
        """Build the corresponding :class:`LatentMIMTrainModule`.

        Args:
            model: The model to train.
            device: The device to train on.
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        return LatentMIMTrainModule(
            model=model,
            device=device,
            **kwargs,
        )


class LatentMIMTrainModule(HeliosTrainModule):
    """A :class:`TrainModule`.

    Initialize the training module.

    Args:
        model: The transformer model to train.
        optim: The corresponding optimizer config.
        rank_batch_size: The rank batch size in instances.
        compile_model: Whether to compile to the model.
        float8_config: Float8 configuration for the model.
        dp_config: Data parallel configuration for the model.
        ac_config: Activation checkpointing configuration for the model.
        loss_fn: Loss function to use.
        compile_loss: Whether to compile the loss function.
        autocast_precision: Enable AMP with this data type.
        max_grad_norm: Clip gradient norms to this value.
        scheduler: Optional learning rate scheduler.
        device: The device to train on.
        state_dict_save_opts: Override state dict options for saving.
        state_dict_load_opts: Override state dict options for loading.
    """

    def __init__(
        self,
        model: Any,
        optim: OptimConfig,
        masking_config: MaskingConfig,
        loss_config: LossConfig,
        rank_batch_size: int,
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
        ema_decay: float = 0.99,
    ):
        """Initialize the training module.

        Args:
            model: The transformer model to train.
            optim: The corresponding optimizer config.
            masking_config: The masking configuration for the model.
            loss_config: The loss configuration for the model.
            rank_batch_size: The rank batch size in instances.
            compile_model: Whether to compile to the model.
            float8_config: Float8 configuration for the model.
            dp_config: Data parallel configuration for the model.
            ac_config: Activation checkpointing configuration for the model.
            loss_fn: Loss function to use.
            compile_loss: Whether to compile the loss function.
            autocast_precision: Enable AMP with this data type.
            max_grad_norm: Clip gradient norms to this value.
            scheduler: Optional learning rate scheduler.
            device: The device to train on.
            state_dict_save_opts: Override state dict options for saving.
            state_dict_load_opts: Override state dict options for loading.
            ema_decay: EMA decay rate for target encoder (default: 0.99).
        """
        super().__init__(
            model=model,
            optim=optim,
            rank_batch_size=rank_batch_size,
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
        )
        self.ema_decay = ema_decay
        self.base_loss = loss_config.build()
        self.masking_strategy = masking_config.build()

    # TODO: Do we always want tokens and masks?
    def loss_fn(self, pred: Any, targets: Any) -> torch.Tensor:
        """Compute the loss between the predicted and target tensors."""
        return self.base_loss.compute(pred, targets)

    def eval_loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the loss between the predicted and target tensors."""
        raise NotImplementedError("eval loss fn not implemented")

    def train_batch(self, batch: HeliosSample, dry_run: bool = False) -> None:
        """Train a batch."""
        # Record how many instances are going to be skipped (masked out).
        # if (instance_mask := batch.get("instance_mask")) is not None and not dry_run:
        #     self.record_metric("train/masked instances", (~instance_mask).sum(), ReduceType.sum)

        # we may want to modify this
        token_budget = self.model.token_budget
        # Smallest h /w must be bigger than the smallest patch size
        h_w_to_sample = list(
            range(self.model.h_w_to_sample_min, self.model.h_w_to_sample_max)
        )

        patch_size = np.random.choice(np.arange(1, self.model.encoder.max_patch_size))
        logger.info(f"Patch size: {patch_size}")
        subsampled_batch = batch.subset(patch_size, token_budget, h_w_to_sample)

        subsampled_batch = subsampled_batch.to_device(self.device)
        logger.info(f"subsampled batch: input {subsampled_batch.sentinel2.shape}")
        masked_batch = self.masking_strategy.apply_mask(subsampled_batch)
        logger.info(
            f"masked batch: input {masked_batch.sentinel2.shape} and mask {masked_batch.sentinel2_mask.shape}"
        )

        # Run Encoder and decoder on the augmented input
        decoded, loss = self.model_forward(masked_batch, patch_size)

        self.trainer.record_metric(
            TRAIN_PATCH_DISC_LOSS_METRIC,
            loss / get_world_size(self.dp_process_group),
            ReduceType.sum,
        )

        # Backpropagate and optimize
        if loss is not None:
            loss.backward()
        # Update target encoder with EMA this should be a callback
        with torch.no_grad():
            for param, target_param in zip(
                self.model.encoder.parameters(), self.model.target_encoder.parameters()
            ):
                target_param.data = (
                    self.ema_decay * target_param.data
                    + (1 - self.ema_decay) * param.data
                )

        del batch  # In case this helps with memory utilization.

        if dry_run:
            self._clear_loss_buffers()
            return

        self._clear_loss_buffers()

    def eval_batch(
        self, batch: dict[str, Any], labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Evaluate a batch."""
        raise NotImplementedError("eval batch not implemented")

    def model_forward(
        self, batch: MaskedHeliosSample, patch_size: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Run a forward pass."""
        with self._model_forward_context():
            decoded = self.model.forward(batch, patch_size)
            with torch.no_grad():
                logger.info("target encoder running here")
                target_output = self.model.target_encoder.forward(
                    batch.unmask(), patch_size=patch_size
                )
            loss = self.loss_fn(decoded, target_output)
            return decoded, loss
