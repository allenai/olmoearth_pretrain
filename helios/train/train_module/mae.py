"""Training and optimizer abstraction for Helios."""

from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from olmo_core.distributed.parallel import DataParallelConfig
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.common import Duration, ReduceType
from olmo_core.train.train_module.transformer import (
    TransformerActivationCheckpointingConfig,
)

from helios.data.constants import Modality
from helios.data.dataset import HeliosSample
from helios.data.transform import TransformConfig
from helios.nn.mae import MAE
from helios.train.loss import LossConfig
from helios.train.masking import MaskedHeliosSample, MaskingConfig
from helios.train.train_module.train_module import (
    HeliosTrainModule,
    HeliosTrainModuleConfig,
)
from helios.train.utils import split_batch

logger = getLogger(__name__)


@dataclass
class MAETrainModuleConfig(HeliosTrainModuleConfig):
    """A configuration class for building :class:`MAETrainModule` instances.

    Args:
        loss_config: The loss configuration for the model.
        masking_config: The masking configuration for the model.
    """

    mae_loss_config: LossConfig | None = field(
        default_factory=lambda: LossConfig(
            loss_config={"type": "mae", "loss_function": "SmoothL1Loss", "beta": 0.1}
        )
    )
    latent_mim_loss_config: LossConfig | None = None
    masking_config: MaskingConfig = field(
        default_factory=lambda: MaskingConfig(strategy_config={"type": "random"})
    )
    token_exit_cfg: dict[str, int] = field(
        default_factory=lambda: {modality: 0 for modality in Modality.names()}
    )
    max_grad_norm: float = 1.0
    warmup_duration: Duration = field(default_factory=lambda: Duration.epochs(2))

    def build(
        self,
        model: MAE,
        device: torch.device | None = None,
    ) -> "MAETrainModule":
        """Build the corresponding :class:`MAETrainModule`.

        Args:
            model: The model to train.
            device: The device to train on.
        """
        kwargs = self.prepare_kwargs()
        return MAETrainModule(
            model=model,
            device=device,
            **kwargs,
        )


class MAETrainModule(HeliosTrainModule):
    """A :class:`TrainModule`.

    Initialize the training module.

    Args:
        model: The transformer model to train.
        optim: The corresponding optimizer config.
        masking_config: The masking configuration for the model.
        loss_config: The loss configuration for the model.
        rank_microbatch_size: The rank microbatch size in instances.
        compile_model: Whether to compile to the model.
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
        token_exit_cfg: The token exit configuration for the model.
        warmup_duration: The warmup duration for the model.
        regularizer_config: An optional regularizer configuration for the model.
        find_unused_parameters: Whether to find unused parameters in the model.
    """

    def __init__(
        self,
        model: MAE,
        optim_config: OptimConfig,
        transform_config: TransformConfig,
        masking_config: MaskingConfig,
        rank_microbatch_size: int,
        token_exit_cfg: dict[str, int],
        compile_model: bool = False,
        dp_config: DataParallelConfig | None = None,
        ac_config: TransformerActivationCheckpointingConfig | None = None,
        compile_loss: bool = False,
        autocast_precision: torch.dtype | None = None,
        max_grad_norm: float | None = None,
        scheduler: Scheduler | None = None,
        device: torch.device | None = None,
        state_dict_save_opts: dist_cp_sd.StateDictOptions | None = None,
        state_dict_load_opts: dist_cp_sd.StateDictOptions | None = None,
        warmup_duration: Duration = Duration.epochs(2),
        mae_loss_config: LossConfig | None = None,
        latent_mim_loss_config: LossConfig | None = None,
        regularizer_config: LossConfig | None = None,
        find_unused_parameters: bool = True,
    ):
        """Initialize the training module.

        Args:
            model: The transformer model to train.
            optim_config: The corresponding optimizer config.
            transform_config: The transform configuration for the model.
            masking_config: The masking configuration for the model.
            mae_loss_config: The loss configuration for mae.
            latent_mim_loss_config: The loss configuration for latent mim.
            rank_microbatch_size: The rank microbatch size in instances.
            compile_model: Whether to compile to the model.
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
            token_exit_cfg: The token exit configuration for the model.
            warmup_duration: The warmup duration for the model.
            regularizer_config: An optional regularizer configuration for the model.
            find_unused_parameters: Whether to find unused parameters in the model.
        """
        super().__init__(
            model=model,
            optim_config=optim_config,
            transform_config=transform_config,
            rank_microbatch_size=rank_microbatch_size,
            compile_model=compile_model,
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
            find_unused_parameters=find_unused_parameters,  # Must be true so that we can deal with missing modalities
        )
        self.masking_strategy = masking_config.build()
        self.token_exit_cfg = token_exit_cfg
        self.mae_loss = mae_loss_config and mae_loss_config.build()
        self.latent_mim_loss = latent_mim_loss_config and latent_mim_loss_config.build()
        self.regularizer = regularizer_config and regularizer_config.build()

        loss_names = [
            loss.name
            for loss in [self.mae_loss, self.latent_mim_loss, self.regularizer]
            if loss is not None
        ]
        self.total_loss_name = "+".join(loss_names)

    def loss_fn(self, pred: Any, targets: Any) -> torch.Tensor:
        """Compute the loss between the predicted and target tensors."""
        return self.base_loss.compute(pred, targets)

    def model_forward(
        self, x: MaskedHeliosSample, patch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the model."""
        with self._model_forward_context():
            latent, decoded, reconstructed = self.model(x, patch_size=patch_size)

            loss = torch.zeros([], device=self.device)
            if self.mae_loss and reconstructed is not None:
                loss += self.mae_loss.compute(reconstructed, x)
            if self.latent_mim_loss and decoded is not None:
                with torch.no_grad():
                    logger.info("Target Encoder forward pass...")
                    target_output = self.model.encoder.forward(
                        x.unmask(),
                        patch_size=patch_size,
                        token_exit_cfg=self.token_exit_cfg,
                    )
                loss += self.latent_mim_loss.compute(decoded, target_output)
            return loss, latent, decoded

    def train_batch(
        self, patch_batch: tuple[int, HeliosSample], dry_run: bool = False
    ) -> None:
        """Train a batch.

        NOTE: Gradient accumulation/microbatching is not invariant for all losses across the same global batch size.

        - All Disc loss with same global batch size but different micro-batch sizes result in different gradients,
        though this matches the implementation in gallileo.
        - If the min hw is too low when subsampling, we may get micro-batches with uneven
        numbers of tokens making the loss for token averaged losses
        like l1 and l2 weight microbatches with less tokens relatively more.

        NOTE: For contrastive losses, the loss is invariant to the global batch size across GPUS as well
        """
        # why is this a tuple?
        patch_size, batch = patch_batch
        self.model.train()
        # Set the maximum number of tokens
        total_batch_loss = torch.zeros([], device=self.device)
        total_batch_reg = torch.zeros([], device=self.device)
        # Split into micro-batches.
        microbatches = split_batch(batch, self.rank_microbatch_size)
        num_microbatches = len(microbatches)
        for microbatch_idx, microbatch in enumerate(microbatches, start=1):
            with self._train_microbatch_context(microbatch_idx, num_microbatches):
                logger.info(
                    f"Training microbatch {microbatch_idx} of {num_microbatches} with batch size {microbatch.batch_size}"
                )
                microbatch = self.transform.apply(microbatch).to_device(self.device)
                masked_batch = self.masking_strategy.apply_mask(
                    microbatch, patch_size=patch_size
                )

                # Run Encoder and decoder on the augmented input
                loss, latent, decoded = self.model_forward(masked_batch, patch_size)
                reg_term = self.compute_regularization(latent)
                if reg_term is not None:
                    loss = loss + reg_term
                    total_batch_reg += (
                        get_local_tensor(reg_term.detach()) / num_microbatches
                    )
                # Scale loss by number of microbatches
                loss = loss / num_microbatches
                loss_val = get_local_tensor(loss.detach())
                total_batch_loss += loss_val

                # Skip bad batches
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logger.warning(
                        f"NaN or Inf detected in loss at microbatch {microbatch_idx}, stopping training for this batch."
                    )
                    break

                loss.backward()

        self.trainer.record_metric(
            f"train/{self.total_loss_name}",
            total_batch_loss,
            ReduceType.mean,
        )
        self.log_regularization(total_batch_reg)

        if dry_run:
            return

        del batch  # In case this helps with memory utilization.
        del masked_batch
