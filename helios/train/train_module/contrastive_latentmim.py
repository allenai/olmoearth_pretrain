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
from helios.nn.flexihelios import TokensAndMasks
from helios.nn.latent_mim import LatentMIM
from helios.train.loss import LossConfig
from helios.train.masking import MaskedHeliosSample, MaskingConfig
from helios.train.train_module.train_module import (
    HeliosTrainModule,
    HeliosTrainModuleConfig,
)
from helios.train.utils import split_batch

logger = getLogger(__name__)


@dataclass
class ContrastiveLatentMIMTrainModuleConfig(HeliosTrainModuleConfig):
    """A configuration class for building :class:`LatentMIMTrainModule` instances.

    Args:
        loss_config: The loss configuration for the model.
        masking_config: The masking configuration for the model.
        ema_decay: EMA decay rate for target encoder (default: 0.99).
    """

    loss_config: LossConfig = field(
        default_factory=lambda: LossConfig(loss_config={"type": "patch_discrimination"})
    )
    masking_config: MaskingConfig = field(
        default_factory=lambda: MaskingConfig(strategy_config={"type": "random"})
    )
    token_exit_cfg: dict[str, int] = field(
        default_factory=lambda: {modality: 0 for modality in Modality.names()}
    )
    warmup_duration: Duration = field(default_factory=lambda: Duration.epochs(2))
    ema_decay: tuple[float, float] = (0.996, 1.0)
    max_grad_norm: float = 1.0
    contrastive_config: LossConfig | None = None

    def build(
        self,
        model: LatentMIM,
        device: torch.device | None = None,
    ) -> "ContrastiveLatentMIMTrainModuleConfig":
        """Build the corresponding :class:`ContrastiveLatentMIMTrainModuleConfig`.

        Args:
            model: The model to train.
            device: The device to train on.
        """
        kwargs = self.prepare_kwargs()
        return ContrastiveLatentMIMTrainModule(
            model=model,
            device=device,
            **kwargs,
        )


class ContrastiveLatentMIMTrainModule(HeliosTrainModule):
    """A :class:`TrainModule`.

    Initialize the training module.

    Args:
        model: The transformer model to train.
        optim: The corresponding optimizer config.
        transform_config: The transform configuration for the model.
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
    """

    def __init__(
        self,
        model: LatentMIM,
        optim_config: OptimConfig,
        transform_config: TransformConfig,
        masking_config: MaskingConfig,
        loss_config: LossConfig,
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
        ema_decay: tuple[float, float] = (0.996, 1.0),
        warmup_duration: Duration = Duration.epochs(2),
        regularizer_config: LossConfig | None = None,
        contrastive_config: LossConfig | None = None,
    ):
        """Initialize the training module.

        Args:
            model: The transformer model to train.
            optim_config: The corresponding optimizer config.
            transform_config: The transform configuration for the model.
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
            ema_decay: EMA decay rate for target encoder, as a tuple of (start_ema_decay, end_ema_decay)
            token_exit_cfg: The token exit configuration for the model.
            warmup_duration: The warmup duration for the model.
            regularizer_config: An optional regularizer configuration for the model.
            contrastive_config: An optional contrastive configration for the model.
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
        )
        self.start_ema, self.end_ema = ema_decay
        self.token_exit_cfg = token_exit_cfg
        self.base_loss = loss_config.build()
        self.masking_strategy = masking_config.build()
        self.regularizer = (
            regularizer_config.build() if regularizer_config is not None else None
        )
        self.contrastive_loss = (
            contrastive_config.build() if contrastive_config is not None else None
        )
        self.total_loss_name = self.base_loss.name
        if self.regularizer is not None:
            self.total_loss_name = f"{self.base_loss.name}+{self.regularizer.name}"

    def loss_fn(self, pred: Any, targets: Any) -> torch.Tensor:
        """Compute the loss between the predicted and target tensors."""
        return self.base_loss.compute(pred, targets)

    def train_batch(
        self, batch: tuple[int, HeliosSample], dry_run: bool = False
    ) -> None:
        """Train a batch.

        NOTE: Gradient accumulation/microbatching is not invariant for all losses across the same global batch size.

        - All Disc loss with same global batch size but different micro-batch sizes result in different gradients,
        though this matches the implementation in gallileo.
        - If the min hw is too low when subsampling, we may get micro-batches with uneven
        numbers of tokens making the loss for token averaged losses
        like l1 and l2 weight microbatches with less tokens relatively more.

        NOTE: For non contrastive losses, the loss is invariant to the global batch size across GPUS as well
        """
        self.update_target_encoder()
        # Set the model to train mode
        self.model.train()
        total_batch_loss = torch.zeros([], device=self.device)
        total_batch_reg = torch.zeros([], device=self.device)
        total_batch_con = torch.tensor(0.0, device=self.device)
        patch_size, batch_data = batch
        # Split into micro-batches.
        microbatches = split_batch(batch_data, self.rank_microbatch_size)
        num_microbatches = len(microbatches)
        for microbatch_idx, microbatch in enumerate(microbatches, start=1):
            with self._train_microbatch_context(microbatch_idx, num_microbatches):
                logger.info(
                    f"Training microbatch {microbatch_idx} of {num_microbatches} with batch size {microbatch.batch_size}"
                )
                masked_batch_a = self.masking_strategy.apply_mask(
                    self.transform.apply(microbatch).to_device(self.device),
                    patch_size=patch_size,
                )
                masked_batch_b = self.masking_strategy.apply_mask(
                    self.transform.apply(microbatch).to_device(self.device),
                    patch_size=patch_size,
                )
                # Run Encoder and decoder on the augmented input
                loss_a, latent_a, decoded_a, target_output_a, pooled_a = (
                    self.model_forward(masked_batch_a, patch_size, self.token_exit_cfg)
                )
                loss_b, latent_b, decoded_b, target_output_b, pooled_b = (
                    self.model_forward(masked_batch_b, patch_size, self.token_exit_cfg)
                )
                loss = (loss_a + loss_b) / 2

                # Scale loss by number of microbatches
                reg_term_a = self.compute_regularization(pooled_a)
                reg_term_b = self.compute_regularization(pooled_b)
                if reg_term_a is not None:
                    assert reg_term_b is not None
                    loss = loss + (reg_term_a + reg_term_b) / 2
                    total_batch_reg += (
                        get_local_tensor(
                            (reg_term_a.detach() + reg_term_b.detach()) / 2
                        )
                        / num_microbatches
                    )

                if self.contrastive_loss is not None:
                    contrastive_loss = self.contrastive_loss.compute(pooled_a, pooled_b)
                    loss += contrastive_loss
                    total_batch_con += (
                        get_local_tensor(contrastive_loss.detach()) / num_microbatches
                    )

                loss = loss / num_microbatches
                loss_val = get_local_tensor(loss.detach())
                total_batch_loss += loss_val

                # Skip bad batches
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logger.warning(
                        f"NaN or Inf detected in loss at microbatch {microbatch_idx}, stopping training for this batch."
                    )
                    del latent_a, latent_b
                    break

                del latent_a, latent_b
                loss.backward()

        self.trainer.record_metric(
            f"train/{self.total_loss_name}",
            total_batch_loss,
            ReduceType.mean,
        )
        self.log_regularization(total_batch_reg)

        if dry_run:
            return

        del batch, batch_data  # In case this helps with memory utilization.
        del masked_batch_a, masked_batch_b

    def model_forward(
        self, batch: MaskedHeliosSample, patch_size: int, token_exit_cfg: dict[str, int]
    ) -> tuple[
        torch.Tensor, TokensAndMasks, TokensAndMasks, TokensAndMasks, torch.Tensor
    ]:
        """Run a forward pass."""
        with self._model_forward_context():
            latent, decoded, latent_projected_and_pooled = self.model.forward(
                batch, patch_size
            )
            with torch.no_grad():
                logger.info("Target Encoder forward pass...")
                target_output, _ = self.model.target_encoder.forward(
                    batch.unmask(),
                    patch_size=patch_size,
                    token_exit_cfg=token_exit_cfg,
                )
            loss = self.loss_fn(decoded, target_output)
            return loss, latent, decoded, target_output, latent_projected_and_pooled
