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
        transform_config: TransformConfig,
        masking_config_a: MaskingConfig,
        masking_config_b: MaskingConfig,
        loss_config_a: LossConfig,
        loss_config_b: LossConfig,
        rank_microbatch_size: int,
        token_exit_cfg_a: dict[str, int],
        token_exit_cfg_b: dict[str, int],
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
    ):
        """Initialize the training module.

        Args:
            model: The transformer model to train.
            optim_config: The corresponding optimizer config.
            transform_config: The transform configuration for the model.
            masking_config_a: The masking configuration for the model.
            masking_config_b: The masking configuration for the model.
            loss_config_a: The loss configuration for the model.
            loss_config_b: The loss configuration for the model.
            rank_microbatch_size: The rank microbatch size in instances.
            compile_model: Whether to compile to the model.
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

        NOTE: For contrastive losses, the loss is invariant to the global batch size across GPUS as well
        """
        self.update_target_encoder()
        # Set the model to train mode
        self.model.train()

        # Set the maximum number of tokens
        total_batch_loss = torch.tensor(0.0, device=self.device)
        # Split into micro-batches.
        patch_size, batch_data = batch
        microbatches = split_batch(batch_data, self.rank_microbatch_size)
        num_microbatches = len(microbatches)
        for microbatch_idx, microbatch in enumerate(microbatches, start=1):
            with self._train_microbatch_context(microbatch_idx, num_microbatches):
                logger.info(
                    f"Training microbatch {microbatch_idx} of {num_microbatches} with batch size {microbatch.batch_size}"
                )
                microbatch = self.transform.apply(microbatch).to_device(self.device)

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

                # Skip bad batches
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logger.warning(
                        f"NaN or Inf detected in loss at microbatch {microbatch_idx}, stopping training for this batch."
                    )
                    del decoded, target_output
                    break

                del decoded, target_output
                loss.backward()

        if dry_run:
            return

        self.trainer.record_metric(
            f"train/{self.base_loss_a.name}+{self.base_loss_b.name}",
            total_batch_loss,
            ReduceType.mean,
        )
        del masked_batch, batch, microbatch, batch_data

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
