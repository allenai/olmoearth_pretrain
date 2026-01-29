"""Training and optimizer abstraction for OlmoEarth Pretrain."""

from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, cast

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from olmo_core.distributed.parallel import DataParallelConfig
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.common import ReduceType

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.nn.mae import MAE
from olmoearth_pretrain.nn.utils import unpack_encoder_output
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.train_module import (
    OlmoEarthTrainModule,
    OlmoEarthTrainModuleConfig,
)
from olmoearth_pretrain.train.utils import split_batch, split_masked_batch

logger = getLogger(__name__)


@dataclass
class MAETrainModuleConfig(OlmoEarthTrainModuleConfig):
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


class MAETrainModule(OlmoEarthTrainModule):
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
        loss_fn: Loss function to use.
        compile_loss: Whether to compile the loss function.
        autocast_precision: Enable AMP with this data type.
        max_grad_norm: Clip gradient norms to this value.
        scheduler: Optional learning rate scheduler.
        device: The device to train on.
        state_dict_save_opts: Override state dict options for saving.
        state_dict_load_opts: Override state dict options for loading.
        token_exit_cfg: The token exit configuration for the model.
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
        compile_loss: bool = False,
        autocast_precision: torch.dtype | None = None,
        max_grad_norm: float | None = None,
        scheduler: Scheduler | None = None,
        device: torch.device | None = None,
        state_dict_save_opts: dist_cp_sd.StateDictOptions | None = None,
        state_dict_load_opts: dist_cp_sd.StateDictOptions | None = None,
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

            loss_fn: Loss function to use.
            compile_loss: Whether to compile the loss function.
            autocast_precision: Enable AMP with this data type.
            max_grad_norm: Clip gradient norms to this value.
            scheduler: Optional learning rate scheduler.
            device: The device to train on.
            state_dict_save_opts: Override state dict options for saving.
            state_dict_load_opts: Override state dict options for loading.
            token_exit_cfg: The token exit configuration for the model.
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
            compile_loss=compile_loss,
            autocast_precision=autocast_precision,
            max_grad_norm=max_grad_norm,
            scheduler=scheduler,
            device=device,
            state_dict_save_opts=state_dict_save_opts,
            state_dict_load_opts=state_dict_load_opts,
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
        self, x: MaskedOlmoEarthSample, patch_size: int
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
                    output_dict = self.model.encoder.forward(
                        x.unmask(),
                        patch_size=patch_size,
                        token_exit_cfg=self.token_exit_cfg,
                    )
                    target_output, _, _ = unpack_encoder_output(output_dict)
                loss += self.latent_mim_loss.compute(decoded, target_output)
            return loss, latent, decoded

    def train_batch(
        self,
        patch_batch: tuple[int, OlmoEarthSample] | tuple[int, MaskedOlmoEarthSample],
        dry_run: bool = False,
    ) -> None:
        """Train a batch.

        NOTE: Gradient accumulation/microbatching is not invariant for all losses across the same global batch size.

        - All Disc loss with same global batch size but different micro-batch sizes result in different gradients,
        though this matches the implementation in gallileo.
        - If the min hw is too low when subsampling, we may get micro-batches with uneven
        numbers of tokens making the loss for token averaged losses
        like l1 and l2 weight microbatches with less tokens relatively more.

        NOTE: For contrastive losses, the loss is invariant to the global batch size across GPUS as well

        Args:
            patch_batch: Either a legacy (patch_size, OlmoEarthSample) tuple or a pre-masked
                (patch_size, MaskedOlmoEarthSample) tuple from the dataloader.
            dry_run: If True, skip metric recording and just run forward/backward.
        """
        patch_size = patch_batch[0]
        batch = patch_batch[1]
        self.model.train()
        # Set the maximum number of tokens
        total_batch_loss = torch.zeros([], device=self.device)
        total_batch_reg = torch.zeros([], device=self.device)

        # Detect batch type and split accordingly
        is_pre_masked = isinstance(batch, MaskedOlmoEarthSample)
        if is_pre_masked:
            masked_microbatches = split_masked_batch(
                cast(MaskedOlmoEarthSample, batch), self.rank_microbatch_size
            )
            num_microbatches = len(masked_microbatches)
        else:
            unmasked_microbatches = split_batch(
                cast(OlmoEarthSample, batch), self.rank_microbatch_size
            )
            num_microbatches = len(unmasked_microbatches)

        for microbatch_idx in range(num_microbatches):
            with self._train_microbatch_context(microbatch_idx, num_microbatches):
                if is_pre_masked:
                    # Pre-masked from dataloader - just move to device
                    microbatch_masked = masked_microbatches[microbatch_idx]
                    logger.info(
                        f"Training microbatch {microbatch_idx} of {num_microbatches} "
                        f"with batch size {microbatch_masked.timestamps.shape[0]} (pre-masked)"
                    )
                    masked_batch = microbatch_masked.to_device(self.device)
                else:
                    # Legacy mode - apply transform and masking on GPU
                    microbatch_unmasked = unmasked_microbatches[microbatch_idx]
                    logger.info(
                        f"Training microbatch {microbatch_idx} of {num_microbatches} with batch size {microbatch_unmasked.batch_size}"
                    )
                    transformed = self.transform.apply(microbatch_unmasked).to_device(
                        self.device
                    )
                    masked_batch = self.masking_strategy.apply_mask(
                        transformed, patch_size=patch_size
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

        if dry_run:
            return

        self.trainer.record_metric(
            f"train/{self.total_loss_name}",
            total_batch_loss,
            ReduceType.mean,
        )
        self.log_regularization(total_batch_reg)

        del batch  # In case this helps with memory utilization.
        del masked_batch
