"""Training and optimizer abstraction for Helios."""

from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

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

from helios.data.constants import Modality
from helios.data.dataset import HeliosSample
from helios.nn.flexihelios import TokensAndMasks
from helios.nn.mae import MAE
from helios.train.loss import LossConfig
from helios.train.masking import MaskingConfig
from helios.train.train_module.train_module import (
    HeliosTrainModule,
    HeliosTrainModuleConfig,
)

logger = getLogger(__name__)


@dataclass
class MAETrainModuleConfig(HeliosTrainModuleConfig):
    """A configuration class for building :class:`MAETrainModule` instances.

    Args:
        loss_config: The loss configuration for the model.
        masking_config: The masking configuration for the model.
    """

    loss_config: LossConfig = field(
        default_factory=lambda: LossConfig(loss_config={"type": "l1"})
    )
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
        kwargs = self.as_dict(exclude_none=True, recurse=False)
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
        token_exit_cfg: The token exit configuration for the model.
    """

    def __init__(
        self,
        model: MAE,
        optim_config: OptimConfig,
        masking_config: MaskingConfig,
        loss_config: LossConfig,
        rank_microbatch_size: int,
        token_exit_cfg: dict[str, int],
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
    ):
        """Initialize the training module.

        Args:
            model: The transformer model to train.
            optim_config: The corresponding optimizer config.
            masking_config: The masking configuration for the model.
            loss_config: The loss configuration for the model.
            rank_microbatch_size: The rank microbatch size in instances.
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
            token_exit_cfg: The token exit configuration for the model.
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
        )
        self.token_exit_cfg = token_exit_cfg
        self.base_loss = loss_config.build()
        self.masking_strategy = masking_config.build()

    def loss_fn(self, pred: Any, targets: Any) -> torch.Tensor:
        """Compute the loss between the predicted and target tensors."""
        return self.base_loss.compute(pred, targets)

    def eval_loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the loss between the predicted and target tensors."""
        raise NotImplementedError("eval loss fn not implemented")

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
        # Set the maximum number of tokens
        token_budget = self.model.token_budget
        h_w_to_sample = list(
            range(self.model.h_w_to_sample_min, self.model.h_w_to_sample_max)
        )
        total_batch_loss = torch.tensor(0.0, device=self.device)

        # Smallest h /w must be bigger than the smallest patch size

        patch_size = self.model.encoder.max_patch_size
        batch = self.model.transform.apply(batch)
        subsampled_batch = batch.subset(patch_size, token_budget, h_w_to_sample)
        subsampled_batch = subsampled_batch.to_device(self.device)
        # Each microbatch should have about the same number of encoded tokens if
        # we mask here
        masked_batch = self.masking_strategy.apply_mask(
            subsampled_batch, patch_size=patch_size
        )

        # Run Encoder and decoder on the augmented input
        reconstructed = self.model(masked_batch, patch_size, self.token_exit_cfg)
        target_output = TokensAndMasks(**batch.as_dict())
        loss = self.loss_fn(reconstructed, target_output)
        loss.backward()

        self.trainer.record_metric(
            f"train/{self.base_loss.name}",
            total_batch_loss / get_world_size(self.dp_process_group),
            ReduceType.mean,
        )

        if dry_run:
            return

        del batch  # In case this helps with memory utilization.
        del masked_batch

    def eval_batch(
        self, batch: dict[str, Any], labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Evaluate a batch."""
        raise NotImplementedError("eval batch not implemented")
