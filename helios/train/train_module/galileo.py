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

from helios.data.constants import (
    Modality,
)
from helios.data.dataset import HeliosSample
from helios.data.transform import TransformConfig
from helios.nn.flexihelios import TokensAndMasks
from helios.nn.galileo import Galileo
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
    mae_loss_config: LossConfig | None = None
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
    contrastive_config: LossConfig | None = None

    def build(
        self,
        model: Galileo,
        device: torch.device | None = None,
    ) -> "GalileoTrainModule":
        """Build the corresponding :class:`LatentMIMTrainModule`.

        Args:
            model: The model to train.
            device: The device to train on.
        """
        kwargs = self.prepare_kwargs()
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
        mae_loss_config: Optional loss config for masked auto-encoding.
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
        model: Galileo,
        optim_config: OptimConfig,
        transform_config: TransformConfig,
        masking_config_a: MaskingConfig,
        masking_config_b: MaskingConfig,
        loss_config_a: LossConfig,
        loss_config_b: LossConfig,
        rank_microbatch_size: int,
        token_exit_cfg_a: dict[str, int],
        token_exit_cfg_b: dict[str, int],
        mae_loss_config: LossConfig | None = None,
        compile_model: bool = False,
        dp_config: DataParallelConfig | None = None,
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
        find_unused_parameters: bool = True,
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
            mae_loss_config: Optional loss config for masked auto-encoding.
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
            regularizer_config: An optional regularizer configuration for the model.
            contrastive_config: An optional contrastive configration for the model.
            find_unused_parameters: Whether to find unused parameters in the model, only used for DDP.
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
            warmup_duration=warmup_duration,
            find_unused_parameters=find_unused_parameters,
        )
        self.start_ema, self.end_ema = ema_decay
        self.token_exit_cfg_a = token_exit_cfg_a
        self.base_loss_a = loss_config_a.build()
        self.masking_strategy_a = masking_config_a.build()
        self.token_exit_cfg_b = token_exit_cfg_b
        self.base_loss_b = loss_config_b.build()
        self.masking_strategy_b = masking_config_b.build()
        self.regularizer = (
            regularizer_config.build() if regularizer_config is not None else None
        )
        self.contrastive_loss = (
            contrastive_config.build() if contrastive_config is not None else None
        )
        self.total_loss_name = f"{self.base_loss_a.name}+{self.base_loss_b.name}"
        if self.regularizer is not None:
            self.total_loss_name = f"{self.total_loss_name}+{self.regularizer.name}"
        self.mae_loss = mae_loss_config.build() if mae_loss_config is not None else None
        if self.mae_loss is not None:
            self.total_loss_name = f"{self.total_loss_name}+{self.mae_loss.name}"

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
        total_mask_a_loss = torch.tensor(0.0, device=self.device)
        total_mask_b_loss = torch.tensor(0.0, device=self.device)
        total_batch_loss = torch.tensor(0.0, device=self.device)
        total_batch_reg = torch.tensor(0.0, device=self.device)
        total_batch_con = torch.tensor(0.0, device=self.device)
        # Split into micro-batches.
        patch_size, batch_data = batch
        microbatches = split_batch(batch_data, self.rank_microbatch_size)
        num_microbatches = len(microbatches)
        for microbatch_idx, microbatch in enumerate(microbatches):
            with self._train_microbatch_context(microbatch_idx, num_microbatches):
                logger.info(
                    f"Training microbatch {microbatch_idx} of {num_microbatches} with batch size {microbatch.batch_size}"
                )

                microbatch = self.transform.apply(microbatch).to_device(self.device)

                masked_batch_a = self.masking_strategy_a.apply_mask(
                    microbatch, patch_size
                )

                masked_batch_b = self.masking_strategy_b.apply_mask(
                    microbatch, patch_size
                )
                loss_a, latent_a, pooled_a, loss_b, latent_b, pooled_b = (
                    self.model_forward(
                        masked_batch_a,
                        masked_batch_b,
                        patch_size,
                        self.token_exit_cfg_a,
                        self.token_exit_cfg_b,
                    )
                )
                loss = (loss_a + loss_b) / 2
                total_mask_a_loss += (
                    get_local_tensor(loss_a.detach()) / num_microbatches
                )
                total_mask_b_loss += (
                    get_local_tensor(loss_b.detach()) / num_microbatches
                )

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
                    logger.info(f"contrastive loss: {contrastive_loss}")
                    loss += contrastive_loss
                    total_batch_con += (
                        get_local_tensor(contrastive_loss.detach()) / num_microbatches
                    )

                loss = loss / num_microbatches
                loss_val = get_local_tensor(loss.detach())
                total_batch_loss += loss_val
                loss.backward()

        if dry_run:
            return

        total_batch_loss = torch.nan_to_num(total_batch_loss, nan=float("inf"))
        total_batch_reg = torch.nan_to_num(total_batch_reg, nan=float("inf"))
        total_batch_con = torch.nan_to_num(total_batch_con, nan=float("inf"))
        total_mask_a_loss = torch.nan_to_num(total_mask_a_loss, nan=float("inf"))
        total_mask_b_loss = torch.nan_to_num(total_mask_b_loss, nan=float("inf"))

        self.trainer.record_metric(
            f"train/{self.total_loss_name}",
            total_batch_loss,
            ReduceType.mean,
        )
        self.trainer.record_metric(
            f"{self.masking_strategy_a.name}_masking_{self.base_loss_a.name}",
            total_mask_a_loss,
            ReduceType.mean,
            namespace="train",
        )
        self.trainer.record_metric(
            f"{self.masking_strategy_b.name}_masking_{self.base_loss_b.name}",
            total_mask_b_loss,
            ReduceType.mean,
            namespace="train",
        )
        self.trainer.record_metric("train/epoch", self.trainer.epoch)
        self.log_regularization(total_batch_reg)

        if self.contrastive_loss is not None:
            self.trainer.record_metric(
                f"train/{self.contrastive_loss.name}",
                total_batch_con,
                ReduceType.mean,
            )
        del batch, microbatch, batch_data

    def model_forward(
        self,
        batch_a: MaskedHeliosSample,
        batch_b: MaskedHeliosSample,
        patch_size: int,
        token_exit_cfg_a: dict[str, int],
        token_exit_cfg_b: dict[str, int],
    ) -> tuple[
        torch.Tensor,
        TokensAndMasks,
        TokensAndMasks,
        torch.Tensor,
        TokensAndMasks,
        TokensAndMasks,
    ]:
        """Run a forward pass."""
        with self._model_forward_context():
            output_dict = self.model(batch_a, batch_b, patch_size)
            latent_a, decoded_a, latent_projected_and_pooled_a, reconstructed_a = (
                output_dict["a"]
            )
            latent_b, decoded_b, latent_projected_and_pooled_b, reconstructed_b = (
                output_dict["b"]
            )

            with torch.no_grad():
                logger.info("target encoder running here")
                target_output_a, _ = self.model.target_encoder.forward(
                    batch_a.unmask(),
                    patch_size=patch_size,
                    token_exit_cfg=token_exit_cfg_a,
                )
                target_output_b, _ = self.model.target_encoder.forward(
                    batch_b.unmask(),
                    patch_size=patch_size,
                    token_exit_cfg=token_exit_cfg_b,
                )

            loss_a = self.loss_fn_a(decoded_a, target_output_a)
            loss_b = self.loss_fn_b(decoded_b, target_output_b)
            if self.mae_loss is not None:
                loss_a += self.mae_loss.compute(reconstructed_a, batch_a)
                loss_b += self.mae_loss.compute(reconstructed_b, batch_b)
            return (
                loss_a,
                latent_a,
                latent_projected_and_pooled_a,
                loss_b,
                latent_b,
                latent_projected_and_pooled_b,
            )
