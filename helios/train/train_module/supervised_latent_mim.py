"""Training and optimizer abstraction for Helios."""

from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from einops import rearrange
from olmo_core.distributed.parallel import DataParallelConfig
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.common import ReduceType
from torch.nn import functional as F

from helios.data.constants import MISSING_VALUE, Modality
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
class SupervisedLatentMIMTrainModuleConfig(HeliosTrainModuleConfig):
    """A configuration class for building :class:`LatentMIMTrainModule` instances.

    Args:
        loss_config: The loss configuration for the model.
        masking_config: The masking configuration for the model.
        ema_decay: EMA decay rate for target encoder (default: 0.99).
    """

    loss_config: LossConfig = field(
        default_factory=lambda: LossConfig(loss_config={"type": "patch_discrimination"})
    )
    mae_loss_config: LossConfig | None = None
    masking_config: MaskingConfig = field(
        default_factory=lambda: MaskingConfig(strategy_config={"type": "random"})
    )
    token_exit_cfg: dict[str, int] = field(
        default_factory=lambda: {modality: 0 for modality in Modality.names()}
    )
    ema_decay: tuple[float, float] = (0.996, 1.0)
    max_grad_norm: float = 1.0

    supervisory_modalities: list[str] | None = None
    compute_accuracies: bool = False
    supervisory_weight: float = 0.1

    def build(
        self,
        model: LatentMIM,
        device: torch.device | None = None,
    ) -> "SupervisedLatentMIMTrainModuleConfig":
        """Build the corresponding :class:`LatentMIMTrainModule`.

        Args:
            model: The model to train.
            device: The device to train on.
        """
        kwargs = self.prepare_kwargs()
        return SupervisedLatentMIMTrainModule(
            model=model,
            device=device,
            **kwargs,
        )


class SupervisedLatentMIMTrainModule(HeliosTrainModule):
    """A :class:`TrainModule`.

    Initialize the training module.

    Args:
        model: The transformer model to train.
        optim: The corresponding optimizer config.
        transform_config: The transform configuration for the model.
        masking_config: The masking configuration for the model.
        loss_config: The loss configuration for the model.
        mae_loss_config: Optional loss config for masked auto-encoding.
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
        supervisory_modalities: bandsets which should only be used for supervision
        supervisory_weight: weight to apply to the supervisory losses
        compute_accuracies: Whether to compute accuracies too
        find_unused_parameters: Whether to find unused parameters in the model, only used for DDP.
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
        regularizer_config: LossConfig | None = None,
        supervisory_modalities: list[str] = [Modality.WORLDCOVER.name],
        supervisory_weight: float = 0.1,
        compute_accuracies: bool = False,
        find_unused_parameters: bool = True,
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
            mae_loss_config: Optional loss config for masked auto-encoding.
            regularizer_config: An optional regularizer configuration for the model.
            supervisory_modalities: Which modalities to use as supervision
            supervisory_weight: weight to apply to the supervisory losses
            compute_accuracies: Whether to compute accuracies too
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
            find_unused_parameters=find_unused_parameters,
        )
        self.start_ema, self.end_ema = ema_decay
        self.token_exit_cfg = token_exit_cfg
        self.base_loss = loss_config.build()
        self.masking_strategy = masking_config.build()
        self.regularizer = (
            regularizer_config.build() if regularizer_config is not None else None
        )

        self.total_loss_name = self.base_loss.name
        if self.regularizer is not None:
            self.total_loss_name = f"{self.base_loss.name}+{self.regularizer.name}"

        self.mae_loss = mae_loss_config.build() if mae_loss_config is not None else None
        if self.mae_loss is not None:
            self.total_loss_name = f"{self.total_loss_name}+{self.mae_loss.name}"

        self.supervisory_modalities = supervisory_modalities
        self.compute_accuracies = compute_accuracies
        self.supervisory_weight = supervisory_weight

    def loss_fn(self, pred: Any, targets: Any) -> torch.Tensor:
        """Compute the loss between the predicted and target tensors."""
        return self.base_loss.compute(pred, targets)

    @staticmethod
    def accuracy_score(pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute accuracy score with missing values."""
        argmax_pred = pred.argmax(dim=-1)
        matches = argmax_pred == targets
        return sum(matches) / len(matches)

    @classmethod
    def supervisory_losses(
        cls,
        supervisory_modalities: dict[str, torch.Tensor],
        probe_outputs: dict[str, torch.Tensor],
        loss: torch.Tensor,
        total_batch_sup: torch.Tensor,
        total_batch_acc: dict[str, torch.Tensor] | None,
        supervisory_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the supervisory losses."""
        loss_fn = torch.nn.CrossEntropyLoss()
        # TODO: this is required; is it okay to be a normal dict key?
        spatial_mask = probe_outputs["mask"]
        for modality, modality_tensor in supervisory_modalities.items():
            modality_spec = Modality.get(modality)
            for idx, bands in enumerate(modality_spec.bandsets_as_indices()):
                modality_bandset = modality_tensor[:, :, :, 0, bands]
                if len(bands) > 1:
                    # then we need to turn it into indices
                    modality_bandset = torch.argmax(
                        modality_bandset, dim=-1, keepdim=True
                    )
                probe_output = probe_outputs[
                    f"{modality}_{idx}"
                ]  # B, H, W, T, Bandsets or 11 if its worldcover
                if probe_output.shape[-3] != modality_bandset.shape[-2]:
                    # this is in case patch_size < max_patch_size
                    probe_output = rearrange(
                        F.interpolate(
                            rearrange(probe_output, "b h w c -> b c h w"),
                            size=(
                                modality_bandset.shape[-3],
                                modality_bandset.shape[-2],
                            ),
                            mode="bilinear",
                            align_corners=True,
                        ),
                        "b c h w -> b h w c",
                    )

                modality_bandset = modality_bandset.long()
                flat_modality_bandset = modality_bandset.flatten().to(
                    probe_output.device
                )
                target_mask = torch.logical_and(
                    spatial_mask.flatten(), (flat_modality_bandset != MISSING_VALUE)
                )
                filtered_modality_bandset = flat_modality_bandset[target_mask]
                filtered_targets = probe_output.flatten(end_dim=-2)[target_mask, :]
                if len(filtered_modality_bandset) == 0:
                    logger.info(f"All values missing for {modality}")
                    continue
                modality_loss = loss_fn(
                    filtered_targets,
                    filtered_modality_bandset,
                )
                if torch.isnan(modality_loss).any():
                    logger.warning(f"NaN in unsupervised loss for {modality}")
                    continue
                loss += supervisory_weight * modality_loss
                total_batch_sup += (
                    get_local_tensor(modality_loss.detach()) * supervisory_weight
                )
                if total_batch_acc is not None:
                    batch_acc = get_local_tensor(
                        cls.accuracy_score(
                            filtered_targets,
                            filtered_modality_bandset,
                        ).detach()
                    )
                    total_batch_acc[f"{modality}_{idx}"] += batch_acc

        return loss, total_batch_sup, total_batch_acc

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
        total_batch_sup = torch.zeros([], device=self.device)
        if self.compute_accuracies:
            total_batch_acc = {}
            for modality in self.supervisory_modalities:
                for idx in range(len(Modality.get(modality).band_sets)):
                    total_batch_acc[f"{modality}_{idx}"] = torch.zeros(
                        [], device=self.device
                    )
        else:
            total_batch_acc = None
        patch_size, batch_data = batch
        # Split into micro-batches.
        microbatches = split_batch(batch_data, self.rank_microbatch_size)
        num_microbatches = len(microbatches)
        for microbatch_idx, microbatch in enumerate(microbatches):
            with self._train_microbatch_context(microbatch_idx, num_microbatches):
                logger.info(
                    f"Training microbatch {microbatch_idx} of {num_microbatches} with batch size {microbatch.batch_size}"
                )
                microbatch, supervisory_modalities = microbatch.pop(
                    self.supervisory_modalities
                )
                # TODO - mark supervisory bandsets as missing, and store the original mask
                microbatch = self.transform.apply(microbatch).to_device(self.device)
                masked_batch = self.masking_strategy.apply_mask(
                    microbatch, patch_size=patch_size
                )
                # Run Encoder and decoder on the augmented input
                loss, latent, decoded, target_output, probe_outputs = (
                    self.model_forward(masked_batch, patch_size, self.token_exit_cfg)
                )
                reg_term = self.compute_regularization(latent)
                if reg_term is not None:
                    loss = loss + reg_term
                    total_batch_reg += (
                        get_local_tensor(reg_term.detach()) / num_microbatches
                    )

                loss, total_batch_sup, total_batch_acc = self.supervisory_losses(
                    supervisory_modalities,
                    probe_outputs,
                    loss,
                    total_batch_sup,
                    total_batch_acc,
                    self.supervisory_weight,
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
                    del latent, decoded, target_output
                    break

                del latent, decoded, target_output
                loss.backward()

        self.trainer.record_metric(
            f"train/{self.total_loss_name}",
            total_batch_loss,
            ReduceType.mean,
        )
        self.trainer.record_metric(
            f"train/supervisory_loss_{'_'.join(self.supervisory_modalities)}",
            total_batch_sup / num_microbatches,
            ReduceType.mean,
        )
        if total_batch_acc is not None:
            for modality_idx, modality_accuracy in total_batch_acc.items():
                self.trainer.record_metric(
                    f"train/supervisory_accuracy_{modality_idx}",
                    # scale by number of microbatches
                    modality_accuracy / num_microbatches,
                    ReduceType.mean,
                )
        self.log_regularization(total_batch_reg)

        if dry_run:
            return

        del batch, batch_data  # In case this helps with memory utilization.
        del masked_batch

    def model_forward(
        self, batch: MaskedHeliosSample, patch_size: int, token_exit_cfg: dict[str, int]
    ) -> tuple[
        torch.Tensor,
        TokensAndMasks,
        TokensAndMasks,
        TokensAndMasks,
        dict[str, torch.Tensor],
    ]:
        """Run a forward pass."""
        with self._model_forward_context():
            output = self.model(batch, patch_size)
            with torch.no_grad():
                logger.info("Target Encoder forward pass...")
                target_output, _, _ = self.model.target_encoder.forward(
                    batch.unmask(),
                    patch_size=patch_size,
                    token_exit_cfg=token_exit_cfg,
                )
            loss = self.loss_fn(output.decoded, target_output)
            if self.mae_loss is not None:
                loss += self.mae_loss.compute(output.reconstructed, batch)
            return (
                loss,
                output.latent,
                output.decoded,
                target_output,
                output.probe_outputs,
            )
