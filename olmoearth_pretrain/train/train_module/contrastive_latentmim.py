"""Training and optimizer abstraction for OlmoEarth Pretrain."""

from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from olmo_core.distributed.parallel import DataParallelConfig
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.common import ReduceType

from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, TokensAndMasks
from olmoearth_pretrain.modalities import Modality
from olmoearth_pretrain.nn.latent_mim import LatentMIM
from olmoearth_pretrain.nn.utils import unpack_encoder_output
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.train_module import (
    MicrobatchTrainOutput,
    OlmoEarthTrainModule,
    OlmoEarthTrainModuleConfig,
)

logger = getLogger(__name__)


@dataclass
class ContrastiveLatentMIMTrainModuleConfig(OlmoEarthTrainModuleConfig):
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
    contrastive_config: LossConfig | None = None
    reinit_targets: bool = False

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


class ContrastiveLatentMIMTrainModule(OlmoEarthTrainModule):
    """A :class:`TrainModule`.

    Initialize the training module.
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
        contrastive_config: LossConfig | None = None,
        find_unused_parameters: bool = True,
        reinit_targets: bool = False,
    ):
        """Initialize the training module.

        Args:
            model: The transformer model to train.
            optim_config: The corresponding optimizer config.
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
            ema_decay: EMA decay rate for target encoder, as a tuple of (start_ema_decay, end_ema_decay)
            token_exit_cfg: The token exit configuration for the model.
            regularizer_config: An optional regularizer configuration for the model.
            contrastive_config: An optional contrastive configration for the model.
            find_unused_parameters: Whether to find unused parameters in the model, only used for DDP.
            reinit_targets: Whether or not to reinitialize the target encoder.
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
        self.contrastive_loss = (
            contrastive_config.build() if contrastive_config is not None else None
        )
        self.total_loss_name = self.base_loss.name
        if self.regularizer is not None:
            self.total_loss_name = f"{self.base_loss.name}+{self.regularizer.name}"

        self.mae_loss = mae_loss_config.build() if mae_loss_config is not None else None
        self._propagate_model_tokenization_config(self.masking_strategy, self.mae_loss)
        if self.mae_loss is not None:
            self.total_loss_name = f"{self.total_loss_name}+{self.mae_loss.name}"
        if reinit_targets:
            if ema_decay != (0.0, 0.0):
                logger.warning(
                    "Applying EMA updates to a randomly initialized target encoder."
                )
            self.model.target_encoder.apply(self.model.target_encoder._init_weights)

    def loss_fn(self, pred: Any, targets: Any) -> torch.Tensor:
        """Compute the loss between the predicted and target tensors."""
        return self.base_loss.compute(pred, targets)

    def train_batch(
        self,
        batch: tuple[int, MaskedOlmoEarthSample, MaskedOlmoEarthSample],
        dry_run: bool = False,
    ) -> None:
        """Train a batch.

        NOTE: Gradient accumulation/microbatching is not invariant for all losses across the same global batch size.

        - All Disc loss with same global batch size but different micro-batch sizes result in different gradients,
        though this matches the implementation in gallileo.
        - If the min hw is too low when subsampling, we may get micro-batches with uneven
        numbers of tokens making the loss for token averaged losses
        like l1 and l2 weight microbatches with less tokens relatively more.

        NOTE: For non contrastive losses, the loss is invariant to the global batch size across GPUS as well

        Args:
            batch: A (patch_size, MaskedOlmoEarthSample_a, MaskedOlmoEarthSample_b) tuple from the dataloader.
            dry_run: If True, skip metric recording and just run forward/backward.
        """
        if not dry_run:
            self.update_target_encoder()
        self.model.train()

        patch_size = batch[0]
        batch_data_a = batch[1]
        batch_data_b = batch[2]

        def step(
            microbatches: tuple[MaskedOlmoEarthSample, ...], microbatch_idx: int
        ) -> MicrobatchTrainOutput:
            masked_batch_a, masked_batch_b = microbatches
            loss_a, _latent_a, _decoded_a, _target_output_a, pooled_a = (
                self.model_forward(masked_batch_a, patch_size, self.token_exit_cfg)
            )
            loss_b, _latent_b, _decoded_b, _target_output_b, pooled_b = (
                self.model_forward(masked_batch_b, patch_size, self.token_exit_cfg)
            )
            loss = (loss_a + loss_b) / 2

            metrics = {}
            if self.contrastive_loss is not None:
                contrastive_loss = self.contrastive_loss.compute(pooled_a, pooled_b)
                loss += contrastive_loss
                metrics["contrastive"] = contrastive_loss

            return MicrobatchTrainOutput(
                loss=loss,
                regularizer_inputs=(pooled_a, pooled_b),
                metrics=metrics,
            )

        totals = self._run_masked_microbatches(
            (batch_data_a, batch_data_b),
            step,
            nonfinite_behavior="warn_break",
        )

        if dry_run:
            return

        self.trainer.record_metric(
            f"train/{self.total_loss_name}",
            totals.loss,
            ReduceType.mean,
        )
        if self.contrastive_loss is not None:
            self.trainer.record_metric(
                f"train/{self.contrastive_loss.name}",
                totals.metrics["contrastive"],
                ReduceType.mean,
            )
        self.log_regularization(totals.regularizer)

        del batch  # In case this helps with memory utilization.

    def model_forward(
        self,
        batch: MaskedOlmoEarthSample,
        patch_size: int,
        token_exit_cfg: dict[str, int],
    ) -> tuple[
        torch.Tensor, TokensAndMasks, TokensAndMasks, TokensAndMasks, torch.Tensor
    ]:
        """Run a forward pass."""
        with self._model_forward_context():
            (
                latent,
                decoded,
                latent_projected_and_pooled,
                reconstructed,
                extra_metrics,
            ) = self.model(batch, patch_size)
            if extra_metrics is not None:
                self.log_extra_metrics(extra_metrics)
            with torch.no_grad():
                logger.debug("Target Encoder forward pass...")
                output_dict = self.model.target_encoder.forward(
                    batch.unmask(),
                    patch_size=patch_size,
                    token_exit_cfg=token_exit_cfg,
                )
                target_output, _, _ = unpack_encoder_output(output_dict)
            loss = self.loss_fn(decoded, target_output)
            if self.mae_loss is not None and reconstructed is not None:
                loss += self.mae_loss.compute(reconstructed, batch)
            return loss, latent, decoded, target_output, latent_projected_and_pooled
