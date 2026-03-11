"""Training and optimizer abstraction for OlmoEarth Pretrain."""

from collections import defaultdict
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from olmo_core.distributed.parallel import DataParallelConfig
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.common import ReduceType

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks
from olmoearth_pretrain.nn.latent_mim import LatentMIM
from olmoearth_pretrain.nn.utils import unpack_encoder_output
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import (
    MaskingConfig,
    propagate_tokenization_config,
)
from olmoearth_pretrain.train.train_module.train_module import (
    OlmoEarthTrainModule,
    OlmoEarthTrainModuleConfig,
)
from olmoearth_pretrain.train.utils import split_masked_batch

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
    target_patch_size: int | None = None
    target_patch_size_by_modality: dict[str, int] | None = None

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
        target_patch_size: int | None = None,
        target_patch_size_by_modality: dict[str, int] | None = None,
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
            target_patch_size: Optional fallback target patch size for all modalities.
            target_patch_size_by_modality: Optional per-modality target patch sizes.
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
        tokenization_config = getattr(self.model.encoder, "tokenization_config", None)
        if tokenization_config is not None:
            propagate_tokenization_config(self.masking_strategy, tokenization_config)
        self.regularizer = (
            regularizer_config.build() if regularizer_config is not None else None
        )
        self.contrastive_loss = (
            contrastive_config.build() if contrastive_config is not None else None
        )
        self.target_patch_size = target_patch_size
        self.target_patch_size_by_modality = target_patch_size_by_modality or {}
        unknown_target_modalities = set(self.target_patch_size_by_modality) - set(
            Modality.names()
        )
        if unknown_target_modalities:
            raise ValueError(
                "target_patch_size_by_modality contains unknown modalities: "
                f"{unknown_target_modalities}"
            )
        invalid_target_patch_sizes = {
            modality: target_patch_size
            for modality, target_patch_size in self.target_patch_size_by_modality.items()
            if target_patch_size < 1
        }
        if invalid_target_patch_sizes:
            raise ValueError(
                "target_patch_size_by_modality must only contain positive patch "
                f"sizes, got: {invalid_target_patch_sizes}"
            )
        self.total_loss_name = self.base_loss.name
        if self.regularizer is not None:
            self.total_loss_name = f"{self.base_loss.name}+{self.regularizer.name}"

        self.mae_loss = mae_loss_config.build() if mae_loss_config is not None else None
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

    def _get_target_patch_size_for_modality(
        self, modality: str, patch_size: int
    ) -> int:
        """Get the target patch size for a modality."""
        default_target_patch_size = self.target_patch_size or patch_size
        return self.target_patch_size_by_modality.get(
            modality, default_target_patch_size
        )

    @staticmethod
    def _subset_batch_to_modalities(
        batch: MaskedOlmoEarthSample, modalities: list[str]
    ) -> MaskedOlmoEarthSample:
        """Keep only the requested modalities in a masked batch."""
        batch_dict = batch.as_dict(include_nones=True)
        modalities_to_keep = set(modalities)
        for modality in batch.modalities:
            if modality in modalities_to_keep:
                continue
            batch_dict[modality] = None
            batch_dict[batch.get_masked_modality_name(modality)] = None
        return MaskedOlmoEarthSample.from_dict(batch_dict)

    @staticmethod
    def _merge_target_outputs(target_outputs: list[TokensAndMasks]) -> TokensAndMasks:
        """Merge target outputs from multiple target-encoder forwards."""
        merged_dict: dict[str, Any] = {}
        for target_output in target_outputs:
            target_output_dict = target_output.as_dict(include_nones=False)
            overlapping_keys = set(merged_dict).intersection(target_output_dict)
            if overlapping_keys:
                raise ValueError(
                    "Attempted to merge duplicate target outputs for keys: "
                    f"{overlapping_keys}"
                )
            merged_dict.update(target_output_dict)
        return TokensAndMasks(**merged_dict)

    def _build_target_output(
        self,
        batch: MaskedOlmoEarthSample,
        patch_size: int,
        token_exit_cfg: dict[str, int],
    ) -> TokensAndMasks:
        """Build target outputs, grouping modalities by target patch size."""
        unmasked_batch = batch.unmask()
        target_modalities_by_patch_size: dict[int, list[str]] = defaultdict(list)
        for modality in unmasked_batch.modalities:
            target_patch_size = self._get_target_patch_size_for_modality(
                modality, patch_size
            )
            target_modalities_by_patch_size[target_patch_size].append(modality)

        target_outputs = []
        for target_patch_size, grouped_modalities in sorted(
            target_modalities_by_patch_size.items()
        ):
            target_batch = self._subset_batch_to_modalities(
                unmasked_batch, grouped_modalities
            )
            output_dict = self.model.target_encoder.forward(
                target_batch,
                patch_size=target_patch_size,
                token_exit_cfg=token_exit_cfg,
            )
            target_output, _, _ = unpack_encoder_output(output_dict)
            target_outputs.append(target_output)
        return self._merge_target_outputs(target_outputs)

    @staticmethod
    def _validate_decoded_target_shapes(
        decoded: TokensAndMasks, target_output: TokensAndMasks
    ) -> None:
        """Fail fast when decoded and target outputs do not align."""
        decoded_modalities = set(decoded.modalities)
        target_modalities = set(target_output.modalities)
        if decoded_modalities != target_modalities:
            raise ValueError(
                "Decoded and target modalities do not match: "
                f"decoded={sorted(decoded_modalities)}, "
                f"target={sorted(target_modalities)}"
            )

        for modality in decoded.modalities:
            decoded_tokens = getattr(decoded, modality)
            target_tokens = getattr(target_output, modality)
            decoded_mask = getattr(decoded, decoded.get_masked_modality_name(modality))
            target_mask = getattr(
                target_output, target_output.get_masked_modality_name(modality)
            )
            if decoded_tokens is None or target_tokens is None:
                raise ValueError(f"Missing token tensor for modality {modality}")
            if decoded_mask is None or target_mask is None:
                raise ValueError(f"Missing mask tensor for modality {modality}")
            if decoded_tokens.shape != target_tokens.shape:
                raise ValueError(
                    f"Decoded and target token shapes differ for modality {modality}: "
                    f"{decoded_tokens.shape} != {target_tokens.shape}"
                )
            if decoded_mask.shape != target_mask.shape:
                raise ValueError(
                    f"Decoded and target mask shapes differ for modality {modality}: "
                    f"{decoded_mask.shape} != {target_mask.shape}"
                )

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
        # Set the model to train mode
        self.model.train()
        total_batch_loss = torch.zeros([], device=self.device)
        total_batch_reg = torch.zeros([], device=self.device)
        total_batch_con = torch.zeros([], device=self.device)

        # Unpack batch
        patch_size = batch[0]
        batch_data_a = batch[1]
        batch_data_b = batch[2]
        microbatches_a = split_masked_batch(batch_data_a, self.rank_microbatch_size)
        microbatches_b = split_masked_batch(batch_data_b, self.rank_microbatch_size)
        num_microbatches = len(microbatches_a)

        for microbatch_idx in range(num_microbatches):
            with self._train_microbatch_context(microbatch_idx, num_microbatches):
                microbatch_a = microbatches_a[microbatch_idx]
                microbatch_b = microbatches_b[microbatch_idx]
                logger.info(
                    f"Training microbatch {microbatch_idx} of {num_microbatches} "
                    f"with batch size {microbatch_a.batch_size}"
                )
                masked_batch_a = microbatch_a.to_device(self.device)
                masked_batch_b = microbatch_b.to_device(self.device)

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

        if dry_run:
            return

        self.trainer.record_metric(
            f"train/{self.total_loss_name}",
            total_batch_loss,
            ReduceType.mean,
        )
        if self.contrastive_loss is not None:
            self.trainer.record_metric(
                f"train/{self.contrastive_loss.name}",
                total_batch_con,
                ReduceType.mean,
            )
        self.log_regularization(total_batch_reg)

        del batch  # In case this helps with memory utilization.
        del masked_batch_a, masked_batch_b

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
                target_output = self._build_target_output(
                    batch,
                    patch_size=patch_size,
                    token_exit_cfg=token_exit_cfg,
                )
            self._validate_decoded_target_shapes(decoded, target_output)
            loss = self.loss_fn(decoded, target_output)
            if self.mae_loss is not None and reconstructed is not None:
                loss += self.mae_loss.compute(reconstructed, batch)
            return loss, latent, decoded, target_output, latent_projected_and_pooled
