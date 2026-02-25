"""Contrastive Latent MIM training module with adversarial discriminator loss."""

from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

import torch
import torch.nn.functional as F
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.parallel.data_parallel import get_dp_mesh
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.optim import AdamWConfig, OptimConfig
from olmo_core.train.common import ReduceType
from torch.distributed.fsdp import MixedPrecisionPolicy

from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.nn.discriminator import TokenDiscriminatorConfig
from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks
from olmoearth_pretrain.nn.latent_mim import LatentMIM
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskValue, MaskingConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModule,
    ContrastiveLatentMIMTrainModuleConfig,
)
from olmoearth_pretrain.train.utils import split_masked_batch

logger = getLogger(__name__)


def _extract_modality_decoded_and_target(
    decoded: TokensAndMasks,
    target: TokensAndMasks,
    modality: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Extract decoded (fake) and target (real) tokens for a specific modality.

    Uses the decoded mask to index both, matching how PatchDiscriminationLoss works:
    the decoder mask indicates which spatial positions were predicted, and the same
    positions in the target output are the ground truth.

    Returns:
        (fake_tokens, real_tokens) each of shape [N, D], or (None, None) if the
        modality is absent or has no decoded tokens.
    """
    decoded_data = getattr(decoded, modality, None)
    target_data = getattr(target, modality, None)
    if decoded_data is None or target_data is None:
        return None, None

    mask_name = TokensAndMasks.get_masked_modality_name(modality)
    decoded_mask = getattr(decoded, mask_name, None)
    if decoded_mask is None:
        return None, None

    flat_decoded = decoded_data.reshape(-1, decoded_data.shape[-1])
    flat_target = target_data.reshape(-1, target_data.shape[-1])
    flat_mask = decoded_mask.reshape(-1)

    selector = flat_mask == MaskValue.DECODER.value
    fake_tokens = flat_decoded[selector]
    real_tokens = flat_target[selector]

    if fake_tokens.shape[0] == 0:
        return None, None
    return fake_tokens, real_tokens


@dataclass
class DiscriminatorContrastiveLatentMIMTrainModuleConfig(
    ContrastiveLatentMIMTrainModuleConfig
):
    """Config for ContrastiveLatentMIM training with an adversarial discriminator.

    Args:
        discriminator_config: Configuration for the token discriminator network.
    """

    discriminator_config: TokenDiscriminatorConfig = field(
        default_factory=lambda: TokenDiscriminatorConfig()
    )

    def build(
        self,
        model: LatentMIM,
        device: torch.device | None = None,
    ) -> "DiscriminatorContrastiveLatentMIMTrainModule":
        kwargs = self.prepare_kwargs()
        return DiscriminatorContrastiveLatentMIMTrainModule(
            model=model,
            device=device,
            **kwargs,
        )


class DiscriminatorContrastiveLatentMIMTrainModule(ContrastiveLatentMIMTrainModule):
    """Extends ContrastiveLatentMIMTrainModule with a GAN-style discriminator on a target modality.

    The discriminator is trained to distinguish "real" tokens (from the EMA target
    encoder) from "fake" tokens (from the decoder) for a specific modality (e.g. NAIP).
    The encoder+decoder receive an adversarial gradient signal encouraging the decoder
    to produce NAIP tokens indistinguishable from the target encoder's.
    """

    def __init__(
        self,
        model: LatentMIM,
        discriminator_config: TokenDiscriminatorConfig,
        optim_config: OptimConfig,
        transform_config: TransformConfig,
        masking_config: MaskingConfig,
        loss_config: LossConfig,
        rank_microbatch_size: int,
        token_exit_cfg: dict[str, int],
        **kwargs: Any,
    ):
        super().__init__(
            model=model,
            optim_config=optim_config,
            transform_config=transform_config,
            masking_config=masking_config,
            loss_config=loss_config,
            rank_microbatch_size=rank_microbatch_size,
            token_exit_cfg=token_exit_cfg,
            **kwargs,
        )
        self.disc_config = discriminator_config
        self.discriminator = discriminator_config.build()
        self.disc_weight = discriminator_config.weight
        self.disc_target_modality = discriminator_config.target_modality
        self.disc_label_smoothing = discriminator_config.label_smoothing
        self.n_disc_steps = discriminator_config.n_disc_steps
        self._disc_step_counter = 0

    def on_attach(self) -> None:
        """Set up discriminator on the correct device with FSDP and its own optimizer."""
        super().on_attach()

        self.discriminator = self.discriminator.to(self.device)

        if self._dp_config is not None and self._dp_config.name == DataParallelType.fsdp:
            dp_mesh = get_dp_mesh(self.world_mesh)
            param_dtype = (
                self._dp_config.param_dtype.as_pt()
                if self._dp_config.param_dtype is not None
                else None
            )
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype,
                reduce_dtype=self._dp_config.reduce_dtype.as_pt(),
            )
            self.discriminator.apply_fsdp(mesh=dp_mesh, mp_policy=mp_policy)
            logger.info("Applied FSDP to discriminator")

        self.disc_optimizer = AdamWConfig(
            lr=self.disc_config.disc_lr,
            weight_decay=self.disc_config.disc_weight_decay,
            fused=False,
        ).build(self.discriminator)

    def _compute_disc_losses(
        self,
        decoded: TokensAndMasks,
        target_output: TokensAndMasks,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Compute discriminator loss and generator adversarial loss for the target modality.

        Returns:
            (disc_loss, gen_adv_loss) — either may be None if the modality is absent.
        """
        fake_tokens, real_tokens = _extract_modality_decoded_and_target(
            decoded, target_output, self.disc_target_modality
        )

        if fake_tokens is None or real_tokens is None:
            return None, None

        real_label = 1.0 - self.disc_label_smoothing
        fake_label = 0.0

        # --- Discriminator loss (detach fake to avoid updating encoder/decoder) ---
        real_logits = self.discriminator(real_tokens.detach())
        fake_logits = self.discriminator(fake_tokens.detach())

        disc_loss_real = F.binary_cross_entropy_with_logits(
            real_logits,
            torch.full_like(real_logits, real_label),
        )
        disc_loss_fake = F.binary_cross_entropy_with_logits(
            fake_logits,
            torch.full_like(fake_logits, fake_label),
        )
        disc_loss = (disc_loss_real + disc_loss_fake) / 2

        # --- Generator adversarial loss (gradients flow to encoder+decoder) ---
        fake_logits_for_gen = self.discriminator(fake_tokens)
        gen_adv_loss = self.disc_weight * F.binary_cross_entropy_with_logits(
            fake_logits_for_gen,
            torch.ones_like(fake_logits_for_gen),
        )

        return disc_loss, gen_adv_loss

    def train_batch(
        self,
        batch: tuple[int, MaskedOlmoEarthSample, MaskedOlmoEarthSample],
        dry_run: bool = False,
    ) -> None:
        """Train a batch with the base contrastive loss plus adversarial discriminator.

        The training loop interleaves discriminator updates with generator updates.
        """
        if not dry_run:
            self.update_target_encoder()

        self.model.train()
        self.discriminator.train()

        total_batch_loss = torch.zeros([], device=self.device)
        total_batch_reg = torch.zeros([], device=self.device)
        total_batch_con = torch.tensor(0.0, device=self.device)
        total_batch_disc = torch.tensor(0.0, device=self.device)
        total_batch_gen_adv = torch.tensor(0.0, device=self.device)

        patch_size = batch[0]
        batch_data_a = batch[1]
        batch_data_b = batch[2]
        microbatches_a = split_masked_batch(batch_data_a, self.rank_microbatch_size)
        microbatches_b = split_masked_batch(batch_data_b, self.rank_microbatch_size)
        num_microbatches = len(microbatches_a)

        self._disc_step_counter += 1
        update_generator = self._disc_step_counter % self.n_disc_steps == 0

        for microbatch_idx in range(num_microbatches):
            with self._train_microbatch_context(microbatch_idx, num_microbatches):
                microbatch_a = microbatches_a[microbatch_idx]
                microbatch_b = microbatches_b[microbatch_idx]
                masked_batch_a = microbatch_a.to_device(self.device)
                masked_batch_b = microbatch_b.to_device(self.device)

                # --- Standard forward passes ---
                loss_a, latent_a, decoded_a, target_output_a, pooled_a = (
                    self.model_forward(masked_batch_a, patch_size, self.token_exit_cfg)
                )
                loss_b, latent_b, decoded_b, target_output_b, pooled_b = (
                    self.model_forward(masked_batch_b, patch_size, self.token_exit_cfg)
                )
                loss = (loss_a + loss_b) / 2

                # Regularization
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

                # Contrastive loss
                if self.contrastive_loss is not None:
                    contrastive_loss = self.contrastive_loss.compute(pooled_a, pooled_b)
                    loss += contrastive_loss
                    total_batch_con += (
                        get_local_tensor(contrastive_loss.detach()) / num_microbatches
                    )

                # --- Discriminator losses ---
                disc_loss_a, gen_adv_loss_a = self._compute_disc_losses(
                    decoded_a, target_output_a
                )
                disc_loss_b, gen_adv_loss_b = self._compute_disc_losses(
                    decoded_b, target_output_b
                )

                # Accumulate discriminator loss for logging
                if disc_loss_a is not None and disc_loss_b is not None:
                    disc_loss = (disc_loss_a + disc_loss_b) / 2
                    total_batch_disc += (
                        get_local_tensor(disc_loss.detach()) / num_microbatches
                    )
                elif disc_loss_a is not None:
                    disc_loss = disc_loss_a
                    total_batch_disc += (
                        get_local_tensor(disc_loss.detach()) / num_microbatches
                    )
                elif disc_loss_b is not None:
                    disc_loss = disc_loss_b
                    total_batch_disc += (
                        get_local_tensor(disc_loss.detach()) / num_microbatches
                    )
                else:
                    disc_loss = None

                # Add generator adversarial loss to the main loss
                if update_generator:
                    if gen_adv_loss_a is not None and gen_adv_loss_b is not None:
                        gen_adv_loss = (gen_adv_loss_a + gen_adv_loss_b) / 2
                        loss = loss + gen_adv_loss
                        total_batch_gen_adv += (
                            get_local_tensor(gen_adv_loss.detach()) / num_microbatches
                        )
                    elif gen_adv_loss_a is not None:
                        loss = loss + gen_adv_loss_a
                        total_batch_gen_adv += (
                            get_local_tensor(gen_adv_loss_a.detach()) / num_microbatches
                        )
                    elif gen_adv_loss_b is not None:
                        loss = loss + gen_adv_loss_b
                        total_batch_gen_adv += (
                            get_local_tensor(gen_adv_loss_b.detach()) / num_microbatches
                        )

                loss = loss / num_microbatches
                loss_val = get_local_tensor(loss.detach())
                total_batch_loss += loss_val

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logger.warning(
                        f"NaN or Inf detected in loss at microbatch {microbatch_idx}, "
                        "stopping training for this batch."
                    )
                    del latent_a, latent_b
                    break

                del latent_a, latent_b

                # Backward for encoder+decoder (generator) loss
                loss.backward()

                # Discriminator backward (separate computation graph via detach)
                if disc_loss is not None:
                    (disc_loss / num_microbatches).backward()

        if dry_run:
            return

        # Step discriminator optimizer
        self.disc_optimizer.step()
        self.disc_optimizer.zero_grad(set_to_none=True)

        # --- Logging ---
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

        self.trainer.record_metric(
            f"train/disc_loss_{self.disc_target_modality}",
            total_batch_disc,
            ReduceType.mean,
        )
        self.trainer.record_metric(
            f"train/gen_adv_loss_{self.disc_target_modality}",
            total_batch_gen_adv,
            ReduceType.mean,
        )

        del batch
        del masked_batch_a, masked_batch_b

    def state_dict(self) -> dict[str, Any]:
        """Include discriminator and its optimizer in the state dict."""
        sd = super().state_dict()
        sd["discriminator"] = self.discriminator.state_dict()
        sd["disc_optim"] = self.disc_optimizer.state_dict()
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load discriminator state alongside the main model."""
        super().load_state_dict(state_dict)
        if "discriminator" in state_dict:
            self.discriminator.load_state_dict(state_dict["discriminator"])
        if "disc_optim" in state_dict:
            self.disc_optimizer.load_state_dict(state_dict["disc_optim"])
