"""Train module for the auxiliary NAIP conditional pix2pix-GAN objective.

Extends :class:`LatentMIMTrainModule` with a generator (part of the model,
trained by the main optimizer alongside the encoder) and a conditional
discriminator (held here with its own optimizer). The discriminator loss only
updates the discriminator; the generator loss (adversarial + L1) backprops
through the encoder.
"""

from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn.functional as F
from olmo_core.distributed.parallel import DataParallelConfig, get_dp_mesh
from olmo_core.distributed.utils import get_local_rank, get_local_tensor
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.common import ReduceType

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.datatypes import (
    MaskedOlmoEarthSample,
    MaskValue,
    TokensAndMasks,
)
from olmoearth_pretrain.nn.naip_gan import (
    NaipGanModel,
    discriminator_adversarial_loss,
    generator_adversarial_loss,
)
from olmoearth_pretrain.nn.utils import unpack_encoder_output
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.latent_mim import (
    LatentMIMTrainModule,
    LatentMIMTrainModuleConfig,
)
from olmoearth_pretrain.train.utils import split_masked_batch

logger = getLogger(__name__)


@dataclass
class NaipGanTrainModuleConfig(LatentMIMTrainModuleConfig):
    """Configuration for :class:`NaipGanTrainModule`.

    Args:
        discriminator_config: Config for the conditional discriminator.
        disc_optim_config: Optimizer config for the discriminator.
        lambda_adv: Weight on the generator adversarial loss.
        lambda_l1: Weight on the generator L1 reconstruction loss.
        gan_loss_type: Adversarial loss variant ("hinge" or "bce").
        gan_warmup_steps: Number of steps before the adversarial terms turn on
            (the L1 reconstruction term is active from step 0).
        naip_modality: Name of the NAIP modality to predict.
    """

    discriminator_config: Config | None = None
    disc_optim_config: OptimConfig | None = None
    lambda_adv: float = 0.1
    lambda_l1: float = 10.0
    gan_loss_type: str = "hinge"
    gan_warmup_steps: int = 0
    naip_modality: str = field(default_factory=lambda: Modality.NAIP_10.name)

    def build(
        self,
        model: NaipGanModel,
        device: torch.device | None = None,
    ) -> "NaipGanTrainModule":
        """Build the corresponding :class:`NaipGanTrainModule`."""
        if self.discriminator_config is None:
            raise ValueError("discriminator_config is required")
        if self.disc_optim_config is None:
            raise ValueError("disc_optim_config is required")
        kwargs = self.prepare_kwargs()
        return NaipGanTrainModule(
            model=model,
            device=device,
            **kwargs,
        )


class NaipGanTrainModule(LatentMIMTrainModule):
    """LatentMIM train module with an auxiliary NAIP conditional GAN branch."""

    def __init__(
        self,
        model: NaipGanModel,
        optim_config: OptimConfig,
        transform_config: TransformConfig,
        masking_config: MaskingConfig,
        loss_config: LossConfig,
        rank_microbatch_size: int,
        token_exit_cfg: dict[str, int],
        discriminator_config: Config,
        disc_optim_config: OptimConfig,
        lambda_adv: float = 0.1,
        lambda_l1: float = 10.0,
        gan_loss_type: str = "hinge",
        gan_warmup_steps: int = 0,
        naip_modality: str = Modality.NAIP_10.name,
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
        find_unused_parameters: bool = True,
    ):
        """Initialize the train module (see config for arg descriptions)."""
        super().__init__(
            model=model,
            optim_config=optim_config,
            transform_config=transform_config,
            masking_config=masking_config,
            loss_config=loss_config,
            rank_microbatch_size=rank_microbatch_size,
            token_exit_cfg=token_exit_cfg,
            mae_loss_config=mae_loss_config,
            compile_model=compile_model,
            dp_config=dp_config,
            compile_loss=compile_loss,
            autocast_precision=autocast_precision,
            max_grad_norm=max_grad_norm,
            scheduler=scheduler,
            device=device,
            state_dict_save_opts=state_dict_save_opts,
            state_dict_load_opts=state_dict_load_opts,
            ema_decay=ema_decay,
            regularizer_config=regularizer_config,
            find_unused_parameters=find_unused_parameters,
        )
        self.lambda_adv = lambda_adv
        self.lambda_l1 = lambda_l1
        self.gan_loss_type = gan_loss_type
        self.gan_warmup_steps = gan_warmup_steps
        self.naip_modality = naip_modality

        # The discriminator is intentionally NOT part of ``self.model`` so it is
        # excluded from the main optimizer (its loss must not update the encoder).
        self.discriminator = discriminator_config.build()
        self.discriminator.to(self.device)
        if self._dp_config is not None:
            # Replicate the (small) discriminator with DDP so its gradients stay
            # in sync across ranks; the main model may be FSDP-sharded.
            from torch.distributed._composable.replicate import replicate

            replicate(
                self.discriminator,
                device_mesh=get_dp_mesh(self.world_mesh),
                find_unused_parameters=find_unused_parameters,
            )
        self.disc_optimizer = disc_optim_config.build(self.discriminator)

        self.total_loss_name = f"{self.total_loss_name}+G_l1+G_adv"

    def zero_grads(self) -> None:
        """Zero gradients for both the main optimizer and the discriminator."""
        super().zero_grads()
        self.disc_optimizer.zero_grad(set_to_none=True)

    def model_forward(  # type: ignore[override]
        self,
        batch: MaskedOlmoEarthSample,
        patch_size: int,
        token_exit_cfg: dict[str, int],
    ) -> tuple[
        torch.Tensor,
        TokensAndMasks,
        TokensAndMasks,
        TokensAndMasks,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Run a forward pass returning the SSL loss plus the GAN tensors."""
        with self._model_forward_context():
            (
                latent,
                decoded,
                _,
                reconstructed,
                extra_metrics,
                pooled,
                fake_naip,
            ) = self.model(batch, patch_size)
            if extra_metrics is not None:
                self.log_extra_metrics(extra_metrics)
            with torch.no_grad():
                logger.info("Target Encoder forward pass...")
                output_dict = self.model.target_encoder.forward(
                    batch.unmask(),
                    patch_size=patch_size,
                    token_exit_cfg=token_exit_cfg,
                )
                target_output, _, _ = unpack_encoder_output(output_dict)
            loss = self.loss_fn(decoded, target_output)
            if self.mae_loss is not None:
                loss += self.mae_loss.compute(reconstructed, batch)
            return loss, latent, decoded, target_output, pooled, fake_naip

    def _extract_real_naip(
        self, batch: MaskedOlmoEarthSample
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Extract the real NAIP image and a per-instance validity mask.

        Returns:
            (image ``[B, C, Hn, Wn]``, valid ``[B]`` bool) or (None, None) if the
            NAIP modality is absent from the batch.
        """
        naip = getattr(batch, self.naip_modality, None)
        mask = getattr(
            batch,
            MaskedOlmoEarthSample.get_masked_modality_name(self.naip_modality),
            None,
        )
        if naip is None or mask is None:
            return None, None
        batch_size = naip.shape[0]
        valid = (mask != MaskValue.MISSING.value).reshape(batch_size, -1).any(dim=1)
        # NAIP is not multitemporal (T == 1); drop time and move channels first.
        naip_img = naip[:, :, :, 0, :].permute(0, 3, 1, 2).contiguous()
        return naip_img, valid

    def train_batch(
        self,
        batch: tuple[int, MaskedOlmoEarthSample],
        dry_run: bool = False,
    ) -> None:
        """Train a batch with the SSL objective plus the NAIP GAN branch."""
        if not dry_run:
            self.update_target_encoder()
        self.model.train()
        self.discriminator.train()

        total_batch_loss = torch.zeros([], device=self.device)
        total_batch_reg = torch.zeros([], device=self.device)
        total_d_loss = torch.zeros([], device=self.device)
        total_g_adv = torch.zeros([], device=self.device)
        total_g_l1 = torch.zeros([], device=self.device)

        patch_size = batch[0]
        batch_data = batch[1]

        adversarial_active = (
            not dry_run and self.trainer.global_step >= self.gan_warmup_steps
        )
        stepped_discriminator = False

        masked_microbatches = split_masked_batch(batch_data, self.rank_microbatch_size)
        num_microbatches = len(masked_microbatches)

        for microbatch_idx in range(num_microbatches):
            with self._train_microbatch_context(microbatch_idx, num_microbatches):
                microbatch_masked = masked_microbatches[microbatch_idx]
                logger.info(
                    f"Training microbatch {microbatch_idx} of {num_microbatches} "
                    f"with batch size {microbatch_masked.batch_size} on rank {get_local_rank()}"
                )
                masked_batch = microbatch_masked.to_device(self.device)

                loss, latent, decoded, target_output, pooled, fake_naip = (
                    self.model_forward(masked_batch, patch_size, self.token_exit_cfg)
                )

                reg_term = self.compute_regularization(latent)
                if reg_term is not None:
                    loss = loss + reg_term
                    total_batch_reg += (
                        get_local_tensor(reg_term.detach()) / num_microbatches
                    )

                d_loss = self._maybe_discriminator_loss(
                    masked_batch,
                    pooled,
                    fake_naip,
                    num_microbatches,
                    adversarial_active,
                )
                gen_loss, g_l1_val, g_adv_val = self._generator_loss(
                    masked_batch, pooled, fake_naip, adversarial_active
                )
                total_g_l1 += g_l1_val / num_microbatches
                total_g_adv += g_adv_val / num_microbatches

                loss = (loss + gen_loss) / num_microbatches
                total_batch_loss += get_local_tensor(loss.detach())

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logger.warning(
                        f"NaN or Inf detected in loss at microbatch {microbatch_idx}, "
                        f"stopping training for this batch."
                    )
                    print(f"rank {get_local_rank()} has nan or inf")

                if d_loss is not None:
                    total_d_loss += get_local_tensor(d_loss.detach())
                    d_loss.backward()
                    stepped_discriminator = True
                loss.backward()

        if not dry_run and stepped_discriminator:
            self.disc_optimizer.step()

        if dry_run:
            self.disc_optimizer.zero_grad(set_to_none=True)
            return

        self.trainer.record_metric(
            f"train/{self.total_loss_name}", total_batch_loss, ReduceType.mean
        )
        self.trainer.record_metric("train/G_l1", total_g_l1, ReduceType.mean)
        self.trainer.record_metric("train/G_adv", total_g_adv, ReduceType.mean)
        self.trainer.record_metric("train/D_loss", total_d_loss, ReduceType.mean)
        self.log_regularization(total_batch_reg)

        del batch, batch_data
        del masked_batch
        del latent, decoded, target_output, pooled, fake_naip

    def _maybe_discriminator_loss(
        self,
        batch: MaskedOlmoEarthSample,
        pooled: torch.Tensor,
        fake_naip: torch.Tensor,
        num_microbatches: int,
        adversarial_active: bool,
    ) -> torch.Tensor | None:
        """Compute the (scaled) discriminator loss on detached inputs.

        Returns None when the adversarial phase is inactive or NAIP is absent.
        Only the discriminator receives gradients from this loss.
        """
        if not adversarial_active or fake_naip is None:
            return None
        real_img, valid = self._extract_real_naip(batch)
        if real_img is None or valid is None or not bool(valid.any()):
            return None
        with self._model_forward_context():
            fake_v = fake_naip[valid].detach()
            cond_v = pooled[valid].detach()
            real_v = F.interpolate(
                real_img[valid].to(fake_naip.dtype),
                size=fake_naip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            real_logits = self.discriminator(real_v, cond_v)
            fake_logits = self.discriminator(fake_v, cond_v)
            d_loss = discriminator_adversarial_loss(
                real_logits, fake_logits, self.gan_loss_type
            )
        return d_loss / num_microbatches

    def _generator_loss(
        self,
        batch: MaskedOlmoEarthSample,
        pooled: torch.Tensor,
        fake_naip: torch.Tensor,
        adversarial_active: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the generator loss (L1 always, adversarial once warmed up).

        Gradients from this loss flow into the generator and encoder but not the
        discriminator (its parameters are frozen during the forward pass here).

        Returns:
            (generator_loss, g_l1_value, g_adv_value); the loss is zero when NAIP
            is absent from the batch.
        """
        zero = torch.zeros([], device=self.device)
        if fake_naip is None:
            return zero, zero, zero
        real_img, valid = self._extract_real_naip(batch)
        if real_img is None or valid is None or not bool(valid.any()):
            return zero, zero, zero

        with self._model_forward_context():
            fake_v = fake_naip[valid]
            cond_v = pooled[valid]
            real_v = F.interpolate(
                real_img[valid].to(fake_naip.dtype),
                size=fake_naip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            g_l1 = F.l1_loss(fake_v, real_v)
            gen_loss = self.lambda_l1 * g_l1
            g_adv = zero
            if adversarial_active:
                # Freeze the discriminator so the generator loss doesn't update it.
                for p in self.discriminator.parameters():
                    p.requires_grad_(False)
                fake_logits = self.discriminator(fake_v, cond_v)
                for p in self.discriminator.parameters():
                    p.requires_grad_(True)
                g_adv = generator_adversarial_loss(fake_logits, self.gan_loss_type)
                gen_loss = gen_loss + self.lambda_adv * g_adv
        return (
            gen_loss,
            get_local_tensor(g_l1.detach()),
            get_local_tensor(g_adv.detach()),
        )

    def _get_state_dict(
        self, sd_options: dist_cp_sd.StateDictOptions
    ) -> dict[str, Any]:
        """Include the discriminator + its optimizer for clean resume."""
        sd = super()._get_state_dict(sd_options)
        sd["discriminator"] = dist_cp_sd.get_model_state_dict(
            self.discriminator, options=sd_options
        )
        sd["disc_optim"] = dist_cp_sd.get_optimizer_state_dict(
            self.discriminator, self.disc_optimizer, options=sd_options
        )
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load model + optim, then the discriminator state if present."""
        super().load_state_dict(state_dict)
        if "discriminator" in state_dict:
            dist_cp_sd.set_model_state_dict(
                self.discriminator,
                state_dict["discriminator"],
                options=self.state_dict_load_opts,
            )
        if "disc_optim" in state_dict:
            dist_cp_sd.set_optimizer_state_dict(
                self.discriminator,
                self.disc_optimizer,
                state_dict["disc_optim"],
                options=self.state_dict_load_opts,
            )
