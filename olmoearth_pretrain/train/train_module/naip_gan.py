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

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn.functional as F
from olmo_core.distributed.parallel import DataParallelConfig
from olmo_core.distributed.utils import (
    get_local_rank,
    get_local_tensor,
    get_rank,
    get_world_size,
)
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.common import ReduceType

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality
from olmoearth_pretrain.data.normalize import load_computed_config
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
from olmoearth_pretrain.nn.pooling import PoolingType, pool_unmasked_tokens
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
        image_log_interval: Log generated-vs-real NAIP images to W&B every this
            many steps (0 disables image logging).
        num_log_images: Max number of paired images to upload each time.
        naip_denorm_std_multiplier: std multiplier used to invert the NAIP
            normalization for display (must match the dataset Normalizer).
        discriminator_cond_source: What the discriminator conditions on. One of
            "online_pooled" (default; the online-encoder pooled embedding),
            "target_pooled" (the target-encoder pooled embedding over the
            unmasked tokens), or "raw_sentinel2" (the raw Sentinel-2 image,
            conditioned on even when it is masked for the online encoder). The
            discriminator's ``cond_mode`` must match: "image" for
            "raw_sentinel2", "embedding" otherwise.
        cond_modality: Modality name used as the raw image condition when
            ``discriminator_cond_source == "raw_sentinel2"``. The full temporal
            stack of all its bands is fed to the discriminator.
    """

    discriminator_config: Config | None = None
    disc_optim_config: OptimConfig | None = None
    lambda_adv: float = 0.1
    lambda_l1: float = 10.0
    gan_loss_type: str = "hinge"
    gan_warmup_steps: int = 0
    naip_modality: str = field(default_factory=lambda: Modality.NAIP_10.name)
    image_log_interval: int = 1000
    num_log_images: int = 10
    naip_denorm_std_multiplier: float = 2.0
    discriminator_cond_source: str = "online_pooled"
    cond_modality: str = field(default_factory=lambda: Modality.SENTINEL2_L2A.name)

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
        image_log_interval: int = 1000,
        num_log_images: int = 10,
        naip_denorm_std_multiplier: float = 2.0,
        discriminator_cond_source: str = "online_pooled",
        cond_modality: str = Modality.SENTINEL2_L2A.name,
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
        self.image_log_interval = image_log_interval
        self.num_log_images = num_log_images
        self.discriminator_cond_source = discriminator_cond_source
        self.cond_modality = cond_modality
        valid_sources = ("online_pooled", "target_pooled", "raw_sentinel2")
        if discriminator_cond_source not in valid_sources:
            raise ValueError(
                f"discriminator_cond_source must be one of {valid_sources}, "
                f"got {discriminator_cond_source}"
            )

        # Precompute per-band (R, G, B) min/max for inverse min-max denormalization
        # of NAIP for W&B display, matching the dataset's computed normalization.
        computed_config = load_computed_config()[self.naip_modality]
        rgb_bands = Modality.get(self.naip_modality).band_order[:3]
        means = np.array([computed_config[b]["mean"] for b in rgb_bands])
        stds = np.array([computed_config[b]["std"] for b in rgb_bands])
        self._naip_rgb_min = means - naip_denorm_std_multiplier * stds
        self._naip_rgb_max = means + naip_denorm_std_multiplier * stds

        # The discriminator is intentionally NOT part of ``self.model`` so it is
        # excluded from the main optimizer (its loss must not update the encoder).
        self.discriminator = discriminator_config.build()
        self.discriminator.to(self.device)
        # The discriminator's condition input must match the chosen source: a raw
        # image ("image") for raw Sentinel-2, otherwise a pooled embedding.
        expected_cond_mode = (
            "image" if discriminator_cond_source == "raw_sentinel2" else "embedding"
        )
        disc_cond_mode = getattr(self.discriminator, "cond_mode", "embedding")
        if disc_cond_mode != expected_cond_mode:
            raise ValueError(
                f"discriminator_cond_source={discriminator_cond_source!r} requires "
                f"the discriminator cond_mode={expected_cond_mode!r}, but got "
                f"{disc_cond_mode!r}"
            )
        # The discriminator is small and replicated (not sharded); rather than
        # wrapping it in DDP we keep it as a plain module and all-reduce its
        # gradients manually once per step (see ``_all_reduce_discriminator_grads``).
        # Broadcast the initial weights from the first rank of the data-parallel
        # group so every replica starts identical (this is what DDP would do at
        # construction time).
        if self._dp_config is not None and get_world_size(self.dp_process_group) > 1:
            src = dist.get_global_rank(self.dp_process_group, 0)
            for tensor in [
                *self.discriminator.parameters(),
                *self.discriminator.buffers(),
            ]:
                dist.broadcast(tensor.detach(), src=src, group=self.dp_process_group)
        self.disc_optimizer = disc_optim_config.build(self.discriminator)

        self.total_loss_name = f"{self.total_loss_name}+G_l1+G_adv"

    def zero_grads(self) -> None:
        """Zero gradients for both the main optimizer and the discriminator."""
        super().zero_grads()
        self.disc_optimizer.zero_grad(set_to_none=True)

    def _all_reduce_discriminator_grads(self) -> None:
        """Average the discriminator gradients across the data-parallel group.

        The discriminator is a plain (non-DDP) replicated module, so its
        gradients are accumulated purely locally during backward. This performs
        the single explicit all-reduce that keeps every replica's update (and
        therefore its weights) identical. Ranks that produced no discriminator
        gradient this step (e.g. no valid NAIP) contribute zeros so that every
        rank still participates in the collective; the sum is divided by the full
        data-parallel world size, matching standard DDP averaging.
        """
        if self._dp_config is None:
            return
        world_size = get_world_size(self.dp_process_group)
        if world_size <= 1:
            return
        for p in self.discriminator.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=self.dp_process_group)
            p.grad /= world_size

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

    def _naip_to_rgb01(self, x: torch.Tensor) -> np.ndarray:
        """Invert NAIP normalization and return displayable RGB in ``[0, 1]``.

        Args:
            x: Normalized NAIP tensor of shape ``[N, C, H, W]`` (C >= 3).

        Returns:
            A ``[N, H, W, 3]`` float array in ``[0, 1]``.
        """
        rgb = x[:, :3].detach().float().cpu().numpy()  # [N, 3, H, W]
        min_vals = self._naip_rgb_min.reshape(1, 3, 1, 1)
        max_vals = self._naip_rgb_max.reshape(1, 3, 1, 1)
        raw = rgb * (max_vals - min_vals) + min_vals
        rgb01 = np.clip(raw / 255.0, 0.0, 1.0)
        return np.transpose(rgb01, (0, 2, 3, 1))  # [N, H, W, 3]

    def _maybe_log_naip_images(
        self, batch: MaskedOlmoEarthSample, fake_naip: torch.Tensor | None
    ) -> None:
        """Occasionally upload paired real/generated NAIP images to W&B."""
        if self.image_log_interval <= 0 or fake_naip is None:
            return
        if self.trainer.global_step % self.image_log_interval != 0:
            return
        if get_rank() != 0:
            return

        real_img, valid = self._extract_real_naip(batch)
        if real_img is None or valid is None or not bool(valid.any()):
            return

        wandb = self._get_wandb()
        if wandb is None:
            return

        num = min(self.num_log_images, int(valid.sum().item()))
        fake_v = fake_naip[valid][:num]
        real_v = F.interpolate(
            real_img[valid][:num].to(fake_v.dtype),
            size=fake_v.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fake_rgb = self._naip_to_rgb01(fake_v)
        real_rgb = self._naip_to_rgb01(real_v)

        images = []
        for i in range(num):
            # Real on the left, generated on the right.
            pair = np.concatenate([real_rgb[i], fake_rgb[i]], axis=1)
            images.append(wandb.Image(pair, caption="left: real | right: generated"))
        wandb.log(
            {"gan/naip_real_vs_fake": images},
            step=self.trainer.global_step,
        )

    def _get_wandb(self) -> Any | None:
        """Return the wandb module from the OlmoEarth W&B callback, if enabled."""
        from olmoearth_pretrain.train.callbacks.wandb import OlmoEarthWandBCallback

        for callback in self.trainer._iter_callbacks():
            if isinstance(callback, OlmoEarthWandBCallback) and callback.enabled:
                return callback.wandb
        return None

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

                if not dry_run and microbatch_idx == 0:
                    with torch.no_grad():
                        self._maybe_log_naip_images(masked_batch, fake_naip)

                reg_term = self.compute_regularization(latent)
                if reg_term is not None:
                    loss = loss + reg_term
                    total_batch_reg += (
                        get_local_tensor(reg_term.detach()) / num_microbatches
                    )

                d_cond, d_cond_valid, d_cond_time_mask = (
                    self._extract_discriminator_cond(
                        masked_batch, pooled, target_output
                    )
                )

                d_loss = self._maybe_discriminator_loss(
                    masked_batch,
                    d_cond,
                    d_cond_valid,
                    fake_naip,
                    num_microbatches,
                    adversarial_active,
                    patch_size,
                    d_cond_time_mask,
                )
                gen_loss, g_l1_val, g_adv_val = self._generator_loss(
                    masked_batch,
                    d_cond,
                    d_cond_valid,
                    fake_naip,
                    adversarial_active,
                    patch_size,
                    d_cond_time_mask,
                )
                total_g_l1 += g_l1_val / num_microbatches
                total_g_adv += g_adv_val / num_microbatches

                # Anchor the generator to the loss with zero weight so its
                # parameters always receive a gradient on every rank. When a rank
                # has no valid NAIP, the generator is otherwise disconnected from
                # the loss (both the L1 and adversarial terms are skipped), so its
                # gradient flow -- and therefore the root model's FSDP gradient
                # reduce-scatter -- would be asymmetric across ranks and could
                # deadlock the collective. The term is exactly zero and does not
                # change any real gradient.
                if fake_naip is not None:
                    loss = loss + 0.0 * fake_naip.sum()

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
                loss.backward()

        if not dry_run and adversarial_active:
            # The discriminator accumulated gradients locally across microbatches;
            # average them across the data-parallel group once, then step. Gating
            # on ``adversarial_active`` (which is derived from the global step and
            # so is identical on every rank) guarantees all ranks participate in
            # the collective together, even if some had no valid NAIP this step.
            self._all_reduce_discriminator_grads()
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

    def _extract_discriminator_cond(
        self,
        batch: MaskedOlmoEarthSample,
        pooled: torch.Tensor,
        target_output: TokensAndMasks,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Return the discriminator condition, validity mask, and time mask.

        The condition depends on ``discriminator_cond_source``:

        * ``online_pooled``: the online-encoder pooled embedding ``[B, H, W, D]``
          (gradients flow to the encoder through the condition path).
        * ``target_pooled``: the target-encoder pooled embedding over the unmasked
          tokens ``[B, H, W, D]`` (no gradient to the encoder, EMA weights).
        * ``raw_sentinel2``: the raw Sentinel-2 temporal stack
          ``[B, T, C, Hs, Ws]``.

        Returns:
            ``(cond, cond_valid, cond_time_mask)``. ``cond`` is None when the raw
            condition is entirely absent from the batch; ``cond_valid`` is a
            ``[B]`` bool mask (None means every instance is valid);
            ``cond_time_mask`` is a ``[B, T]`` bool mask of valid timesteps for
            the raw stack (None for the pooled embedding modes).
        """
        source = self.discriminator_cond_source
        if source == "online_pooled":
            return pooled, None, None
        if source == "target_pooled":
            target_pooled = pool_unmasked_tokens(
                target_output, PoolingType.MEAN, spatial_pooling=True
            )
            return target_pooled, None, None
        # raw_sentinel2
        return self._extract_raw_s2_cond(batch)

    def _extract_raw_s2_cond(
        self, batch: MaskedOlmoEarthSample
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Extract the raw image temporal stack and validity masks.

        The raw modality tensor retains its full pixel values regardless of the
        online-encoder mask, so an instance can be conditioned on even when the
        modality is masked for the online encoder (any non-MISSING mask counts as
        valid). All bands and all timesteps are returned; missing entries (the
        MISSING_VALUE sentinel) are zeroed and excluded from the temporal mean via
        the returned per-timestep validity mask.

        Returns:
            ``(stack [B, T, C, H, W], valid [B], time_valid [B, T])`` or
            ``(None, None, None)`` if the modality is absent from the batch.
        """
        modality = self.cond_modality
        raw = getattr(batch, modality, None)
        mask = getattr(
            batch,
            MaskedOlmoEarthSample.get_masked_modality_name(modality),
            None,
        )
        if raw is None or mask is None:
            return None, None, None
        batch_size = raw.shape[0]
        # [B, H, W, T, C] -> [B, T, C, H, W] (all bands).
        stack = raw.permute(0, 3, 4, 1, 2)
        present = stack != MISSING_VALUE
        # Zero the sentinel so the per-timestep convs don't see huge values.
        stack = torch.where(present, stack, torch.zeros_like(stack))
        # Per-timestep validity (any present pixel/band).
        time_valid = present.reshape(batch_size, stack.shape[1], -1).any(dim=2)
        # Per-instance validity.
        valid = (mask != MaskValue.MISSING.value).reshape(batch_size, -1).any(dim=1)
        return stack, valid, time_valid

    def _maybe_discriminator_loss(
        self,
        batch: MaskedOlmoEarthSample,
        cond: torch.Tensor | None,
        cond_valid: torch.Tensor | None,
        fake_naip: torch.Tensor,
        num_microbatches: int,
        adversarial_active: bool,
        patch_size: int,
        cond_time_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Compute the (scaled) discriminator loss on detached inputs.

        Returns None when the adversarial phase is inactive, NAIP is absent, or no
        instance has both valid NAIP and a valid condition. Only the
        discriminator receives gradients from this loss.
        """
        if not adversarial_active or fake_naip is None or cond is None:
            return None
        real_img, valid = self._extract_real_naip(batch)
        if real_img is None or valid is None:
            return None
        if cond_valid is not None:
            valid = valid & cond_valid
        if not bool(valid.any()):
            return None
        with self._model_forward_context():
            fake_v = fake_naip[valid].detach()
            cond_v = cond[valid].detach()
            time_v = None if cond_time_mask is None else cond_time_mask[valid]
            real_v = F.interpolate(
                real_img[valid].to(fake_naip.dtype),
                size=fake_naip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            # Single discriminator forward on the concatenated real+fake batch
            # (one forward is slightly cheaper than two separate passes).
            images = torch.cat([real_v, fake_v], dim=0)
            conds = torch.cat([cond_v, cond_v], dim=0)
            time_mask = None if time_v is None else torch.cat([time_v, time_v], dim=0)
            logits = self.discriminator(
                images,
                conds,
                patch_size=patch_size,
                cond_time_mask=time_mask,
            )
            real_logits, fake_logits = logits.chunk(2, dim=0)
            d_loss = discriminator_adversarial_loss(
                real_logits, fake_logits, self.gan_loss_type
            )
        return d_loss / num_microbatches

    def _generator_loss(
        self,
        batch: MaskedOlmoEarthSample,
        cond: torch.Tensor | None,
        cond_valid: torch.Tensor | None,
        fake_naip: torch.Tensor,
        adversarial_active: bool,
        patch_size: int,
        cond_time_mask: torch.Tensor | None = None,
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
        if real_img is None or valid is None:
            return zero, zero, zero
        if cond_valid is not None:
            valid = valid & cond_valid
        if not bool(valid.any()):
            return zero, zero, zero

        with self._model_forward_context():
            fake_v = fake_naip[valid]
            real_v = F.interpolate(
                real_img[valid].to(fake_naip.dtype),
                size=fake_naip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            g_l1 = F.l1_loss(fake_v, real_v)
            gen_loss = self.lambda_l1 * g_l1
            g_adv = zero
            if adversarial_active and cond is not None:
                cond_v = cond[valid]
                time_v = None if cond_time_mask is None else cond_time_mask[valid]
                # Freeze the discriminator so the generator loss doesn't update
                # it: without this, ``loss.backward()`` (which includes g_adv)
                # would accumulate adversarial gradients into the discriminator's
                # params, which would then be applied by ``disc_optimizer``.
                for p in self.discriminator.parameters():
                    p.requires_grad_(False)
                fake_logits = self.discriminator(
                    fake_v,
                    cond_v,
                    patch_size=patch_size,
                    cond_time_mask=time_v,
                )
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
