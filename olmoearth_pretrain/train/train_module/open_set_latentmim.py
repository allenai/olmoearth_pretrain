"""Latent-MIM train module with a supervised open-set probe.

Extends :class:`LatentMIMTrainModule` (the single-forward recipe used by the
register-bottleneck / perceiver runs) by adding a supervised segmentation +
regression loss (see
:class:`olmoearth_pretrain.train.open_set_probe.OpenSetProbe`) on top of the
self-supervised objective. The probe reads the encoder's *spatial latent grid*
(the register/Perceiver bottleneck output), so the supervised gradient flows
back into the bottleneck and the encoder.

The probe itself lives inside the model
(:class:`olmoearth_pretrain.nn.open_set_latent_mim.OpenSetLatentMIM`) so that the
DDP gradient all-reduce and the optimizer cover its parameters.
"""

from dataclasses import dataclass
from logging import getLogger
from typing import Any

import torch
import torch.distributed as dist
from olmo_core.distributed.utils import get_world_size

from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, TokensAndMasks
from olmoearth_pretrain.train.train_module.latent_mim import (
    LatentMIMTrainModule,
    LatentMIMTrainModuleConfig,
)

logger = getLogger(__name__)


class OpenSetLatentMIMTrainModule(LatentMIMTrainModule):
    """Latent-MIM plus a supervised open-set probe loss on the spatial latent."""

    # The single-forward (1fwd) recipe runs one view per batch.
    _NUM_AUGMENTED_VIEWS = 1

    def __init__(
        self,
        *args: Any,
        sup_loss_weight: float = 1.0,
        freeze_backbone_until_step: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize, extracting the supervised loss weight.

        Args:
            *args: Positional arguments forwarded to the base train module.
            sup_loss_weight: Scalar weight applied to the combined supervised
                (CE + MSE) loss when added to the self-supervised objective.
            freeze_backbone_until_step: If > 0, train ONLY the open-set probe
                until this global step, then unfreeze the rest of the model
                (the linear-probe warmup of the mid-training recipe, for runs
                initialized from a pretrained checkpoint via
                ``trainer.load_path``).
            **kwargs: Keyword arguments forwarded to the base train module.
        """
        super().__init__(*args, **kwargs)
        self.sup_loss_weight = sup_loss_weight
        self.freeze_backbone_until_step = freeze_backbone_until_step
        self._supervised_metrics: dict[str, tuple[float, int]] | None = None
        # Params frozen at init (e.g. FrozenTargetProjection copies) must never
        # be flipped trainable by the freeze schedule.
        self._always_frozen_param_ids = {
            id(p) for p in self.model.parameters() if not p.requires_grad
        }
        self._backbone_frozen: bool | None = None

    def _apply_freeze_schedule(self) -> None:
        """Freeze/unfreeze the non-probe params based on the global step.

        Before ``freeze_backbone_until_step`` only the open-set probe trains;
        afterwards the whole model does. ``requires_grad`` is toggled lazily
        AFTER the optimizer was built, so the backbone params remain in the
        optimizer throughout and simply resume updating on unfreeze (AdamW,
        fused included, skips params whose ``grad`` is ``None``). The flip is
        keyed on the global step, so all DP ranks toggle together and the
        replicated-DDP gradient all-reduce sees identical grad sets on every
        rank. Params that were already frozen at init (the projection-only
        target copies) are never unfrozen.
        """
        if self.freeze_backbone_until_step <= 0:
            return
        freeze = self.trainer.global_step < self.freeze_backbone_until_step
        if freeze == self._backbone_frozen:
            return
        probe_param_ids = {id(p) for p in self.model.open_set_probe.parameters()}
        num_toggled = 0
        for p in self.model.parameters():
            if id(p) in probe_param_ids or id(p) in self._always_frozen_param_ids:
                continue
            p.requires_grad_(not freeze)
            num_toggled += 1
        self._backbone_frozen = freeze
        logger.info(
            "open-set freeze schedule: %s %d backbone params at step %d "
            "(freeze_backbone_until_step=%d)",
            "froze" if freeze else "unfroze",
            num_toggled,
            self.trainer.global_step,
            self.freeze_backbone_until_step,
        )

    def train_batch(
        self,
        batch: tuple[int, MaskedOlmoEarthSample],
        dry_run: bool = False,
    ) -> None:
        """Train a batch and record supervised metrics once for the full batch."""
        self._apply_freeze_schedule()
        self._supervised_metrics = {}
        try:
            super().train_batch(batch, dry_run=dry_run)
            if not dry_run:
                self._flush_supervised_metrics()
        finally:
            self._supervised_metrics = None

    def _accumulate_supervised_metrics(self, metrics: dict[str, float]) -> None:
        """Accumulate metrics emitted by each view and microbatch forward."""
        if self._supervised_metrics is None:
            raise RuntimeError(
                "supervised metrics can only be recorded during train_batch"
            )
        for key in ("open_set_ce", "open_set_mse"):
            value = metrics.get(key, 0.0)
            sample_count = metrics.get(f"{key}_samples", 0.0)
            patch_count = metrics.get(f"{key}_patches", 0.0)
            for suffix, increment in (
                ("", value * sample_count),
                ("_samples", sample_count),
                ("_patches", patch_count),
            ):
                total, count = self._supervised_metrics.get(key + suffix, (0.0, 0))
                self._supervised_metrics[key + suffix] = (
                    total + increment,
                    count + 1,
                )

    def _flush_supervised_metrics(self) -> None:
        """Log globally sample-weighted metrics once for the full batch."""
        if not self._supervised_metrics:
            return
        keys = [
            "open_set_ce",
            "open_set_ce_samples",
            "open_set_ce_patches",
            "open_set_mse",
            "open_set_mse_samples",
            "open_set_mse_patches",
        ]
        totals = torch.tensor(
            [self._supervised_metrics[key][0] for key in keys],
            dtype=torch.float64,
            device=self.device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals, group=self.dp_process_group)
        (
            ce_sum,
            ce_samples,
            ce_patches,
            mse_sum,
            mse_samples,
            mse_patches,
        ) = totals.tolist()
        metrics = {
            "open_set_ce": ce_sum / ce_samples if ce_samples else 0.0,
            "open_set_ce_samples": ce_samples / self._NUM_AUGMENTED_VIEWS,
            "open_set_ce_patches": ce_patches / self._NUM_AUGMENTED_VIEWS,
            "open_set_mse": mse_sum / mse_samples if mse_samples else 0.0,
            "open_set_mse_samples": mse_samples / self._NUM_AUGMENTED_VIEWS,
            "open_set_mse_patches": mse_patches / self._NUM_AUGMENTED_VIEWS,
        }
        self.log_extra_metrics(
            {f"train/{key}": value for key, value in metrics.items()},
            reduce_type=None,
        )

    def _global_sample_counts(self, metrics: dict[str, float]) -> dict[str, float]:
        """Sum labeled classification and regression sample counts across DP ranks."""
        counts = torch.tensor(
            [
                metrics.get("open_set_ce_samples", 0.0),
                metrics.get("open_set_mse_samples", 0.0),
            ],
            dtype=torch.float64,
            device=self.device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, group=self.dp_process_group)
        return {
            "open_set_ce": float(counts[0]),
            "open_set_mse": float(counts[1]),
        }

    def _combine_supervised_losses(
        self,
        losses: dict[str, torch.Tensor],
        metrics: dict[str, float],
    ) -> torch.Tensor:
        """Combine local sample means so DP averaging yields global sample means."""
        loss = losses["zero_touch"]
        global_counts = self._global_sample_counts(metrics)
        world_size = get_world_size(self.dp_process_group)
        for key in ("open_set_ce", "open_set_mse"):
            local_count = metrics.get(f"{key}_samples", 0.0)
            global_count = global_counts[key]
            if key in losses and global_count > 0:
                loss = loss + losses[key] * local_count * world_size / global_count
        return loss

    def model_forward(
        self,
        batch: MaskedOlmoEarthSample,
        patch_size: int,
        token_exit_cfg: dict[str, int],
    ) -> tuple[
        torch.Tensor,
        TokensAndMasks,
        TokensAndMasks,
        TokensAndMasks,
        dict[str, Any] | None,
    ]:
        """Run the base forward, then add the supervised probe loss."""
        loss, latent, decoded, target_output, extra_metrics = super().model_forward(
            batch, patch_size, token_exit_cfg
        )
        # The probe reads the encoder's spatial latent grid (the Perceiver/register
        # bottleneck output), which the model forward stashes on the model.
        register_grid = getattr(self.model, "last_register_grid", None)
        if register_grid is None:
            raise RuntimeError(
                "OpenSetLatentMIMTrainModule requires the encoder register "
                "bottleneck: the open-set probe reads the spatial latent grid "
                "(set use_register_bottleneck=True on the encoder/decoder configs)"
            )
        # The probe lives inside the model so DDP/optimizer cover its params. It always
        # returns a probe-connected loss (a zero-touch term when a rank has no labeled
        # patches) so every rank produces gradients for the probe params each step.
        # Re-enter the model forward context because the base method has already exited
        # it. Production DDP uses bf16 autocast, and the probe's fp32 parameters must be
        # autocast together with the encoder's bf16 spatial latent grid.
        with self._model_forward_context():
            sup_losses, sup_metrics = self.model.open_set_probe(register_grid, batch)
        sup_loss = self._combine_supervised_losses(sup_losses, sup_metrics)
        loss = loss + self.sup_loss_weight * sup_loss
        if sup_metrics:
            self._accumulate_supervised_metrics(sup_metrics)
        return loss, latent, decoded, target_output, extra_metrics


@dataclass
class OpenSetLatentMIMTrainModuleConfig(LatentMIMTrainModuleConfig):
    """Configuration for :class:`OpenSetLatentMIMTrainModule`."""

    sup_loss_weight: float = 1.0
    freeze_backbone_until_step: int = 0

    def build(
        self,
        model: Any,
        device: torch.device | None = None,
    ) -> "OpenSetLatentMIMTrainModule":
        """Build the corresponding :class:`OpenSetLatentMIMTrainModule`."""
        kwargs = self.prepare_kwargs()
        return OpenSetLatentMIMTrainModule(
            model=model,
            device=device,
            **kwargs,
        )
