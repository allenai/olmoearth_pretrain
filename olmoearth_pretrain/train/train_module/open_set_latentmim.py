"""Contrastive latent-MIM train module with a supervised open-set probe.

Extends :class:`ContrastiveLatentMIMTrainModule` by adding a supervised
segmentation + regression loss (see
:class:`olmoearth_pretrain.train.open_set_probe.OpenSetProbe`) on top of the
self-supervised objective. The probe reads the *online* encoder output, so the
supervised gradient flows back into the encoder.

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
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModule,
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = getLogger(__name__)


class OpenSetLatentMIMTrainModule(ContrastiveLatentMIMTrainModule):
    """Contrastive latent-MIM plus a supervised open-set probe loss."""

    _NUM_AUGMENTED_VIEWS = 2

    def __init__(self, *args: Any, sup_loss_weight: float = 1.0, **kwargs: Any) -> None:
        """Initialize, extracting the supervised loss weight.

        Args:
            *args: Positional arguments forwarded to the base train module.
            sup_loss_weight: Scalar weight applied to the combined supervised
                (CE + MSE) loss when added to the self-supervised objective.
            **kwargs: Keyword arguments forwarded to the base train module.
        """
        super().__init__(*args, **kwargs)
        self.sup_loss_weight = sup_loss_weight
        self._supervised_metrics: dict[str, tuple[float, int]] | None = None

    def train_batch(
        self,
        batch: tuple[int, MaskedOlmoEarthSample, MaskedOlmoEarthSample],
        dry_run: bool = False,
    ) -> None:
        """Train a batch and record supervised metrics once for the full batch."""
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
            patch_count = metrics.get(f"{key}_patches", 0.0)
            total, count = self._supervised_metrics.get(key, (0.0, 0))
            self._supervised_metrics[key] = (
                total + value * patch_count,
                count + 1,
            )
            patch_key = f"{key}_patches"
            patch_total, patch_count_entries = self._supervised_metrics.get(
                patch_key, (0.0, 0)
            )
            self._supervised_metrics[patch_key] = (
                patch_total + patch_count,
                patch_count_entries + 1,
            )

    def _flush_supervised_metrics(self) -> None:
        """Log globally patch-weighted metrics once for the full batch."""
        if not self._supervised_metrics:
            return
        totals = torch.tensor(
            [
                self._supervised_metrics["open_set_ce"][0],
                self._supervised_metrics["open_set_ce_patches"][0],
                self._supervised_metrics["open_set_mse"][0],
                self._supervised_metrics["open_set_mse_patches"][0],
            ],
            dtype=torch.float64,
            device=self.device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals, group=self.dp_process_group)
        ce_sum, ce_count, mse_sum, mse_count = totals.tolist()
        metrics = {
            "open_set_ce": ce_sum / ce_count if ce_count else 0.0,
            "open_set_ce_patches": ce_count / self._NUM_AUGMENTED_VIEWS,
            "open_set_mse": mse_sum / mse_count if mse_count else 0.0,
            "open_set_mse_patches": mse_count / self._NUM_AUGMENTED_VIEWS,
        }
        self.log_extra_metrics(
            {f"train/{key}": value for key, value in metrics.items()},
            reduce_type=None,
        )

    def _global_patch_counts(self, metrics: dict[str, float]) -> dict[str, float]:
        """Sum valid classification and regression patch counts across DP ranks."""
        counts = torch.tensor(
            [
                metrics.get("open_set_ce_patches", 0.0),
                metrics.get("open_set_mse_patches", 0.0),
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
        """Combine local means so DP gradient averaging yields global patch means."""
        loss = losses["zero_touch"]
        global_counts = self._global_patch_counts(metrics)
        world_size = get_world_size(self.dp_process_group)
        for key in ("open_set_ce", "open_set_mse"):
            local_count = metrics.get(f"{key}_patches", 0.0)
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
        torch.Tensor, TokensAndMasks, TokensAndMasks, TokensAndMasks, torch.Tensor
    ]:
        """Run the base forward, then add the supervised probe loss."""
        loss, latent, decoded, target_output, pooled = super().model_forward(
            batch, patch_size, token_exit_cfg
        )
        # The probe lives inside the model so DDP/optimizer cover its params. It always
        # returns a probe-connected loss (a zero-touch term when a rank has no labeled
        # patches) so every rank produces gradients for the probe params each step.
        # Re-enter the model forward context because the base method has already exited
        # it. Production DDP uses bf16 autocast, and the probe's fp32 parameters must be
        # autocast together with the encoder's bf16 latent representations.
        with self._model_forward_context():
            sup_losses, sup_metrics = self.model.open_set_probe(latent, batch)
        sup_loss = self._combine_supervised_losses(sup_losses, sup_metrics)
        loss = loss + self.sup_loss_weight * sup_loss
        if sup_metrics:
            self._accumulate_supervised_metrics(sup_metrics)
        return loss, latent, decoded, target_output, pooled


@dataclass
class OpenSetLatentMIMTrainModuleConfig(ContrastiveLatentMIMTrainModuleConfig):
    """Configuration for :class:`OpenSetLatentMIMTrainModule`."""

    sup_loss_weight: float = 1.0

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
