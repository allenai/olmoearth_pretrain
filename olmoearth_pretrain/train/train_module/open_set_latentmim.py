"""Latent-MIM train module with a supervised open-set probe.

Extends :class:`LatentMIMTrainModule` (the single-forward recipe used by the v1.3
Perceiver runs) by adding a supervised segmentation + regression loss (see
:class:`olmoearth_pretrain.train.open_set_probe.OpenSetProbe`) on top of the
self-supervised objective. The probe reads the encoder's *register grid* (the
Perceiver output, ``[B, n_h, n_w, D]``), so the supervised gradient flows back
into the Perceiver and the encoder.

The probe itself lives inside the model
(:class:`olmoearth_pretrain.nn.open_set_latent_mim.OpenSetLatentMIM`) so that the
DDP gradient all-reduce and the optimizer cover its parameters. Its loss is a
per-rank mean over labeled samples and its metrics ride on ``extra_metrics``,
exactly like the v1.3 map supervision heads.

Post-training schedule
----------------------

With ``freeze_backbone_until_step = N > 0`` the module runs a single-run,
two-phase schedule for continuing from a pretrained checkpoint whose model lacks
the probe:

1. ``global_step < N`` (linear-probe phase): every parameter except the probe is
   frozen. Only the encoder runs (under ``no_grad``) to produce the register grid;
   the decoder / target encoder / map supervision / student losses are skipped
   since none of them can produce a gradient.
2. ``global_step >= N`` (fine-tune phase): the backbone unfreezes and the probe is
   frozen (``freeze_probe_after_unfreeze``). The full v1.3 forward runs (latent-MIM
   + map supervision + student distillation) plus the supervised loss, whose
   gradient now reaches the backbone through the fixed probe.
"""

from dataclasses import dataclass
from logging import getLogger
from typing import Any

import torch

from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, TokensAndMasks
from olmoearth_pretrain.nn.utils import unpack_encoder_output
from olmoearth_pretrain.train.train_module.latent_mim import (
    LatentMIMTrainModule,
    LatentMIMTrainModuleConfig,
)

logger = getLogger(__name__)


class OpenSetLatentMIMTrainModule(LatentMIMTrainModule):
    """Latent-MIM plus a supervised open-set probe loss on the register grid."""

    def __init__(
        self,
        *args: Any,
        sup_loss_weight: float = 1.0,
        freeze_backbone_until_step: int = 0,
        freeze_probe_after_unfreeze: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize, extracting the supervised loss weight and freeze schedule.

        Args:
            *args: Positional arguments forwarded to the base train module.
            sup_loss_weight: Scalar weight applied to the combined supervised
                (CE + MSE) loss when added to the self-supervised objective.
            freeze_backbone_until_step: If > 0, train ONLY the open-set probe
                until this global step, then unfreeze the rest of the model (the
                linear-probe phase of the post-training recipe, for runs
                initialized from a pretrained checkpoint via ``init_weights_path``).
            freeze_probe_after_unfreeze: If True (and the schedule is active), the
                probe is frozen once the backbone unfreezes, so the second phase
                trains the backbone against a fixed probe.
            **kwargs: Keyword arguments forwarded to the base train module.
        """
        super().__init__(*args, **kwargs)
        self.sup_loss_weight = sup_loss_weight
        self.freeze_backbone_until_step = freeze_backbone_until_step
        self.freeze_probe_after_unfreeze = freeze_probe_after_unfreeze
        self.total_loss_name = f"{self.total_loss_name}+open_set"
        # Params frozen at init (e.g. FrozenTargetProjection copies) must never
        # be flipped trainable by the freeze schedule.
        self._always_frozen_param_ids = {
            id(p) for p in self.model.parameters() if not p.requires_grad
        }
        self._backbone_frozen: bool | None = None

    # ------------------------------------------------------------------
    # Freeze schedule
    # ------------------------------------------------------------------
    @property
    def backbone_frozen(self) -> bool:
        """Whether the current step is in the probe-only phase."""
        return bool(self._backbone_frozen)

    def _apply_freeze_schedule(self) -> None:
        """Freeze/unfreeze the backbone and probe based on the global step.

        Before ``freeze_backbone_until_step`` only the open-set probe trains;
        afterwards the backbone trains and (with ``freeze_probe_after_unfreeze``)
        the probe is frozen. ``requires_grad`` is toggled lazily AFTER the optimizer
        was built, so every param remains in the optimizer throughout and simply
        resumes/stops updating on the flip (AdamW, fused included, skips params
        whose ``grad`` is ``None``). The flip is keyed on the global step, so all
        DP ranks toggle together and the replicated-DDP gradient all-reduce sees
        identical grad sets on every rank. Params that were already frozen at init
        (the projection-only target copies) are never unfrozen.
        """
        if self.freeze_backbone_until_step <= 0:
            return
        freeze = self.trainer.global_step < self.freeze_backbone_until_step
        if freeze == self._backbone_frozen:
            return
        probe_trainable = freeze or not self.freeze_probe_after_unfreeze
        probe_param_ids = {id(p) for p in self.model.open_set_probe.parameters()}
        num_toggled = 0
        for p in self.model.parameters():
            if id(p) in self._always_frozen_param_ids:
                continue
            if id(p) in probe_param_ids:
                p.requires_grad_(probe_trainable)
                continue
            p.requires_grad_(not freeze)
            num_toggled += 1
        self._backbone_frozen = freeze
        logger.info(
            "open-set freeze schedule: %s %d backbone params at step %d "
            "(freeze_backbone_until_step=%d, probe %s)",
            "froze" if freeze else "unfroze",
            num_toggled,
            self.trainer.global_step,
            self.freeze_backbone_until_step,
            "trainable" if probe_trainable else "frozen",
        )

    def train_batch(
        self,
        batch: tuple[int, MaskedOlmoEarthSample],
        dry_run: bool = False,
    ) -> None:
        """Apply the freeze schedule, then train the batch."""
        self._apply_freeze_schedule()
        super().train_batch(batch, dry_run=dry_run)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _probe_loss(
        self, register_grid: torch.Tensor | None, batch: MaskedOlmoEarthSample
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run the probe on the register grid; return its weighted loss + metrics."""
        if register_grid is None:
            raise RuntimeError(
                "OpenSetLatentMIMTrainModule requires the encoder Perceiver: the "
                "open-set probe reads the register grid (set "
                "encoder_config.perceiver_config)"
            )
        # The probe lives inside the model so DDP/optimizer cover its params. It
        # always returns a probe-connected loss (a zero-touch term when a rank has no
        # labeled patches) so every rank produces gradients for the probe params each
        # step. Production DDP uses bf16 autocast, and the probe's fp32 parameters
        # must be autocast together with the encoder's bf16 register grid.
        with self._model_forward_context():
            sup_loss, sup_metrics = self.model.open_set_probe(register_grid, batch)
        # Logged as open_set/{ce,ce_samples,ce_patches,mse,...}; the base train_batch
        # averages each key over the microbatches that reported it, so the counts are
        # per-microbatch means and the losses are means over labeled microbatches.
        metrics: dict[str, Any] = {
            f"open_set/{key.removeprefix('open_set_')}": value
            for key, value in sup_metrics.items()
        }
        metrics["open_set/backbone_frozen"] = float(self.backbone_frozen)
        return self.sup_loss_weight * sup_loss, metrics

    def _probe_only_forward(
        self,
        batch: MaskedOlmoEarthSample,
        patch_size: int,
    ) -> tuple[
        torch.Tensor,
        TokensAndMasks,
        TokensAndMasks,
        TokensAndMasks,
        dict[str, Any] | None,
    ]:
        """Linear-probe phase forward: frozen encoder -> registers -> probe loss.

        Everything but the probe is frozen, so the decoder, target encoder, map
        supervision and student losses cannot produce gradients and are skipped.
        The ``latent`` slot of the return tuple is the encoder output (so the
        regularizer hook still has something to look at); the ``decoded`` and
        ``target`` slots reuse it, as there is no decoder pass.
        """
        with self._model_forward_context(), torch.no_grad():
            output_dict = self.model.encoder(batch, patch_size=patch_size)
        register_grid = output_dict.get("registers")
        latent, _, _ = unpack_encoder_output(output_dict)
        loss, metrics = self._probe_loss(register_grid, batch)
        return loss, latent, latent, latent, metrics

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
        if self.backbone_frozen:
            return self._probe_only_forward(batch, patch_size)

        loss, latent, decoded, target_output, extra_metrics = super().model_forward(
            batch, patch_size, token_exit_cfg
        )
        # The probe reads the encoder's register grid, which the model forward
        # stashes on the model.
        sup_loss, sup_metrics = self._probe_loss(
            getattr(self.model, "last_register_grid", None), batch
        )
        extra_metrics = {**(extra_metrics or {}), **sup_metrics}
        return loss + sup_loss, latent, decoded, target_output, extra_metrics


@dataclass
class OpenSetLatentMIMTrainModuleConfig(LatentMIMTrainModuleConfig):
    """Configuration for :class:`OpenSetLatentMIMTrainModule`."""

    sup_loss_weight: float = 1.0
    freeze_backbone_until_step: int = 0
    freeze_probe_after_unfreeze: bool = True

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
