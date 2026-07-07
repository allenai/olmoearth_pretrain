"""Callback that logs the ERA5 encoder's learned per-band mask embedding.

The encoder's ``mask_embed`` parameter (shape ``[1, 1, V]``) is one learned
scalar per input band. This logs it to wandb as a bar plot (band on the
x-axis) on a step cadence. No-op when ``use_mask_embed`` is disabled.
"""

from __future__ import annotations

from dataclasses import dataclass

from olmo_core.distributed.utils import get_rank
from olmo_core.train.callbacks.callback import Callback, CallbackConfig
from olmo_core.train.trainer import Trainer

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.train.callbacks.era5_evaluator_callback import (
    _get_encoder,
    _get_wandb_callback,
)


class Era5MaskEmbedVizCallback(Callback):
    """Logs the learned per-band mask embedding as a wandb bar plot."""

    def __init__(self, log_interval: int = 1000) -> None:
        """Store the logging cadence (in steps) for the mask-embedding plot."""
        super().__init__()
        self.log_interval = log_interval

    def post_step(self) -> None:
        """Log the mask-embedding bar plot when the interval has elapsed."""
        if self.log_interval > 0 and self.step % self.log_interval == 0:
            self._log(self.step)

    def _log(self, step: int) -> None:
        if get_rank() != 0:
            return
        wandb_callback = _get_wandb_callback(self.trainer)
        mask_embed = getattr(_get_encoder(self.trainer), "mask_embed", None)
        if wandb_callback is None or mask_embed is None:
            return

        values = mask_embed.detach().float().reshape(-1).cpu().tolist()
        bands = Modality.ERA5L_DAY_10.band_order
        wandb = wandb_callback.wandb

        # The mask embedding has one entry per input channel. For a raw-input
        # encoder that is one per band; for an SWT-input encoder it is
        # ``V * n_bands`` entries in var-major layout (``c = v*n_bands + s``),
        # so build composite ``<band>_s<scale>`` labels.
        if len(values) == len(bands):
            labels = list(bands)
        elif len(bands) > 0 and len(values) % len(bands) == 0:
            n_bands = len(values) // len(bands)
            labels = [
                f"{band}_s{scale}" for band in bands for scale in range(n_bands)
            ]
        else:
            labels = [str(i) for i in range(len(values))]

        table = wandb.Table(
            data=list(zip(labels, values)), columns=["band", "mask_embed"]
        )
        wandb.log(
            {"mask_embed/per_band": wandb.plot.bar(table, "band", "mask_embed")},
            step=step,
        )


@dataclass
class Era5MaskEmbedVizCallbackConfig(CallbackConfig):
    """Config for the ERA5 mask-embedding visualization callback."""

    enabled: bool = True
    log_interval: int = 1000

    def build(self, trainer: Trainer) -> Callback | None:
        """Build the callback, or return None if disabled."""
        if not self.enabled:
            return None
        return Era5MaskEmbedVizCallback(log_interval=self.log_interval)
