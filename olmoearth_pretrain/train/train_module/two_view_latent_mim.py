"""Two-view LatentMIM: per-view latent MIM plus an instance-contrastive term (E5 arm C).

The batch is ``(patch_size, masked_a, masked_b)``, where row ``i`` of the two views is
the same sample under two independent date-draws (see
``olmoearth_pretrain.data.collate.collate_two_draw_masked_batched``). Each view runs
:class:`LatentMIMTrainModule`'s own forward and loss unchanged (pixel targets,
projection-only target, supervision heads, detached student, regularizer); the two
per-view losses are averaged, and the contrastive loss between the two views' pooled
instance embeddings is added exactly as
:class:`~olmoearth_pretrain.train.train_module.contrastive_latentmim.ContrastiveLatentMIMTrainModule`
does it: ``loss = (loss_a + loss_b) / 2 + contrastive(pooled_a, pooled_b)``, with
in-microbatch negatives.
"""

from dataclasses import dataclass
from logging import getLogger
from typing import Any

import torch
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.train.common import ReduceType

from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.nn.latent_mim import LatentMIM
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.train_module.latent_mim import (
    LatentMIMTrainModule,
    LatentMIMTrainModuleConfig,
)
from olmoearth_pretrain.train.utils import split_masked_batch

logger = getLogger(__name__)


@dataclass
class TwoViewLatentMIMTrainModuleConfig(LatentMIMTrainModuleConfig):
    """:class:`LatentMIMTrainModuleConfig` plus a contrastive loss between two views.

    Args:
        contrastive_config: Loss between the views' pooled embeddings (e.g. v1.2's
            ``InfoNCE`` with ``weight=0.05``). None trains the two views with MIM only.
    """

    contrastive_config: LossConfig | None = None

    def build(
        self,
        model: LatentMIM,
        device: torch.device | None = None,
    ) -> "TwoViewLatentMIMTrainModule":
        """Build the corresponding :class:`TwoViewLatentMIMTrainModule`."""
        kwargs = self.prepare_kwargs()
        return TwoViewLatentMIMTrainModule(
            model=model,
            device=device,
            **kwargs,
        )


class TwoViewLatentMIMTrainModule(LatentMIMTrainModule):
    """LatentMIM on two views of each sample with an instance-contrastive term."""

    def __init__(
        self, *args: Any, contrastive_config: LossConfig | None = None, **kwargs: Any
    ):
        """Initialize; all other arguments are :class:`LatentMIMTrainModule`'s."""
        super().__init__(*args, **kwargs)
        self.contrastive_loss = (
            contrastive_config.build() if contrastive_config is not None else None
        )

    def train_batch(
        self,
        batch: tuple[int, MaskedOlmoEarthSample, MaskedOlmoEarthSample],
        dry_run: bool = False,
    ) -> None:
        """Train on a ``(patch_size, masked_a, masked_b)`` batch.

        Logged: ``train/<total_loss_name>`` is the mean over the two views of the
        per-view LatentMIM loss (comparable to a one-view run), ``train/contrastive``
        the weighted contrastive term, ``train/total`` their sum (what is minimized).
        """
        if not dry_run:
            self.update_target_encoder()
        self.model.train()
        total_batch_view_loss = torch.zeros([], device=self.device)
        total_batch_reg = torch.zeros([], device=self.device)
        total_batch_con = torch.zeros([], device=self.device)
        total_batch_loss = torch.zeros([], device=self.device)
        total_nonfinite = torch.zeros([], device=self.device)
        accumulated_extra_metrics: dict[str, Any] = {}
        extra_metric_counts: dict[str, int] = {}

        patch_size = batch[0]
        microbatches_a = split_masked_batch(batch[1], self.rank_microbatch_size)
        microbatches_b = split_masked_batch(batch[2], self.rank_microbatch_size)
        num_microbatches = len(microbatches_a)
        assert len(microbatches_b) == num_microbatches, "views differ in batch size"

        for microbatch_idx in range(num_microbatches):
            with self._train_microbatch_context(microbatch_idx, num_microbatches):
                masked_batch_a = microbatches_a[microbatch_idx].to_device(self.device)
                masked_batch_b = microbatches_b[microbatch_idx].to_device(self.device)
                logger.info(
                    f"Training microbatch {microbatch_idx} of {num_microbatches} "
                    f"with batch size {masked_batch_a.batch_size} (x2 views)"
                )

                view_losses = []
                pooled = []
                for masked_batch in (masked_batch_a, masked_batch_b):
                    loss_v, latent_v, _, _, extra_metrics, pooled_v = (
                        self.model_forward(
                            masked_batch,
                            patch_size,
                            self.token_exit_cfg,
                            return_pooled=True,
                        )
                    )
                    if extra_metrics is not None:
                        self.accumulate_extra_metrics(
                            accumulated_extra_metrics,
                            extra_metric_counts,
                            extra_metrics,
                        )
                    reg_term = self.compute_regularization(latent_v)
                    if reg_term is not None:
                        loss_v = loss_v + reg_term
                        total_batch_reg += (
                            get_local_tensor(reg_term.detach()) / num_microbatches / 2
                        )
                    view_losses.append(loss_v)
                    pooled.append(pooled_v)
                    del latent_v

                loss = (view_losses[0] + view_losses[1]) / 2
                total_batch_view_loss += (
                    get_local_tensor(loss.detach()) / num_microbatches
                )

                if self.contrastive_loss is not None:
                    # fp32, outside autocast, like the per-view losses.
                    with torch.autocast(torch.device(self.device).type, enabled=False):
                        contrastive = self.contrastive_loss.compute(
                            pooled[0].float(), pooled[1].float()
                        )
                    loss = loss + contrastive
                    total_batch_con += (
                        get_local_tensor(contrastive.detach()) / num_microbatches
                    )

                loss = loss / num_microbatches
                total_batch_loss += get_local_tensor(loss.detach())
                total_nonfinite += (~torch.isfinite(loss.detach())).any().float()
                loss.backward()

        if dry_run:
            return

        self.trainer.record_metric(
            f"train/{self.total_loss_name}", total_batch_view_loss, ReduceType.mean
        )
        self.trainer.record_metric("train/total", total_batch_loss, ReduceType.mean)
        if self.contrastive_loss is not None:
            self.trainer.record_metric(
                "train/contrastive", total_batch_con, ReduceType.mean
            )
        self.trainer.record_metric(
            "train/nonfinite_loss_microbatches", total_nonfinite, ReduceType.sum
        )
        self.log_accumulated_extra_metrics(
            accumulated_extra_metrics, extra_metric_counts
        )
        self.log_regularization(total_batch_reg)

        del batch, masked_batch_a, masked_batch_b
