"""Train module for the Set-Latent Perceiver (SLP) SSL encoder.

Much thinner than :class:`LatentMIMTrainModule`: the SLP masks internally and
computes its own soft-InfoNCE loss against frozen random targets, so there is no
separate target-encoder forward and ``update_target_encoder`` is a no-op. Each
microbatch is simply ``loss, metrics = model(batch, ...)`` -> ``backward``.

Distributed correctness (spec S6.5): the nested-K draw uses a rank-free seed
(``global_step``) so K is identical across DDP ranks in a step, while the mask
draw uses a per-rank seed so distinct data gets distinct masks.
"""

from dataclasses import dataclass
from logging import getLogger

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from olmo_core.distributed.parallel import DataParallelConfig
from olmo_core.distributed.utils import get_local_tensor, get_rank
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.common import ReduceType

from olmoearth_pretrain.config import require_olmo_core
from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.nn.set_latent_perceiver import SetLatentPerceiver
from olmoearth_pretrain.train.train_module.train_module import (
    OlmoEarthTrainModule,
    OlmoEarthTrainModuleConfig,
)
from olmoearth_pretrain.train.utils import split_masked_batch

require_olmo_core("Set-Latent Perceiver training")

logger = getLogger(__name__)

# Strides keeping mask seeds disjoint across (step, microbatch, rank): steps
# occupy the low bits, microbatches shift by 2**20 (no collision for runs under
# ~1M steps), ranks by 2**40 (global rank, so multi-node ranks stay distinct).
_MASK_SEED_MICROBATCH_STRIDE = 2**20
_MASK_SEED_RANK_STRIDE = 2**40


@dataclass
class SetLatentPerceiverTrainModuleConfig(OlmoEarthTrainModuleConfig):
    """Configuration for :class:`SetLatentPerceiverTrainModule`."""

    max_grad_norm: float = 1.0

    def build(
        self,
        model: SetLatentPerceiver,
        device: torch.device | None = None,
    ) -> "SetLatentPerceiverTrainModule":
        """Build the corresponding :class:`SetLatentPerceiverTrainModule`."""
        kwargs = self.prepare_kwargs()
        if kwargs.pop("regularizer_config", None) is not None:
            raise ValueError(
                "regularizer_config is not supported by the SLP train loop"
            )
        return SetLatentPerceiverTrainModule(model=model, device=device, **kwargs)


class SetLatentPerceiverTrainModule(OlmoEarthTrainModule):
    """Thin SSL train module driving :class:`SetLatentPerceiver`."""

    def __init__(
        self,
        model: SetLatentPerceiver,
        optim_config: OptimConfig,
        transform_config: TransformConfig,
        rank_microbatch_size: int,
        compile_model: bool = False,
        dp_config: DataParallelConfig | None = None,
        compile_loss: bool = False,
        autocast_precision: torch.dtype | None = None,
        max_grad_norm: float | None = None,
        scheduler: Scheduler | None = None,
        device: torch.device | None = None,
        state_dict_save_opts: dist_cp_sd.StateDictOptions | None = None,
        state_dict_load_opts: dist_cp_sd.StateDictOptions | None = None,
        find_unused_parameters: bool = True,
    ):
        """Initialize the train module. See ``OlmoEarthTrainModule`` for args."""
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

    def update_target_encoder(self) -> None:
        """No-op: the SLP uses fixed frozen random targets (no EMA teacher)."""
        return None

    def train_batch(
        self,
        batch: tuple[int, MaskedOlmoEarthSample],
        dry_run: bool = False,
    ) -> None:
        """Train a batch (a ``(patch_size, MaskedOlmoEarthSample)`` tuple)."""
        self.model.train()
        patch_size = batch[0]
        batch_data = batch[1]

        # Rank-free K seed (same across ranks per step); per-rank mask seed.
        step = int(getattr(self.trainer, "global_step", 0))
        k_seed = step
        mask_seed = step + get_rank() * _MASK_SEED_RANK_STRIDE

        masked_microbatches = split_masked_batch(batch_data, self.rank_microbatch_size)
        num_microbatches = len(masked_microbatches)

        total_batch_loss = torch.zeros([], device=self.device)
        total_correct = 0
        total_targets = 0

        for microbatch_idx in range(num_microbatches):
            with self._train_microbatch_context(microbatch_idx, num_microbatches):
                masked_batch = masked_microbatches[microbatch_idx].to_device(
                    self.device
                )
                with self._model_forward_context():
                    loss, metrics = self.model(
                        masked_batch,
                        patch_size,
                        mask_seed=mask_seed
                        + microbatch_idx * _MASK_SEED_MICROBATCH_STRIDE,
                        k_seed=k_seed,
                    )
                loss = loss / num_microbatches
                total_batch_loss += get_local_tensor(loss.detach())
                total_correct += sum(metrics["group_correct"].values())
                total_targets += sum(metrics["group_total"].values())

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    # Fail fast: backpropagating a non-finite loss NaNs the
                    # clip coefficient and then every parameter, silently.
                    raise RuntimeError(
                        f"NaN/Inf loss at step {step}, microbatch "
                        f"{microbatch_idx}, rank {get_rank()}"
                    )
                loss.backward()

        if dry_run:
            return

        self.trainer.record_metric("train/loss", total_batch_loss, ReduceType.mean)
        top1 = (total_correct / total_targets) if total_targets > 0 else 0.0
        self.trainer.record_metric(
            "train/top1", torch.tensor(top1, device=self.device), ReduceType.mean
        )
        self.trainer.record_metric(
            "train/target_count",
            torch.tensor(float(total_targets), device=self.device),
            ReduceType.mean,
        )
