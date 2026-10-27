"""Per-module-group gradient norms, logged every ``interval`` steps.

The total grad norm hides which part of the model is misbehaving (the d768 narrow
arms: the slot merge's GRU carried ~900x the gradient of the d256 arm's while the
trunk looked normal). Parameters are grouped by name prefix; the norms are taken
over all ranks (sharded gradients' squared norms are summed).
"""

from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from olmo_core.train.callbacks import Callback
from olmo_core.train.common import ReduceType

DEFAULT_GROUPS = (
    "encoder.patch_embeddings",
    "encoder.narrow_stage",
    "encoder.slot_merge.gru",
    "encoder.slot_merge",
    "encoder.narrow_up",
    "encoder.blocks",
    "encoder.perceiver",
    "decoder",
)


@dataclass
class GroupGradNormCallback(Callback):
    """Log ``grad_norm/<group>`` for each name-prefix group (first match wins)."""

    interval: int = 50
    groups: tuple[str, ...] = field(default=DEFAULT_GROUPS)

    def pre_optim_step(self) -> None:  # noqa: D102
        if self.trainer.global_step % self.interval:
            return
        model = self.trainer.train_module.model
        sums = None
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            group = next(
                (i for i, g in enumerate(self.groups) if name.startswith(g)), None
            )
            if group is None:
                continue
            g = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
            if sums is None:
                sums = torch.zeros(
                    len(self.groups), device=g.device, dtype=torch.float32
                )
            sums[group] += g.float().pow(2).sum()
        if sums is None:
            return
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sums)
        for prefix, value in zip(self.groups, sums.sqrt().tolist()):
            self.trainer.record_metric(
                f"grad_norm/{prefix.replace('encoder.', '')}", value, ReduceType.max
            )
