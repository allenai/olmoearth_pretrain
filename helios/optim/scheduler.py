from olmo_core.optim.scheduler import Scheduler
from dataclasses import dataclass
from typing import Union
import torch


@dataclass
class LinearDecay(Scheduler):
    """
    Linear decay scheduler that decays from initial_lr to min_lr over a specified number of steps.

    Args:
        decay_steps: Number of steps over which to decay the learning rate
        min_lr: Minimum learning rate to decay to
        start_step: Step at which to start the decay (default: 0)
    """

    decay_steps: int = 30000
    min_lr: float = 0.0
    start_step: int = 0

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        if current < self.start_step:
            return initial_lr
        if current >= self.start_step + self.decay_steps:
            return self.min_lr

        # Linear decay from initial_lr to min_lr over decay_steps
        decay_ratio = (current - self.start_step) / self.decay_steps
        return initial_lr - (initial_lr - self.min_lr) * decay_ratio