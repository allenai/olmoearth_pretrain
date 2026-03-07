"""Learnable loss weights using homoscedastic uncertainty (Kendall et al., 2018).

Each loss component is weighted by ``exp(-s) * loss + s`` where ``s = log(sigma^2)``
is a learnable scalar parameter.  The ``+s`` regularizer prevents any weight from
collapsing to zero.  Tasks with inherently larger losses learn a larger ``s``,
automatically down-weighting them.

Reference:
    Kendall, Gal & Cipolla, "Multi-Task Learning Using Uncertainty to Weigh
    Losses for Scene Geometry and Semantics", CVPR 2018.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from olmoearth_pretrain.config import Config

logger = logging.getLogger(__name__)


class LearnableLossWeights(nn.Module):
    """Per-loss-component learnable uncertainty weights."""

    def __init__(self, loss_names: list[str]) -> None:
        """Per-loss-component learnable uncertainty weights."""
        super().__init__()
        # s = log(sigma^2), initialised to 0 so initial weight = exp(0) = 1.
        self.log_vars = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(1)) for name in loss_names}
        )

    def weight_loss(self, name: str, loss: Tensor) -> Tensor:
        """Apply uncertainty weighting: exp(-s) * loss + s."""
        s = self.log_vars[name]
        return (torch.exp(-s) * loss + s).squeeze()


@dataclass
class LearnableLossWeightsConfig(Config):
    """Configuration for :class:`LearnableLossWeights`.

    Args:
        loss_names: Names of the loss components that will receive learnable weights.
    """

    loss_names: list[str] = field(default_factory=list)

    def build(self) -> LearnableLossWeights:
        """Build the learnable loss weights module."""
        return LearnableLossWeights(loss_names=self.loss_names)
