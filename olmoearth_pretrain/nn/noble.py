"""NOBLE: Nonlinear lOw-rank Branch for Linear Enhancement.

Implementation based on: "NOBLE: Accelerating Transformers with Nonlinear Low-Rank Branches"
(arXiv:2603.06492)

NOBLE augments transformer linear layers with nonlinear low-rank branches that compute:
    output = xW + σ(xW_down)W_up

Where σ is a learnable nonlinearity. The best performing activation is CosNet,
a two-layer cosine nonlinearity with learnable frequency and phase.

Key results from the paper:
- Up to 1.47x step speedup to reach baseline eval loss
- As low as 4% additional parameters with 7% step time overhead
- Up to 1.22x net wallclock speedup

Note: The paper found that Mixup/CutMix and stochastic augmentations interfere
with NOBLE's benefits. Consider disabling these when using NOBLE.
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from olmoearth_pretrain.config import Config


class CosNet(nn.Module):
    """CosNet: Two-layer cosine nonlinearity with learnable frequency and phase.
    
    Computes: cos(w2 * (cos(w1 * x + b1) @ W_mid) + b2)
    
    where w1, w2 are learnable frequency scalars and b1, b2 are learnable phase biases.
    """

    def __init__(self, hidden_dim: int, init_freq: float = 1.0):
        """Initialize CosNet.
        
        Args:
            hidden_dim: Dimension of the hidden space (bottleneck dimension).
            init_freq: Initial value for the learnable frequency parameters.
        """
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor(init_freq))
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.w_mid = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w2 = nn.Parameter(torch.tensor(init_freq))
        self.b2 = nn.Parameter(torch.zeros(hidden_dim))
        
        nn.init.orthogonal_(self.w_mid.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CosNet.
        
        Args:
            x: Input tensor of shape (..., hidden_dim)
            
        Returns:
            Output tensor of shape (..., hidden_dim)
        """
        x = torch.cos(self.w1 * x + self.b1)
        x = self.w_mid(x)
        x = torch.cos(self.w2 * x + self.b2)
        return x


class GELUActivation(nn.Module):
    """Simple GELU wrapper for consistency with other activations."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.gelu(x)


class NobleBranch(nn.Module):
    """Low-rank nonlinear branch for NOBLE.
    
    Computes: σ(xW_down)W_up
    
    where σ is a learnable nonlinearity (CosNet by default).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        activation: Literal["cosnet", "gelu"] = "cosnet",
        init_scale: float = 0.01,
    ):
        """Initialize the NOBLE branch.
        
        Args:
            in_features: Input dimension.
            out_features: Output dimension.
            rank: Rank of the low-rank approximation (bottleneck dimension).
            activation: Type of activation ("cosnet" or "gelu").
            init_scale: Scale for initializing W_up to near-zero.
        """
        super().__init__()
        self.w_down = nn.Linear(in_features, rank, bias=False)
        
        if activation == "cosnet":
            self.activation = CosNet(rank)
        elif activation == "gelu":
            self.activation = GELUActivation()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.w_up = nn.Linear(rank, out_features, bias=False)
        
        nn.init.kaiming_normal_(self.w_down.weight, mode="fan_out", nonlinearity="linear")
        nn.init.normal_(self.w_up.weight, std=init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the NOBLE branch.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        return self.w_up(self.activation(self.w_down(x)))


class NobleLinear(nn.Module):
    """Linear layer augmented with a NOBLE branch.
    
    Computes: xW + b + σ(xW_down)W_up
    
    where the base linear layer computes xW + b and the NOBLE branch adds
    the nonlinear low-rank term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank: int | None = None,
        rank_ratio: float = 0.25,
        activation: Literal["cosnet", "gelu"] = "cosnet",
        init_scale: float = 0.01,
    ):
        """Initialize NobleLinear.
        
        Args:
            in_features: Input dimension.
            out_features: Output dimension.
            bias: Whether to include bias in the main linear layer.
            rank: Explicit rank for the low-rank branch. If None, uses rank_ratio.
            rank_ratio: Ratio of rank to min(in_features, out_features). Default 0.25.
            activation: Activation type for the branch ("cosnet" or "gelu").
            init_scale: Scale for W_up initialization.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        if rank is None:
            rank = max(1, int(min(in_features, out_features) * rank_ratio))
        
        self.branch = NobleBranch(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            activation=activation,
            init_scale=init_scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        return self.linear(x) + self.branch(x)


@dataclass
class NobleConfig(Config):
    """Configuration for NOBLE (Nonlinear Low-Rank Branches).
    
    NOBLE augments transformer linear layers with nonlinear low-rank branches
    that can accelerate training by up to 1.47x.
    
    Attributes:
        enabled: Whether to enable NOBLE branches.
        rank_ratio: Ratio of branch rank to layer dimension. Default 0.25 means
            the bottleneck is 25% of min(in_features, out_features).
        activation: Activation function for the branch. "cosnet" performs best
            according to the paper.
        init_scale: Scale for initializing W_up weights. Small values ensure
            the branch starts near identity.
        apply_to_qkv: Apply NOBLE to Q, K, V projections in attention.
        apply_to_proj: Apply NOBLE to output projection in attention.
        apply_to_mlp: Apply NOBLE to MLP linear layers.
    """
    
    enabled: bool = True
    rank_ratio: float = 0.25
    activation: Literal["cosnet", "gelu"] = "cosnet"
    init_scale: float = 0.01
    apply_to_qkv: bool = True
    apply_to_proj: bool = True
    apply_to_mlp: bool = True

    def validate(self) -> None:
        """Validate the config."""
        if self.rank_ratio <= 0 or self.rank_ratio > 1:
            raise ValueError(f"rank_ratio must be in (0, 1], got {self.rank_ratio}")
        if self.activation not in ("cosnet", "gelu"):
            raise ValueError(f"activation must be 'cosnet' or 'gelu', got {self.activation}")
        if self.init_scale <= 0:
            raise ValueError(f"init_scale must be positive, got {self.init_scale}")

    def make_linear(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        layer_type: Literal["qkv", "proj", "mlp"] = "mlp",
    ) -> nn.Module:
        """Create a linear layer, optionally with NOBLE branch.
        
        Args:
            in_features: Input dimension.
            out_features: Output dimension.
            bias: Whether to include bias.
            layer_type: Type of layer ("qkv", "proj", or "mlp").
            
        Returns:
            Either nn.Linear or NobleLinear depending on config.
        """
        should_apply = self.enabled and (
            (layer_type == "qkv" and self.apply_to_qkv) or
            (layer_type == "proj" and self.apply_to_proj) or
            (layer_type == "mlp" and self.apply_to_mlp)
        )
        
        if should_apply:
            return NobleLinear(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
                rank_ratio=self.rank_ratio,
                activation=self.activation,
                init_scale=self.init_scale,
            )
        else:
            return nn.Linear(in_features, out_features, bias=bias)


def get_noble_config(noble_config: NobleConfig | None) -> NobleConfig:
    """Get a NobleConfig, returning a disabled one if None."""
    if noble_config is None:
        return NobleConfig(enabled=False)
    return noble_config
