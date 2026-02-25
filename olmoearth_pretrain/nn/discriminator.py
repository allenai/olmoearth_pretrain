"""Token-level discriminator for adversarial training."""

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from olmoearth_pretrain.config import Config

logger = logging.getLogger(__name__)


class TokenDiscriminator(nn.Module):
    """MLP discriminator that classifies token embeddings as real or fake.

    Operates on individual token embeddings from the decoder output space.
    Real tokens come from the target encoder; fake tokens come from the decoder.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = input_dim
        for i in range(num_layers - 1):
            linear = nn.Linear(in_features, hidden_dim)
            if use_spectral_norm:
                linear = nn.utils.parametrizations.spectral_norm(linear)
            layers.append(linear)
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            in_features = hidden_dim

        final_linear = nn.Linear(hidden_dim, 1)
        if use_spectral_norm:
            final_linear = nn.utils.parametrizations.spectral_norm(final_linear)
        layers.append(final_linear)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify tokens as real or fake.

        Args:
            x: Token embeddings of shape [N, D] where N is the number of tokens.

        Returns:
            Logits of shape [N, 1]. Positive values indicate "real".
        """
        return self.net(x)

    def apply_fsdp(
        self,
        mesh: DeviceMesh | None = None,
        mp_policy: MixedPrecisionPolicy | None = None,
    ) -> None:
        """Apply FSDP wrapping to the discriminator."""
        if mp_policy is None:
            mp_policy = MixedPrecisionPolicy()
        fully_shard(self, mesh=mesh, mp_policy=mp_policy)


@dataclass
class TokenDiscriminatorConfig(Config):
    """Configuration for the token discriminator.

    Args:
        input_dim: Dimension of input token embeddings (should match decoder output dim).
        hidden_dim: Hidden layer dimension.
        num_layers: Number of MLP layers.
        use_spectral_norm: Whether to apply spectral normalization for training stability.
        target_modality: Which modality to apply the discriminator to.
        weight: Scalar weight for the adversarial loss.
        disc_lr: Learning rate for the discriminator optimizer.
        disc_weight_decay: Weight decay for the discriminator optimizer.
        label_smoothing: One-sided label smoothing for real labels (e.g., 0.9 instead of 1.0).
        n_disc_steps: Number of discriminator updates per generator update.
    """

    input_dim: int = 768
    hidden_dim: int = 256
    num_layers: int = 3
    use_spectral_norm: bool = True
    target_modality: str = "naip_10"
    weight: float = 0.1
    disc_lr: float = 1e-4
    disc_weight_decay: float = 0.01
    label_smoothing: float = 0.0
    n_disc_steps: int = 1

    def build(self) -> TokenDiscriminator:
        """Build the discriminator."""
        return TokenDiscriminator(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            use_spectral_norm=self.use_spectral_norm,
        )
