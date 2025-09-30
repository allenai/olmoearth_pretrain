"""Single file architecture for Tessera model https://github.com/ucam-eo/tessera/tree/a883aa12392eb9fc237ae4c29824318760e138a2/tessera_infer/src/models."""

import math

import numpy as np
import torch
import torch.nn as nn

# Constants from config in https://github.com/ucam-eo/tessera/blob/alpha_version_1.0/tessera_infer/configs/multi_tile_infer_config.py
LATENT_DIM = 128
FUSION_METHOD = "concat"


class AttentionPooling(nn.Module):
    """Attention-based pooling layer that learns to weight sequence elements.

    This module applies attention mechanism to pool variable-length sequences
    into fixed-size representations by learning attention weights over the
    sequence dimension.
    """

    def __init__(self, input_dim: int) -> None:
        """Initialize the attention pooling layer.

        Args:
            input_dim: Input feature dimension for the attention query.
        """
        super().__init__()
        self.query = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention pooling to input sequence.

        Args:
            x: Input tensor of shape (B, seq_len, dim) where B is batch size,
               seq_len is sequence length, and dim is feature dimension.

        Returns:
            Pooled tensor of shape (B, dim) representing the weighted average
            of the input sequence.
        """
        # x: (B, seq_len, dim)
        w = torch.softmax(self.query(x), dim=1)  # (B, seq_len, 1)
        return (w * x).sum(dim=1)


class TemporalAwarePooling(nn.Module):
    """Temporal-aware pooling that considers temporal context through GRU.

    This module first processes the input sequence through a GRU to capture
    temporal dependencies, then applies attention pooling to the contextualized
    features.
    """

    def __init__(self, input_dim: int) -> None:
        """Initialize the temporal-aware pooling layer.

        Args:
            input_dim: Input feature dimension for both GRU and attention.
        """
        super().__init__()
        self.query = nn.Linear(input_dim, 1)
        self.temporal_context = nn.GRU(input_dim, input_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal-aware pooling to input sequence.

        Args:
            x: Input tensor of shape (B, seq_len, dim) where B is batch size,
               seq_len is sequence length, and dim is feature dimension.

        Returns:
            Pooled tensor of shape (B, dim) representing the temporal-aware
            weighted average of the input sequence.
        """
        # First capture temporal context through RNN
        x_context, _ = self.temporal_context(x)
        # Then calculate attention weights
        w = torch.softmax(self.query(x_context), dim=1)
        return (w * x).sum(dim=1)


class TemporalEncoding(nn.Module):
    """Learnable temporal encoding using Fourier features for day-of-year input.

    This module generates temporal encodings for day-of-year values using
    learnable frequency parameters and projects them to the target dimension.
    """

    def __init__(self, d_model: int, num_freqs: int = 64) -> None:
        """Initialize the temporal encoding layer.

        Args:
            d_model: Target dimension for the temporal encoding.
            num_freqs: Number of frequency components for Fourier features.
        """
        super().__init__()
        self.num_freqs = num_freqs
        self.d_model = d_model

        # Learnable frequency parameters (more flexible than fixed frequencies)
        self.freqs = nn.Parameter(
            torch.exp(torch.linspace(0, np.log(365.0), num_freqs))
        )

        # Project Fourier features to the target dimension through a linear layer
        self.proj = nn.Linear(2 * num_freqs, d_model)
        self.phase = nn.Parameter(torch.zeros(1, 1, d_model))  # Learnable phase offset

    def forward(self, doy: torch.Tensor) -> torch.Tensor:
        """Generate temporal encoding for day-of-year values.

        Args:
            doy: Day-of-year tensor of shape (B, seq_len, 1) where values
                 are in range [0, 365].

        Returns:
            Temporal encoding tensor of shape (B, seq_len, d_model).
        """
        # doy: (B, seq_len, 1)
        t = doy / 365.0 * 2 * np.pi  # Normalize to the 0-2π range

        # Generate multi-frequency sine/cosine features
        t_scaled = t * self.freqs.view(1, 1, -1)  # (B, seq_len, num_freqs)
        sin = torch.sin(t_scaled + self.phase[..., : self.num_freqs])
        cos = torch.cos(t_scaled + self.phase[..., self.num_freqs : 2 * self.num_freqs])

        # Concatenate and project to the target dimension
        encoding = torch.cat([sin, cos], dim=-1)  # (B, seq_len, 2*num_freqs)
        return self.proj(encoding)  # (B, seq_len, d_model)


class TemporalPositionalEncoder(nn.Module):
    """Sinusoidal positional encoding for temporal sequences.

    This module generates positional encodings for day-of-year values using
    sinusoidal functions, similar to the original Transformer positional encoding.
    """

    def __init__(self, d_model: int) -> None:
        """Initialize the temporal positional encoder.

        Args:
            d_model: Dimension of the positional encoding.
        """
        super().__init__()
        self.d_model = d_model

    def forward(self, doy: torch.Tensor) -> torch.Tensor:
        """Generate positional encoding for day-of-year values.

        Args:
            doy: Day-of-year tensor of shape (B, T) containing DOY values (0-365).

        Returns:
            Positional encoding tensor of shape (B, T, d_model).
        """
        # doy: [B, T] tensor containing DOY values (0-365)
        position = doy.unsqueeze(-1).float()  # Ensure float type
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / self.d_model)
        )
        div_term = div_term.to(doy.device)

        pe = torch.zeros(doy.shape[0], doy.shape[1], self.d_model, device=doy.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


class TransformerEncoder(nn.Module):
    """Transformer encoder for processing temporal satellite data sequences.

    This module processes satellite data sequences with temporal information,
    using a transformer encoder with temporal positional encoding and
    temporal-aware pooling.
    """

    def __init__(
        self,
        band_num: int,
        latent_dim: int,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 20,
    ) -> None:
        """Initialize the transformer encoder.

        Args:
            band_num: Number of spectral bands in the input data.
            latent_dim: Target latent dimension for the output representation.
            nhead: Number of attention heads in the transformer.
            num_encoder_layers: Number of transformer encoder layers.
            dim_feedforward: Dimension of the feedforward network.
            dropout: Dropout probability for regularization.
            max_seq_len: Maximum sequence length (currently unused but kept for compatibility).
        """
        super().__init__()
        # Total input dimension: bands
        input_dim = band_num

        # Embedding to increase dimension
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, latent_dim * 4),
        )

        # Temporal Encoder for DOY as position encoding
        self.temporal_encoder = TemporalPositionalEncoder(d_model=latent_dim * 4)

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim * 4,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Temporal Aware Pooling
        self.attn_pool = TemporalAwarePooling(latent_dim * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process temporal satellite data through the transformer encoder.

        Args:
            x: Input tensor of shape (B, seq_len, band_num + 1) where the last
               column contains day-of-year values and the remaining columns
               contain spectral band values.

        Returns:
            Processed tensor of shape (B, latent_dim) representing the
            temporal-aware encoding of the input sequence.
        """
        # x: (B, seq_len, 10 bands + 1 doy)
        # Split bands and doy
        bands = x[:, :, :-1]  # All columns except last one
        doy = x[:, :, -1]  # Last column is DOY
        # Embedding of bands
        bands_embedded = self.embedding(bands)  # (B, seq_len, latent_dim*4)
        temporal_encoding = self.temporal_encoder(doy)
        # Add temporal encoding to embedded bands (instead of random positional encoding)
        x = bands_embedded + temporal_encoding
        x = self.transformer_encoder(x)
        x = self.attn_pool(x)
        return x


# Requires day of year as input


class MultimodalBTInferenceModel(torch.nn.Module):
    """Model for the inference phase, containing only two Transformer encoders (S2 + S1), without the projection head.

    This model processes Sentinel-2 and Sentinel-1 data through separate
    transformer encoders and fuses the representations using either sum or
    concatenation.
    """

    def __init__(
        self,
        s2_backbone: TransformerEncoder,
        s1_backbone: TransformerEncoder,
        fusion_method: str,
        latent_dim: int,
    ) -> None:
        """Initialize the multimodal inference model.

        Args:
            s2_backbone: Transformer encoder for Sentinel-2 data.
            s1_backbone: Transformer encoder for Sentinel-1 data.
            fusion_method: Method for fusing representations ('sum' or 'concat').
            latent_dim: Target latent dimension for the output.
        """
        super().__init__()
        self.s2_backbone = s2_backbone
        self.s1_backbone = s1_backbone
        self.fusion_method = fusion_method
        self.latent_dim = latent_dim
        if fusion_method == "concat":
            in_dim = 8 * latent_dim
        elif fusion_method == "sum":
            in_dim = 4 * latent_dim
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        self.in_dim = in_dim
        # the dim reducer only works if we have both s2_x and s1_x
        self.dim_reducer = nn.Sequential(
            nn.Linear(self.in_dim, latent_dim),
        )

    def forward(self, s2_x: torch.Tensor, s1_x: torch.Tensor) -> torch.Tensor:
        """Process Sentinel-2 and Sentinel-1 data through the multimodal model.

        Args:
            s2_x: Sentinel-2 data tensor of shape (batch, seq_len_s2, band_num_s2).
            s1_x: Sentinel-1 data tensor of shape (batch, seq_len_s1, band_num_s1).

        Returns:
            Fused representation tensor of shape (batch, latent_dim).

        Raises:
            ValueError: If either s2_x or s1_x is None, as both are required
                       for the dimension reducer to work correctly.
        """
        if s2_x is None or s1_x is None:
            raise ValueError(
                "Both s2_x and s1_x must be provided, otherwise the dim reducer will  mismatch dim as the weights use concat method."
            )
        s2_repr = self.s2_backbone(s2_x)  # (batch, latent_dim)
        s1_repr = self.s1_backbone(s1_x)  # (batch, latent_dim)

        if self.fusion_method == "sum":
            fused = s2_repr + s1_repr
        elif self.fusion_method == "concat":
            fused = torch.cat([s2_repr, s1_repr], dim=-1)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        fused = self.dim_reducer(fused)
        return fused


def build_inference_model() -> MultimodalBTInferenceModel:
    """Build a pre-configured multimodal inference model for Tessera.

    Creates a model with the default configuration used in the Tessera paper,
    including separate transformer encoders for Sentinel-2 and Sentinel-1 data
    with concatenation-based fusion.

    Returns:
        Configured MultimodalBTInferenceModel instance ready for inference.
    """
    # Note: After enhancement, the number of S2 data channels is fixed at 12 (10 original bands + 2 doy features), and the number of S1 data channels is 4 (2+2)
    ###############Sentinel-2##################
    s2_backbone_ssl = TransformerEncoder(
        band_num=10,
        latent_dim=LATENT_DIM,
        nhead=8,
        num_encoder_layers=8,
        dim_feedforward=4096,
        dropout=0.1,
        max_seq_len=40,
    )
    ###############Sentinel-1##################
    s1_backbone_ssl = TransformerEncoder(
        band_num=2,
        latent_dim=LATENT_DIM,
        nhead=8,
        num_encoder_layers=8,
        dim_feedforward=4096,
        dropout=0.1,
        max_seq_len=40,
    )

    inference_model = MultimodalBTInferenceModel(
        s2_backbone_ssl,
        s1_backbone_ssl,
        FUSION_METHOD,
        latent_dim=LATENT_DIM,
    )
    return inference_model
