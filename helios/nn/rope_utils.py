"""Utility functions for rope positional encoding."""
import torch
from logging import getLogger

logger = getLogger(__name__)


# Code based on https://github.com/naver-ai/rope-vit/blob/main/self-attn/rope_self_attn.py
def init_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    """Initializes 2D frequency tensors for positional encoding.

    Args:
        dim: The dimension of the model.
        num_heads: The number of attention heads.
        theta: A parameter controlling the frequency range.
        rotate: Whether to randomly rotate the frequencies.

    Returns:
        A tensor containing the 2D frequencies.
    """
    freqs_x = []
    freqs_y = []
    # Calculate the magnitudes for the frequencies.
    # The `mag` tensor determines the scale of the sinusoidal functions.
    # It creates a geometric progression of values based on `theta` and `dim`.
    # `torch.arange(0, dim, 4)[: (dim // 4)]` generates a sequence of numbers
    # up to `dim // 4` with a step of 4, which are then scaled by `dim`
    # and used as exponents for `theta`.
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        # Generate random angles for rotation if `rotate` is True, otherwise use zero angles.
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        # Calculate the x-component of the frequencies.
        # This involves taking the cosine of the angles and `pi/2 + angles`,
        # scaling by `mag`, and concatenating the results.
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        # Calculate the y-component of the frequencies.
        # This is similar to `fx` but uses the sine function.
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    # Stack the x and y frequency components for all heads.
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    # Stack the x and y frequencies together.
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs

def init_t_xy(end_x: int, end_y: int):
    """Initializes 1D tensors representing x and y coordinates.

    Args:
        end_x: The maximum x-coordinate value.
        end_y: The maximum y-coordinate value.

    Returns:
        A tuple containing the x and y coordinate tensors.
    """
    # Create a 1D tensor representing a flattened grid of coordinates.
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    # Calculate the x-coordinates using the modulo operator.
    t_x = (t % end_x).float()
    # Calculate the y-coordinates using integer division.
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y

def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
    """
    Computes complex numbers (cisoids) for rotary positional embeddings
    by combining 2D spatial frequencies with x and y coordinate tensors.

    Args:
        freqs: Tensor of shape [2, num_heads, dim // num_heads // 2].
               The 2D frequency components for x and y axes.
        t_x: Tensor of shape [batch_size, N], representing x coordinates.
        t_y: Tensor of shape [batch_size, N], representing y coordinates.
        num_heads: Number of attention heads.

    Returns:
        A tensor of shape [batch_size, num_heads, N, dim // num_heads // 2]
        containing complex positional encodings.
    """
    batch_size, seq_len = t_x.shape
    # Get per-head feature dimension
    _, _, dim_per_head_half = freqs.view(2, num_heads, -1).shape
    logger.info(f"freqs shape: {freqs.shape}")
    with torch.cuda.amp.autocast(enabled=False):
        # Compute the frequency projections along x and y axes
        x_proj = t_x.unsqueeze(-1).float() @ freqs[0].unsqueeze(-2)  # [B, N, dim//(2*num_heads)]
        y_proj = t_y.unsqueeze(-1).float() @ freqs[1].unsqueeze(-2)  # [B, N, dim//(2*num_heads)]

        # Combine and convert to complex
        angles = x_proj + y_proj  # [B, N, dim//(2*num_heads)]
        cis = torch.polar(torch.ones_like(angles), angles)  # [B, N, dim//(2*num_heads)]

        # Reshape to [B, H, N, dim_per_head_half]
        cis = cis.view(batch_size, seq_len, num_heads, dim_per_head_half).permute(0, 2, 1, 3)
        logger.info(f"cis shape: {cis.shape}")
    return cis


def apply_rotarty_embed_to_matrix(freqs_cis: torch.Tensor, x: torch.Tensor):
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x).to(x.device)
