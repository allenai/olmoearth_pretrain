"""Attention Components for Helios."""

from logging import getLogger
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributed.fsdp import fully_shard
from torch.jit import Final

try:
    import flash_attn
except ImportError:
    flash_attn = None

logger = getLogger(__name__)


@torch._dynamo.disable()
def dispatch_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Dispatch flash attention.

    Modeled after olmo core but doesnt flatten internally
    """
    if flash_attn is None:
        raise RuntimeError("flash-attn is required!")

    if cu_seqlens is not None:
        if cu_seqlens_q is None:
            cu_seqlens_q = cu_seqlens
        if cu_seqlens_k is None:
            cu_seqlens_k = cu_seqlens
    if max_seqlen is not None:
        if max_seqlen_q is None:
            max_seqlen_q = max_seqlen
        if max_seqlen_k is None:
            max_seqlen_k = max_seqlen

    varlen = all(
        x is not None for x in (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    )

    if varlen:
        assert q.ndim == 3, "q must be pre-packed"
        logger.debug("using varlen")

        return flash_attn.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    else:
        return flash_attn.flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
        )


class Attention(nn.Module):
    """Multi-head attention module with optional cross-attention support.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads. Defaults to 8.
        qkv_bias: Enable bias for QKV projections. Defaults to False.
        qk_norm: Apply normalization to Q and K. Defaults to False.
        attn_drop: Attention dropout rate. Defaults to 0.0.
        proj_drop: Output projection dropout rate. Defaults to 0.0.
        norm_layer: Normalization layer. Defaults to nn.LayerNorm.
        cross_attn: Enable cross-attention. Defaults to False.
    """

    fast_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        cross_attn: bool = False,
        use_flash_attn: bool = False,
    ) -> None:
        """Initialize the attention module.

        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Enable bias for QKV projections
            qk_norm: Apply normalization to Q and K
            attn_drop: Attention dropout rate
            proj_drop: Output projection dropout rate
            norm_layer: Normalization layer
            cross_attn: Enable cross-attention
            use_flash_attn: Use flash attention
        """
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.cross_attn = cross_attn
        self.use_flash_attn = use_flash_attn
        self.fast_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        n: int,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute scaled dot product attention.

        Args:
            q: Query tensor of shape (B, H, N, D)
            k: Key tensor of shape (B, H, N, D)
            v: Value tensor of shape (B, H, N, D)
            n: Number of tokens
            attn_mask: Attention mask. Defaults to None.
            cu_seqlens: Optional cumulative sequence lengths for the input tensor needed for varlen flash attention
            cu_seqlens_q: Optional cumulative sequence lengths for the query tensor, needed for cross varlen flash attention
            cu_seqlens_k: Optional cumulative sequence lengths for the key tensor, needed for cross varlen flash attention
            max_seqlen: Optional maximum sequence length for the input tensor, needed for varlen flash attention
            max_seqlen_q: Optional maximum sequence length for the query tensor, needed for cross varlen flash attention
            max_seqlen_k: Optional maximum sequence length for the key tensor, needed for cross varlen flash attention

        Returns:
            Output tensor of shape (B, H, N, D)
        """
        if self.use_flash_attn:
            x = dispatch_flash_attn(
                q,
                k,
                v,
                cu_seqlens=cu_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen=max_seqlen,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=False,
            )
            # Output is (B, Nq, H, D), transpose back to (B, H, Nq, D)
            # matching the transpose of the other attention implementations that need to be transposed back
            x = x.transpose(1, 2)
        elif self.fast_attn:
            if attn_mask is not None:
                if len(attn_mask.shape) == 2:
                    # attn_mask has shape [B, N] and we want to expand it to [B, num_heads, N, N]
                    attn_mask = attn_mask[:, None, None].repeat(
                        (1, self.num_heads, n, 1)
                    )
                elif len(attn_mask.shape) == 3:
                    # attn_mask has shape [B, N, N] and we want to expand it to [B, num_heads, N, N]
                    attn_mask = attn_mask[:, None].repeat(1, self.num_heads, 1, 1)
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                # a value of True indicates that the element should take part in attention
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p,
            )
        else:
            # Backward Compatible for older PyTorch versions
            if attn_mask is not None:
                raise NotImplementedError
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        return x

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, C) or (B* N , C) if packed
            y: Second input for cross-attention. Defaults to None.
            attn_mask: Attention mask. Defaults to None.
            cu_seqlens: Optional cumulative sequence lengths for the input tensor needed for varlen flash attention
            cu_seqlens_q: Optional cumulative sequence lengths for the query tensor, needed for cross varlen flash attention
            cu_seqlens_k: Optional cumulative sequence lengths for the key tensor, needed for cross varlen flash attention
            max_seqlen: Optional maximum sequence length for the input tensor, needed for varlen flash attention
            max_seqlen_q: Optional maximum sequence length for the query tensor, needed for cross varlen flash attention
            max_seqlen_k: Optional maximum sequence length for the key tensor, needed for cross varlen flash attention

        Returns:
            Output tensor of shape (B, N, C) or (B* N , C) if packed
        """
        original_shape = x.shape

        q = self.q(x)

        if y is None:
            assert not self.cross_attn
            k = self.k(x)
            v = self.v(x)
        else:
            assert self.cross_attn
            k = self.k(y)
            v = self.v(y)
        if not self.use_flash_attn:
            q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
            k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
            v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        else:
            q = rearrange(q, "bn (h d) -> bn h d", h=self.num_heads)
            # Flash attention only supports k v heads that divide the number of query heads
            k = rearrange(k, "bn (h d) -> bn h d", h=self.num_heads)
            v = rearrange(v, "bn (h d) -> bn h d", h=self.num_heads)
        # logger.info(f"q shape: {q.shape} k shape: {k.shape} v shape: {v.shape}")

        q, k = self.q_norm(q), self.k_norm(k)
        x = self.sdpa(
            q,
            k,
            v,
            n=original_shape[
                -2
            ],  # supposed to be the number of tokens in each sample with padding
            cu_seqlens=cu_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen=max_seqlen,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            attn_mask=attn_mask,
        )
        x = x.transpose(1, 2).reshape(original_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP module used in Vision Transformer, MLP-Mixer and related networks.

    Args:
        in_features: Number of input features
        hidden_features: Hidden dimension. Defaults to None.
        out_features: Output dimension. Defaults to None.
        act_layer: Activation layer. Defaults to nn.GELU.
        bias: Enable bias in linear layers. Defaults to True.
        drop: Dropout rate. Defaults to 0.0.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: nn.Module = nn.GELU,
        bias: bool = True,
        drop: float = 0.0,
    ) -> None:
        """Initialize the MLP module.

        Args:
            in_features: Number of input features
            hidden_features: Hidden dimension. Defaults to None.
            out_features: Output dimension. Defaults to None.
            act_layer: Activation layer. Defaults to nn.GELU.
            bias: Enable bias in linear layers. Defaults to True.
            drop: Dropout rate. Defaults to 0.0.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LayerScale(nn.Module):
    """Learnable scaling layer.

    Args:
        dim: Input dimension
        init_values: Initial scaling value. Defaults to 1e-5.
        inplace: Perform scaling operation in-place. Defaults to False.
    """

    def __init__(
        self, dim: int, init_values: float = 1e-5, inplace: bool = False
    ) -> None:
        """Initialize the LayerScale module.

        Args:
            dim: Input dimension
            init_values: Initial scaling value
            inplace: Perform scaling operation in-place
        """
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Scaled output tensor
        """
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample when applied in main path of residual blocks.

    This is a regularization technique that randomly drops entire layers/paths during training
    to prevent overfitting. During inference, all paths are kept.

    Args:
        drop_prob: Probability of dropping the path. Defaults to None.

    References:
        Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382)
    """

    def __init__(self, drop_prob: float) -> None:
        """Initialize the DropPath module.

        Args:
            drop_prob: Probability of dropping the path. Defaults to None.
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying stochastic depth to input tensor.

        Args:
            x: Input tensor of any shape (B, ...)

        Returns:
            Tensor with same shape as input, with paths randomly dropped during training
        """
        if self.drop_prob is None or self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1, ...)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor


class Block(nn.Module):
    """Transformer block with self/cross attention and MLP.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of mlp hidden dim to input dim. Default: 4.0
        qkv_bias: Add bias to qkv projections. Default: False
        qk_norm: Apply normalization to q,k. Default: False
        drop: Dropout rate. Default: 0.0
        attn_drop: Attention dropout rate. Default: 0.0
        drop_path: Drop path rate. Default: 0.0
        init_values: Layer scale initialization value. Default: None
        act_layer: Activation layer. Default: nn.GELU
        norm_layer: Normalization layer. Default: nn.LayerNorm
        cross_attn: Whether to use cross attention. Default: False
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: float | None = None,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        cross_attn: bool = False,
        use_flash_attn: bool = False,
    ) -> None:
        """Initialize the Transformer block.

        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of mlp hidden dim to input dim
            qkv_bias: Add bias to qkv projections
            qk_norm: Apply normalization to q,k
            drop: Dropout rate
            attn_drop: Attention dropout rate
            drop_path: Drop path rate
            init_values: Layer scale initialization value
            act_layer: Activation layer
            norm_layer: Normalization layer
            cross_attn: Whether to use cross attention
            use_flash_attn: Whether to use flash attention
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
            cross_attn=cross_attn,
            use_flash_attn=use_flash_attn,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, C)
            y: Optional context tensor for cross attention of shape (B, M, C)
            attn_mask: Optional attention mask tensor
            cu_seqlens: Optional cumulative sequence lengths for the input tensor needed for varlen flash attention
            cu_seqlens_q: Optional cumulative sequence lengths for the query tensor, needed for cross varlen flash attention
            cu_seqlens_k: Optional cumulative sequence lengths for the key tensor, needed for cross varlen flash attention
            max_seqlen: Optional maximum sequence length for the input tensor, needed for varlen flash attention
            max_seqlen_q: Optional maximum sequence length for the query tensor, needed for cross varlen flash attention
            max_seqlen_k: Optional maximum sequence length for the key tensor, needed for cross varlen flash attention

        Returns:
            Output tensor of shape (B, N, C)
        """
        x = x + self.drop_path(
            self.ls1(
                self.attn(
                    x=self.norm1(x),
                    y=y,
                    cu_seqlens=cu_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen=max_seqlen,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    attn_mask=attn_mask,
                )
            )
        )

        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x

    def apply_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP to the model."""
        fully_shard(self, **fsdp_kwargs)

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        self.compile()


class AttnPool(nn.Module):
    """Multi-query attention pooling with gated averaging.

    Args:
        in_dim (int): token dim (must be divisible by 64; head_dim=64).
        hidden_dim (int): MLP hidden/out dim (defaults to in_dim unless mlp_ratio provided).
        mlp_ratio (float|None): if set, hidden_dim := int(in_dim * mlp_ratio)
        num_queries (int): number of learned queries per (t,s) group.
        gate_temperature (float): temperature for softmax gating (>0).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int | None = None,
        mlp_ratio: float | None = None,
        num_heads: int | None = None,
        num_queries: int = 1,
        gate_temperature: float = 1.0,
    ) -> None:
        """Initialize the Attn pooling layer."""
        super().__init__()
        if num_heads is None:
            assert in_dim % 64 == 0, "in_dim must be divisible by 64"
            self.num_heads: int = in_dim // 64
        else:
            self.num_heads = int(num_heads)
        self.num_queries: int = num_queries
        self.gate_temperature: float = gate_temperature

        # k learned queries (k, D)
        self.query_tokens: nn.Parameter = nn.Parameter(torch.empty(num_queries, in_dim))

        # shared KV projection
        self.kv: nn.Linear = nn.Linear(in_dim, in_dim * 2)

        # output MLP (+ optional expansion via mlp_ratio)
        if mlp_ratio is not None:
            hidden_dim = int(in_dim * mlp_ratio)
        hidden_dim = hidden_dim or in_dim
        self.out_layer: Mlp = Mlp(in_dim, hidden_dim)
        self.out_norm = nn.LayerNorm(in_dim)

        # gating over k query outputs (maps D -> 1 per query)
        self.gate: nn.Linear | None = (
            nn.Linear(in_dim, 1, bias=False) if num_queries > 1 else None
        )

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights for the probe."""
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        nn.init.trunc_normal_(self.kv.weight, std=0.02)
        nn.init.zeros_(self.kv.bias)

        nn.init.zeros_(self.kv.bias)
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)  # start near uniform mix

    def forward(
        self, feat_tokens: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Apply attention pooling to the tokens."""
        Bc, N, D = feat_tokens.shape
        H = self.num_heads
        Dh = D // H

        # queries: [B*, k, D] -> [B*, H, k, Dh]
        q = (
            self.query_tokens[None, :, :]
            .expand(Bc, -1, -1)
            .reshape(Bc, self.num_queries, H, Dh)
        )
        q = rearrange(q, "b k h d -> b h k d")

        # K/V: [B*, N, D] -> [2, B*, H, N, Dh]
        feat_tokens = feat_tokens.to(self.kv.weight.dtype)
        kv = self.kv(feat_tokens).reshape(Bc, N, 2, H, Dh)
        kv = rearrange(kv, "b n two h d -> two b h n d")
        k, v = torch.unbind(kv, dim=0)  # [B*, H, N, Dh] each

        # mask -> [B*, H, k, N] (broadcastable is fine, but expand for clarity)
        attn_mask = None
        if mask is not None:
            m = mask[:, None, None, :]  # [B*,1,1,N]
            attn_mask = m.expand(Bc, H, self.num_queries, N)

        # H100 chunking on batch axis
        max_size = 63488
        x_chunks = []
        for i in range(0, Bc, max_size):
            q_chunk = q[i : i + max_size, ...]
            k_chunk = k[i : i + max_size, ...]
            v_chunk = v[i : i + max_size, ...]
            m_chunk = (
                attn_mask[i : i + max_size, ...] if attn_mask is not None else None
            )
            # SDPA expects [B,H,Q,D] x [B,H,K,D] -> [B,H,Q,D]
            x_chunk = F.scaled_dot_product_attention(
                q_chunk, k_chunk, v_chunk, attn_mask=m_chunk
            )
            x_chunks.append(x_chunk)

        # [B*, H, k, Dh] -> [B*, k, D]
        x = torch.cat(x_chunks, dim=0)
        o = rearrange(x, "b h k d -> b k (h d)")

        # gated average across k, or pass-through if k=1
        if self.num_queries > 1 and self.gate is not None:
            o_for_gate = F.layer_norm(o, (D,))  # normalize only for gating
            logits = self.gate(o_for_gate).squeeze(-1)  # [B*, k]
            w = torch.softmax(logits, dim=1)
            z = (w.unsqueeze(-1) * o).sum(dim=1)  # mix the *unnormalized* values
        else:
            z = o.squeeze(1)

        # MLP + LN head
        z = self.out_norm(self.out_layer(z))
        return z
