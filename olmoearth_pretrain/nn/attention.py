"""Attention Components for OlmoEarth Pretrain."""

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
                attn_mask = attn_mask[:, None, None].repeat((1, self.num_heads, n, 1))
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


class SparseMoE(nn.Module):
    """Sparse Mixture of Experts layer.

    Replaces a single Mlp with multiple experts + a learned router.
    Uses top-k gating to select which experts process each token.

    Args:
        in_features: Number of input features
        hidden_features: Hidden dimension. Defaults to None.
        out_features: Output dimension. Defaults to None.
        num_experts: Number of expert networks. Defaults to 8.
        top_k: Number of experts each token is routed to. Defaults to 2.
        act_layer: Activation layer. Defaults to nn.GELU.
        bias: Enable bias in linear layers. Defaults to True.
        drop: Dropout rate. Defaults to 0.0.
        aux_loss_weight: Weight for load balancing auxiliary loss. Defaults to 0.01.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        num_experts: int = 8,
        top_k: int = 2,
        act_layer: nn.Module = nn.GELU,
        bias: bool = True,
        drop: float = 0.0,
        aux_loss_weight: float = 0.01,
    ) -> None:
        """Initialize the SparseMoE module."""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self._aux_loss: torch.Tensor = torch.tensor(0.0)

        # Router: projects input to num_experts logits
        self.router = nn.Linear(in_features, num_experts, bias=False)

        # Create experts (each is an Mlp)
        self.experts = nn.ModuleList(
            [
                Mlp(
                    in_features=in_features,
                    hidden_features=hidden_features,
                    out_features=out_features,
                    act_layer=act_layer,
                    bias=bias,
                    drop=drop,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with efficient batched sparse routing.

        Uses sorted token grouping for better memory access patterns and fewer
        kernel launches compared to boolean masking per expert.

        Args:
            x: Input tensor of shape (B, N, D) or (B*N, D)

        Returns:
            Output tensor of same shape as input
        """
        orig_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])  # (num_tokens, D)
        num_tokens = x_flat.shape[0]

        # Compute router logits and probabilities
        router_logits = self.router(x_flat)  # (num_tokens, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts per token
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # Normalize the selected probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Compute auxiliary load balancing loss
        if self.training:
            self._aux_loss = self._compute_aux_loss(router_probs, top_k_indices)

        # Flatten top-k selections: (num_tokens * top_k,)
        flat_indices = top_k_indices.view(-1)  # Which expert for each (token, k) pair
        flat_probs = top_k_probs.view(-1)  # Weight for each (token, k) pair

        # Repeat input for each top-k selection: (num_tokens * top_k, D)
        x_repeat = x_flat.repeat_interleave(self.top_k, dim=0)

        # Sort by expert index for contiguous memory access patterns
        sorted_indices, sort_order = torch.sort(flat_indices)
        sorted_probs = flat_probs[sort_order]
        sorted_x = x_repeat[sort_order]

        # Find boundaries for each expert using bincount
        expert_counts = torch.bincount(sorted_indices, minlength=self.num_experts)
        expert_offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=x.device),
                torch.cumsum(expert_counts, dim=0),
            ]
        )

        # Process each expert's tokens in a batch (contiguous slices, not boolean masks)
        sorted_outputs = torch.zeros_like(sorted_x)
        for expert_idx in range(self.num_experts):
            start = expert_offsets[expert_idx].item()
            end = expert_offsets[expert_idx + 1].item()
            if start < end:
                sorted_outputs[start:end] = self.experts[expert_idx](
                    sorted_x[start:end]
                )

        # Apply weights
        sorted_outputs = sorted_outputs * sorted_probs.unsqueeze(-1)

        # Unsort back to original order
        outputs = torch.zeros_like(sorted_outputs)
        outputs[sort_order] = sorted_outputs

        # Reshape and sum over top-k dimension
        outputs = outputs.view(num_tokens, self.top_k, -1)
        output = outputs.sum(dim=1)

        return output.view(orig_shape)

    def _compute_aux_loss(
        self, router_probs: torch.Tensor, top_k_indices: torch.Tensor
    ) -> torch.Tensor:
        """Compute load balancing auxiliary loss.

        Encourages balanced expert utilization across tokens.

        Args:
            router_probs: Router probabilities of shape (num_tokens, num_experts)
            top_k_indices: Selected expert indices of shape (num_tokens, top_k)

        Returns:
            Scalar auxiliary loss tensor
        """
        num_tokens = router_probs.shape[0]
        expert_counts = torch.zeros(
            self.num_experts, device=router_probs.device, dtype=router_probs.dtype
        )
        for k in range(self.top_k):
            expert_counts.scatter_add_(
                0,
                top_k_indices[:, k],
                torch.ones(
                    num_tokens, device=router_probs.device, dtype=router_probs.dtype
                ),
            )
        expert_fraction = expert_counts / (num_tokens * self.top_k)

        # Average routing probability to each expert
        mean_probs = router_probs.mean(dim=0)

        # Aux loss = num_experts * sum(fraction_i * prob_i)
        aux_loss = self.num_experts * (expert_fraction * mean_probs).sum()
        return self.aux_loss_weight * aux_loss

    @property
    def aux_loss(self) -> torch.Tensor:
        """Get the auxiliary loss from the last forward pass."""
        return self._aux_loss


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
        self.compile(dynamic=False, mode="max-autotune-no-cudagraphs", fullgraph=True)


class MoEBlock(nn.Module):
    """Transformer block with Sparse MoE instead of MLP.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of mlp hidden dim to input dim. Default: 4.0
        num_experts: Number of expert networks. Default: 8
        top_k: Number of experts each token is routed to. Default: 2
        aux_loss_weight: Weight for load balancing auxiliary loss. Default: 0.01
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
        num_experts: int = 8,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
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
        """Initialize the MoE Transformer block."""
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
        self.moe = SparseMoE(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_weight=aux_loss_weight,
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
            cu_seqlens: Optional cumulative sequence lengths for varlen flash attention
            cu_seqlens_q: Optional cumulative sequence lengths for query
            cu_seqlens_k: Optional cumulative sequence lengths for key
            max_seqlen: Optional maximum sequence length
            max_seqlen_q: Optional maximum sequence length for query
            max_seqlen_k: Optional maximum sequence length for key

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

        x = x + self.drop_path(self.ls2(self.moe(self.norm2(x))))
        return x

    @property
    def aux_loss(self) -> torch.Tensor:
        """Get the auxiliary loss from the MoE layer."""
        return self.moe.aux_loss

    def apply_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP to the model."""
        fully_shard(self, **fsdp_kwargs)

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        self.compile(dynamic=False, mode="max-autotune-no-cudagraphs", fullgraph=True)
