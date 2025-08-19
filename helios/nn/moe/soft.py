"""Soft MoE (Mixture of Experts) implementation.

Mostly from
https://raw.githubusercontent.com/lucidrains/soft-moe-pytorch/refs/heads/main/soft_moe_pytorch/soft_moe.py.
"""

from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import pack, rearrange, unpack
from torch import Tensor, einsum, nn
from torch.nn import Module

from helios.nn.moe.distributed import (
    AllGather,
    gather_sizes,
    has_only_one_value,
    split_by_rank,
)

# helper functions


def exists(val: Any) -> bool:
    """Check if a value exists (is not None).

    Args:
        val: The value to check.

    Returns:
        bool: True if the value is not None, False otherwise.
    """
    return val is not None


def default(val: Any, d: Any) -> Any:
    """Return the value if it exists, otherwise return the default.

    Args:
        val: The value to check.
        d: The default value to return if val is None.

    Returns:
        Any: The value if it exists, otherwise the default.
    """
    return val if exists(val) else d


def divisible_by(num: int, den: int) -> bool:
    """Check if a number is divisible by another.

    Args:
        num: The numerator.
        den: The denominator.

    Returns:
        bool: True if num is divisible by den, False otherwise.
    """
    return (num % den) == 0


def chunk_num(num: int, chunks: int) -> list[int]:
    """Divide a number into approximately equal chunks.

    Args:
        num: The number to divide.
        chunks: The number of chunks to create.

    Returns:
        List[int]: List of chunk sizes that sum to num.
    """
    num_per_chunk, remainder = divmod(num, chunks)

    out = []
    for i in range(chunks):
        n = num_per_chunk
        out.append(n + int(i < remainder))

    return out


def pack_one(t: Tensor, pattern: str) -> tuple[Tensor, tuple[int, ...]]:
    """Pack a single tensor using einops pattern.

    Args:
        t: The tensor to pack.
        pattern: The einops pattern to use.

    Returns:
        Tuple[Tensor, Tuple[int, ...]]: Packed tensor and its shape.
    """
    return pack([t], pattern)


def unpack_one(t: Tensor, ps: tuple[int, ...], pattern: str) -> Tensor:
    """Unpack a single tensor using einops pattern.

    Args:
        t: The tensor to unpack.
        ps: The packed shape.
        pattern: The einops pattern to use.

    Returns:
        Tensor: The unpacked tensor.
    """
    return unpack(t, ps, pattern)[0]


def l2norm(t: Tensor) -> Tensor:
    """Apply L2 normalization to a tensor.

    Args:
        t: The tensor to normalize.

    Returns:
        Tensor: The L2 normalized tensor.
    """
    return F.normalize(t, dim=-1)


def cumsum_exclusive(t: Tensor, dim: int = -3) -> Tensor:
    """Compute exclusive cumulative sum along a dimension.

    Args:
        t: The input tensor.
        dim: The dimension along which to compute the cumulative sum.

    Returns:
        Tensor: The exclusive cumulative sum.

    Raises:
        AssertionError: If dim is not negative.
    """
    assert dim < 0
    num_pad_dims = -dim - 1
    pre_padding = (0, 0) * num_pad_dims
    return F.pad(t, (*pre_padding, 1, -1)).cumsum(dim=dim)


def log(t: Tensor, eps: float = 1e-20) -> Tensor:
    """Compute the natural logarithm with a minimum value.

    Args:
        t: The input tensor.
        eps: The minimum value to clamp to.

    Returns:
        Tensor: The natural logarithm of the clamped tensor.
    """
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t: Tensor) -> Tensor:
    """Generate Gumbel noise for the given tensor.

    Args:
        t: The input tensor.

    Returns:
        Tensor: Gumbel noise with the same shape as t.
    """
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


# norm


class LayerNorm(nn.Module):
    """Layer normalization module with learnable parameters.

    This module applies layer normalization with learnable gamma and beta parameters.
    """

    def __init__(self, dim: int) -> None:
        """Initialize the LayerNorm module.

        Args:
            dim: The dimension to normalize over.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the layer normalization.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Tensor: Normalized tensor with the same shape as x.
        """
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class RMSNorm(Module):
    """Root Mean Square normalization module.

    This module applies RMS normalization with a learnable gamma parameter.
    """

    def __init__(self, dim: int) -> None:
        """Initialize the RMSNorm module.

        Args:
            dim: The dimension to normalize over.
        """
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the RMS normalization.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Tensor: Normalized tensor with the same shape as x.
        """
        return l2norm(x) * self.scale * self.gamma


# expert


def FeedForward(dim: int, mult: int | float = 4, dropout: float = 0.0) -> nn.Sequential:
    """Create a feedforward neural network.

    Args:
        dim: The input and output dimension.
        mult: The multiplier for the hidden dimension.
        dropout: The dropout rate.

    Returns:
        nn.Sequential: A feedforward network with GELU activation.
    """
    dim_hidden = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim),
    )


class GEGLU(Module):
    """Gated Linear Unit with GELU activation.

    This module implements a gated linear unit where the gate uses GELU activation.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the GEGLU module.

        Args:
            x: Input tensor of shape (..., 2 * dim).

        Returns:
            Tensor: Output tensor of shape (..., dim).
        """
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


def GLUFeedForward(
    dim: int, mult: int | float = 4, dropout: float = 0.0
) -> nn.Sequential:
    """Create a feedforward neural network with GLU activation.

    Args:
        dim: The input and output dimension.
        mult: The multiplier for the hidden dimension.
        dropout: The dropout rate.

    Returns:
        nn.Sequential: A feedforward network with GLU activation.
    """
    dim_hidden = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.Linear(dim, dim_hidden * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim),
    )


# experts


class Experts(nn.Module):
    """A module that manages multiple expert networks for distributed training.

    This module handles the distribution of experts across multiple devices
    and manages the routing of inputs to the appropriate experts.
    """

    def __init__(
        self,
        experts: list[nn.Module],
        is_distributed: bool | None = None,
        offload_unused_experts_to_cpu: bool = True,
    ) -> None:
        """Initialize the Experts module.

        Args:
            experts: List of expert modules.
            is_distributed: Whether to use distributed training. If None,
                          automatically detected from torch.distributed.
            offload_unused_experts_to_cpu: Whether to move unused experts to CPU
                                         to save GPU memory.
        """
        super().__init__()
        self.num_experts = len(experts)
        self.experts = nn.ModuleList(experts)

        self.is_distributed = is_distributed
        if not exists(self.is_distributed):
            self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1

        # whether to offload unused experts to cpu, will require optimizer handles conversion of gradients to right device when accumulating
        self.offload_unused_experts_to_cpu = offload_unused_experts_to_cpu

        self.all_gather = AllGather()
        self.register_buffer("dummy", torch.ones(1), persistent=False)

    @property
    def device(self) -> torch.device:
        """Get the device of the dummy buffer.

        Returns:
            torch.device: The device of the module.
        """
        return self.dummy.device

    def all_experts_to_cpu_besides(
        self, selection: int | slice | list[nn.Module]
    ) -> None:
        """Move all experts to CPU except those in the selection.

        Args:
            selection: The experts to keep on the current device. Can be an int,
                     slice, or list of expert modules.
        """
        if not self.offload_unused_experts_to_cpu:
            return

        if isinstance(selection, int):
            experts = [self.experts[selection]]
        elif isinstance(selection, slice):
            experts = self.experts[selection]
        else:
            experts = selection

        experts_set = set(experts)

        for expert in self.experts:
            device = self.device if expert in experts_set else "cpu"
            expert.to(device)

    def forward(self, x: Tensor, is_distributed: bool | None = None) -> Tensor:
        """Forward pass through the experts.

        Args:
            x: Input tensor of shape (batch, experts, seq_len, dim).
            is_distributed: Whether to use distributed training. If None, uses
                          the default setting.

        Returns:
            Tensor: Output tensor with the same shape as the input.

        Note:
            einops notation:
            b - batch
            r - rank (device / machines)
            e - experts
            n - sequence (number of tokens per expert)
            d - feature dimension
        """
        is_distributed = default(is_distributed, self.is_distributed)
        shape, num_experts = x.shape, self.num_experts

        # for now naively all gather across batch dimension if distributed, optimize later

        if is_distributed:
            seq_sizes = gather_sizes(x, dim=-2)
            assert has_only_one_value(seq_sizes), (
                "number of tokens per expert must be the same"
            )

            x, batch_sizes = self.all_gather(x)
            total_batch_size = x.shape[0]

            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        # the experts in use on the rank

        if is_distributed:
            if world_size <= num_experts:
                num_experts_across_ranks = chunk_num(num_experts, world_size)
                start_indices = cumsum_exclusive(
                    torch.tensor(num_experts_across_ranks), dim=-1
                )

                num_experts_per_rank = num_experts_across_ranks[rank]
                num_experts_batches_across_ranks = [
                    i * total_batch_size for i in num_experts_across_ranks
                ]

                expert_start_index = start_indices[rank].item()
            else:
                num_batch_chunks = world_size // num_experts
                total_ranks_in_use = num_batch_chunks * num_experts

                expert_start_index = rank // num_batch_chunks

                batch_splits = chunk_num(total_batch_size, num_batch_chunks)
                num_experts_batches_across_ranks = list(batch_splits * num_experts)

                # for now, remaining machines just process nothing

                remain_ranks = world_size % num_experts
                num_experts_batches_across_ranks += [0] * remain_ranks

                num_experts_per_rank = int(rank < total_ranks_in_use)

            assert len(num_experts_batches_across_ranks) == world_size

            expert_slice = slice(
                expert_start_index, expert_start_index + num_experts_per_rank
            )
        else:
            num_experts_per_rank = num_experts
            expert_slice = slice(0, num_experts)

        # if distributed, each machine only handles subset of experts and batch

        x = rearrange(x, "b e n d -> e b n d")

        if is_distributed:
            x, expert_batch_packed_shape = pack_one(x, "* n d")
            x_split = x.split(num_experts_batches_across_ranks, dim=0)
            x = split_by_rank(x_split)

            if num_experts_per_rank > 0:
                x = rearrange(x, "(e b) n d -> e b n d", e=num_experts_per_rank)
            else:
                x = x.reshape(num_experts, *x.shape)

        # get the experts in use

        self.all_experts_to_cpu_besides(expert_slice)

        experts = self.experts[expert_slice]

        # route tokens to appropriate experts

        outs_list = []
        for expert, expert_input in zip(experts, x):
            out = expert(expert_input)
            outs_list.append(out)

        if len(outs_list) > 0:
            outs = torch.stack(outs_list)
        else:
            outs = torch.empty_like(x).requires_grad_()

        # all gather across merged expert batches dimensions
        # then split the batch dimension back

        if is_distributed:
            outs = rearrange(outs, "e b n d -> (e b) n d")
            outs, _ = self.all_gather(outs)
            outs = unpack_one(outs, expert_batch_packed_shape, "* n d")

        outs = rearrange(outs, "e b n d -> b e n d")

        if is_distributed:
            if batch_sizes is not None:
                outs_split = outs.split(batch_sizes.tolist())
                outs = split_by_rank(outs_split)

        assert outs.shape == shape
        return outs


# task conditioning (deepseek-v3 style expert bias)


class TaskBias(nn.Module):
    """Task-conditioned expert bias, to be added to the expert logits.

    Would recommend a simple linear transformation since the task embeddings
    are pretrained so that cosine similarity corresponds to task similarity.
    """

    def __init__(
        self,
        task_dim: int,
        num_experts: int,
        hidden: int = 0,
        dropout: float = 0.0,
        zero_init: bool = False,
    ):
        """Initialize the TaskBias module.

        Args:
            task_dim: The dimension of the task embedding.
            num_experts: The number of experts.
            hidden: The dimension of the hidden layer.
            dropout: The dropout rate.
            zero_init: Whether to zero-initialize the last-layer parameters.
        """
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(task_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_experts),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(task_dim, num_experts),
                nn.Dropout(dropout),
            )

        if zero_init:
            # zero-init: start with no effect
            # don't use this, symmetry breaking is important
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
            self.net[-1]._skip_reinit = True

    def forward(self, task_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass of the TaskBias module.

        Args:
            task_emb: The task embedding tensor of shape (B, D) or (D,).

        Returns:
            Tensor: The expert bias tensor of shape (B, E).
        """
        if task_emb.dim() == 1:
            task_emb = task_emb.unsqueeze(0)
        return self.net(task_emb)  # (B, E)


# main class


class SoftMoE(Module):
    """Soft Mixture of Experts (MoE) module.

    This module implements a soft mixture of experts where tokens are softly
    assigned to experts using learned routing weights.
    """

    def __init__(
        self,
        dim: int,
        *,
        task_dim: int | None = None,
        seq_len: int | None = None,
        num_experts: int = 4,
        num_slots: int | None = None,
        expert_mult: int | float = 4,
        dropout: float = 0.0,
        geglu: bool = False,
        is_distributed: bool | None = None,
        offload_unused_experts_to_cpu: bool = True,
        use_layernorm: bool = False,
        add_noise: bool = False,
        noise_mult: float = 1.0,
    ) -> None:
        """Initialize the SoftMoE module.

        Args:
            dim: The input and output dimension.
            task_dim: The dimension of the task embedding.
            seq_len: The sequence length. Must be provided if num_slots is not.
            num_experts: The number of experts.
            num_slots: The number of slots per expert. Must be provided if seq_len is not.
            expert_mult: The multiplier for expert hidden dimensions.
            dropout: The dropout rate.
            geglu: Whether to use GLU activation in experts.
            is_distributed: Whether to use distributed training.
            offload_unused_experts_to_cpu: Whether to move unused experts to CPU.
            use_layernorm: Whether to use LayerNorm instead of RMSNorm.
            add_noise: Whether to add Gumbel noise to the routing logits.
            noise_mult: Multiplier for the Gumbel noise.

        Raises:
            AssertionError: If neither seq_len nor num_slots is provided, or if both are provided.
        """
        super().__init__()
        assert exists(seq_len) ^ exists(num_slots), (
            "either seq_len, or num_slots must be passed into SoftMoE"
        )

        if exists(seq_len):
            if seq_len is not None:
                num_slots = default(num_slots, seq_len // num_experts)
        elif exists(num_slots):
            if num_slots is not None:
                seq_len = num_slots * num_experts
        else:
            raise ValueError("Either seq_len or num_slots must be provided")

        norm_klass = LayerNorm if use_layernorm else RMSNorm
        self.norm: Callable = norm_klass(dim)  # type: ignore

        self.slot_norm: Callable = norm_klass(dim)  # type: ignore
        self.slot_embeds = nn.Parameter(torch.randn(num_experts, num_slots, dim))

        expert_klass = GLUFeedForward if geglu else FeedForward

        self.experts = Experts(
            experts=[
                expert_klass(dim=dim, mult=expert_mult, dropout=dropout)
                for _ in range(num_experts)
            ],
            is_distributed=is_distributed,
            offload_unused_experts_to_cpu=offload_unused_experts_to_cpu,
        )

        self.num_experts = num_experts
        self.num_slots = num_slots

        if task_dim is not None:
            # self.task_bias = TaskBias(task_dim, num_experts * num_slots)
            self.task_bias = nn.Linear(task_dim, dim)
        else:
            self.task_bias = None

        self.add_noise = add_noise
        self.noise_mult = noise_mult

    def forward(
        self,
        x: Tensor,
        task_emb: Tensor | None = None,
        mask: Tensor | None = None,
        weight_key: Tensor | None = None,
        return_load_balance_loss: bool = True,
        return_dispatch_weights: bool = True,
        return_combine_weights: bool = True,
    ) -> dict[str, Tensor]:
        """Forward pass of the SoftMoE module.

        Args:
            x: Input tensor of shape (batch, seq_len, dim) or (batch, dim) for single token.
            task_emb: Optional task embedding tensor of shape (B, task_dim) or (task_dim,).
            mask: Optional mask tensor of shape (batch, seq_len) to mask out padding tokens.
            weight_key: Tensor of shape (batch, seq_len, dim) with which to compute the dispatch and combine
                weights. If not specified, use the input tokens.
            return_load_balance_loss: Whether to return the load balance loss.
            return_dispatch_weights: Whether to return the dispatch weights along with the output.
            return_combine_weights: Whether to return the combine weights along with the output.

        Returns:
            dict with key "outputs" (output tensor) and optionally "load_balance_loss",
            "dispatch_weights", and "combine_weights".

        Note:
            einstein notation
            b - batch
            n - sequence length
            e - number of experts
            s - number of slots per expert
            d - feature dimension
        """
        is_single_token = x.ndim == 2
        is_image = x.ndim == 4

        if is_image:
            x = rearrange(x, "b d h w -> b h w d")
            x, ps = pack([x], "b * d")  # type: ignore
        elif is_single_token:
            x = rearrange(x, "b d -> b 1 d")

        # following Algorithm 1, with the normalization they proposed, but with scaling of both (the now popular rmsnorm + gamma)
        x = self.norm(x)
        slot_embeds = self.slot_norm(self.slot_embeds)

        dispatch_logits = einsum("b n d, e s d -> b n e s", x, slot_embeds)

        # if weight_key is not provided, use task embedding to compute it
        if task_emb is not None and weight_key is None:
            weight_key = self.task_bias(task_emb)
            weight_key = weight_key.unsqueeze(1).repeat(1, x.shape[1], 1)

        # compute combine weights from weight_key (or x by default)
        if weight_key is not None:
            assert weight_key.shape == x.shape, (
                "weight_key must be (batch_size, seq_len, dim)"
            )
            combine_logits = einsum("b n d, e s d -> b n e s", weight_key, slot_embeds)
        else:
            combine_logits = dispatch_logits

        print("weight distribution", F.softmax(combine_logits.sum(dim=-1), dim=-1).mean(dim=(0, 1)).detach().cpu().numpy())

        # add task expert bias (see deepseek-v3 auxiliary-free loss balancing)
        # if task_emb is not None and self.task_bias is not None:
        #     task_bias = self.task_bias(task_emb)
        #     task_bias = rearrange(task_bias, 'b (e s) -> b () e s', e=self.num_experts, s=self.num_slots)
        #     print("without task bias", F.softmax(combine_logits.sum(dim=-1), dim=-1).mean(dim=(0, 1)))
        #     print("combine norm", combine_logits.sum(dim=-1).norm(), "task bias norm", task_bias.norm())
        #     # combine_logits += task_bias * 10
        #     task_bias = task_bias.repeat(1, combine_logits.shape[1], 1, 1)
        #     combine_logits = task_bias
        #     print("with task bias", F.softmax(combine_logits.sum(dim=-1), dim=-1).mean(dim=(0, 1)))

        # noised dispatch and combine gate logits, with annealing if needed
        if self.add_noise:
            dispatch_logits = (
                dispatch_logits + gumbel_noise(dispatch_logits) * self.noise_mult
            )
            combine_logits = (
                combine_logits + gumbel_noise(combine_logits) * self.noise_mult
            )

        # account for key padding mask
        if exists(mask):
            mask = rearrange(mask, "b n -> b n 1 1")
            fill_value = -torch.finfo(dispatch_logits.dtype).max
            dispatch_logits = dispatch_logits.masked_fill(~mask, fill_value)
            combine_logits = combine_logits.masked_fill(~mask, fill_value)

        # get dispatch and combine weights (softmax across right dimensions)
        dispatch_weights = dispatch_logits.softmax(dim=1)

        combine_weights = rearrange(combine_logits, "b n e s -> b n (e s)")
        combine_weights = combine_weights.softmax(dim=-1)

        # derive slots by weighted average of input tokens using the dispatch weights from above
        slots = einsum("b n d, b n e s -> b e s d", x, dispatch_weights)

        # route the slots per expert to each expert
        out = self.experts(slots)

        # combine back out
        out = rearrange(out, " b e s d -> b (e s) d")
        out = einsum("b s d, b n s -> b n d", out, combine_weights)

        if is_image:
            (out,) = unpack(out, ps, "b * d")  # type: ignore
            out = rearrange(out, "b h w d -> b d h w")
        elif is_single_token:
            out = rearrange(out, "b 1 d -> b d")

        # compute the load balance loss per layer if requested
        info = {"outputs": out}
        if return_load_balance_loss:
            # penalize negative entropy of the expert combine weights
            # this is negative, so be careful when adding it to the total loss
            sizes = (self.num_experts, self.num_slots)
            unflat = combine_weights.unflatten(dim=-1, sizes=sizes).sum(dim=-1)
            distr = torch.distributions.Categorical(probs=unflat)
            info["load_balance_loss"] = -distr.entropy().mean()
        if return_dispatch_weights:
            info["dispatch_weights"] = dispatch_weights
        if return_combine_weights:
            info["combine_weights"] = combine_weights
        return info
