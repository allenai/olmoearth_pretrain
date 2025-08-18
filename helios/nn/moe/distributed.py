"""Distributed training utilities for Soft MoE.

This module provides utilities for distributed training of Soft MoE models,
including all-gather operations and rank-based tensor splitting.

Copied from
https://raw.githubusercontent.com/lucidrains/soft-moe-pytorch/refs/heads/main/soft_moe_pytorch/distributed.py.
"""

from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.autograd import Function


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


def pad_dim_to(t: Tensor, length: int, dim: int = 0) -> Tensor:
    """Pad a tensor along a specific dimension to a target length.

    Args:
        t: The input tensor.
        length: The target length to pad to.
        dim: The dimension to pad along.

    Returns:
        Tensor: The padded tensor with the specified dimension padded to length.
    """
    pad_length = length - t.shape[dim]
    zero_pairs = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length))


def all_gather_same_dim(t: Tensor) -> list[Tensor]:
    """Gather tensors from all processes when they have the same dimension.

    Args:
        t: The tensor to gather from all processes.

    Returns:
        List[Tensor]: List of tensors gathered from all processes.

    Note:
        This function assumes all processes have tensors with the same shape.
    """
    world_size = dist.get_world_size()
    t = t.contiguous()
    gathered_tensors = [
        torch.empty_like(t, device=t.device, dtype=t.dtype) for i in range(world_size)
    ]
    dist.all_gather(gathered_tensors, t)
    return gathered_tensors


def gather_sizes(t: Tensor, *, dim: int) -> Tensor:
    """Gather the sizes of tensors along a specific dimension from all processes.

    Args:
        t: The input tensor.
        dim: The dimension to gather sizes for.

    Returns:
        Tensor: Tensor containing the sizes from all processes.
    """
    size = torch.tensor(t.shape[dim], device=t.device, dtype=torch.long)
    sizes = all_gather_same_dim(size)
    return torch.stack(sizes)


def has_only_one_value(t: Tensor) -> bool:
    """Check if all values in a tensor are the same.

    Args:
        t: The input tensor.

    Returns:
        bool: True if all values in the tensor are identical, False otherwise.
    """
    return (t == t[0]).all()


def all_gather_variable_dim(
    t: Tensor, dim: int = 0, sizes: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Gather tensors from all processes when they may have different dimensions.

    Args:
        t: The tensor to gather from all processes.
        dim: The dimension along which tensors may vary.
        sizes: Optional pre-computed sizes tensor. If None, will be computed.

    Returns:
        Tuple[Tensor, Tensor]: The gathered tensors and the sizes tensor.

    Note:
        This function handles the case where tensors from different processes
        may have different sizes along the specified dimension.
    """
    device = t.device

    if not exists(sizes):
        sizes = gather_sizes(t, dim=dim)

    if has_only_one_value(sizes):
        gathered_tensors = all_gather_same_dim(t)
        gathered_tensors = torch.cat(gathered_tensors, dim=dim)
        return gathered_tensors, sizes

    # Add null check for sizes
    if sizes is None:
        raise ValueError("sizes cannot be None")

    max_size = sizes.amax().item()

    padded_t = pad_dim_to(t, max_size, dim=dim)
    gathered_tensors = all_gather_same_dim(padded_t)

    gathered_tensors = torch.cat(gathered_tensors, dim=dim)
    seq = torch.arange(max_size, device=device)

    mask = rearrange(seq, "j -> 1 j") < rearrange(sizes, "i -> i 1")
    mask = rearrange(mask, "i j -> (i j)")
    seq = torch.arange(mask.shape[-1], device=device)
    indices = seq[mask]

    # Convert gathered_tensors to tensor before calling index_select
    if isinstance(gathered_tensors, list):
        gathered_tensors = torch.cat(gathered_tensors, dim=dim)

    gathered_tensors = gathered_tensors.index_select(dim, indices)  # type: ignore

    return gathered_tensors, sizes


class AllGatherFunction(Function):
    """Custom autograd function for all-gather operations.

    This function provides gradient support for all-gather operations
    by implementing custom forward and backward passes.
    """

    @staticmethod
    def forward(
        ctx: Any, x: Tensor, dim: int, sizes: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        """Forward pass of the all-gather function.

        Args:
            ctx: The context object for storing information for backward pass.
            x: The input tensor to gather.
            dim: The dimension along which to gather.
            sizes: Optional pre-computed sizes tensor.

        Returns:
            Tuple[Tensor, Tensor]: The gathered tensor and the sizes tensor.
        """
        x, batch_sizes = all_gather_variable_dim(x, dim=dim, sizes=sizes)
        ctx.batch_sizes = batch_sizes.tolist()
        ctx.dim = dim
        return x, batch_sizes

    @staticmethod
    def backward(ctx: Any, grads: Tensor, _: Any) -> tuple[Tensor, None, None]:
        """Backward pass of the all-gather function.

        Args:
            ctx: The context object containing information from forward pass.
            grads: The gradient tensor.
            _: Unused parameter for compatibility.

        Returns:
            Tuple[Tensor, None, None]: The gradient for the input tensor and None for other inputs.
        """
        batch_sizes, rank = ctx.batch_sizes, dist.get_rank()
        grads_by_rank = grads.split(batch_sizes, dim=ctx.dim)
        return grads_by_rank[rank], None, None


class AllGather(nn.Module):
    """A module that performs all-gather operations across distributed processes.

    This module provides a convenient interface for gathering tensors from
    all processes in a distributed training setup.
    """

    def __init__(self, *, dim: int = 0) -> None:
        """Initialize the AllGather module.

        Args:
            dim: The dimension along which to gather tensors.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor, sizes: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Forward pass of the all-gather operation.

        Args:
            x: The input tensor to gather from all processes.
            sizes: Optional pre-computed sizes tensor.

        Returns:
            Tuple[Tensor, Tensor]: The gathered tensor and the sizes tensor.
        """
        return AllGatherFunction.apply(x, self.dim, sizes)


def split_by_rank(x: list[Tensor]) -> Tensor:
    """Split a list of tensors and return the tensor corresponding to the current rank.

    Args:
        x: List of tensors, one per rank.

    Returns:
        Tensor: The tensor corresponding to the current process rank.

    Note:
        This function assumes the list has one tensor per rank and returns
        the tensor corresponding to the current process rank.
    """
    rank = dist.get_rank()
    return x[rank]
