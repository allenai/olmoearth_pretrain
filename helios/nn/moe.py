from __future__ import annotations
from olmo_core.config import Config
from collections import namedtuple
from typing import Tuple
from torch.autograd import Function
from dataclasses import dataclass
import torch
from torch import autocast
from torch import Tensor
from torch.nn import Module, ModuleList
from torch import nn, einsum
import torch.nn.functional as F

import einx
from typing import Any, NamedTuple
from einops import rearrange, repeat, reduce, pack, unpack
import logging
from helios.nn.attention import Attention, LayerScale, DropPath, fully_shard
from helios.dataset.utils import get_modality_specs_from_names
from helios.nn.flexihelios import BASE_GSD, Encoder, TokensAndMasks, get_cumulative_sequence_lengths
from helios.train.masking import MaskedHeliosSample, MaskValue
from helios.data.constants import ModalitySpec, Modality
import torch.distributed as dist
logger = logging.getLogger(__name__)

# constants

MIN_EXPERT_CAPACITY = 4

class MixtureOfExpertsReturn(NamedTuple):
    outputs: Tensor
    total_aux_loss: Tensor
    balance_loss: Tensor
    router_z_loss: Tensor

    def update_outputs(self, x: Tensor) -> "MixtureOfExpertsReturn":
        return MixtureOfExpertsReturn(
            outputs=x,
            total_aux_loss=self.total_aux_loss,
            balance_loss=self.balance_loss,
            router_z_loss=self.router_z_loss
        )

# helper functions

def pad_dim_to(t, length, dim = 0):
    pad_length = length - t.shape[dim]
    zero_pairs = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length))

def all_gather_same_dim(t):
    t = t.contiguous()
    world_size = dist.get_world_size()
    gathered_tensors = [torch.empty_like(t, device = t.device, dtype = t.dtype) for i in range(world_size)]
    dist.all_gather(gathered_tensors, t)
    return gathered_tensors

def gather_sizes(t, *, dim):
    size = torch.tensor(t.shape[dim], device = t.device, dtype = torch.long)
    sizes = all_gather_same_dim(size)
    return torch.stack(sizes)

def has_only_one_value(t):
    return (t == t[0]).all()

def all_gather_variable_dim(t, dim = 0, sizes = None):
    device, rank, world_size = t.device, dist.get_rank(), dist.get_world_size()

    if not exists(sizes):
        sizes = gather_sizes(t, dim = dim)

    if has_only_one_value(sizes):
        gathered_tensors = all_gather_same_dim(t)
        gathered_tensors = torch.cat(gathered_tensors, dim = dim)
        return gathered_tensors, sizes

    max_size = sizes.amax().item()

    padded_t = pad_dim_to(t, max_size, dim = dim)
    gathered_tensors = all_gather_same_dim(padded_t)

    gathered_tensors = torch.cat(gathered_tensors, dim = dim)
    seq = torch.arange(max_size, device = device)

    mask = rearrange(seq, 'j -> 1 j') < rearrange(sizes, 'i -> i 1')
    mask = rearrange(mask, 'i j -> (i j)')
    seq = torch.arange(mask.shape[-1], device = device)
    indices = seq[mask]

    gathered_tensors = gathered_tensors.index_select(dim, indices)

    return gathered_tensors, sizes

class AllGatherFunction(Function):
    @staticmethod
    def forward(ctx, x, dim, sizes):
        x, batch_sizes = all_gather_variable_dim(x, dim = dim, sizes = sizes)
        ctx.batch_sizes = batch_sizes.tolist()
        ctx.dim = dim
        return x, batch_sizes

    @staticmethod
    def backward(ctx, grads, _):
        batch_sizes, rank = ctx.batch_sizes, dist.get_rank()
        grads_by_rank = grads.split(batch_sizes, dim = ctx.dim)
        return grads_by_rank[rank], None, None

class AllGather(nn.Module):
    def __init__(self, *, dim = 0):
        super().__init__()
        self.dim = dim

    def forward(self, x, sizes = None):
        return AllGatherFunction.apply(x, self.dim, sizes)

def split_by_rank(x):
    rank = dist.get_rank()
    out = x[rank]

    if isinstance(x, tuple):
        sizes = tuple(map(lambda t: t.shape[0], x))
    else:
        sizes = (x.shape[1],) * x.shape[0]

    sizes = torch.tensor(sizes, device = out.device, dtype = torch.long)
    return out, sizes

def exists(val):
    return val is not None

def default(val, default):
    if exists(val):
        return val

    return default() if callable(default) else default

def divisible_by(num, den):
    return (num % den) == 0

def chunk_num(num, chunks):
    num_per_chunk, remainder = divmod(num, chunks)

    out = []
    for i in range(chunks):
        n = num_per_chunk
        out.append(n + int(i < remainder))

    return out

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def cast_tuple(el, len = 1):
    return el if isinstance(el, tuple) else ((el,) * len)

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

# tensor related helper functions

def cumsum_exclusive(t, dim = -3):
    assert dim < 0
    num_pad_dims = -dim - 1
    pre_padding = (0, 0) * num_pad_dims
    return F.pad(t, (*pre_padding, 1, -1)).cumsum(dim = dim)

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# pytorch one hot throws an error if there are out of bound indices.
# tensorflow, in contrast, does not throw an error

def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    one_hot_classes = max(max_index + 1, max_length)
    return F.one_hot(indexes, one_hot_classes)[..., :max_length]

# rms normalization

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale

# expert class
# best performing was ff geglu with multiplicative bias (just after gating)

class GEGLU(Module):
    def __init__(
        self,
        dim,
        mult_bias = True
    ):
        super().__init__()
        self.mult_bias = nn.Parameter(torch.ones(dim)) if mult_bias else 1.

    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x * self.mult_bias

class Expert(Module):
    def __init__(
        self,
        dim,
        hidden_mult = 4,
        mult_bias = True,
        prenorm = False
    ):
        super().__init__()
        dim_hidden = int(dim * hidden_mult * 2 / 3)

        self.net = Sequential(
            RMSNorm(dim) if prenorm else None,
            nn.Linear(dim, dim_hidden * 2),
            GEGLU(dim_hidden, mult_bias = mult_bias),
            nn.Linear(dim_hidden, dim)
        )

        self.apply(self.init_)

    def init_(self, module):
        if isinstance(module, nn.Linear):
            dim = module.weight.shape[0]
            std = dim ** -0.5

            module.weight.data.uniform_(-std, std)
            module.bias.data.uniform_(-std, std)

    def forward(self, x):
        return self.net(x)

class Experts(Module):
    def __init__(
        self,
        experts,
        is_distributed = None,
        allow_var_seq_len = False # whether to handle variable sequence length
    ):
        super().__init__()
        self.num_experts = len(experts)
        self.experts = ModuleList(experts)

        # distributed related settings

        self.is_distributed = is_distributed
        if not exists(self.is_distributed):
            self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1

        self.all_gather = AllGather()

        self.allow_var_seq_len = allow_var_seq_len

        # device tracker, since need to manually move experts not in use to CPU in distributed

        self.register_buffer('dummy', torch.ones(1), persistent = False)

    @property
    def device(self):
        return self.dummy.device

    def all_experts_to_cpu_besides(self, selection):
        if isinstance(selection, int):
            experts = [self.experts[selection]]
        if isinstance(selection, slice):
            experts = self.experts[selection]
        else:
            experts = selection

        experts_set = set(experts)

        for expert in self.experts:
            device = self.device if expert in experts_set else 'cpu'
            expert.to(device)

    def forward(
        self,
        x,
        is_distributed = None
    ):
        """
        einops notation:
        b - batch
        r - rank (device / machines)
        e - experts
        n - sequence (number of tokens per expert)
        d - feature dimension
        """

        # declare some variables

        is_distributed = default(is_distributed, self.is_distributed)
        shape, num_experts = x.shape, self.num_experts
        seq_len = shape[-2]

        # for now naively all gather across batch dimension if distributed, optimize later

        world_size = 1
        rank = 0

        if is_distributed:
            seq_sizes = gather_sizes(x, dim = -2)
            var_seq_len = not has_only_one_value(seq_sizes)

            assert self.allow_var_seq_len or not var_seq_len, 'number of tokens per expert must be the same - if you want the framework to handle it, set `allow_var_seq_len = True` on `Experts`'

            # if variable sequence length, pad

            if var_seq_len:
                max_seq_size = seq_sizes.amax().item()
                x = pad_dim_to(x, max_seq_size, dim = -2)

            # gather and concat across batches, accounting for variable batch sizes

            x, batch_sizes = self.all_gather(x)
            total_batch_size = batch_sizes.sum().item()

            world_size = dist.get_world_size()
            rank = dist.get_rank()

        # the experts in use on the rank

        num_experts_per_rank = num_experts
        expert_slice = slice(0, num_experts)

        if is_distributed:
            if world_size <= num_experts:
                num_experts_across_ranks = chunk_num(num_experts, world_size)
                start_indices = cumsum_exclusive(torch.tensor(num_experts_across_ranks), dim = -1)

                num_experts_per_rank = num_experts_across_ranks[rank]
                num_experts_batches_across_ranks = tuple(i * total_batch_size for i in num_experts_across_ranks)

                expert_start_index = start_indices[rank].item()
            else:
                num_batch_chunks = world_size // num_experts
                total_ranks_in_use = num_batch_chunks * num_experts

                expert_start_index = rank // num_batch_chunks

                batch_splits = chunk_num(total_batch_size, num_batch_chunks)
                num_experts_batches_across_ranks = batch_splits * num_experts

                # for now, remaining machines just process nothing

                remain_ranks = world_size % num_experts
                num_experts_batches_across_ranks += (0,) * remain_ranks

                num_experts_per_rank = int(rank < total_ranks_in_use)

            assert len(num_experts_batches_across_ranks) == world_size

            expert_slice = slice(expert_start_index, expert_start_index + num_experts_per_rank)

        # if distributed, each machine only handles subset of experts and batch

        x = rearrange(x, 'b e n d -> e b n d')

        if is_distributed:
            x, expert_batch_packed_shape = pack_one(x, '* n d')

            x = x.split(num_experts_batches_across_ranks, dim = 0)
            x, experts_per_rank_sizes = split_by_rank(x)

            if num_experts_per_rank > 0:
                x = rearrange(x, '(e b) n d -> e b n d', e = num_experts_per_rank)
            else:
                x = x.reshape(num_experts, *x.shape)

        # get the experts in use

        self.all_experts_to_cpu_besides(expert_slice)

        experts = self.experts[expert_slice]

        # route tokens to appropriate experts

        outs = []

        for expert, expert_input in zip(experts, x):
            out = expert(expert_input)
            outs.append(out)

        if len(outs) > 0:
            outs = torch.stack(outs)
        else:
            outs = torch.empty_like(x, requires_grad = self.training)

        # all gather across merged expert batches dimensions
        # then split the batch dimension back

        if is_distributed:
            outs = rearrange(outs, 'e b n d -> (e b) n d')
            outs, _ = self.all_gather(outs, sizes = experts_per_rank_sizes)
            outs = unpack_one(outs, expert_batch_packed_shape, '* n d')

        outs = rearrange(outs, 'e b n d -> b e n d')

        if is_distributed:
            outs = outs.split(batch_sizes.tolist())
            outs, _ = split_by_rank(outs)

            # account for padded sequence length
            outs = outs[..., :seq_len, :]

        assert outs.shape == shape
        return outs

@autocast('cuda', enabled = False)
def topk(x, k):
    """
    differentiable top-k on last dimension
    """

    values, indices = torch.topk(x, k = k, dim = -1)
    return values, indices

# the below code is almost all transcribed from the official tensorflow version, from which the papers are written
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/moe.py

# gating network

class TopNGating(Module):

    def __init__(
        self,
        dim,
        num_gates,
        eps = 1e-9,
        top_n = 2,
        threshold_train: float | Tuple[float, ...] = 0.2,
        threshold_eval: float | Tuple[float, ...] = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        straight_through_dispatch_tensor = True
    ):
        super().__init__()
        self.eps = eps
        self.num_gates = num_gates
        self.to_gates = nn.Linear(dim, num_gates, bias = False)

        assert top_n >= 2, 'must be 2 or more experts'
        self.top_n = top_n
        top_n_minus_1 = top_n - 1

        threshold_train = cast_tuple(threshold_train, top_n_minus_1)
        threshold_eval = cast_tuple(threshold_eval, top_n_minus_1)

        assert len(threshold_train) == len(threshold_eval) == top_n_minus_1

        self.register_buffer('threshold_train', torch.tensor([eps, *threshold_train]))
        self.register_buffer('threshold_eval', torch.tensor([eps, *threshold_eval]))

        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

        self.straight_through_dispatch_tensor = straight_through_dispatch_tensor
        self.register_buffer('zero', torch.zeros((1,)), persistent = False)

    def forward(
        self,
        x,
        routing_tokens,
        noise_gates = False,
        noise_mult = 1.
    ):
        """
        einstein notation:

        b - batch
        n - sequence
        e - experts
        k - top-n experts
        c - capacity
        """

        *_, b, group_size, dim, dtype, top_n, num_gates, eps = *x.shape, x.dtype, self.top_n, self.num_gates, self.eps

        # threshold, capacity depending on training or eval

        suffix = 'train' if self.training else 'eval'

        threshold = getattr(self, f'threshold_{suffix}')
        capacity_factor = getattr(self, f'capacity_factor_{suffix}')

        # Each sequence sends (at most?) expert_capacity positions to each expert.
        # Static expert_capacity dimension is needed for expert batch sizes

        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)

        # gate logits and gates
        if routing_tokens is not None:
            gate_logits = self.to_gates(routing_tokens)  # B, 1, num_experts
            gate_logits = gate_logits.repeat(1, x.shape[1], 1)
        else:
            gate_logits = self.to_gates(x)
        maybe_noised_gate_logits = gate_logits

        if noise_gates:
            noise = gumbel_noise(maybe_noised_gate_logits)
            maybe_noised_gate_logits = maybe_noised_gate_logits + noise * noise_mult

        raw_gates = maybe_noised_gate_logits.softmax(dim = -1)

        # find top N experts per position

        gates, gate_indices = torch.topk(raw_gates, k = top_n, dim = -1)

        # move the top-n dimension to be first

        gates = rearrange(gates, '... k -> k ...')
        gate_indices = rearrange(gate_indices, '... k -> k ...')

        # masks

        one_hot_gate_indices = F.one_hot(gate_indices, num_gates)
        mask = one_hot_gate_indices.float()

        mask_1 = mask[0] # needed for balancing loss

        # normalize top-n gate scores

        denom = reduce(gates, 'k ... -> 1 ...', 'sum').clamp(min = eps)
        gates = gates / denom

        # best performing policy was to route to the second expert, with probability of min(1., score / threshold), where score = gate2 / (gate1 + gate2)
        # optimal threshold was ~ 0.2
        # generalized to more than 2 experts

        probs = torch.zeros_like(gates).uniform_(0., 1.)

        should_route = probs < einx.divide('k b n, k -> k b n', gates, threshold.clamp(min = eps))

        # tokens should always be routed to first expert
        # threshold for first expert already set to very small number, but just in case

        should_route[0, ...] = True

        mask *= rearrange(should_route.float(), '... -> ... 1')

        mask_cumsum = cumsum_exclusive(mask, dim = -2) # along sequence dimension

        # compute assignment to experts - (batch, seq, experts)

        # This is the position within the expert's mini-batch for this sequence

        positions = []
        prev_expert_count = 0.

        for n in range(self.top_n):
            position_in_expert = (mask_cumsum[n] + prev_expert_count) * mask[n]

            # Remove the elements that don't fit. (batch, sequence, experts)
            mask[n] *= (position_in_expert < expert_capacity_f).float()

            # How many examples in this sequence go to this expert - needed for the next iteration as offset
            prev_expert_count = reduce(mask[n], '... n e -> ... 1 e', 'sum') + prev_expert_count

            # (batch, sequence)
            position_in_expert = reduce(position_in_expert, '... n e -> ... n', 'sum')
            positions.append(position_in_expert)

        positions = torch.stack(positions)

        # (k, batch, sequence) - mostly ones, but zeros where something didn't fit
        mask_flat = reduce(mask, '... n e -> ... n', 'sum')

        # (k, batch, sequence) - weighted assignment
        # following https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py#L1903
        gates = gates * mask_flat

        # (batch, sequence, experts, expert_capacity)

        combine_tensor = einx.multiply(
            'k b n, k b n, k b n e, k b n c -> k b n e c',
            gates,
            mask_flat,
            one_hot_gate_indices,
            safe_one_hot(positions.long(), expert_capacity)
        )

        combine_tensor = reduce(combine_tensor, 'k b n e c -> b n e c', 'sum')

        # dispatch tensor

        dispatch_tensor = combine_tensor.bool().type(dtype)

        if self.straight_through_dispatch_tensor:
            dispatch_tensor = dispatch_tensor + combine_tensor - combine_tensor.detach()

        # balance losses - (batch, experts)
        # We want to equalize the fraction of the batch assigned to each expert

        if self.training:
            density_1 = reduce(mask_1, '... n e -> ... e', 'mean')
            density_1_proxy = reduce(raw_gates, '... n e -> ... e', 'mean') # Something continuous that is correlated with what we want to equalize.

            balance_loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)
        else:
            balance_loss = self.zero

        # calculate the router z-loss proposed in paper

        if self.training:
            router_z_loss = torch.logsumexp(gate_logits, dim = -1)
            router_z_loss = torch.square(router_z_loss)
            router_z_loss = router_z_loss.mean()
        else:
            router_z_loss = self.zero

        return dispatch_tensor, combine_tensor, balance_loss, router_z_loss

# plain mixture of experts

class MoE(Module):

    def __init__(self,
        dim,
        num_experts = 16,
        expert_hidden_mult = 4,
        threshold_train = 0.2,
        threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        gating_top_n = 2,
        balance_loss_coef = 1e-2,
        router_z_loss_coef = 1e-3,
        experts: Module | None = None,
        straight_through_dispatch_tensor = True,
        differentiable_topk = False,
        differentiable_topk_fused = True,
        is_distributed = None,
        allow_var_seq_len = False
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts

        self.gate = TopNGating(
            dim,
            top_n = gating_top_n,
            num_gates = num_experts,
            straight_through_dispatch_tensor = straight_through_dispatch_tensor,
            threshold_train = threshold_train,
            threshold_eval = threshold_eval,
            capacity_factor_train = capacity_factor_train,
            capacity_factor_eval = capacity_factor_eval
        )

        experts = default(experts, lambda: [Expert(dim = dim, hidden_mult = expert_hidden_mult) for _ in range(num_experts)])

        self.experts = Experts(
            experts,
            is_distributed = is_distributed,
            allow_var_seq_len = allow_var_seq_len
        )

        self.balance_loss_coef = balance_loss_coef
        self.router_z_loss_coef = router_z_loss_coef

    def forward(
        self,
        x,
        routing_tokens,
        noise_gates = False,
        noise_mult = 1.
    ):
        dispatch_tensor, combine_tensor, balance_loss, router_z_loss = self.gate(x, routing_tokens, noise_gates = noise_gates, noise_mult = noise_mult)

        # dispatch

        expert_inputs = einsum('b n d, b n e c -> b e c d', x, dispatch_tensor)

        # feed the expert inputs through the experts.

        expert_outputs = self.experts(expert_inputs)

        # combine

        output = einsum('b e c d, b n e c -> b n d', expert_outputs, combine_tensor)

        # losses

        weighted_balance_loss = balance_loss * self.balance_loss_coef
        weighted_router_z_loss = router_z_loss * self.router_z_loss_coef

        # combine the losses

        total_aux_loss = weighted_balance_loss + weighted_router_z_loss

        return MixtureOfExpertsReturn(output, total_aux_loss, balance_loss, router_z_loss)

# sparse moe block
# in particular, they found that adding a feedforward before or after greatly stabilized the training and improved results

class SparseMoEBlock(Module):

    def __init__(
        self,
        moe: MoE,
        *,
        add_ff_before = False,
        add_ff_after = True
    ):
        super().__init__()
        dim = moe.dim

        self.moe = moe
        self.moe_prenorm = RMSNorm(dim)

        self.ff_before = Expert(dim, prenorm = True) if add_ff_before else None
        self.ff_after = Expert(dim, prenorm = True) if add_ff_after else None

    def forward(
        self,
        x,
        routing_tokens,
        noise_gates = False,
        noise_mult = 1.
    ):

        # feedforward before

        if exists(self.ff_before):
            x = self.ff_before(x) + x

        # mixture of experts layer

        residual = x

        moe_out, total_aux_loss, balance_loss, router_z_loss = self.moe(self.moe_prenorm(x), routing_tokens=routing_tokens, noise_gates = noise_gates, noise_mult = noise_mult)

        x = moe_out + residual

        # feedforward after

        if exists(self.ff_after):
            x = self.ff_after(x) + x

        return MixtureOfExpertsReturn(x, total_aux_loss, balance_loss, router_z_loss)


class SparseMoEBlockWithAttn(Module):
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
        self.mlp = SparseMoEBlock(moe=MoE(dim=dim, num_experts=8, expert_hidden_mult=mlp_ratio))
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        routing_tokens: Tensor,
        y: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> MixtureOfExpertsReturn:
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
        moe_return: MixtureOfExpertsReturn = self.mlp(self.norm2(x), routing_tokens)
        x = x + self.drop_path(self.ls2(moe_return.outputs))
        return moe_return.update_outputs(x)

    def apply_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP to the model."""
        fully_shard(self, **fsdp_kwargs)

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        self.compile()


class MoEEncoder(Encoder):
    """Encoder module that processes masked input samples into token representations."""

    cross_attn: bool = False

    def __init__(
        self,
        embedding_size: int,
        max_patch_size: int,
        min_patch_size: int,
        num_heads: int,
        mlp_ratio: float,
        depth: int,
        drop_path: float,
        supported_modalities: list[ModalitySpec],
        max_sequence_length: int,
        learnable_channel_embeddings: bool = True,
        random_channel_embeddings: bool = False,
        num_projection_layers: int = 1,
        aggregate_then_project: bool = True,
        use_flash_attn: bool = False,
        frozen_patch_embeddings: bool = False,
        qk_norm: bool = False,
    ):
        """Initialize the encoder.

        Args:
            embedding_size: Size of token embeddings
            max_patch_size: Maximum patch size for patchification
            min_patch_size: Minimum patch size for patchification
            num_heads: Number of attention heads
            mlp_ratio: Ratio for MLP hidden dimension
            depth: Number of transformer layers
            drop_path: Drop path rate
            supported_modalities: list documenting modalities used in a given model instantiation
            max_sequence_length: Maximum sequence length
            learnable_channel_embeddings: Whether to use learnable channel embeddings
            random_channel_embeddings: Initialize channel embeddings randomly (zeros if False)
            num_projection_layers: The number of layers to use in the projection. If >1, then
                a ReLU activation will be applied between layers
            aggregate_then_project: If True, then we will average the tokens before applying
                the projection. If False, we will apply the projection first.
            use_flash_attn: Whether to use flash attention
            frozen_patch_embeddings: If True, we freeze the embedding layer, as recommended in
                https://arxiv.org/pdf/2104.02057, Section 4.2
            qk_norm: Whether to apply normalization to Q and K in attention
        """
        super().__init__(
            embedding_size=embedding_size,
            max_patch_size=max_patch_size,
            min_patch_size=min_patch_size,
            depth=depth,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
            learnable_channel_embeddings=learnable_channel_embeddings,
            drop_path=drop_path,
            supported_modalities=supported_modalities,
            use_flash_attn=use_flash_attn,
            random_channel_embeddings=random_channel_embeddings,
            qk_norm=qk_norm,
        )

        # todo - make this only the later blocks
        self.blocks = nn.ModuleList(
            [
                SparseMoEBlockWithAttn(
                    embedding_size,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_norm=qk_norm,
                    norm_layer=nn.LayerNorm,  # TODO: This should be configurable
                    cross_attn=self.cross_attn,
                    drop_path=drop_path,
                    use_flash_attn=self.use_flash_attn,
                )
                for _ in range(depth)
            ]
        )

    def apply_attn(  # type: ignore
        self,
        x: dict[str, Tensor],
        timestamps: Tensor,
        patch_size: int,
        input_res: int,
        token_exit_cfg: dict[str, int] | None = None,
        always_pass_none_mask_to_transformer: bool = False,
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, int, Tensor]:
        """Apply the attention to the tokens and masks."""
        tokens_only_dict, original_masks_dict, modalities_to_dims_dict = (
            self.split_tokens_masks_and_dims(x)
        )
        # remove the latlons from the attention sequence
        latlons = tokens_only_dict["latlon"]
        original_masks_dict["latlon_mask"] = (
            torch.ones_like(original_masks_dict["latlon_mask"])
            * MaskValue.DECODER.value
        )
        exit_ids_seq = self.create_exit_seqs(
            tokens_only_dict, original_masks_dict, token_exit_cfg
        )
        # exited tokens are just the linear projection
        exited_tokens, _ = self.collapse_and_combine_hwtc(x)

        tokens_dict = self.composite_encodings.forward(
            tokens_only_dict,
            timestamps,
            patch_size,
            input_res,
        )
        tokens_dict.update(original_masks_dict)

        tokens, mask = self.collapse_and_combine_hwtc(tokens_dict)

        bool_mask = mask == MaskValue.ONLINE_ENCODER.value

        tokens, indices, new_mask, seq_lengths, max_seqlen = self.remove_masked_tokens(
            tokens, bool_mask
        )
        if exit_ids_seq is not None:
            exit_ids_seq, _, _, _, _ = self.remove_masked_tokens(
                exit_ids_seq, bool_mask
            )
            # still linear projections
            exited_tokens, _, _, _, _ = self.remove_masked_tokens(
                exited_tokens, bool_mask
            )
        cu_seqlens = get_cumulative_sequence_lengths(seq_lengths)
        # Pack x tokens
        if self.use_flash_attn:
            og_shape = tokens.shape
            tokens = self.pack_tokens(tokens, new_mask)

        attn_mask = self.get_attn_or_none_mask(
            new_mask, always_pass_none_mask_to_transformer
        )
        total_aux_losses = []
        # Apply attn with varying encoder depths
        for i_blk, blk in enumerate(self.blocks):
            # Skip the zeroth block because we want to use the exited tokens that don't have encodings as this allows trivial solution of predicting the shared encodings
            if (exit_ids_seq is not None) and (i_blk > 0):
                # this should only ever be called by the target encoder,
                # in a torch.no_grad context
                assert exited_tokens is not None
                # If a token should exit, then we update the exit token with the current token at the same position
                exited_tokens = torch.where(
                    condition=(exit_ids_seq == i_blk),
                    input=tokens,
                    other=exited_tokens,
                )
            # we take the inverse of the mask because a value
            # of True indicates the value *should* take part in
            # attention
            # WARNING: THIS MAY CHANGE DEPENDING ON THE ATTENTION IMPLEMENTATION
            tokens, total_aux_loss, _, _ = blk(
                x=tokens,
                routing_tokens=latlons,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                # we will have to specify k and q lens for cross attention
                attn_mask=attn_mask,
            )
            total_aux_losses.append(total_aux_loss)

        if self.use_flash_attn:
            tokens = self.unpack_tokens(tokens, new_mask, og_shape)

        if exit_ids_seq is not None:
            # this should only ever be called by the target encoder,
            # in a torch.no_grad context
            assert exited_tokens is not None
            # full depth
            # IMPORTANT: write this to x
            tokens = torch.where(
                condition=(exit_ids_seq == (i_blk + 1)),  # 2 for full depth
                input=tokens,
                other=exited_tokens,
            )
        # we apply the norm before we add the removed tokens,
        # so that the norm is only computed against "real" tokens
        tokens = self.norm(tokens)
        # we don't care about the mask returned by add_removed_tokens, since we will
        # just use the original, unclipped mask here
        tokens, _ = self.add_removed_tokens(tokens, indices, new_mask)
        tokens_per_modality_dict = self.split_and_expand_per_modality(
            tokens, modalities_to_dims_dict
        )
        # merge original masks and the processed tokens
        tokens_per_modality_dict.update(original_masks_dict)
        return (  # type: ignore
            tokens_per_modality_dict,
            torch.stack(total_aux_losses),
        )

    # TODO: we want to have a single API for the encoder and decoder
    def forward(
        self,
        x: MaskedHeliosSample,
        patch_size: int,
        input_res: int = BASE_GSD,
        token_exit_cfg: dict | None = None,
        always_pass_none_mask_to_transformer: bool = False,
    ) -> tuple[TokensAndMasks, torch.Tensor]:
        """Process masked input samples into token representations.

        Args:
            x: Masked input sample containing the data to be encoded
            patch_size: Size of patches to divide the input into
            input_res: Resolution of the input data
            token_exit_cfg: Configuration for token exit
            always_pass_none_mask_to_transformer: Whether to always pass None as the mask to the transformer, this enables torch based flash attention

        Returns:
            TokensAndMasks containing the encoded representations and their masks
        """
        # TODO: Add step to validate the exit config is valid
        patchified_tokens_and_masks = self.patch_embeddings.forward(x, patch_size)
        if token_exit_cfg is None or any(
            [exit_depth > 0 for exit_depth in token_exit_cfg.values()]
        ):
            (
                patchified_tokens_and_masks,
                total_aux_losses
            ) = self.apply_attn(
                x=patchified_tokens_and_masks,
                timestamps=x.timestamps,
                patch_size=patch_size,
                input_res=input_res,
                token_exit_cfg=token_exit_cfg,
                always_pass_none_mask_to_transformer=always_pass_none_mask_to_transformer,
            )
        else:
            total_aux_losses = torch.empty(1)

        output = TokensAndMasks(**patchified_tokens_and_masks)
        return (  # type: ignore
            output,
            self.project_and_aggregate(output),
            total_aux_losses,
        )


@dataclass
class MoEEncoderConfig(Config):
    """Configuration for the Encoder."""

    supported_modality_names: list[str]
    embedding_size: int = 16
    # This is the base patch size for the patch embedder
    max_patch_size: int = 8
    min_patch_size: int = 1
    num_heads: int = 2
    mlp_ratio: float = 1.0
    depth: int = 2
    drop_path: float = 0.1
    max_sequence_length: int = 12
    learnable_channel_embeddings: bool = True
    random_channel_embeddings: bool = False
    num_projection_layers: int = 1
    aggregate_then_project: bool = True
    use_flash_attn: bool = False
    frozen_patch_embeddings: bool = False
    qk_norm: bool = False

    def validate(self) -> None:
        """Validate the configuration."""
        if len(self.supported_modalities) == 0:
            raise ValueError("At least one modality must be added!")
        else:
            for modality in self.supported_modalities:
                if modality not in Modality.values():
                    raise ValueError(f"Modality {modality} is not supported")

    @property
    def supported_modalities(self) -> list[ModalitySpec]:
        """Get the supported modalities."""
        return get_modality_specs_from_names(self.supported_modality_names)

    def build(self) -> "Encoder":
        """Build the encoder."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        # supported_modality_names is replaced by supported_modalities
        kwargs.pop("supported_modality_names")
        kwargs["supported_modalities"] = self.supported_modalities
        logger.info(f"Encoder kwargs: {kwargs}")
        return MoEEncoder(**kwargs)
