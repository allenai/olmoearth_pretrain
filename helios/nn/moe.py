"""MoE for Helios."""

import copy
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange
from olmo_core.config import Config
from torch import Tensor

from helios.data.constants import Modality, ModalitySpec
from helios.dataset.utils import get_modality_specs_from_names
from helios.nn.attention import Block, Mlp
from helios.nn.flexihelios import (
    BASE_GSD,
    Encoder,
    TokensAndMasks,
    get_cumulative_sequence_lengths,
)
from helios.train.masking import MaskedHeliosSample, MaskValue

logger = logging.getLogger(__name__)


def clone_module_list(module: nn.Module, n: int) -> nn.ModuleList:
    """Clone Module.

    Make a `nn.ModuleList` with clones of a given module
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class SwitchFeedForward(nn.Module):
    """Routing among multiple FFNs."""

    def __init__(
        self,
        *,
        capacity_factor: float = 1.0,
        drop_tokens: bool = False,
        n_experts: int = 4,
        expert: Mlp,
        d_model: int,
        route_with_latlons: bool = True,
    ) -> None:
        """Routing among multiple FFNs.

        * `capacity_factor` is the capacity of each expert as a factor relative to ideally balanced load
        * `drop_tokens` specifies whether to drop tokens if more tokens are routed to an expert than the capacity
        * `n_experts` is the number of experts
        * `expert` is the expert layer, a [FFN module](../feed_forward.html)
        * `d_model` is the number of features in a token embedding
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `dropout` is dropout probability in the FFN
        """
        super().__init__()

        self.capacity_factor = capacity_factor
        self.n_experts = n_experts
        self.drop_tokens = drop_tokens

        # make copies of the FFNs
        self.experts = clone_module_list(expert, n_experts)
        # Routing layer and softmax
        self.switch = nn.Linear(d_model, n_experts)
        self.softmax = nn.Softmax(dim=-1)
        self.route_with_latlons = route_with_latlons

    def forward(  # type: ignore
        self, x: torch.Tensor, routing_tokens: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, int, Tensor]:
        """X is the input to the switching module with shape `[batch_size, seq_len, d_model]`."""
        # Get routing probabilities for each of the tokens.
        # $$p_i(x) = \frac{e^{h(x)_i}}{\sum^N_j e^{h(x)_j}}$$
        # where $N$ is the number of experts `n_experts` and
        # $h(\cdot)$ is the linear transformation of token embeddings.

        # Initialize an empty list of dropped tokens
        dropped: list[Tensor] = []

        if self.route_with_latlons:
            # assuming only a single routing token
            routing_tokens = routing_tokens[:, 0]  # B, D
            route_prob = self.softmax(self.switch(routing_tokens))  # B, D
            # we will index according to batches, so no need to flatten
            # Get indexes of tokens going to each expert

            # Get the maximum routing probabilities and the routes.
            # We route to the expert with highest probability
            route_prob_max, routes = torch.max(route_prob, dim=-1)
            indexes_list = [
                torch.eq(routes, i).nonzero(as_tuple=True)[0]
                for i in range(self.n_experts)
            ]  # B, num_experts
            # Initialize an empty tensor to store outputs
            final_output = x.new_zeros(x.shape)
            if self.drop_tokens:
                raise ValueError("drop_tokens unsupported for route_with_latlons")
            # Number of *instances* routed to each expert.
            counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])
            # Get outputs of the expert FFNs
            expert_output = [
                self.experts[i](x[indexes_list[i], :, :]) for i in range(self.n_experts)
            ]
            # Assign to final output
            for i in range(self.n_experts):
                final_output[indexes_list[i], :] = expert_output[i]
            # Multiply by the expert outputs by the probabilities $y = p_i(x) E_i(x)$
            final_output = final_output * route_prob_max.view(-1, 1, 1)

        else:
            x = rearrange(x, "b n d -> n b d")
            # Capture the shape to change shapes later
            seq_len, batch_size, d_model = x.shape
            # Flatten the sequence and batch dimensions
            # this flattening happens as [B1, B1, B1, ... B_N, B_N, B_N]
            x = x.reshape(-1, d_model)
            route_prob = self.softmax(self.switch(x))

            # Get the maximum routing probabilities and the routes.
            # We route to the expert with highest probability
            route_prob_max, routes = torch.max(route_prob, dim=-1)

            # Get indexes of tokens going to each expert
            indexes_list = [
                torch.eq(routes, i).nonzero(as_tuple=True)[0]
                for i in range(self.n_experts)
            ]

            # Initialize an empty tensor to store outputs
            final_output = x.new_zeros(x.shape)

            # Capacity of each expert.
            # $$\mathrm{expert\;capacity} =
            # \frac{\mathrm{tokens\;per\;batch}}{\mathrm{number\;of\;experts}}
            # \times \mathrm{capacity\;factor}$$
            capacity = int(self.capacity_factor * len(x) / self.n_experts)
            # Number of tokens routed to each expert.
            counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])

            # Only drop tokens if `drop_tokens` is `True`.
            if self.drop_tokens:
                # Drop tokens in each of the experts
                for i in range(self.n_experts):
                    # Ignore if the expert is not over capacity
                    if len(indexes_list[i]) <= capacity:
                        continue
                    # Shuffle indexes before dropping
                    indexes_list[i] = indexes_list[i][
                        torch.randperm(len(indexes_list[i]))
                    ]
                    # Collect the tokens over capacity as dropped tokens
                    dropped.append(indexes_list[i][capacity:])
                    # Keep only the tokens upto the capacity of the expert
                    indexes_list[i] = indexes_list[i][:capacity]

            # Get outputs of the expert FFNs
            expert_output = [
                self.experts[i](x[indexes_list[i], :]) for i in range(self.n_experts)
            ]

            # Assign to final output
            for i in range(self.n_experts):
                final_output[indexes_list[i], :] = expert_output[i]

            # Pass through the dropped tokens
            if dropped:
                dropped = torch.cat(dropped)
                final_output[dropped, :] = x[dropped, :]

            # Multiply by the expert outputs by the probabilities $y = p_i(x) E_i(x)$
            final_output = final_output * route_prob_max.view(-1, 1)

            # Change the shape of the final output back to `[batch_size, seq_len, d_model]`
            final_output = rearrange(
                final_output.view(seq_len, batch_size, d_model), "n b d -> b n d"
            )

        # Return
        #
        # * the final output
        # * number of tokens routed to each expert
        # * sum of probabilities for each expert
        # * number of tokens dropped.
        # * routing probabilities of the selected experts
        #
        # These are used for the load balancing loss and logging
        return final_output, counts, route_prob.sum(0), len(dropped), route_prob_max


class SwitchBlock(Block):
    """Transformer block with self/cross attention and an MoE MLP.

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
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            init_values=init_values,
            act_layer=act_layer,
            norm_layer=norm_layer,
            cross_attn=cross_attn,
            use_flash_attn=use_flash_attn,
        )

        self.mlp = SwitchFeedForward(d_model=dim, expert=self.mlp)

    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        routing_tokens: torch.Tensor,
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
            routing_tokens: routing tokens, shape (b, 1, C)
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
        x, counts, route_prob, n_dropped, route_prob_max = self.mlp(
            self.norm2(x), routing_tokens
        )
        x = x + self.drop_path(self.ls2(x))
        return x, counts, route_prob, n_dropped, route_prob_max


class SwitchEncoder(Encoder):
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
            num_projection_layers=num_projection_layers,
            aggregate_then_project=aggregate_then_project,
            frozen_patch_embeddings=frozen_patch_embeddings,
        )

        self.blocks = nn.ModuleList(
            [
                SwitchBlock(
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
        counts, route_prob, n_dropped, route_prob_max = [], [], [], []
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
            tokens, f, p, n_d, p_max = blk(
                x=tokens,
                routing_tokens=latlons,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                # we will have to specify k and q lens for cross attention
                attn_mask=attn_mask,
            )
            counts.append(f)
            route_prob.append(p)
            n_dropped.append(n_d)
            route_prob_max.append(p_max)

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
            torch.stack(counts),
            torch.stack(route_prob),
            n_dropped,
            torch.stack(route_prob_max),
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
                counts,
                route_prob,
                n_dropped,
                route_prob_max,
            ) = self.apply_attn(
                x=patchified_tokens_and_masks,
                timestamps=x.timestamps,
                patch_size=patch_size,
                input_res=input_res,
                token_exit_cfg=token_exit_cfg,
                always_pass_none_mask_to_transformer=always_pass_none_mask_to_transformer,
            )
        else:
            counts = torch.empty(1)
            route_prob = torch.empty(1)
            n_dropped = 0
            route_prob_max = torch.empty(1)

        output = TokensAndMasks(**patchified_tokens_and_masks)
        return (  # type: ignore
            output,
            self.project_and_aggregate(output),
            counts,
            route_prob,
            n_dropped,
            route_prob_max,
        )


@dataclass
class SwitchEncoderConfig(Config):
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
        return SwitchEncoder(**kwargs)
