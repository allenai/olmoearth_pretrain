"""APT-enabled encoder that uses adaptive patching.

This wraps an existing Encoder and replaces uniform patching with APT.
"""

import copy
import logging
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from olmoearth_pretrain.data.constants import Modality, BASE_GSD
from olmoearth_pretrain.nn.apt.adaptive_patch_embed import AdaptivePatchEmbed
from olmoearth_pretrain.datatypes import MaskValue
from olmoearth_pretrain.nn.apt.config import APTConfig
from olmoearth_pretrain.nn.apt.partitioner import PatchDescriptor, QuadtreePartitioner
from olmoearth_pretrain.nn.apt.scorers import EntropyScorer
from olmoearth_pretrain.nn.flexi_patch_embed import FlexiPatchEmbed
from olmoearth_pretrain.nn.flexi_vit import Encoder, TokensAndMasks
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)


def get_modalities_to_process(
    available_modalities: list[str], supported_modality_names: list[str]
) -> list[str]:
    """Get the modalities to process."""
    modalities_to_process = set(supported_modality_names).intersection(
        set(available_modalities)
    )
    return list(modalities_to_process)


def return_modalities_from_dict(
    per_modality_input_tokens: dict[str, Tensor],
) -> list[str]:
    """Return the modalities from a dictionary of per modality input tokens."""
    return [
        key for key in per_modality_input_tokens.keys() if not key.endswith("_mask")
    ]


class APTEncoder(Encoder):
    """APT-enabled encoder that uses adaptive patching."""

    def __init__(
        self,
        *args: Any,
        apt_num_scales: int = 2,
        apt_base_patch_size: int = 4,
        apt_modality: str = "sentinel2_l2a",
        apt_conv_init: str = "average",
        **kwargs: Any,
    ):
        """Initialize the APT encoder.

        Args:
            *args: Passed to Encoder.
            apt_num_scales: Number of APT scales (1 = base only, 2 = base + 2x, etc.)
            apt_base_patch_size: Smallest patch size in base-patch-grid units.
            apt_modality: Which modality to apply adaptive patching to.
            apt_conv_init: Weight init for ConvDownsample. "kaiming" or "average".
            **kwargs: Passed to Encoder.
        """
        super().__init__(*args, **kwargs)
        self.apt_modality = apt_modality

        # Compute the token width after flattening band_sets into D
        modality_spec = Modality.get(apt_modality)
        num_bandsets = len(modality_spec.band_sets)
        token_dim = self.embedding_size * num_bandsets

        self.adaptive_patch_embed = AdaptivePatchEmbed(
            num_scales=apt_num_scales,
            embedding_size=token_dim,
            base_patch_size=apt_base_patch_size,
            conv_init=apt_conv_init,
        )

    def collapse_and_combine_hwtc_apt(self, x: dict[str, Tensor], apt_tokens: Tensor, apt_positions: Tensor, apt_modality: str) -> tuple[Tensor, Tensor]:
        """Collapse and combine the tokens and masks for the APT modality."""
        tokens, masks = [], []
        available_modalities = return_modalities_from_dict(x)
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )
        for modality in modalities_to_process:
            masked_modality_name = MaskedOlmoEarthSample.get_masked_modality_name(
                modality
            )
            if modality == apt_modality:
                # Pad per-sample APT tokens to same length and stack into a batch
                max_across_samples = max(t.shape[0] for t in apt_tokens)
                padded_tokens = []
                padded_masks = []
                for apt_token_sample in apt_tokens:
                    n_tok = apt_token_sample.shape[0]
                    device = apt_token_sample.device
                    dtype = apt_token_sample.dtype
                    if n_tok < max_across_samples:
                        pad = torch.zeros(max_across_samples - n_tok, apt_token_sample.shape[-1], device=device, dtype=dtype)
                        padded_tokens.append(torch.cat([apt_token_sample, pad], dim=0))
                        padded_masks.append(torch.cat([
                            torch.full((n_tok,), MaskValue.ONLINE_ENCODER.value, device=device),
                            torch.full((max_across_samples - n_tok,), MaskValue.MISSING.value, device=device),
                        ]))
                    else:
                        padded_tokens.append(apt_token_sample)
                        padded_masks.append(
                            torch.full((n_tok,), MaskValue.ONLINE_ENCODER.value, device=device)
                        )
                tokens.append(torch.stack(padded_tokens))  # [B, max_tokens, D]
                masks.append(torch.stack(padded_masks))    # [B, max_tokens]
                continue

            x_modality = x[modality]
            x_modality_mask = x[masked_modality_name]
            tokens.append(rearrange(x_modality, "b ... d -> b (...) d"))
            masks.append(rearrange(x_modality_mask, "b ... -> b (...)"))
        tokens = torch.cat(tokens, dim=1)
        masks = torch.cat(masks, dim=1)

        return tokens, masks

    @staticmethod
    def split_and_expand_per_modality(
        x: Tensor,
        modalities_to_dims_dict: dict,
        apt_modality: str | None = None,
        apt_positions: list[Tensor] | None = None,
        apt_original_dims: tuple | None = None,
        apt_num_bandsets: int = 1,
    ) -> dict[str, Tensor]:
        """Split tokens per modality, scattering APT tokens back to the spatial grid.

        Non-APT modalities use the standard reshape.
        APT modality tokens are scattered back to [B, H, W, T, B_S, D] using
        positions from adaptive_patch_embed, so coarse tokens are broadcast
        to all grid positions they cover.
        """
        tokens_only_dict: dict[str, Tensor] = {}
        tokens_reshaped = 0

        for modality, dims in modalities_to_dims_dict.items():
            middle_dims = dims[1:-1]
            num_tokens = math.prod(middle_dims)
            modality_tokens = x[:, tokens_reshaped : tokens_reshaped + num_tokens]

            if modality == apt_modality and apt_positions is not None:
                # Scatter flat APT tokens back to the original spatial grid
                b, h, w, t, bs, d = apt_original_dims

                output = torch.zeros(b, h, w, t, bs, d, device=x.device, dtype=x.dtype)

                for bi in range(b):
                    positions = apt_positions[bi]  # [N_i, 4]

                    for pi in range(positions.shape[0]):
                        y, xp, size, ti = positions[pi].tolist()
                        tok_start = pi * apt_num_bandsets
                        patch_tokens = modality_tokens[bi, tok_start : tok_start + apt_num_bandsets]  # [bs, d]

                        if size == 1:
                            output[bi, y, xp, ti] = patch_tokens
                        else:
                            output[bi, y : y + size, xp : xp + size, ti] = (
                                patch_tokens.unsqueeze(0).unsqueeze(0).expand(size, size, apt_num_bandsets, d)
                            )

                tokens_only_dict[modality] = output
            else:
                x_modality = modality_tokens.view(x.shape[0], *middle_dims, x.shape[-1])
                tokens_only_dict[modality] = x_modality

            tokens_reshaped += num_tokens

        return tokens_only_dict

    def apply_attn(
        self,
        x: dict[str, Tensor],
        patch_descs: list[list[PatchDescriptor]],
        timestamps: Tensor,
        patch_size: int,
        input_res: int,
        token_exit_cfg: dict[str, int] | None = None,
        fast_pass: bool = False,
    ) -> tuple[dict[str, Tensor], dict[str, Any] | None]:
        """Apply the attention to the tokens and masks."""
        tokens_only_dict, original_masks_dict, modalities_to_dims_dict = (
            self.split_tokens_masks_and_dims(x)
        )
        # already a no-op but we could remove entirely
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

        # Apply adaptive token merging on the post-encoding tokens for the APT modality
        apt_modality_tokens = tokens_dict[self.apt_modality]
        num_bandsets = apt_modality_tokens.shape[-2] if apt_modality_tokens.ndim == 6 else 1
        # Flatten band_sets into D: [B, H, W, T, B_S, D] -> [B, H, W, T, B_S*D]
        if apt_modality_tokens.ndim == 6:
            apt_modality_tokens = rearrange(apt_modality_tokens, "b h w t bs d -> b h w t (bs d)")
        all_tokens, all_positions = self.adaptive_patch_embed.forward(apt_modality_tokens, patch_descs)
        # Unfold band_sets back into sequence dim: [N_i, bs*d] -> [N_i*bs, d]
        # so each band_set is a separate token with D=embedding_size, matching the standard path
        if num_bandsets > 1:
            all_tokens = [rearrange(t, "n (bs d) -> (n bs) d", bs=num_bandsets) for t in all_tokens]

        # Save original spatial dims before overwriting for flat bookkeeping
        apt_original_dims = modalities_to_dims_dict[self.apt_modality]

        # Update dims dict: APT tokens are flat [B, N, D] for collapse/expand bookkeeping
        max_apt_tokens = max(t.shape[0] for t in all_tokens)
        modalities_to_dims_dict[self.apt_modality] = (
            apt_modality_tokens.shape[0], max_apt_tokens, self.embedding_size,
        )

        tokens, mask = self.collapse_and_combine_hwtc_apt(tokens_dict, all_tokens, all_positions, self.apt_modality)

        tokens, indices, new_mask, seq_lengths, max_seqlen, bool_mask = (
            self._maybe_remove_masked_tokens(tokens, mask, fast_pass)
        )

        if exit_ids_seq is not None:
            exit_ids_seq, _, _, _, _ = self.remove_masked_tokens(
                exit_ids_seq, bool_mask
            )
            # still linear projections
            exited_tokens, _, _, _, _ = self.remove_masked_tokens(
                exited_tokens, bool_mask
            )

        # Pack x tokens
        if self.use_flash_attn:
            cu_seqlens = get_cumulative_sequence_lengths(seq_lengths)
            og_shape = tokens.shape
            tokens = self.pack_tokens(tokens, new_mask)
        else:
            cu_seqlens = None

        attn_mask = self._maybe_get_attn_mask(
            new_mask,
            fast_pass=fast_pass,
        )

        if self.has_register_tokens:
            tokens, attn_mask = self.add_register_tokens_and_masks(tokens, attn_mask)

        # Log active token count before attention (use original mask, not new_mask which is None during fast_pass)
        num_active = (mask == MaskValue.ONLINE_ENCODER.value).sum().item()
        num_total = mask.numel()
        logger.info(
            f"APT attention input: tokens={tokens.shape}, "
            f"active={int(num_active)}/{num_total} "
            f"({100*num_active/num_total:.1f}%)"
        )

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

            tokens = blk(
                x=tokens,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                # we will have to specify k and q lens for cross attention
                attn_mask=attn_mask,
            )

        if self.has_register_tokens:
            tokens, register_tokens = self.pop_register_tokens(tokens)
            token_norm_stats = (
                self.get_token_norm_stats(tokens, register_tokens)
                if self.log_token_norm_stats
                else None
            )
        else:
            token_norm_stats = None

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
        tokens = self._maybe_add_removed_tokens(tokens, indices, new_mask, fast_pass)

        tokens_per_modality_dict = self.split_and_expand_per_modality(
            tokens,
            modalities_to_dims_dict,
            apt_modality=self.apt_modality,
            apt_positions=all_positions,
            apt_original_dims=apt_original_dims,
            apt_num_bandsets=num_bandsets,
        )
        # merge original masks and the processed tokens
        tokens_per_modality_dict.update(original_masks_dict)
        return tokens_per_modality_dict, token_norm_stats


    def forward(
        self,
        x: MaskedOlmoEarthSample,
        patch_size: int,
        patch_descs: list[list[PatchDescriptor]],
        input_res: int = BASE_GSD,
        token_exit_cfg: dict | None = None,
        fast_pass: bool = False,
    ) -> dict[str, Any]:
        """Process masked input samples into token representations.

        Args:
            x: Masked input sample containing the data to be encoded
            patch_size: Size of patches to divide the input into
            input_res: Resolution of the input data
            token_exit_cfg: Configuration for token exit
            fast_pass: Whether to always pass None as the mask to the transformer, this enables torch based flash attention, and skips mask construciton and sorting

        Returns:
            TokensAndMasks containing the encoded representations and their masks
        """
        if fast_pass and token_exit_cfg is not None:
            raise ValueError("token_exit_cfg cannot be set when fast_pass is True")

        patchified_tokens_and_masks = self.patch_embeddings.forward(x, patch_size)
        if token_exit_cfg is None or any(
            [exit_depth > 0 for exit_depth in token_exit_cfg.values()]
        ):
            patchified_tokens_and_masks, token_norm_stats = self.apply_attn(
                x=patchified_tokens_and_masks,
                timestamps=x.timestamps,
                patch_size=patch_size,
                input_res=input_res,
                token_exit_cfg=token_exit_cfg,
                fast_pass=fast_pass,
                patch_descs=patch_descs,
            )
        else:
            token_norm_stats = {}
        output = TokensAndMasks(**patchified_tokens_and_masks)
        output_dict: dict[str, Any] = {
            "tokens_and_masks": output,
        }
        if token_norm_stats:
            output_dict["token_norm_stats"] = token_norm_stats

        if not fast_pass:
            output_dict["project_aggregated"] = self.project_and_aggregate(output)
        return output_dict

class APTEncoderWrapper(nn.Module):
    """Encoder with APT adaptive patching.

    Wraps an existing Encoder and uses APT for the patching stage.
    Currently only supports single-modality (e.g., sentinel2_l2a) for simplicity.
    """

    def __init__(
        self,
        encoder: Encoder,
        apt_config: APTConfig,
        apt_modality: str = "sentinel2_l2a",
        apt_bandset_idx: int = 0,
    ):
        """Initialize APT encoder.

        Args:
            encoder: The base encoder (Encoder or APTEncoder) to wrap
            apt_config: APT configuration
            apt_modality: Which modality to apply APT to (others use uniform patching)
            apt_bandset_idx: Which bandset index to use for APT (default: 0, first bandset)
        """
        super().__init__()
        self.encoder = encoder
        self.apt_config = apt_config
        self.apt_modality = apt_modality
        self.apt_bandset_idx = apt_bandset_idx

        # Get the modality spec and bandset indices
        modality_spec = Modality.get(apt_modality)
        bandsets = modality_spec.bandsets_as_indices()
        if apt_bandset_idx >= len(bandsets):
            raise ValueError(
                f"apt_bandset_idx {apt_bandset_idx} out of range for {apt_modality} "
                f"which has {len(bandsets)} bandsets"
            )
        self.bandset_indices = bandsets[apt_bandset_idx]
        logger.info(
            f"APT using bandset {apt_bandset_idx} with band indices {self.bandset_indices}"
        )

        # Build APT components - scorer uses its own configured bands (e.g., RGB)
        self.scorer = EntropyScorer(
            num_bins=apt_config.scorer.num_bins,
            bands=apt_config.scorer.bands,  # Scorer uses configured bands (e.g., 0,1,2 for RGB)
            normalizer=None,
        )
        self.partitioner = QuadtreePartitioner(
            scorer=self.scorer,
            base_patch_size=apt_config.partitioner.base_patch_size,
            num_scales=apt_config.partitioner.num_scales,
            thresholds=apt_config.partitioner.thresholds,
        )

        # Stats tracking
        self.total_tokens = 0
        self.total_uniform_tokens = 0
        self.num_samples = 0
        self.all_entropy_scores: list[float] = []  # Collect scores for distribution analysis
        self.log_entropy_distribution_every: int = 10  # Log distribution every N samples

        # Convert base Encoder -> APTEncoder with pretrained weights
        self._load_encoder_into_apt_encoder()

    def _load_encoder_into_apt_encoder(self) -> None:
        """Convert base Encoder -> APTEncoder, loading pretrained weights non-strictly.

        Existing encoder weights are preserved. New APT-specific layers
        (adaptive_patch_embed) are randomly initialized.
        """
        if isinstance(self.encoder, APTEncoder):
            logger.info("Encoder is already an APTEncoder, skipping conversion")
            return

        base_encoder = self.encoder
        pretrained_sd = base_encoder.state_dict()
        device = next(base_encoder.parameters()).device
        logger.info(f"Loading encoder into apt encoder with device {device}")

        # Extract architecture config from the base encoder
        block0 = base_encoder.blocks[0]
        num_proj_layers = sum(
            1 for m in base_encoder.project_and_aggregate.projection
            if isinstance(m, nn.Linear)
        )

        apt_encoder = APTEncoder(
            embedding_size=base_encoder.embedding_size,
            max_patch_size=base_encoder.max_patch_size,
            min_patch_size=base_encoder.min_patch_size,
            num_heads=block0.attn.num_heads,
            mlp_ratio=block0.mlp.fc1.out_features / base_encoder.embedding_size,
            depth=len(base_encoder.blocks),
            drop_path=getattr(block0.drop_path, "drop_prob", 0.0),
            supported_modalities=base_encoder.supported_modalities,
            max_sequence_length=base_encoder.max_sequence_length,
            num_register_tokens=base_encoder.num_register_tokens,
            learnable_channel_embeddings=base_encoder.learnable_channel_embeddings,
            random_channel_embeddings=base_encoder.random_channel_embeddings,
            num_projection_layers=num_proj_layers,
            aggregate_then_project=base_encoder.project_and_aggregate.aggregate_then_project,
            use_flash_attn=base_encoder.use_flash_attn,
            qk_norm=not isinstance(block0.attn.q_norm, nn.Identity),
            log_token_norm_stats=base_encoder.log_token_norm_stats,
            tokenization_config=getattr(base_encoder, "tokenization_config", None),
            # APT-specific
            apt_num_scales=self.apt_config.partitioner.num_scales,
            apt_base_patch_size=self.apt_config.partitioner.base_patch_size,
            apt_modality=self.apt_modality,
            apt_conv_init=self.apt_config.embed.conv_init,
        )

        # Load pretrained weights non-strictly — missing keys are new APT layers
        missing, unexpected = apt_encoder.load_state_dict(pretrained_sd, strict=False)
        logger.info(
            f"Converted Encoder -> APTEncoder (non-strict): "
            f"{len(missing)} new keys (randomly init), "
            f"{len(unexpected)} unexpected keys"
        )
        if missing:
            logger.info(f"New APT keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys not loaded: {unexpected}")

        self.encoder = apt_encoder.to(device)

    @property
    def embedding_size(self) -> int:
        """Forward embedding_size from underlying encoder."""
        return self.encoder.embedding_size

    def _log_entropy_distribution(self) -> None:
        """Log entropy score distribution statistics."""
        if not self.all_entropy_scores:
            return

        scores = np.array(self.all_entropy_scores)
        percentiles = np.percentile(scores, [0, 10, 25, 50, 75, 90, 100])
        threshold = self.apt_config.partitioner.thresholds[0]

        # Count how many are above/below threshold
        above = np.sum(scores > threshold)
        below = np.sum(scores <= threshold)
        total = len(scores)

        logger.info(
            f"\n{'='*60}\n"
            f"ENTROPY SCORE DISTRIBUTION (n={total} patches)\n"
            f"{'='*60}\n"
            f"  Threshold: {threshold:.2f}\n"
            f"  Above threshold (→4px): {above} ({100*above/total:.1f}%)\n"
            f"  Below threshold (→8px): {below} ({100*below/total:.1f}%)\n"
            f"  \n"
            f"  Percentiles:\n"
            f"    Min (p0):   {percentiles[0]:.3f}\n"
            f"    p10:        {percentiles[1]:.3f}\n"
            f"    p25:        {percentiles[2]:.3f}\n"
            f"    Median:     {percentiles[3]:.3f}\n"
            f"    p75:        {percentiles[4]:.3f}\n"
            f"    p90:        {percentiles[5]:.3f}\n"
            f"    Max (p100): {percentiles[6]:.3f}\n"
            f"  \n"
            f"  Mean: {scores.mean():.3f}, Std: {scores.std():.3f}\n"
            f"{'='*60}"
        )
        # Clear for next batch
        self.all_entropy_scores = []

    def _convert_dtensor_to_local(self, module: nn.Module) -> nn.Module:
        """Convert any DTensor parameters in a module to regular tensors.

        This is needed because FSDP/distributed training uses DTensors, but
        APT needs to run convolutions with regular tensors.
        """
        # Check if DTensor is available
        try:
            from torch.distributed.tensor import DTensor
        except ImportError:
            return module  # No DTensor support, return as-is

        # Deep copy to avoid modifying the original
        module_copy = copy.deepcopy(module)

        # Convert all DTensor parameters to regular tensors
        for name, param in list(module_copy.named_parameters()):
            if isinstance(param.data, DTensor):
                # Get the full tensor from the DTensor
                local_tensor = param.data.full_tensor()
                # Create a new parameter with the local tensor
                new_param = nn.Parameter(local_tensor, requires_grad=param.requires_grad)
                # Set it on the module
                parts = name.split(".")
                target = module_copy
                for part in parts[:-1]:
                    target = getattr(target, part)
                setattr(target, parts[-1], new_param)

        # Also handle buffers
        for name, buf in list(module_copy.named_buffers()):
            if isinstance(buf, DTensor):
                local_tensor = buf.full_tensor()
                parts = name.split(".")
                target = module_copy
                for part in parts[:-1]:
                    target = getattr(target, part)
                target.register_buffer(parts[-1], local_tensor)

        return module_copy

    def forward(
        self,
        x: MaskedOlmoEarthSample,
        patch_size: int,
        input_res: int = 10,
        token_exit_cfg: dict | None = None,
        fast_pass: bool = False,
    ) -> dict[str, Any]:
        """Forward pass with APT for the target modality.

        For the APT modality, uses adaptive patching.
        For other modalities, falls back to standard uniform patching.
        """
        # APT produces variable-length sequences with padding — masking is required
        if fast_pass:
            logger.debug("APT requires masking; overriding fast_pass=True -> False")
            fast_pass = False

        # we need to get the patchified tokens normally for all the modalities and then the other tokens for the APT modality
        # we also will neeed the positions for the apt modality so that we can properly handle the composite encodings for it

        # Get image data for APT modality: [B, H, W, T, C]
        apt_modality_data = getattr(x, self.apt_modality)

        b, h, w, t, c = apt_modality_data.shape


        #TODO: THis is a slow and bad implementation, we need to improve it
        # Run APT partitioning and embedding per sample
        batch_patch_descs = []
        for bi in range(b):
            sample_image = apt_modality_data[bi]  # [H, W, T, C]

            # Partition each timestep and collect patches
            patch_descs = []

            for ti in range(t):
                # Cast to float32 before numpy (numpy doesn't support bfloat16)
                #TODO: we should not be doign this here anyways so lets just leave it
                frame = sample_image[:, :, ti, :].cpu().float().numpy()  # [H, W, C]
                # Use configured bands for scoring (scorer handles band selection internally)
                patches = self.partitioner.partition(frame, timestep=ti)
                patch_descs.append(patches)

                # Collect entropy scores for distribution analysis
                if patches:
                    self.all_entropy_scores.extend([p.score for p in patches])

                # Log detailed patch info for first few samples
                if self.num_samples < 5 and ti == 0 and patches:

                    scale_counts = {}
                    for p in patches:
                        scale_counts[p.scale] = scale_counts.get(p.scale, 0) + 1
                    uniform_count = (h // self.partitioner.base_patch_size) ** 2
                    logger.info(
                        f"APT Partition [sample={self.num_samples}, t={ti}]: "
                        f"image={h}x{w}, base_patch={self.partitioner.base_patch_size}, "
                        f"patches={len(patches)} (uniform={uniform_count}), "
                        f"by_scale={scale_counts}, "
                        f"sizes={[self.partitioner.base_patch_size * (2**s) for s in scale_counts.keys()]}"
                    )

                    # Update stats
                    self.total_tokens += len(patches)
                    uniform_count = (h // self.partitioner.base_patch_size) * (
                        w // self.partitioner.base_patch_size
                    )
                    self.total_uniform_tokens += uniform_count

            batch_patch_descs.append(patch_descs)


        self.num_samples += b

        # Log stats periodically
        if self.num_samples % 50 == 0 and self.total_uniform_tokens > 0:
            reduction = 1 - (self.total_tokens / self.total_uniform_tokens)
            logger.info(
                f"APT Stats: samples={self.num_samples}, "
                f"tokens={self.total_tokens}, "
                f"uniform_would_be={self.total_uniform_tokens}, "
                f"reduction={reduction:.1%}"
            )
            # Log entropy distribution
            self._log_entropy_distribution()

        # all tokens and all positions we should get a dict witht the tokens and masks we will need to input into the mode

        # For now, fall back to standard encoder for actual processing
        # TODO: Integrate APT tokens into the transformer blocks
        # The variable-length token sequences need special handling
        # we will need to jsut mask the tokens to properly handle this so that it works
        # Packing needs to happen so we will need to generate the masks out of the token sequences so that w
        # TODO: basically we need to figure out how to handle the compositie encodings
        # TODO: We need to  then get to the point that after collapse and combine hwtc everything is the same so that this works as well
        return self.encoder(
            x=x,
            patch_size=patch_size,
            patch_descs=batch_patch_descs,
            input_res=input_res,
            token_exit_cfg=token_exit_cfg,
            fast_pass=fast_pass,
        )

    def get_apt_stats(self) -> dict[str, float]:
        """Get APT token reduction statistics."""
        if self.total_uniform_tokens == 0:
            return {"reduction_ratio": 0.0, "num_samples": 0}

        return {
            "total_tokens": self.total_tokens,
            "total_uniform_tokens": self.total_uniform_tokens,
            "reduction_ratio": 1 - (self.total_tokens / self.total_uniform_tokens),
            "num_samples": self.num_samples,
            "avg_tokens_per_sample": self.total_tokens / max(1, self.num_samples),
        }

    def reset_stats(self) -> None:
        """Reset APT statistics."""
        self.total_tokens = 0
        self.total_uniform_tokens = 0
        self.num_samples = 0
