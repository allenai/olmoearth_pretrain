"""APT-enabled encoder that uses adaptive patching.

This wraps an existing Encoder and replaces uniform patching with APT.
"""

import copy
import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.apt.adaptive_patch_embed import AdaptivePatchEmbed
from olmoearth_pretrain.nn.apt.config import APTConfig
from olmoearth_pretrain.nn.apt.partitioner import QuadtreePartitioner
from olmoearth_pretrain.nn.apt.scorers import EntropyScorer
from olmoearth_pretrain.nn.flexi_patch_embed import FlexiPatchEmbed
from olmoearth_pretrain.nn.flexi_vit import Encoder, TokensAndMasks
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)


class APTEncoder(nn.Module):
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
            encoder: The base encoder to wrap
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

        # Get the FlexiPatchEmbed for the specific bandset
        base_patch_embed = self._get_base_patch_embed(apt_modality, apt_bandset_idx)
        if base_patch_embed is not None:
            self.adaptive_patch_embed = AdaptivePatchEmbed(
                base_patch_embed=base_patch_embed,
                num_scales=apt_config.partitioner.num_scales,
                embedding_size=encoder.embedding_size,
                base_patch_size=apt_config.partitioner.base_patch_size,
            )
            # Verify input channels match the bandset
            self.in_chans = base_patch_embed.proj.in_channels
            if self.in_chans != len(self.bandset_indices):
                logger.warning(
                    f"Patch embed expects {self.in_chans} channels but bandset has "
                    f"{len(self.bandset_indices)} bands - using patch embed's in_channels"
                )
            logger.info(
                f"APT patch embed expects {self.in_chans} input channels, "
                f"using bands {self.bandset_indices}"
            )
        else:
            logger.warning(
                f"Could not find patch embed for {apt_modality}, APT disabled"
            )
            self.adaptive_patch_embed = None
            self.in_chans = None

        # Stats tracking
        self.total_tokens = 0
        self.total_uniform_tokens = 0
        self.num_samples = 0
        self.all_entropy_scores: list[float] = []  # Collect scores for distribution analysis
        self.log_entropy_distribution_every: int = 10  # Log distribution every N samples

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

    def _get_base_patch_embed(
        self, modality: str, bandset_idx: int = 0
    ) -> FlexiPatchEmbed | None:
        """Extract the FlexiPatchEmbed for a specific bandset from the encoder.

        Args:
            modality: The modality name (e.g., "sentinel2_l2a")
            bandset_idx: Which bandset's embed to get (default: 0)

        Returns:
            The FlexiPatchEmbed for the specified bandset, or None if not found
        """
        patch_embeddings = self.encoder.patch_embeddings
        if hasattr(patch_embeddings, "per_modality_embeddings"):
            per_mod = patch_embeddings.per_modality_embeddings
            if modality in per_mod:
                mod_embeds = per_mod[modality]
                # Get the specific bandset's embed (e.g., modality__0, modality__1, etc.)
                target_key = f"{modality}__{bandset_idx}"
                for name, embed in mod_embeds.items():
                    if isinstance(embed, FlexiPatchEmbed) and name == target_key:
                        # Convert DTensor params to regular tensors
                        logger.info(f"Found patch embed for {target_key}")
                        return self._convert_dtensor_to_local(embed)
                # Fallback: if exact key not found, try first match
                logger.warning(
                    f"Could not find {target_key}, trying first available embed"
                )
                for name, embed in mod_embeds.items():
                    if isinstance(embed, FlexiPatchEmbed):
                        logger.info(f"Using fallback patch embed: {name}")
                        return self._convert_dtensor_to_local(embed)
        return None

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
        if self.adaptive_patch_embed is None:
            # Fall back to standard encoder
            return self.encoder(x, patch_size, input_res, token_exit_cfg, fast_pass)

        apt_modality_data = getattr(x, self.apt_modality)

        # Get image data for APT modality: [B, H, W, T, C]
        image_data = apt_modality_data
        if image_data.ndim == 4:
            # [B, H, W, C] -> add time dim
            image_data = image_data.unsqueeze(3)

        b, h, w, t, c = image_data.shape

        # Run APT partitioning and embedding per sample
        all_tokens = []
        all_positions = []
        token_counts = []

        for bi in range(b):
            sample_image = image_data[bi]  # [H, W, T, C]

            # Partition each timestep and collect patches
            sample_tokens = []
            sample_positions = []

            for ti in range(t):
                # Cast to float32 before numpy (numpy doesn't support bfloat16)
                frame = sample_image[:, :, ti, :].cpu().float().numpy()  # [H, W, C]
                # Use configured bands for scoring (scorer handles band selection internally)
                patches = self.partitioner.partition(frame, timestep=ti)

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

                if patches:
                    # Convert to tensor and embed
                    frame_tensor = sample_image[:, :, ti, :].unsqueeze(0)  # [1, H, W, C]

                    # Convert DTensor to regular tensor if needed
                    try:
                        from torch.distributed.tensor import DTensor

                        if isinstance(frame_tensor, DTensor):
                            frame_tensor = frame_tensor.full_tensor()
                    except ImportError:
                        pass

                    # Select only the bands for this bandset
                    # The bandset_indices tells us which channels to use (e.g., [0,1,2,3])
                    if self.bandset_indices is not None:
                        frame_tensor = frame_tensor[..., self.bandset_indices]

                    # Ensure we're on the right device and dtype
                    embed_device = next(self.adaptive_patch_embed.parameters()).device
                    embed_dtype = next(self.adaptive_patch_embed.parameters()).dtype
                    frame_tensor = frame_tensor.to(device=embed_device, dtype=embed_dtype)

                    tokens, positions = self.adaptive_patch_embed(
                        frame_tensor, patches
                    )
                    sample_tokens.append(tokens)
                    sample_positions.append(positions)

                    # Update stats
                    self.total_tokens += len(patches)
                    uniform_count = (h // self.partitioner.base_patch_size) * (
                        w // self.partitioner.base_patch_size
                    )
                    self.total_uniform_tokens += uniform_count

            if sample_tokens:
                all_tokens.append(torch.cat(sample_tokens, dim=0))
                all_positions.append(torch.cat(sample_positions, dim=0))
                token_counts.append(sum(t.shape[0] for t in sample_tokens))

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

        # For now, fall back to standard encoder for actual processing
        # TODO: Integrate APT tokens into the transformer blocks
        # The variable-length token sequences need special handling
        return self.encoder(x, patch_size, input_res, token_exit_cfg, fast_pass)

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
