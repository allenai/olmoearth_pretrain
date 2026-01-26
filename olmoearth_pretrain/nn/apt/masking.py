"""APT Masking Strategy integration.

Provides AdaptiveMaskingStrategy that applies APT patch assignment
before delegating to an existing masking strategy.
"""

import logging
from typing import Any

import torch

from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.apt.partitioner import PatchDescriptor
from olmoearth_pretrain.train.masking import MaskingStrategy

logger = logging.getLogger(__name__)


class AdaptiveMaskingStrategy(MaskingStrategy):
    """Masking strategy that applies APT adaptive patches.

    This is a wrapper that:
    1. Uses precomputed APT patch descriptors (or computes them if not present)
    2. Converts base-grid masks to adaptive patch masks
    3. Ensures large patches get consistent mask values
    """

    def __init__(
        self,
        base_strategy: MaskingStrategy,
        apt_modalities: list[str],
        base_patch_size: int = 16,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
    ):
        """Initialize the adaptive masking strategy.

        Args:
            base_strategy: Base masking strategy to use for mask value assignment
            apt_modalities: List of modality names that should use APT
            base_patch_size: Base patch size in pixels
            encode_ratio: Ratio of tokens to encode
            decode_ratio: Ratio of tokens to decode
        """
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio
        self.base_strategy = base_strategy
        self.apt_modalities = apt_modalities
        self.base_patch_size = base_patch_size

    def apply_mask(
        self,
        batch: OlmoEarthSample,
        patch_size: int | None = None,
        **kwargs: Any,
    ) -> MaskedOlmoEarthSample:
        """Apply masking to the input data.

        If APT patch descriptors are present in the batch, uses them to
        create adaptive masks. Otherwise, falls back to base strategy.

        Args:
            batch: Input data of type OlmoEarthSample
            patch_size: Patch size for spatial masking
            **kwargs: Additional arguments

        Returns:
            MaskedOlmoEarthSample containing the masked data and mask
        """
        # Check if APT partition is available
        apt_partition = getattr(batch, "apt_partition", None)

        if apt_partition is None:
            # Fall back to base strategy
            logger.debug("No APT partition found, using base strategy")
            return self.base_strategy.apply_mask(batch, patch_size, **kwargs)

        # First apply base strategy to get initial masks
        masked_sample = self.base_strategy.apply_mask(batch, patch_size, **kwargs)

        # Now adapt masks for APT modalities
        return self._adapt_masks_for_apt(masked_sample, apt_partition, patch_size)

    def _adapt_masks_for_apt(
        self,
        masked_sample: MaskedOlmoEarthSample,
        apt_partition: Any,
        patch_size: int | None,
    ) -> MaskedOlmoEarthSample:
        """Adapt masks based on APT patch descriptors.

        For each APT modality:
        - Large patches should have consistent mask values across their coverage
        - If any base-grid cell is MISSING, the whole large patch is MISSING
        - Otherwise, use majority vote or first cell's mask value

        Args:
            masked_sample: Sample with initial masks from base strategy
            apt_partition: APT partition result with patch descriptors
            patch_size: Base patch size

        Returns:
            Adapted MaskedOlmoEarthSample
        """
        masked_dict = masked_sample.as_dict(return_none=False)

        for modality_name in self.apt_modalities:
            if modality_name not in masked_dict:
                continue

            mask_name = MaskedOlmoEarthSample.get_masked_modality_name(modality_name)
            if mask_name not in masked_dict:
                continue

            modality_mask = masked_dict[mask_name]
            patch_descriptors = self._get_descriptors_for_modality(
                apt_partition, modality_name
            )

            if not patch_descriptors:
                continue

            # Adapt the mask
            adapted_mask = self._adapt_mask_for_patches(
                modality_mask, patch_descriptors, patch_size or self.base_patch_size
            )
            masked_dict[mask_name] = adapted_mask

        return MaskedOlmoEarthSample(**masked_dict)

    def _get_descriptors_for_modality(
        self,
        apt_partition: Any,
        modality_name: str,
    ) -> list[PatchDescriptor]:
        """Extract patch descriptors for a modality from APT partition.

        Args:
            apt_partition: APT partition result
            modality_name: Name of the modality

        Returns:
            List of patch descriptors
        """
        # Handle different APT partition formats
        if hasattr(apt_partition, "patch_descriptors"):
            descriptors = apt_partition.patch_descriptors
            if isinstance(descriptors, list) and len(descriptors) > 0:
                if isinstance(descriptors[0], list):
                    # Temporal: flatten all timesteps
                    return [p for timestep in descriptors for p in timestep]
                return descriptors
        elif isinstance(apt_partition, dict):
            return apt_partition.get(modality_name, [])
        return []

    def _adapt_mask_for_patches(
        self,
        mask: torch.Tensor,
        patch_descriptors: list[PatchDescriptor],
        patch_size: int,
    ) -> torch.Tensor:
        """Adapt a mask tensor based on patch descriptors.

        Args:
            mask: Original mask tensor [B, H, W, ...] or [H, W, ...]
            patch_descriptors: List of patch descriptors
            patch_size: Base patch size

        Returns:
            Adapted mask tensor
        """
        # Handle batch dimension
        had_batch = mask.ndim >= 3 and mask.shape[0] > 1
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        adapted_mask = mask.clone()
        b = adapted_mask.shape[0]

        for bi in range(b):
            for desc in patch_descriptors:
                size_in_base = desc.size // self.base_patch_size
                if size_in_base <= 1:
                    continue  # Base patch, no adaptation needed

                # Get the region this patch covers
                y_start = desc.y * patch_size
                y_end = y_start + desc.size
                x_start = desc.x * patch_size
                x_end = x_start + desc.size

                # Ensure indices are within bounds
                if y_end > adapted_mask.shape[1] or x_end > adapted_mask.shape[2]:
                    continue

                region = mask[bi, y_start:y_end, x_start:x_end, ...]

                # Determine mask value for the whole region
                # Rule: if any cell is MISSING, whole patch is MISSING
                # Otherwise, use the most common non-MISSING value
                flat_region = region.flatten()

                if (flat_region == MaskValue.MISSING.value).any():
                    mask_value = MaskValue.MISSING.value
                else:
                    # Use first non-missing value (consistent assignment)
                    mask_value = flat_region[0].item()

                # Apply uniform mask value to the region
                adapted_mask[bi, y_start:y_end, x_start:x_end, ...] = mask_value

        if not had_batch and adapted_mask.shape[0] == 1:
            adapted_mask = adapted_mask.squeeze(0)

        return adapted_mask


class APTAwareMaskingConfig:
    """Configuration helper for APT-aware masking.

    Use this to configure masking strategies that work with APT.
    """

    def __init__(
        self,
        base_strategy_type: str = "random",
        apt_modalities: list[str] | None = None,
        base_patch_size: int = 16,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
        base_strategy_kwargs: dict | None = None,
    ):
        """Initialize APT masking config.

        Args:
            base_strategy_type: Type of base masking strategy
            apt_modalities: Modalities to apply APT to
            base_patch_size: Base patch size
            encode_ratio: Encode ratio for masking
            decode_ratio: Decode ratio for masking
            base_strategy_kwargs: Additional kwargs for base strategy
        """
        self.base_strategy_type = base_strategy_type
        self.apt_modalities = apt_modalities or ["sentinel2_l2a"]
        self.base_patch_size = base_patch_size
        self.encode_ratio = encode_ratio
        self.decode_ratio = decode_ratio
        self.base_strategy_kwargs = base_strategy_kwargs or {}

    def build(self) -> AdaptiveMaskingStrategy:
        """Build the adaptive masking strategy.

        Returns:
            Configured AdaptiveMaskingStrategy
        """
        from olmoearth_pretrain.train.masking import MASKING_STRATEGY_REGISTRY

        # Build base strategy
        base_strategy_cls = MASKING_STRATEGY_REGISTRY.get_class(self.base_strategy_type)
        base_strategy = base_strategy_cls(
            encode_ratio=self.encode_ratio,
            decode_ratio=self.decode_ratio,
            **self.base_strategy_kwargs,
        )

        return AdaptiveMaskingStrategy(
            base_strategy=base_strategy,
            apt_modalities=self.apt_modalities,
            base_patch_size=self.base_patch_size,
            encode_ratio=self.encode_ratio,
            decode_ratio=self.decode_ratio,
        )
