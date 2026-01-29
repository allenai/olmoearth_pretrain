"""Masking module."""

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from class_registry import ClassRegistry
from einops import rearrange, repeat

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality, ModalitySpec
from olmoearth_pretrain.datatypes import (
    MaskedOlmoEarthSample,
    MaskValue,
    OlmoEarthSample,
)
from olmoearth_pretrain.decorators import experimental
from olmoearth_pretrain.nn.tokenization import TokenizationConfig
from olmoearth_pretrain.types import ArrayTensor

logger = logging.getLogger(__name__)

# all bandset indices should be tuples of (modality, bandset_idx) so we can create a power set of these combinations from it
ALL_BANDSET_IDXS: list[tuple[str, int]] = []
for modality in Modality.values():
    for bandset_idx in range(modality.num_band_sets):
        ALL_BANDSET_IDXS.append((modality.name, bandset_idx))


class MaskingStrategy:
    """Abstract base class for masking strategies.

    Be sure to implement apply_mask in subclasses.
    """

    tokenization_config: TokenizationConfig | None = None

    @property
    def name(self) -> str:
        """Return the name of the masking strategy."""
        return self.__class__.__name__.replace("MaskingStrategy", "").lower()

    @property
    def encode_ratio(self) -> float:
        """Return the encode ratio."""
        if not hasattr(self, "_encode_ratio"):
            raise AttributeError("Encode ratio not set")
        return self._encode_ratio

    @property
    def decode_ratio(self) -> float:
        """Return the decode ratio."""
        if not hasattr(self, "_decode_ratio"):
            raise AttributeError("Decode ratio not set")
        return self._decode_ratio

    def _get_num_bandsets(self, modality_name: str) -> int:
        """Get the number of bandsets for a modality, using tokenization config if available."""
        if self.tokenization_config is not None:
            return self.tokenization_config.get_num_bandsets(modality_name)
        return Modality.get(modality_name).num_band_sets

    def _get_bandset_indices(self, modality_name: str) -> list[list[int]]:
        """Get the bandset indices for a modality, using tokenization config if available."""
        if self.tokenization_config is not None:
            return self.tokenization_config.get_bandset_indices(modality_name)
        return Modality.get(modality_name).bandsets_as_indices()

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply masking to the input data.

        Args:
            batch: Input data of type OlmoEarthSample
            patch_size: Optional patch size for spatial masking strategies
            **kwargs: Additional arguments for maskings
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_missing_mask(
        self, instance: torch.Tensor, modality: ModalitySpec, mask: torch.Tensor
    ) -> torch.Tensor:
        """Get the missing mask for the input data."""
        missing_mask = mask.new_zeros(mask.shape, dtype=torch.bool)
        bandset_indices = self._get_bandset_indices(modality.name)
        for i, band_set_indices in enumerate(bandset_indices):
            instance_band_set = instance[..., band_set_indices]
            missing_mask_band_set = instance_band_set == MISSING_VALUE
            missing_mask_band_set_any = missing_mask_band_set.any(dim=-1)
            # If any band in the band set is missing, set the whole band set to missing
            missing_mask[..., i] = missing_mask_band_set_any
        return missing_mask

    def fill_mask_with_missing_values(
        self, instance: torch.Tensor, mask: torch.Tensor, modality: ModalitySpec
    ) -> torch.Tensor:
        """Apply a missing mask to the input data."""
        missing_mask = self.get_missing_mask(instance, modality, mask)
        # If we are changing the mask, we need to clone it as it may be a view of a masked used by different modalities
        if missing_mask.any():
            output_mask = mask.clone()
            output_mask[missing_mask] = MaskValue.MISSING.value
        else:
            output_mask = mask
        return output_mask

    def _create_random_mask(
        self,
        modality: ModalitySpec,
        shape: torch.Size,
        patch_size_at_16: int,
        device: torch.device | None = None,
        encode_ratio: float | None = None,
        decode_ratio: float | None = None,
    ) -> ArrayTensor:
        mask_shape = list(shape)
        mask_shape[-1] = self._get_num_bandsets(modality.name)
        if modality.is_spatial:
            patch_size = patch_size_at_16 * modality.image_tile_size_factor
            mask_shape[1] //= patch_size
            mask_shape[2] //= patch_size

        if modality.is_spatial or modality.is_multitemporal:
            b = shape[0]
            num_tokens = math.prod(mask_shape[1:])
        else:
            num_tokens = math.prod(mask_shape[:-1])

        if encode_ratio is None:
            encode_ratio = self.encode_ratio
        if decode_ratio is None:
            decode_ratio = self.decode_ratio

        encode_tokens = int(num_tokens * encode_ratio)
        decode_tokens = int(num_tokens * decode_ratio)
        target_tokens = int(num_tokens - (encode_tokens + decode_tokens))
        flat_mask_tokens = torch.cat(
            [
                torch.full(
                    (encode_tokens,), MaskValue.ONLINE_ENCODER.value, device=device
                ),
                torch.full((decode_tokens,), MaskValue.DECODER.value, device=device),
                torch.full(
                    (target_tokens,), MaskValue.TARGET_ENCODER_ONLY.value, device=device
                ),
            ]
        )

        if modality.is_spatial or modality.is_multitemporal:
            masks = [
                flat_mask_tokens[torch.randperm(num_tokens, device=device)]
                for i in range(b)
            ]
            flat_mask_tokens = torch.stack(masks)
        else:
            flat_mask_tokens = flat_mask_tokens[
                torch.randperm(num_tokens, device=device)
            ]

        mask = flat_mask_tokens.view(*mask_shape)
        if modality.is_spatial:
            mask = repeat(
                mask, "b h w ... -> b (h hp) (w wp) ...", hp=patch_size, wp=patch_size
            )

        return mask

    def _random_fill_unmasked(
        self,
        mask: torch.Tensor,
        modality: ModalitySpec,
        patch_size_at_16: int,
        encode_ratio: float | None = None,
        decode_ratio: float | None = None,
    ) -> ArrayTensor:
        """This function assumes B=1."""
        assert mask.shape[0] == 1, (
            f"_random_fill_unmasked does not support B != 1, got input shape {mask.shape}"
        )
        device = mask.device
        if modality.is_spatial:
            patch_size = patch_size_at_16 * modality.image_tile_size_factor
            # the first two dimensions are spatial; lets turn them
            # from h, w to p_h, p_w
            mask = mask[:, 0::patch_size, 0::patch_size]

        original_shape = mask.shape
        # this only works because we assume B = 1
        flat_mask = mask.flatten()  # N tokens
        not_missing_tokens = flat_mask != MaskValue.MISSING.value
        num_not_missing_tokens = sum(not_missing_tokens)

        if encode_ratio is None:
            encode_ratio = self.encode_ratio
        if decode_ratio is None:
            decode_ratio = self.decode_ratio

        if num_not_missing_tokens == 1:
            encode_tokens = 1
            decode_tokens = 0
        else:
            encode_tokens = int(num_not_missing_tokens * encode_ratio)
            decode_tokens = int(num_not_missing_tokens * decode_ratio)

        target_tokens = int(num_not_missing_tokens - (encode_tokens + decode_tokens))
        flat_mask_tokens = torch.cat(
            [
                torch.full(
                    (encode_tokens,), MaskValue.ONLINE_ENCODER.value, device=device
                ),
                torch.full((decode_tokens,), MaskValue.DECODER.value, device=device),
                torch.full(
                    (target_tokens,), MaskValue.TARGET_ENCODER_ONLY.value, device=device
                ),
            ]
        )

        flat_mask_tokens = flat_mask_tokens[
            torch.randperm(num_not_missing_tokens, device=device)
        ]
        flat_mask[not_missing_tokens] = flat_mask_tokens
        mask = flat_mask.view(*original_shape)
        if modality.is_spatial:
            mask = repeat(
                mask, "b h w ... -> b (h hp) (w wp) ...", hp=patch_size, wp=patch_size
            )

        return mask


MASKING_STRATEGY_REGISTRY = ClassRegistry[MaskingStrategy]()


@MASKING_STRATEGY_REGISTRY.register("time")
class TimeMaskingStrategy(MaskingStrategy):
    """Time structured random masking of the input data."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
    ) -> None:
        """Initialize the masking strategy."""
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio

    def _create_temporal_mask(
        self,
        shape: torch.Size,
        timesteps_with_at_least_one_modality: torch.Tensor,
        device: torch.device | None = None,
    ) -> ArrayTensor:
        b = shape[0]
        t = shape[-2]
        # timesteps withat least one modality are the only ones we can put as either encoder and decoder randomly pick from those instead
        # can we relax the all sample contraint here as we are doing per sample stuff anyways
        present_t = timesteps_with_at_least_one_modality.shape[0]  # across all samples
        assert present_t >= 3
        logger.info(f"Present timesteps: {present_t}")
        encode_times = max(int(self.encode_ratio * present_t), 1)
        decode_times = max(int(self.decode_ratio * present_t), 1)
        target_times = present_t - encode_times - decode_times
        logger.info(
            f"Encode times: {encode_times}, Decode times: {decode_times}, Target times: {target_times}"
        )
        # Create mask values only for the encodable timesteps
        encodable_mask_values = torch.cat(
            [
                torch.full(
                    (encode_times,), MaskValue.ONLINE_ENCODER.value, device=device
                ),
                torch.full((decode_times,), MaskValue.DECODER.value, device=device),
                torch.full(
                    (target_times,), MaskValue.TARGET_ENCODER_ONLY.value, device=device
                ),
            ]
        )

        # Create masks for each sample in the batch
        masks = [
            torch.full(
                (t,), MaskValue.TARGET_ENCODER_ONLY.value, device=device
            ).index_put_(
                (timesteps_with_at_least_one_modality,),
                encodable_mask_values[torch.randperm(present_t, device=device)],
            )
            for _ in range(b)
        ]

        mask = torch.stack(masks)
        return mask

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply random masking to the input data.

        Masking happens temporally, with whole time steps having the same mask. Non-temporal data is randomly masked.

        Args:
            batch: Input data of type OlmoEarthSample
            patch_size: patch size applied to sample
            **kwargs: Additional arguments for maskings

        Returns:
            MaskedOlmoEarthSample containing the masked data and mask
        """
        if patch_size is None:
            raise ValueError("patch_size must be provided for time masking")
        output_dict: dict[str, ArrayTensor | None] = {}
        temporal_mask = None
        timesteps_with_at_least_one_modality = (
            batch.timesteps_with_at_least_one_modality
        )
        num_valid_timesteps = timesteps_with_at_least_one_modality.shape[0]
        for modality_name in batch.modalities:
            instance = getattr(batch, modality_name)
            if instance is None:
                # set instance and mask to None
                output_dict[modality_name] = None
                output_dict[
                    MaskedOlmoEarthSample.get_masked_modality_name(modality_name)
                ] = None
            else:
                if modality_name == "timestamps":
                    output_dict[modality_name] = instance
                    continue

                if isinstance(instance, torch.Tensor):
                    device: torch.device | None = instance.device
                else:
                    device = None

                modality = Modality.get(modality_name)
                shape = instance.shape
                if not modality.is_multitemporal or num_valid_timesteps < 3:
                    mask = self._create_random_mask(modality, shape, patch_size, device)
                else:
                    if temporal_mask is None:
                        # if there are timesteps that we wouldn't want to pick we should call a seprate mask creation function
                        logger.info(
                            f"Creating temporal mask for modality {modality.name}"
                        )
                        temporal_mask = self._create_temporal_mask(
                            shape, timesteps_with_at_least_one_modality, device
                        )
                    b_s = self._get_num_bandsets(modality.name)
                    b, h, w = list(shape[:-2]) + [1] * (3 - len(shape[:-2]))
                    # Repeat shares a view of the temporal masks so if we don't clone future changes may propogate across modalities
                    mask = repeat(
                        temporal_mask, "b t -> b h w t b_s", h=h, w=w, b_s=b_s
                    )
                    mask = mask.view(*shape[:-1], b_s).clone()
                # After setting up encoder and decoder masks, fill in missing values

                mask = self.fill_mask_with_missing_values(instance, mask, modality)
                output_dict[modality_name] = instance
                output_dict[
                    MaskedOlmoEarthSample.get_masked_modality_name(modality_name)
                ] = mask
        return MaskedOlmoEarthSample(**output_dict)


@MASKING_STRATEGY_REGISTRY.register("space")
class SpaceMaskingStrategy(MaskingStrategy):
    """Spatially structured random masking of the input data."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
    ) -> None:
        """Initialize the masking strategy."""
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio

    def _create_patch_spatial_mask(
        self,
        modality: ModalitySpec,
        shape: torch.Size,
        patch_size_at_16: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Create a h_p x w_p spatial mask.

        Here, h_p and w_p are the number of patches along height and width dimension
        respectively.

        The mask computed here is modality-agnostic, but we still expect a specific
        modality to be passed since it will be used to compute h_p/w_p. The mask will
        then need to be resized using _resize_spatial_mask_for_modality to the
        modality's patch size.

        Args:
            modality: the modality we are using to compute h_p/w_p.
            shape: the shape of the image for that modality.
            patch_size_at_16: the patch size measured in 10 m/pixel pixels.
            device: the device to use.
        """
        if not modality.is_spatial:
            raise ValueError("Non-spatial modality {modality}")

        b, h, w = shape[:3]

        patch_size = patch_size_at_16 * modality.image_tile_size_factor
        assert (h % patch_size == 0) and (w % patch_size == 0)
        h_p = h // patch_size
        w_p = w // patch_size

        patches = h_p * w_p
        encode_patches = int(self.encode_ratio * patches)
        decode_patches = int(self.decode_ratio * patches)
        target_patches = patches - encode_patches - decode_patches

        flat_mask = torch.cat(
            [
                torch.full(
                    (encode_patches,), MaskValue.ONLINE_ENCODER.value, device=device
                ),
                torch.full((decode_patches,), MaskValue.DECODER.value, device=device),
                torch.full(
                    (target_patches,),
                    MaskValue.TARGET_ENCODER_ONLY.value,
                    device=device,
                ),
            ]
        )

        masks = [flat_mask[torch.randperm(patches, device=device)] for i in range(b)]
        random_batch_mask = torch.stack(masks)
        return rearrange(random_batch_mask, "b (h w) -> b h w", h=h_p, w=w_p)

    def _resize_spatial_mask_for_modality(
        self,
        patch_mask: torch.Tensor,
        modality: ModalitySpec,
        patch_size_at_16: int,
    ) -> ArrayTensor:
        """Resize the mask computed by _create_patch_spatial_mask for the given modality.

        Args:
            patch_mask: the mask computed by _create_patch_spatial_mask.
            modality: the modality to compute the mask for.
            patch_size_at_16: the patch size measured in 10 m/pixel pixels.
        """
        if not modality.is_spatial:
            raise ValueError("Non-spatial modality {modality}")

        patch_size = patch_size_at_16 * modality.image_tile_size_factor
        mask = repeat(
            patch_mask, "b h w -> b (h hps) (w wps)", hps=patch_size, wps=patch_size
        )
        return mask

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply random masking to the input data.

        Masking happens in patchified form, with whole patches having the same mask. Non-spatial data is randomly masked.

        Args:
            batch: Input data of type OlmoEarthSample
            patch_size: patch size applied to sample, at an image_tile_size_factor == 16
            **kwargs: Additional arguments for maskings

        Returns:
            MaskedOlmoEarthSample containing the masked data and mask
        """
        if patch_size is None:
            raise ValueError("patch_size must be provided for space masking")
        output_dict: dict[str, ArrayTensor | None] = {}
        patch_spatial_mask = None
        # Same spatial mask for all modalities
        for modality_name in batch.modalities:
            instance = getattr(batch, modality_name)
            if instance is None:
                # set instance and mask to None
                output_dict[modality_name] = None
                output_dict[
                    MaskedOlmoEarthSample.get_masked_modality_name(modality_name)
                ] = None
                continue

            if modality_name == "timestamps":
                output_dict[modality_name] = instance
                continue

            if isinstance(instance, torch.Tensor):
                device: torch.device | None = instance.device
            else:
                device = None

            modality = Modality.get(modality_name)
            shape = instance.shape
            if not modality.is_spatial:
                logger.warning(
                    f"Modality {modality.name} is not spatial, random masking strategy will be applied"
                )
                mask = self._create_random_mask(modality, shape, patch_size, device)
            else:
                if patch_spatial_mask is None:
                    logger.info(f"Creating spatial mask for modality {modality.name}")
                    patch_spatial_mask = self._create_patch_spatial_mask(
                        modality, shape, patch_size, device
                    )
                resized_spatial_mask = self._resize_spatial_mask_for_modality(
                    patch_spatial_mask, modality, patch_size
                )

                if resized_spatial_mask.shape[0:3] != shape[0:3]:
                    raise ValueError(
                        f"Mismached shapes for {modality.name}: "
                        f"computed mask {mask.shape} but image shape is {shape}"
                    )

                if len(shape) == 5:
                    t = shape[-2]
                else:
                    t = 1
                b_s = self._get_num_bandsets(modality.name)
                # Mask is a view of the spatial mask, so changes to mask will change spatial_mask
                mask = repeat(resized_spatial_mask, "... -> ... t b_s", t=t, b_s=b_s)
                mask = mask.view(*shape[:-1], b_s).clone()
            mask = self.fill_mask_with_missing_values(instance, mask, modality)

            # Keep data as is
            output_dict[modality_name] = instance
            output_dict[
                MaskedOlmoEarthSample.get_masked_modality_name(modality_name)
            ] = mask
        return MaskedOlmoEarthSample(**output_dict)


@MASKING_STRATEGY_REGISTRY.register("space_time")
class SpaceTimeMaskingStrategy(MaskingStrategy):
    """Randomly select space or time masking and apply it to the input data."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
    ) -> None:
        """Initialize the masking strategy."""
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio
        self.generator = np.random.default_rng(0)

        self.space_strategy = SpaceMaskingStrategy(encode_ratio, decode_ratio)
        self.time_strategy = TimeMaskingStrategy(encode_ratio, decode_ratio)

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply space or time masking to the input data."""
        has_enough_timesteps = batch.valid_time >= 3
        # I need a timestamp mask

        if not has_enough_timesteps:
            logger.debug(f"Valid time: {batch.valid_time}, Time: {batch.time}")
        if (self.generator.random() < 0.5) or (not has_enough_timesteps):
            logger.info("Applying space masking")
            return self.space_strategy.apply_mask(batch, patch_size, **kwargs)
        else:
            logger.info("Applying time masking")
            return self.time_strategy.apply_mask(batch, patch_size, **kwargs)


@MASKING_STRATEGY_REGISTRY.register("random_space")
class RandomSpaceMaskingStrategy(MaskingStrategy):
    """Randomly select space or random masking."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
    ) -> None:
        """Initialize the masking strategy."""
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio
        self.generator = np.random.default_rng(0)

        self.random_strategy = RandomMaskingStrategy(encode_ratio, decode_ratio)
        self.space_strategy = SpaceMaskingStrategy(encode_ratio, decode_ratio)

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply space or time masking to the input data."""
        if self.generator.random() < 0.5:
            logger.info("Applying space masking")
            return self.space_strategy.apply_mask(batch, patch_size, **kwargs)
        else:
            logger.info("Applying random masking")
            return self.random_strategy.apply_mask(batch, patch_size, **kwargs)


class ModalityCrossMaskingStrategy(MaskingStrategy):
    """Abstract class for masking strategies that select a seperate set of bandsets to encode and decode on top of another masking strategy."""

    def __init__(
        self,
        strategy: MaskingStrategy,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
        allow_encoding_decoding_same_bandset: bool = False,
        min_encoded_bandsets: int | None = None,
        max_encoded_bandsets: int | None = None,
        min_decoded_bandsets: int | None = None,
        max_decoded_bandsets: int | None = None,
        only_decode_modalities: list[str] = [],
    ) -> None:
        """Initialize the masking strategy.

        Args:
            strategy: The base masking strategy to apply before cross-modality masking.
            encode_ratio: Ratio of tokens to encode (default: 0.5). Used by the base strategy.
            decode_ratio: Ratio of tokens to decode (default: 0.5). Used by the base strategy.
            allow_encoding_decoding_same_bandset: If True, allows the same bandset to be both
                encoded and decoded. If False (default), encoded and decoded bandsets are disjoint.
            min_encoded_bandsets: Minimum number of bandsets to encode per sample. If None (default),
                encodes all available bandsets when there are 3+ modalities, or 1 bandset when there are 2 modalities.
            max_encoded_bandsets: Maximum number of bandsets to encode per sample. If None (default),
                encodes all available bandsets.
            min_decoded_bandsets: Minimum number of bandsets to decode per sample. Only used when
                allow_encoding_decoding_same_bandset=True. If None (default), uses 1.
            max_decoded_bandsets: Maximum number of bandsets to decode per sample. Only used when
                allow_encoding_decoding_same_bandset=True. If None (default), uses all available bandsets.
            only_decode_modalities: List of modality names that should only be used for decoding,
                never for encoding. Empty list by default (all modalities can be encoded).
        """
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio
        self.strategy = strategy
        self.allow_encoding_decoding_same_bandset = allow_encoding_decoding_same_bandset
        if min_encoded_bandsets is None:
            assert max_encoded_bandsets is None, (
                "max_encoded_bandsets must be set if min_encoded_bandsets is set"
            )
        else:
            assert min_encoded_bandsets > 1, (
                "min_encoded_bandsets must be greater than 1 so that we don't only  \
                encode a modality that is randomly masked on batch dimension ie latlon"
            )
        self.min_encoded_bandsets = min_encoded_bandsets
        self.max_encoded_bandsets = max_encoded_bandsets
        self.min_decoded_bandsets = min_decoded_bandsets
        self.max_decoded_bandsets = max_decoded_bandsets
        self.only_decode_modalities = only_decode_modalities

    def get_sample_present_modalities_bandsets(
        self, batch: MaskedOlmoEarthSample
    ) -> list[list[tuple[str, int]]]:
        """Get the modalities that are present for each sample."""
        masked_sample_dict = batch.as_dict(return_none=False)
        batch_size = batch.timestamps.shape[0]
        present_modalities_bandsets: list[list[tuple[str, int]]] = [
            [] for _ in range(batch_size)
        ]
        for modality in batch.modalities:
            if modality == "timestamps":
                continue
            modality_mask_name = MaskedOlmoEarthSample.get_masked_modality_name(
                modality
            )
            modality_mask = masked_sample_dict[modality_mask_name]
            missing_values_mask = modality_mask == MaskValue.MISSING.value
            # Find the samples where the modality is completely missing
            is_modality_completely_missing_for_samples = torch.all(
                missing_values_mask.view(batch_size, -1), dim=1
            )
            is_modality_present_for_samples = (
                ~is_modality_completely_missing_for_samples
            )
            num_bandsets = modality_mask.shape[-1]

            present_sample_indices = torch.where(is_modality_present_for_samples)[0]
            for sample_idx in present_sample_indices:
                sample_idx = sample_idx.item()
                for bandset_idx in range(num_bandsets):
                    # check if that modality bandset has any encoded tokens if it has no encoded tokens it is not present
                    is_any_tokens_encoded_for_sample = (
                        torch.sum(
                            modality_mask[sample_idx, ..., bandset_idx]
                            == MaskValue.ONLINE_ENCODER.value
                        )
                        > 0
                    )
                    # only say something is present if it has any encoded tokens
                    # A little hacky but basically means that we leave the bandset untouched for encoding and decoding
                    if (
                        not is_any_tokens_encoded_for_sample
                        and modality not in self.only_decode_modalities
                    ):
                        continue
                    present_modalities_bandsets[sample_idx].append(
                        (modality, bandset_idx)
                    )
        return present_modalities_bandsets

    def select_encoded_decoded_bandsets(
        self, present_modalities_bandsets: list[list[tuple[str, int]]]
    ) -> list[tuple[set[tuple[str, int]], set[tuple[str, int]]]]:
        """Select the encoded and decoded bandsets for each sample."""
        encoded_decoded_bandsets: list[
            tuple[set[tuple[str, int]], set[tuple[str, int]]]
        ] = []
        for sample_idx in range(len(present_modalities_bandsets)):
            present_modalities_bandsets_for_sample = present_modalities_bandsets[
                sample_idx
            ]
            # If there is only one modality, we only encode not decode
            if len(present_modalities_bandsets_for_sample) == 1:
                encoded_bandset_idxs = set(present_modalities_bandsets_for_sample)
                decoded_bandset_idxs = set()
            # If there are two modalities, we encode one and decode the other
            elif len(present_modalities_bandsets_for_sample) == 2:
                encoded_bandset_idxs = set([present_modalities_bandsets_for_sample[0]])
                decoded_bandset_idxs = set([present_modalities_bandsets_for_sample[1]])
            # If there are more than two modalities, we randomly select some to encode and the rest to decode
            else:
                # Select Indices to Encode
                num_present_modalities = len(present_modalities_bandsets_for_sample)
                encodable_modality_bandsets = [
                    modality_bandset
                    for modality_bandset in present_modalities_bandsets_for_sample
                    if modality_bandset[0] not in self.only_decode_modalities
                ]
                num_encodable_modality_bandsets = len(encodable_modality_bandsets)
                # if min and max are none we will always encode all encodable bandsets
                # if min is none, max must be none
                upper_limit = num_encodable_modality_bandsets
                if not self.allow_encoding_decoding_same_bandset:
                    # Otherwise no decoding will be done
                    upper_limit -= 1
                if self.max_encoded_bandsets is None:
                    max_encoded_bandsets = upper_limit
                else:
                    max_encoded_bandsets = min(self.max_encoded_bandsets, upper_limit)

                if self.min_encoded_bandsets is None:
                    min_encoded_bandsets = num_encodable_modality_bandsets
                else:
                    min_encoded_bandsets = min(
                        self.min_encoded_bandsets, num_encodable_modality_bandsets
                    )
                # Ensure min is less than max
                min_encoded_bandsets = min(min_encoded_bandsets, max_encoded_bandsets)
                num_bandsets_to_encode = np.random.randint(
                    min_encoded_bandsets, max_encoded_bandsets + 1
                )
                encoded_idxs = np.random.choice(
                    len(encodable_modality_bandsets),
                    size=num_bandsets_to_encode,
                    replace=False,
                )
                encoded_bandset_idxs = set(
                    [encodable_modality_bandsets[i] for i in encoded_idxs]
                )
                # Select Indices to Decode
                min_decoded_bandsets = min(
                    self.min_decoded_bandsets or 1, num_present_modalities
                )
                max_decoded_bandsets = min(
                    self.max_decoded_bandsets or num_present_modalities,
                    num_present_modalities,
                )
                if self.allow_encoding_decoding_same_bandset:
                    # Otherwise randomly choose between min and max
                    num_decoded_bandsets = np.random.randint(
                        min_decoded_bandsets, max_decoded_bandsets + 1
                    )
                    decoded_idxs = np.random.choice(
                        len(present_modalities_bandsets_for_sample),
                        size=num_decoded_bandsets,
                        replace=False,
                    )
                    decoded_bandset_idxs = set(
                        [
                            present_modalities_bandsets_for_sample[i]
                            for i in decoded_idxs
                        ]
                    )
                else:
                    available_decoded_bandset_idxs = list(
                        set(present_modalities_bandsets_for_sample)
                        - encoded_bandset_idxs
                    )
                    num_decoded_bandsets = len(available_decoded_bandset_idxs)
                    min_decoded_bandsets = min(
                        min_decoded_bandsets, num_decoded_bandsets
                    )
                    max_decoded_bandsets = min(
                        max_decoded_bandsets, num_decoded_bandsets
                    )
                    # select the decoded bandsets
                    decoded_idxs = np.random.choice(
                        len(available_decoded_bandset_idxs),
                        size=num_decoded_bandsets,
                        replace=False,
                    )
                    decoded_bandset_idxs = set(
                        [available_decoded_bandset_idxs[i] for i in decoded_idxs]
                    )
            encoded_decoded_bandsets.append(
                (encoded_bandset_idxs, decoded_bandset_idxs)
            )
        return encoded_decoded_bandsets

    def overide_strategy_mask(self, modality_spec: ModalitySpec) -> bool:
        """Overide the mask for a modality depending on the strategy being modality cross masked.

        e.g in time masking, static in time data is randomly masked but we want that data to be either used to predict temporally masked data or
        predicted from temporal data.
        """
        return False

    def apply_bandset_mask_rules(
        self,
        masked_batch: MaskedOlmoEarthSample,
        encoded_decoded_bandsets: list[
            tuple[set[tuple[str, int]], set[tuple[str, int]]]
        ],
        present_modalities_bandsets: list[list[tuple[str, int]]],
        patch_size: int,
    ) -> MaskedOlmoEarthSample:
        """Compute masks for each band set based on the encode and decode selections.

        The encoded and decoded bandsets are typically computed by the select_encoded_decoded_bandsets method.

        Args:
            masked_batch: The masked batch to apply the mask to.
            encoded_decoded_bandsets: The encoded and decoded bandsets for each sample.
            present_modalities_bandsets: The present modalities and bandsets for each sample.
            patch_size: The patch size being applied

        Returns:
            The masked batch with the masks applied.
        """
        masked_batch_dict = masked_batch.as_dict(return_none=False)
        num_encoded: None | torch.Tensor = None
        num_decoded: None | torch.Tensor = None
        for modality in masked_batch.modalities:
            if modality == "timestamps":
                continue
            masked_modality_name = MaskedOlmoEarthSample.get_masked_modality_name(
                modality
            )
            modality_spec = Modality.get(modality)
            modality_mask = masked_batch_dict[masked_modality_name]
            # with 1-12 patch size I got a run time aliasing error when writing to the modality mask
            out_modality_mask = modality_mask.clone()
            num_bandsets = modality_mask.shape[-1]

            for sample_idx in range(masked_batch.timestamps.shape[0]):
                encoded_bandset_idxs, decoded_bandset_idxs = encoded_decoded_bandsets[
                    sample_idx
                ]
                available_modalities = [
                    modality_bandset[0]
                    for modality_bandset in present_modalities_bandsets[sample_idx]
                ]
                if modality not in available_modalities:
                    logger.debug(
                        f"Modality {modality} not present for sample {sample_idx}"
                    )
                    continue

                for bandset_idx in range(num_bandsets):
                    is_encoded = (modality, bandset_idx) in encoded_bandset_idxs
                    is_decoded = (modality, bandset_idx) in decoded_bandset_idxs

                    # For different masking strategies, some modalities may not be able to follow the structured masking strategy
                    # e.g static in space is randomly masked in space masking
                    # e.g static in time is randomly masked in time masking
                    # By setting to all encode or decode depending on the strategy,
                    # the modality the structure of the strategy is maintained
                    if self.overide_strategy_mask(modality_spec):
                        if is_encoded:
                            forced_mask_value = MaskValue.ONLINE_ENCODER.value
                        elif is_decoded:
                            forced_mask_value = MaskValue.DECODER.value
                        else:
                            continue
                        logger.debug(
                            f"Setting {modality} bandset {bandset_idx} to {forced_mask_value}"
                        )
                        not_missing_mask = (
                            modality_mask[sample_idx, ..., bandset_idx]
                            != MaskValue.MISSING.value
                        )
                        out_modality_mask[sample_idx, ..., bandset_idx] = torch.where(
                            not_missing_mask,
                            forced_mask_value,
                            modality_mask[sample_idx, ..., bandset_idx],
                        )
                        continue

                    if not is_encoded:
                        # Supress all encoded values for a not encoded bandset
                        online_encoder_mask = (
                            modality_mask[sample_idx, ..., bandset_idx]
                            == MaskValue.ONLINE_ENCODER.value
                        )

                        out_modality_mask[sample_idx, ..., bandset_idx] = torch.where(
                            online_encoder_mask.clone(),
                            MaskValue.TARGET_ENCODER_ONLY.value,
                            modality_mask[sample_idx, ..., bandset_idx],
                        )
                        continue

                    if not is_decoded:
                        decoder_mask = (
                            modality_mask[sample_idx, ..., bandset_idx]
                            == MaskValue.DECODER.value
                        )

                        out_modality_mask[sample_idx, ..., bandset_idx] = torch.where(
                            decoder_mask,
                            MaskValue.TARGET_ENCODER_ONLY.value,
                            modality_mask[sample_idx, ..., bandset_idx],
                        )
            # check we have more than 0 encoded and decoded tokens.
            # this should happen very rarely (only in the S2-only ablation when
            # the h, w is small)
            flat_mask = torch.flatten(out_modality_mask, start_dim=1)
            encoded_for_modality = (flat_mask == MaskValue.ONLINE_ENCODER.value).sum(
                dim=-1
            )
            decoded_for_modality = (flat_mask == MaskValue.DECODER.value).sum(dim=-1)
            if num_encoded is None:
                num_encoded = encoded_for_modality
            else:
                num_encoded += encoded_for_modality
            if num_decoded is None:
                num_decoded = decoded_for_modality
            else:
                num_decoded += decoded_for_modality
            masked_batch_dict[masked_modality_name] = out_modality_mask
        # Again - no_encoded_indices and no_decoded_indices should have length > 0 very rarely
        # (so far this has only happened when we ablate S2 only, and have a small h, w), so these
        # loops should not be entered very often.
        no_encoded_indices = torch.argwhere(num_encoded == 0)
        no_decoded_indices = torch.argwhere(num_decoded == 0)
        for i in no_encoded_indices:
            for key, val in masked_batch_dict.items():
                if key.endswith("mask"):
                    modality_mask = val[i]
                    modality_name = MaskedOlmoEarthSample.get_unmasked_modality_name(
                        key
                    )
                    if modality_name in self.only_decode_modalities:
                        continue
                    modality_spec = Modality.get(modality_name)
                    masked_batch_dict[key][i] = self._random_fill_unmasked(
                        modality_mask, modality_spec, patch_size
                    )
        for i in no_decoded_indices:
            for key, val in masked_batch_dict.items():
                if key.endswith("mask"):
                    modality_mask = val[i]
                    modality_name = MaskedOlmoEarthSample.get_unmasked_modality_name(
                        key
                    )
                    if modality_name in self.only_decode_modalities:
                        continue
                    modality_spec = Modality.get(modality_name)
                    masked_batch_dict[key][i] = self._random_fill_unmasked(
                        modality_mask, modality_spec, patch_size
                    )
        masked_batch = MaskedOlmoEarthSample(**masked_batch_dict)

        return masked_batch

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply space masking to the input data."""
        if patch_size is None:
            # this is because we use a random-masking proxy in case of
            # no encoded or decoded tokens.
            raise ValueError("patch_size must be provided for cross masking")

        masked_sample = self.strategy.apply_mask(batch, patch_size, **kwargs)
        present_modalities_bandsets = self.get_sample_present_modalities_bandsets(
            masked_sample
        )
        encoded_decoded_bandsets = self.select_encoded_decoded_bandsets(
            present_modalities_bandsets
        )
        masked_sample = self.apply_bandset_mask_rules(
            masked_sample,
            encoded_decoded_bandsets,
            present_modalities_bandsets,
            patch_size,
        )

        return masked_sample


@MASKING_STRATEGY_REGISTRY.register("modality_cross_space")
class ModalityCrossSpaceMaskingStrategy(ModalityCrossMaskingStrategy):
    """Randomly select a modality and apply space masking to it."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
        allow_encoding_decoding_same_bandset: bool = False,
        min_encoded_bandsets: int = 2,
        max_encoded_bandsets: int | None = None,
        min_decoded_bandsets: int | None = None,
        max_decoded_bandsets: int | None = None,
        only_decode_modalities: list[str] = [],
    ) -> None:
        """Initialize the masking strategy."""
        space_strategy = SpaceMaskingStrategy(encode_ratio, decode_ratio)
        super().__init__(
            strategy=space_strategy,
            encode_ratio=encode_ratio,
            decode_ratio=decode_ratio,
            allow_encoding_decoding_same_bandset=allow_encoding_decoding_same_bandset,
            min_encoded_bandsets=min_encoded_bandsets,
            max_encoded_bandsets=max_encoded_bandsets,
            min_decoded_bandsets=min_decoded_bandsets,
            max_decoded_bandsets=max_decoded_bandsets,
            only_decode_modalities=only_decode_modalities,
        )

    def overide_strategy_mask(self, modality_spec: ModalitySpec) -> bool:
        """Overide the random mask  for the given modality by the encoding and decoding bandsets."""
        # For space masking non spatial data is randomly masked but we want to use the encoding and decoding bandsets
        # to determine the mask for the non spatial data
        return not modality_spec.is_spatial


@experimental(
    "This masking strategy is experimental and may not work with all combinations of modalities"
)
@MASKING_STRATEGY_REGISTRY.register("modality_cross_time")
class ModalityCrossTimeMaskingStrategy(ModalityCrossMaskingStrategy):
    """Randomly select a modality and apply time masking to it."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
        allow_encoding_decoding_same_bandset: bool = False,
        min_encoded_bandsets: int = 2,
        max_encoded_bandsets: int | None = None,
        min_decoded_bandsets: int | None = None,
        max_decoded_bandsets: int | None = None,
        only_decode_modalities: list[str] = [],
    ) -> None:
        """Initialize the masking strategy."""
        space_strategy = SpaceMaskingStrategy(encode_ratio, decode_ratio)
        super().__init__(
            strategy=space_strategy,
            encode_ratio=encode_ratio,
            decode_ratio=decode_ratio,
            allow_encoding_decoding_same_bandset=allow_encoding_decoding_same_bandset,
            min_encoded_bandsets=min_encoded_bandsets,
            max_encoded_bandsets=max_encoded_bandsets,
            min_decoded_bandsets=min_decoded_bandsets,
            max_decoded_bandsets=max_decoded_bandsets,
            only_decode_modalities=only_decode_modalities,
        )

    def overide_strategy_mask(self, modality_spec: ModalitySpec) -> bool:
        """Overide the random mask  for the given modality by the encoding and decoding bandsets."""
        # For time masking static data is randomly masked but we want to use the encoding and decoding bandsets
        # to determine the mask for the static data
        return not modality_spec.is_spatial


@experimental(
    "This masking strategy is experimental and may not work with all combinations of modalities"
)
@MASKING_STRATEGY_REGISTRY.register("modality_cross_space_time")
class ModalityCrossSpaceTimeMaskingStrategy(MaskingStrategy):
    """Randomly apply space cross modality masking and time cross modality masking."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
        allow_encoding_decoding_same_bandset: bool = False,
        min_encoded_bandsets: int = 2,
        max_encoded_bandsets: int | None = None,
        min_decoded_bandsets: int | None = None,
        max_decoded_bandsets: int | None = None,
        only_decode_modalities: list[str] = [],
    ) -> None:
        """Initialize the masking strategy."""
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio
        self.time_strategy = ModalityCrossTimeMaskingStrategy(
            encode_ratio,
            decode_ratio,
            allow_encoding_decoding_same_bandset=allow_encoding_decoding_same_bandset,
            min_encoded_bandsets=min_encoded_bandsets,
            max_encoded_bandsets=max_encoded_bandsets,
            min_decoded_bandsets=min_decoded_bandsets,
            max_decoded_bandsets=max_decoded_bandsets,
            only_decode_modalities=only_decode_modalities,
        )
        self.space_strategy = ModalityCrossSpaceMaskingStrategy(
            encode_ratio,
            decode_ratio,
            allow_encoding_decoding_same_bandset=allow_encoding_decoding_same_bandset,
            min_encoded_bandsets=min_encoded_bandsets,
            max_encoded_bandsets=max_encoded_bandsets,
            min_decoded_bandsets=min_decoded_bandsets,
            max_decoded_bandsets=max_decoded_bandsets,
            only_decode_modalities=only_decode_modalities,
        )
        self.generator = np.random.default_rng(0)

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply space and time cross modality masking to the input data."""
        has_enough_timesteps = batch.valid_time >= 3
        if (self.generator.random() < 0.5) or (not has_enough_timesteps):
            logger.info("Applying space masking")
            return self.space_strategy.apply_mask(batch, patch_size, **kwargs)
        else:
            logger.info("Applying time masking")
            return self.time_strategy.apply_mask(batch, patch_size, **kwargs)


@MASKING_STRATEGY_REGISTRY.register("random")
class RandomMaskingStrategy(MaskingStrategy):
    """Randomly masks the input data."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
    ) -> None:
        """Initialize the masking strategy."""
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply random masking to the input data.

        All Masking happens in unpatchified form and not grouped across bandsets
        as the modality data is unpatchified and not grouped across bandsets

        The mask created for the space-time varying modality will be different than
        for the static modality.

        For space-time varying data, we will mask out the same ratio of values for
        all the instances in the batch. However, since a static modality might have
        very few tokens in a batch (e.g. 1 for latlons) instead we mask out a certain
        ratios of values across the entire batch.

        Args:
            batch: Input data of type OlmoEarthSample
            patch_size: patch size applied to sample
            **kwargs: Additional arguments for maskings

        Returns:
            MaskedOlmoEarthSample containing the masked data and mask
        """
        if patch_size is None:
            raise ValueError("patch_size must be provided for random masking")
        output_dict: dict[str, ArrayTensor | None] = {}
        for modality_name in batch.modalities:
            instance = getattr(batch, modality_name)
            if instance is None:
                # set instance and mask to None
                output_dict[modality_name] = None
                output_dict[
                    MaskedOlmoEarthSample.get_masked_modality_name(modality_name)
                ] = None
            else:
                if modality_name == "timestamps":
                    output_dict[modality_name] = instance
                    continue

                if isinstance(instance, torch.Tensor):
                    device: torch.device | None = instance.device
                else:
                    device = None
                modality = Modality.get(modality_name)
                mask = self._create_random_mask(
                    modality, instance.shape, patch_size, device
                )
                mask = self.fill_mask_with_missing_values(instance, mask, modality)
                output_dict[modality_name] = instance
                output_dict[
                    MaskedOlmoEarthSample.get_masked_modality_name(modality_name)
                ] = mask
        return MaskedOlmoEarthSample(**output_dict)


@MASKING_STRATEGY_REGISTRY.register("modality_cross_random")
class ModalityCrossRandomMaskingStrategy(ModalityCrossMaskingStrategy):
    """Randomly select a modality and apply random masking to it."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
        allow_encoding_decoding_same_bandset: bool = False,
        min_encoded_bandsets: int = 2,
        max_encoded_bandsets: int | None = None,
        min_decoded_bandsets: int | None = None,
        max_decoded_bandsets: int | None = None,
        only_decode_modalities: list[str] = [],
    ) -> None:
        """Initialize the masking strategy."""
        random_strategy = RandomMaskingStrategy(encode_ratio, decode_ratio)
        super().__init__(
            strategy=random_strategy,
            encode_ratio=encode_ratio,
            decode_ratio=decode_ratio,
            allow_encoding_decoding_same_bandset=allow_encoding_decoding_same_bandset,
            min_encoded_bandsets=min_encoded_bandsets,
            max_encoded_bandsets=max_encoded_bandsets,
            min_decoded_bandsets=min_decoded_bandsets,
            max_decoded_bandsets=max_decoded_bandsets,
            only_decode_modalities=only_decode_modalities,
        )


@MASKING_STRATEGY_REGISTRY.register("random_increasing")
class RandomIncreasingMaskingStrategy(RandomMaskingStrategy):
    """Gradually increase the masked tokens (reduce encode ratio)."""

    def __init__(
        self,
        initial_encode_ratio: float = 0.5,
        final_encode_ratio: float = 0.1,
        initial_decode_ratio: float = 0.5,
        final_decode_ratio: float = 0.9,
        steps: int = 1000,
    ) -> None:
        """Initialize the masking strategy."""
        super().__init__(initial_encode_ratio, initial_decode_ratio)
        self.initial_encode_ratio = initial_encode_ratio
        self.final_encode_ratio = final_encode_ratio
        self.initial_decode_ratio = initial_decode_ratio
        self.final_decode_ratio = final_decode_ratio
        self.steps = steps
        self.elapsed = 0

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply masking while changing the encode and decode ratio over time."""
        self.elapsed += 1
        if self.elapsed >= self.steps:
            self._encode_ratio = self.final_encode_ratio
            self._decode_ratio = self.final_decode_ratio
        else:
            factor = self.elapsed / self.steps
            self._encode_ratio = (
                self.initial_encode_ratio
                + (self.final_encode_ratio - self.initial_encode_ratio) * factor
            )
            self._decode_ratio = (
                self.initial_decode_ratio
                + (self.final_decode_ratio - self.initial_decode_ratio) * factor
            )
        return super().apply_mask(batch, patch_size, **kwargs)


@MASKING_STRATEGY_REGISTRY.register("random_range")
class RandomRangeMaskingStrategy(MaskingStrategy):
    """Randomly masks the input data."""

    def __init__(
        self,
        min_encode_ratio: float = 0.1,
        max_encode_ratio: float = 0.5,
        min_decode_ratio: float | None = None,
        max_decode_ratio: float | None = None,
    ) -> None:
        """Initialize the masking strategy.

        Args:
            min_encode_ratio: lower bound of range to sample encode ratio.
            max_encode_ratio: upper bound of range to sample encode ratio.
            min_decode_ratio: lower bound of range to sample decode ratio. If None, the
                decode ratio is 1 - (sampled encode ratio).
            max_decode_ratio: upper bound of range to sample decode ratio.
        """
        self.min_encode_ratio = min_encode_ratio
        self.max_encode_ratio = max_encode_ratio
        self.min_decode_ratio = min_decode_ratio
        self.max_decode_ratio = max_decode_ratio
        self._encode_ratio = (min_encode_ratio + max_encode_ratio) / 2

        if min_decode_ratio is not None and max_decode_ratio is not None:
            self._decode_ratio = (min_decode_ratio + max_decode_ratio) / 2
        elif min_decode_ratio is not None or max_decode_ratio is not None:
            raise ValueError(
                "min_decode_ratio and max_decode_ratio must be both None or both not None"
            )
        else:
            self._decode_ratio = 1 - self._encode_ratio

        self.generator = np.random.default_rng(0)

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply random masking to the input data.

        All Masking happens in unpatchified form and not grouped across bandsets
        as the modality data is unpatchified and not grouped across bandsets

        The mask created for the space-time varying modality will be different than
        for the static modality.

        For space-time varying data, we will mask out the same ratio of values for
        all the instances in the batch. However, since a static modality might have
        very few tokens in a batch (e.g. 1 for latlons) instead we mask out a certain
        ratios of values across the entire batch.

        Args:
            batch: Input data of type OlmoEarthSample
            patch_size: patch size applied to sample
            **kwargs: Additional arguments for maskings

        Returns:
            MaskedOlmoEarthSample containing the masked data and mask
        """
        if patch_size is None:
            raise ValueError("patch_size must be provided for random masking")
        output_dict: dict[str, ArrayTensor | None] = {}
        for modality_name in batch.modalities:
            instance = getattr(batch, modality_name)
            if instance is None:
                # set instance and mask to None
                output_dict[modality_name] = None
                output_dict[
                    MaskedOlmoEarthSample.get_masked_modality_name(modality_name)
                ] = None
            else:
                if modality_name == "timestamps":
                    output_dict[modality_name] = instance
                    continue

                if isinstance(instance, torch.Tensor):
                    device: torch.device | None = instance.device
                else:
                    device = None

                modality = Modality.get(modality_name)

                if modality.is_spatial or modality.is_multitemporal:
                    # Create masks per element so that we can leverage _create_random_mask
                    # while also ensuring each example can have its own encode and decode
                    # ratios.
                    batch_size = instance.shape[0]
                    example_encode_ratios = self.generator.uniform(
                        self.min_encode_ratio, self.max_encode_ratio, (batch_size,)
                    )
                    if self.min_decode_ratio is not None:
                        example_decode_ratios = self.generator.uniform(
                            self.min_decode_ratio, self.max_decode_ratio, (batch_size,)
                        )
                    else:
                        example_decode_ratios = 1 - example_encode_ratios

                    example_masks = []
                    for batch_idx in range(batch_size):
                        example_masks.append(
                            self._create_random_mask(
                                modality,
                                instance[batch_idx : batch_idx + 1].shape,
                                patch_size,
                                device,
                                encode_ratio=example_encode_ratios[batch_idx],
                                decode_ratio=example_decode_ratios[batch_idx],
                            )
                        )
                    mask = torch.cat(example_masks, dim=0)

                else:
                    # For ones that could be single token we just pass the whole batch.
                    mask = self._create_random_mask(
                        modality, instance.shape, patch_size, device
                    )

                mask = self.fill_mask_with_missing_values(instance, mask, modality)
                output_dict[modality_name] = instance
                output_dict[
                    MaskedOlmoEarthSample.get_masked_modality_name(modality_name)
                ] = mask
        return MaskedOlmoEarthSample(**output_dict)


@MASKING_STRATEGY_REGISTRY.register("selectable_modality")
class SelectableModalityMaskingStrategy(MaskingStrategy):
    """Like modality masking but we mask some for decoding and others fully.

    Plus we also apply random masking for the remaining modalities.
    """

    def __init__(
        self,
        decodable_modalities: list[str],
        fully_mask_modalities: list[str],
        max_to_mask: int,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
    ) -> None:
        """Initialize the masking strategy."""
        self.decodable_modalities = decodable_modalities
        self.fully_mask_modalities = fully_mask_modalities
        self.max_to_mask = max_to_mask
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio
        self.generator = np.random.default_rng(0)
        self.random_strategy = RandomMaskingStrategy(encode_ratio, decode_ratio)

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply random masking, plus mask certain additional modalities."""
        # First apply random masking.
        masked_sample = self.random_strategy.apply_mask(batch, patch_size, **kwargs)

        # Choose additional modalities to mask entirely (either set DECODER or
        # MISSING).
        all_modalities = self.decodable_modalities + self.fully_mask_modalities
        modality_indices = np.arange(len(all_modalities))
        self.generator.shuffle(modality_indices)
        num_to_mask = self.generator.integers(self.max_to_mask + 1)
        cur_mask_modalities = [
            all_modalities[idx] for idx in modality_indices[0:num_to_mask]
        ]

        logger.debug("Decided to mask modalities: %s", cur_mask_modalities)
        for modality in cur_mask_modalities:
            if modality in self.decodable_modalities:
                value = MaskValue.DECODER.value
            else:
                value = MaskValue.MISSING.value
            logger.debug("Filling modality %s mask with %s", modality, value)
            getattr(
                masked_sample, MaskedOlmoEarthSample.get_masked_modality_name(modality)
            )[:] = value

        return masked_sample


@MASKING_STRATEGY_REGISTRY.register("selectable_random_range_modality")
class SelectableRandomRangeModalityMaskingStrategy(MaskingStrategy):
    """Like modality masking but we mask some for decoding and others fully.

    Plus we also apply random range masking for the remaining modalities.
    """

    def __init__(
        self,
        decodable_modalities: list[str],
        fully_mask_modalities: list[str],
        max_to_mask: int,
        min_encode_ratio: float = 0.1,
        max_encode_ratio: float = 0.5,
        min_decode_ratio: float | None = None,
        max_decode_ratio: float | None = None,
    ) -> None:
        """Initialize the masking strategy."""
        self.decodable_modalities = decodable_modalities
        self.fully_mask_modalities = fully_mask_modalities
        self.max_to_mask = max_to_mask
        self.generator = np.random.default_rng(0)
        self.random_strategy = RandomRangeMaskingStrategy(
            min_encode_ratio, max_encode_ratio, min_decode_ratio, max_decode_ratio
        )
        self._encode_ratio = self.random_strategy._encode_ratio
        self._decode_ratio = self.random_strategy._decode_ratio

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply random masking, plus mask certain additional modalities."""
        # First apply random range masking.
        masked_sample = self.random_strategy.apply_mask(batch, patch_size, **kwargs)

        # Decide how many and which modalities to mask per example.
        all_modalities = self.decodable_modalities + self.fully_mask_modalities
        batch_size = getattr(batch, all_modalities[0]).shape[0]

        for batch_idx in range(batch_size):
            # Choose additional modalities to mask entirely (either set DECODER or
            # MISSING).
            modality_indices = np.arange(len(all_modalities))
            self.generator.shuffle(modality_indices)
            num_to_mask = self.generator.integers(self.max_to_mask + 1)
            cur_mask_modalities = [
                all_modalities[idx] for idx in modality_indices[0:num_to_mask]
            ]

            for modality in cur_mask_modalities:
                if modality in self.decodable_modalities:
                    value = MaskValue.DECODER.value
                else:
                    value = MaskValue.MISSING.value
                getattr(
                    masked_sample,
                    MaskedOlmoEarthSample.get_masked_modality_name(modality),
                )[batch_idx] = value

        return masked_sample


class FixedModalityMaskingStrategy(MaskingStrategy):
    """Abstract class for masking strategies always mask certain modalities on top of another masking strategy."""

    def __init__(
        self,
        strategy: MaskingStrategy,
        decoded_modalities: list[str],
        randomize_missing_modalities: list[str] = [],
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
    ) -> None:
        """Initialize the masking strategy."""
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio
        self.strategy = strategy
        self.decoded_modalities = decoded_modalities
        self.randomize_missing_modalities = randomize_missing_modalities
        self.generator = np.random.default_rng(0)

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply masking to the input data."""
        # Apply other strategy first.
        masked_sample = self.strategy.apply_mask(batch, patch_size, **kwargs)

        # Now mark the decoded_modalities for decoding, similar to SelectableModalityMaskingStrategy.
        for modality in self.decoded_modalities:
            mask = getattr(
                masked_sample, MaskedOlmoEarthSample.get_masked_modality_name(modality)
            )
            if mask is None:
                continue
            instance = getattr(masked_sample, modality)
            mask[:] = MaskValue.DECODER.value
            mask = self.fill_mask_with_missing_values(
                instance, mask, Modality.get(modality)
            )

        # Randomly decide whether to mark the randomize_missing_modalities as missing.
        # We do this on a per-instance basis since we want to make sure we don't mark
        # all the modalities for that instance missing.
        if len(self.randomize_missing_modalities) > 0:
            batch_size = getattr(batch, self.randomize_missing_modalities[0]).shape[0]
            for batch_idx in range(batch_size):
                cur_available_modalities = []
                for modality in self.randomize_missing_modalities:
                    mask = getattr(
                        masked_sample,
                        MaskedOlmoEarthSample.get_masked_modality_name(modality),
                    )
                    # We check it is available everywhere since if it is missing in
                    # some patches and we mask a different modality then we might end
                    # up with no data for that spatial patch.
                    is_available = torch.all(mask != MaskValue.MISSING.value)
                    if is_available:
                        cur_available_modalities.append(modality)

                if len(cur_available_modalities) <= 1:
                    continue

                # Pick a subset to actually mask. We leave at least one unmasked.
                modality_indices = np.arange(len(cur_available_modalities))
                self.generator.shuffle(modality_indices)
                num_to_mask = self.generator.integers(len(cur_available_modalities))
                cur_mask_modalities = [
                    cur_available_modalities[idx]
                    for idx in modality_indices[0:num_to_mask]
                ]

                for modality in cur_mask_modalities:
                    getattr(
                        masked_sample,
                        MaskedOlmoEarthSample.get_masked_modality_name(modality),
                    )[batch_idx] = MaskValue.MISSING.value

        return masked_sample


@MASKING_STRATEGY_REGISTRY.register("random_fixed_modality")
class RandomFixedModalityMaskingStrategy(FixedModalityMaskingStrategy):
    """Fixed modality masking + random masking."""

    def __init__(
        self,
        decoded_modalities: list[str],
        randomize_missing_modalities: list[str] = [],
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
    ) -> None:
        """Initialize the masking strategy."""
        super().__init__(
            strategy=RandomMaskingStrategy(encode_ratio, decode_ratio),
            decoded_modalities=decoded_modalities,
            randomize_missing_modalities=randomize_missing_modalities,
            encode_ratio=encode_ratio,
            decode_ratio=decode_ratio,
        )


@MASKING_STRATEGY_REGISTRY.register("random_with_decode")
class RandomWithDecodeMaskingStrategy(MaskingStrategy):
    """Random masking strategy that separates band sets into encode-only and decode-only roles.

    This masking strategy does two things:

    1. For all only_decode_modalities, all non-missing tokens are assigned MaskValue.DECODE
    2. For all other band sets, we randomly select which to encode and which to decode at
       an instance level. Random masking is then applied per instance per bandset.

    The ratio of encoded tokens will be < encode_ratio. For encode_ratio == 0.5,
    we'd encode between 7% and 26% of tokens, from 1000 simulated masks.

    Conversely, the ratio of decoded tokens can be >> decode_ratio, since we decode everything
    we can from the only_decode_modalities. For a decode_ratio == 0.5, we'd encode between
    26% and 92% of tokens, from 1000 simulated masks.
    """

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
        only_decode_modalities: list[str] = [],
    ):
        """Random masking strategy except for decode modalities, which only get decoded."""
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio
        self.only_decode_modalities = only_decode_modalities
        self.generator = np.random.default_rng(0)

    def apply_mask(
        self, batch: OlmoEarthSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedOlmoEarthSample:
        """Apply masking to the input data.

        This function has three parts:

        1. First, we create masks for all the present modalities. These masks have
           two values: MISSING and DECODER. This allows us to keep track of which values
           are missing, and also handles mask creation for all the only_decode_modalities.
        2. Now, we are dealing with *not* only_decode_modalities (i.e. modalities that can
           be either encoded or decoded). We do this in two steps:

           For each instance in the batch, we:

           i. Populate encode_decode_bandsets. This list tells us which bandsets for this
              instance have at least one non-missing token.
           ii. Split encode_decode_bandsets into encode-only or decode-only. We then randomly
              select tokens to encode / decode within that bandset.
        """
        if patch_size is None:
            raise ValueError("patch_size must be provided for random masking")
        output_dict: dict[str, ArrayTensor | None] = {"timestamps": batch.timestamps}
        none_modalites: list[str] = []
        for modality_name in batch.modalities:
            instance = getattr(batch, modality_name)
            if instance is None:
                none_modalites.append(modality_name)
                # set instance and mask to None
                output_dict[modality_name] = None
                output_dict[
                    MaskedOlmoEarthSample.get_masked_modality_name(modality_name)
                ] = None
            elif modality_name == "timestamps":
                continue
            else:
                if isinstance(instance, torch.Tensor):
                    device: torch.device | None = instance.device
                else:
                    device = None
                modality = Modality.get(modality_name)

                mask_shape = instance.shape[:-1] + (len(modality.band_sets),)
                mask = torch.full(
                    mask_shape, fill_value=MaskValue.DECODER.value, device=device
                )
                mask = self.fill_mask_with_missing_values(instance, mask, modality)
                # if its a decode only modality, we will decode every token that isn't missing.
                # for now we will store *everything* as decode-only if its not missing. We'll modify
                # this later for the other bands
                output_dict[modality_name] = instance
                output_dict[
                    MaskedOlmoEarthSample.get_masked_modality_name(modality_name)
                ] = mask
                # TODO - should we apply a random mask with the decode only ratio?

        # now for the trickier encode-decode modalities
        encode_decode_modalities = [
            m
            for m in batch.modalities
            if m not in self.only_decode_modalities + ["timestamps"] + none_modalites
        ]
        for i in range(batch.batch_size):
            encode_decode_bandsets: list[tuple[str, int]] = []

            for modality_name in encode_decode_modalities:
                # 1s where its not missing, 0s elsewhere
                # also we can type: ignore this because we know it will be
                # a tensor now, since we filled it in just above.
                not_missing = (
                    output_dict[
                        MaskedOlmoEarthSample.get_masked_modality_name(modality_name)
                    ][i]  # type: ignore
                    != MaskValue.MISSING.value
                )
                for bandset_idx in range(not_missing.shape[-1]):
                    if not_missing[..., bandset_idx].sum() >= 1:
                        encode_decode_bandsets.append((modality_name, bandset_idx))

            if len(encode_decode_bandsets) == 1:
                modality_name, bandset_idx = encode_decode_bandsets[0]
                masked_modality_name = MaskedOlmoEarthSample.get_masked_modality_name(
                    modality_name
                )  # type: ignore
                # random masking for the bandset
                output_dict[masked_modality_name][
                    i : i + 1, ..., bandset_idx : bandset_idx + 1  # type: ignore
                ] = self._random_fill_unmasked(
                    output_dict[masked_modality_name][
                        i : i + 1, ..., bandset_idx : bandset_idx + 1
                    ],  # type: ignore
                    Modality.get(modality_name),
                    patch_size,
                    self.encode_ratio,
                    self.decode_ratio,
                )
            else:
                # for now, lets assume encode_ratio + decode_ratio = 1
                self.generator.shuffle(encode_decode_bandsets)
                num_encode = math.ceil(len(encode_decode_bandsets) * self.encode_ratio)
                encode_bandsets = encode_decode_bandsets[:num_encode]
                decode_bandsets = encode_decode_bandsets[num_encode:]

                for modality_name, bandset_idx in encode_bandsets:
                    masked_modality_name = (
                        MaskedOlmoEarthSample.get_masked_modality_name(modality_name)
                    )
                    output_dict[masked_modality_name][
                        i : i + 1, ..., bandset_idx : bandset_idx + 1  # type: ignore
                    ] = self._random_fill_unmasked(
                        output_dict[masked_modality_name][
                            i : i + 1, ..., bandset_idx : bandset_idx + 1  # type: ignore
                        ],
                        Modality.get(modality_name),
                        patch_size,
                        self.encode_ratio,
                        0,
                    )
                for modality_name, bandset_idx in decode_bandsets:
                    masked_modality_name = (
                        MaskedOlmoEarthSample.get_masked_modality_name(modality_name)
                    )
                    output_dict[masked_modality_name][
                        i : i + 1, ..., bandset_idx : bandset_idx + 1  # type: ignore
                    ] = self._random_fill_unmasked(
                        output_dict[masked_modality_name][
                            i : i + 1, ..., bandset_idx : bandset_idx + 1  # type: ignore
                        ],
                        Modality.get(modality_name),
                        patch_size,
                        0,
                        self.decode_ratio,
                    )

        return MaskedOlmoEarthSample(**output_dict)


@dataclass
class MaskingConfig(Config):
    """Configuration for masking strategies.

    Args:
        strategy_config: Masking strategy to use in the format of
        {
            "type": "random", # registry key
            # rest of init kwargs
        }
    """

    strategy_config: dict[str, Any]

    def build(self) -> MaskingStrategy:
        """Build a MaskingStrategy from the config."""
        mask_strategy_key = self.strategy_config.pop("type")
        return MASKING_STRATEGY_REGISTRY.get_class(mask_strategy_key)(
            **self.strategy_config
        )
