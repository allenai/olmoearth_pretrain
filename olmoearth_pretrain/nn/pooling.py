"""Pooling operations for TokensAndMasks."""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskValue

if TYPE_CHECKING:
    from olmoearth_pretrain.datatypes import TokensAndMasks

logger = logging.getLogger(__name__)


class PoolingType(StrEnum):
    """Strategy for pooling the tokens."""

    MAX = "max"
    MEAN = "mean"


def pool_spatially_and_concat_modalities(tokens_and_masks: TokensAndMasks) -> Tensor:
    """Pool the modalities across time to get spatial features and concatenate.

    Args:
        tokens_and_masks: The tokens and masks to pool.

    Returns:
        Concatenated spatial features with shape [B, H*W, num_modalities, D].
    """
    spatial_stacked_features = []
    for attr_name in tokens_and_masks.modalities:
        if Modality.get(attr_name).is_spatial:
            mask_attr_name = tokens_and_masks.get_masked_modality_name(attr_name)
            masked_attr = getattr(tokens_and_masks, mask_attr_name)
            if masked_attr is None:
                continue
            if (masked_attr == MaskValue.ONLINE_ENCODER.value).all():
                attr = getattr(tokens_and_masks, attr_name)
                pooled_attr = torch.mean(attr, dim=(-3))
                spatial_stacked_features.append(pooled_attr)
    if len(spatial_stacked_features) == 0:
        raise ValueError("Missing unmasked spatial modalities for spatial pooling.")
    spatial_stacked_features = torch.cat(spatial_stacked_features, dim=-2)
    return spatial_stacked_features


def pool_spatially(
    tokens_and_masks: TokensAndMasks, pooling_type: PoolingType
) -> Tensor:
    """Pool the modalities across time to get spatial features.

    Args:
        tokens_and_masks: The tokens and masks to pool.
        pooling_type: The pooling strategy (MEAN or MAX).

    Returns:
        Pooled spatial features.
    """
    spatial_average = []
    for attr_name in tokens_and_masks.modalities:
        if Modality.get(attr_name).is_spatial:
            mask_attr_name = tokens_and_masks.get_masked_modality_name(attr_name)
            masked_attr = getattr(tokens_and_masks, mask_attr_name)
            if masked_attr is None:
                continue
            if (masked_attr == MaskValue.ONLINE_ENCODER.value).all():
                attr = getattr(tokens_and_masks, attr_name)
                if pooling_type == PoolingType.MEAN:
                    spatial_average.append(torch.mean(attr, dim=(-2, -3)))
                else:
                    spatial_average.append(
                        torch.max(torch.max(attr, dim=-2).values, dim=-2).values
                    )
    if len(spatial_average) == 0:
        raise ValueError("Missing unmasked spatial modalities for spatial pooling.")
    spatial_average_t = torch.stack(spatial_average, dim=-1)
    if pooling_type == PoolingType.MEAN:
        return spatial_average_t.mean(dim=-1)
    else:
        return spatial_average_t.max(dim=-1).values


def pool_instance_wise(
    tokens_and_masks: TokensAndMasks, pooling_type: PoolingType
) -> Tensor:
    """Pool all the tokens in the instance.

    Args:
        tokens_and_masks: The tokens and masks to pool.
        pooling_type: The pooling strategy (MEAN or MAX).

    Returns:
        Pooled instance features with shape [B, D].
    """
    x, mask = tokens_and_masks.flatten_all_tokens_and_masks()
    assert isinstance(x, Tensor) and isinstance(mask, Tensor)
    mask = (mask == MaskValue.ONLINE_ENCODER.value).long()
    x_for_pooling = x * mask.unsqueeze(-1)
    if pooling_type == PoolingType.MAX:
        x_for_pooling = x_for_pooling.masked_fill(
            ~mask.bool().unsqueeze(-1), -float("inf")
        )
        return x_for_pooling.max(dim=1).values
    elif pooling_type == PoolingType.MEAN:
        num_encoded_tokens = torch.sum(mask, -1, keepdim=True)
        logger.debug(f"num_encoded_tokens: {num_encoded_tokens}")
        if (num_encoded_tokens == 0).any():
            raise ValueError(
                f"num_encoded_tokens is 0 for some samples {num_encoded_tokens}"
            )
        return x_for_pooling.sum(dim=1) / num_encoded_tokens
    else:
        raise ValueError(f"Invalid pooling type: {pooling_type}")


def pool_unmasked_tokens(
    tokens_and_masks: TokensAndMasks,
    pooling_type: PoolingType | None = None,
    spatial_pooling: bool = False,
    concat_features: bool = False,
) -> Tensor:
    """Pool the unmasked tokens.

    Args:
        tokens_and_masks: The tokens and masks to pool.
        pooling_type: Pooling type for the tokens. Defaults to MAX.
        spatial_pooling: Whether to keep the spatial dimensions when pooling.
        concat_features: Whether to concatenate the features instead of averaging.

    Returns:
        Pooled features tensor.
    """
    if pooling_type is None:
        pooling_type = PoolingType.MAX

    if concat_features and spatial_pooling:
        return pool_spatially_and_concat_modalities(tokens_and_masks)
    if concat_features:
        raise ValueError("concat_features is not supported for non-spatial pooling")
    if not spatial_pooling:
        return pool_instance_wise(tokens_and_masks, pooling_type)
    else:
        return pool_spatially(tokens_and_masks, pooling_type)
