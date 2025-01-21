"""Collator functions for the dataset."""

import logging
from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch.nn import functional as F

from helios.data.constants import DATA_SOURCE_TO_VARIATION_TYPE
from helios.data.masking import apply_random_masking

logger = logging.getLogger(__name__)


class CollatedOutput(NamedTuple):
    """Output of the collate function.

    Based on presto-v3
    """

    space_time_x: torch.Tensor  # Spatiotime features
    space_x: torch.Tensor  # space-only features
    time_x: torch.Tensor  # time-only features
    static_x: torch.Tensor  # Static features
    space_time_mask: torch.Tensor  # Mask for spatiotime features
    space_mask: torch.Tensor  # Mask for space features
    time_mask: torch.Tensor  # Mask for time features
    static_mask: torch.Tensor  # Mask for static features
    time_info: torch.Tensor  # Time information associated with each index
    patch_size: float  # Size of patches
    collate_info: dict | None  # Additional collation metadata


def variable_time_collate_fn(
    items: Sequence[dict],
    pad_token_value: float = 0.0,
    patch_size: int = 4,
    encode_ratio: float = 0.5,
    decode_ratio: float = 0.5,
) -> CollatedOutput:
    """Collate function for inputs with variable time data."""
    assert items
    max_len = max(item["num_timesteps"] for item in items)
    logger.debug(f"Max len: {max_len}")
    all_space_time_x: list[torch.Tensor] = []
    all_space_x: list[torch.Tensor] = []
    all_time_x: list[torch.Tensor] = []
    all_static_x: list[torch.Tensor] = []
    for item in items:
        num_timesteps = item["num_timesteps"]
        for data_source_name, array in item["data_arrays"].items():
            variation_type = DATA_SOURCE_TO_VARIATION_TYPE[data_source_name]
            # Note: layers are not necessarily sorted in time order
            if variation_type == "space_time_varying":
                space_time_x = torch.tensor(array)
                # h, w,t,c
                pad_shape = (0, 0, 0, int(max_len - num_timesteps), 0, 0)
                all_space_time_x.append(
                    F.pad(
                        space_time_x,
                        pad_shape,
                        value=pad_token_value,
                    )
                )
            elif variation_type == "time_varying":
                raise NotImplementedError("Time varying data not implemented")
            elif variation_type == "space_varying":
                raise NotImplementedError("Space varying data not implemented")
            elif variation_type == "static":
                raise NotImplementedError("Static data not implemented")
            else:
                raise ValueError(
                    f"Unknown data source variation type: {variation_type}"
                )
    space_time_x = (
        torch.stack(all_space_time_x) if all_space_time_x else torch.tensor([])
    )
    space_x = torch.stack(all_space_x) if all_space_x else torch.tensor([])
    time_x = torch.stack(all_time_x) if all_time_x else torch.tensor([])
    static_x = torch.stack(all_static_x) if all_static_x else torch.tensor([])
    # patching and masking can occur here for now
    # We need to figure out how mmany patches per dimension for the space stuff
    # Then figure out what the new image sized will be which is patch size times the space patches per dim
    b, h, w, t, c = space_time_x.shape
    assert h == w, "space dimensions must be equal"
    # Unneeded until we add subset code and the like
    # space_dim_size = h
    # space_patches_per_dim = int(space_dim_size / patch_size)
    # patched_image_size = patch_size * space_patches_per_dim

    # TODO: COllecting the time info
    time_info = torch.tensor([])
    # Random Masking only

    masked_output = apply_random_masking(
        space_time_x=space_time_x,
        space_x=space_x,
        time_x=time_x,
        static_x=static_x,
        time_info=time_info,
        patch_size=patch_size,
        encode_ratio=encode_ratio,
        decode_ratio=decode_ratio,
    )

    collated_output = CollatedOutput(
        space_time_x=masked_output.space_time_x,
        space_x=masked_output.space_x,
        time_x=masked_output.time_x,
        static_x=masked_output.static_x,
        space_time_mask=masked_output.space_time_mask,
        space_mask=masked_output.space_mask,
        time_mask=masked_output.time_mask,
        static_mask=masked_output.static_mask,
        time_info=masked_output.time_info,
        patch_size=patch_size,
        collate_info=None,
    )

    return collated_output
