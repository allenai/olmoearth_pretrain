"""Shared normalization functions for eval sets."""

import logging
from enum import Enum

import numpy as np

from .constants import EVAL_S2_BAND_NAMES

logger = logging.getLogger(__name__)


def impute_normalization_stats(
    band_info: dict,
    imputes: list[tuple[str, str]],
    all_bands: list[str] = EVAL_S2_BAND_NAMES,
) -> dict:
    """For certain eval sets, the normalization stats (band_info) may be incomplete.

    This function imputes it so that len(new_band_info) == len(all_bands).
    """
    # band_info is a dictionary with band names as keys and statistics (mean / std) as values
    if not imputes:
        return band_info

    names_list = list(band_info.keys())
    if any(impute[1] in names_list for impute in imputes):
        raise ValueError("Cannot impute: band already present in band_info.")

    new_band_info: dict = {}
    for band_name in all_bands:
        new_band_info[band_name] = {}
        if band_name in names_list:
            # we have the band, so use it
            new_band_info[band_name] = band_info[band_name]
        else:
            # we don't have the band, so impute it
            for impute in imputes:
                src, tgt = impute
                if tgt == band_name:
                    # we have a match!
                    new_band_info[band_name] = band_info[src]
                    break

    return new_band_info


class NormMethod(str, Enum):
    """Normalization methods."""

    NORM_NO_CLIP = "norm_no_clip"
    NORM_YES_CLIP = "norm_yes_clip"
    NORM_YES_CLIP_3_STD = "norm_yes_clip_3_std"
    NORM_YES_CLIP_2_STD = "norm_yes_clip_2_std"
    NORM_YES_CLIP_3_STD_INT = "norm_yes_clip_3_std_int"
    NORM_YES_CLIP_2_STD_INT = "norm_yes_clip_2_std_int"
    NORM_YES_CLIP_INT = "norm_yes_clip_int"
    NORM_YES_CLIP_MIN_MAX_INT = "norm_yes_clip_min_max_int"
    STANDARDIZE = "standardize"
    NO_NORM = "no_norm"


def _get_normalization_bounds(
    method: NormMethod,
    means: np.ndarray,
    stds: np.ndarray,
    mins: np.ndarray | None,
    maxs: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate normalization bounds based on method."""
    bounds_config = {
        NormMethod.NORM_YES_CLIP: 1.0,
        NormMethod.NORM_YES_CLIP_INT: 1.0,
        NormMethod.NORM_YES_CLIP_3_STD: 3.0,
        NormMethod.NORM_YES_CLIP_3_STD_INT: 3.0,
        NormMethod.NORM_YES_CLIP_2_STD: 2.0,
        NormMethod.NORM_YES_CLIP_2_STD_INT: 2.0,
        NormMethod.NORM_NO_CLIP: 1.0,
    }
    if method == NormMethod.NORM_YES_CLIP_MIN_MAX_INT:
        if mins is None or maxs is None:
            raise ValueError("No mins/maxs provided")
        return mins, maxs

    std_mult = bounds_config[method]
    return means - std_mult * stds, means + std_mult * stds


def _apply_clip_and_quantize(
    image: np.ndarray, method: NormMethod, original_dtype: np.dtype
) -> np.ndarray:
    """Apply post-processing based on normalization method."""
    # Methods that need clipping
    clip_methods = {
        NormMethod.NORM_YES_CLIP,
        NormMethod.NORM_YES_CLIP_3_STD,
        NormMethod.NORM_YES_CLIP_2_STD,
    }

    # Methods that need integer quantization
    int_methods = {
        NormMethod.NORM_YES_CLIP_INT,
        NormMethod.NORM_YES_CLIP_3_STD_INT,
        NormMethod.NORM_YES_CLIP_2_STD_INT,
        NormMethod.NORM_YES_CLIP_MIN_MAX_INT,
    }

    if method in clip_methods:
        image = np.clip(image, 0, 1)
    elif method in int_methods:
        # Scale, quantize to 8-bit, then scale back
        image = image * 255
        image = np.clip(image, 0, 255).astype(np.uint8)
        image = image.astype(original_dtype) / 255

    return image


def normalize_bands(
    image: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    mins: np.ndarray | None = None,
    maxs: np.ndarray | None = None,
    method: str = NormMethod.NORM_NO_CLIP,
) -> np.ndarray:
    """Normalize an image with given statistics using the specified method."""
    if isinstance(method, str):
        method = NormMethod(method)

    logger.debug(f"Normalizing image with method {method}")
    original_dtype = image.dtype

    # Handle simple cases first
    if method == NormMethod.NO_NORM:
        return image

    if method == NormMethod.STANDARDIZE:
        return (image - means) / stds

    # Range-based normalization
    min_value, max_value = _get_normalization_bounds(method, means, stds, mins, maxs)
    normalized = (image - min_value) / (max_value - min_value)

    return _apply_clip_and_quantize(normalized, method, original_dtype)
