"""Shared normalization functions for eval sets."""

from enum import StrEnum

import numpy as np

from .constants import EVAL_S2_BAND_NAMES


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


class NormMethod(StrEnum):
    """Normalization methods."""

    NORM_NO_CLIP = "norm_no_clip"
    NORM_YES_CLIP = "norm_yes_clip"
    NORM_YES_CLIP_INT = "norm_yes_clip_int"
    STANDARDIZE = "standardize"
    NO_NORM = "no_norm"


def normalize_bands(
    image: np.ndarray,
    means: np.array,
    stds: np.array,
    method: str = NormMethod.NORM_NO_CLIP,
) -> np.ndarray:
    """Normalize an image with given mean and std arrays, and a normalization method."""
    original_dtype = image.dtype
    if method == NormMethod.NO_NORM:
        print("No normalization")
        # If the normalization method is model specific we may want to defer normalization to the model e.g dinoV2
        return image
    elif method == NormMethod.STANDARDIZE:
        image = (image - means) / stds
    else:
        min_value = means - stds
        max_value = means + stds
        image = (image - min_value) / (max_value - min_value)

        if method == NormMethod.NORM_YES_CLIP:
            image = np.clip(image, 0, 1)
        elif method == NormMethod.NORM_YES_CLIP_INT:
            # same as clipping between 0 and 1 but rounds to the nearest 1/255
            image = image * 255  # scale
            image = np.clip(image, 0, 255).astype(np.uint8)  # convert to 8-bit integers
            image = (
                image.astype(original_dtype) / 255
            )  # back to original_dtype between 0 and 1
        elif method == NormMethod.NORM_NO_CLIP:
            pass
        else:
            valid_methods = [m.value for m in NormMethod]
            raise ValueError(f"norm type must be one of {valid_methods}, not {method}")
    return image
