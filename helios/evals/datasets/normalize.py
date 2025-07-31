"""Shared normalization functions for eval sets."""

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


def normalize_bands(
    image: np.ndarray, means: np.array, stds: np.array, method: str = "norm_no_clip"
) -> np.ndarray:
    """Normalize an image with given mean and std arrays, and a normalization method."""
    original_dtype = image.dtype

    if method == "standardize":
        image = (image - means) / stds
    else:
        min_value = means - stds
        max_value = means + stds
        image = (image - min_value) / (max_value - min_value)

        if method == "norm_yes_clip":
            image = np.clip(image, 0, 1)
        elif method == "norm_yes_clip_int":
            # same as clipping between 0 and 1 but rounds to the nearest 1/255
            image = image * 255  # scale
            image = np.clip(image, 0, 255).astype(np.uint8)  # convert to 8-bit integers
            image = (
                image.astype(original_dtype) / 255
            )  # back to original_dtype between 0 and 1
        elif method == "norm_no_clip":
            pass
        else:
            raise ValueError(
                f"norm type must be norm_yes_clip, norm_yes_clip_int, norm_no_clip, or standardize, not {method}"
            )
    return image
