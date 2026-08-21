"""Shared wiring for the cloud-aware patch-discrimination (cloud-skip) arms.

Two knobs turn the cloud skip on for an otherwise-unchanged run:

* the **dataset** must be pointed at the precomputed OmniCloudMask sidecars
  (:func:`apply_cloud_cache`), which is what makes the dataloader carry the
  ``<modality>_cloud`` side-payloads at all; and
* the **masking strategy** must be given a ``cloud_skip_threshold``
  (:func:`apply_cloud_skip_threshold`), above which a DECODER token's cloud
  fraction gets it reassigned to MISSING and dropped from the patch-disc loss.

The threshold has to be set on BOTH masking configs. The dataloader's copy is the
operative one -- masking runs inside the collate fn -- but the train module keeps its
own, and setting only one leaves the saved run config disagreeing with what actually
ran. See ``olmoearth_pretrain.data.cloud_mask_cache`` and
``RandomTimeWithDecodeMaskingStrategy._apply_cloud_skip``.
"""

from typing import TypeVar

from olmoearth_pretrain.data.cloud_mask_cache import default_cache_dir
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig

# Both OlmoEarthDataLoaderConfig and the train module configs carry a masking_config.
HasMaskingConfig = TypeVar("HasMaskingConfig")


def apply_cloud_cache(config: OlmoEarthDatasetConfig) -> OlmoEarthDatasetConfig:
    """Point the dataset at the cloud sidecars sitting beside its own h5 dir.

    ``default_cache_dir`` swaps the ``h5py_data*`` path component for
    ``cloud_masks_omnicloudmask``, so the cache is resolved FROM the configured
    ``h5py_dir`` rather than hardcoded -- a run on a different h5 set gets a
    different (and, if uncomputed, simply absent) cache instead of silently
    reading another dataset's clouds.
    """
    config.cloud_cache_dir = default_cache_dir(config.h5py_dir)
    return config


def apply_cloud_skip_threshold(
    config: HasMaskingConfig, threshold: float
) -> HasMaskingConfig:
    """Record the per-token cloud-skip threshold on a config's masking strategy."""
    config.masking_config.strategy_config["cloud_skip_threshold"] = threshold  # type: ignore[attr-defined]
    return config
