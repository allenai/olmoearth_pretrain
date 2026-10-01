"""Dataset module for OlmoEarth Pretrain."""

from __future__ import annotations

import hashlib
import logging
import shutil
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, NamedTuple

import h5py

# hdf5 plugin is needed to decompress the data for certain compression types
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd
from olmo_core.data.utils import get_rng
from torch.utils.data import Dataset
from upath import UPath

from olmoearth_pretrain._compat import (
    deprecated_class_alias as _deprecated_class_alias,
)
from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import (
    MAX_SEQUENCE_LENGTH,
    MISSING_VALUE,
    WORLDCOVER_CLASSES,
    Modality,
    ModalitySpec,
)
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.dataset.convert_to_h5py import ConvertToH5py
from olmoearth_pretrain.datatypes import (
    OlmoEarthSample,
)
from olmoearth_pretrain.nn.tokenization import TokenizationConfig
from olmoearth_pretrain.types import ArrayTensor

logger = logging.getLogger(__name__)


# =============================================================================
# Subsetting Functions
# =============================================================================


def compute_bandset_rates(
    modalities: list[str],
    tokenization_config: TokenizationConfig | None = None,
    exclude_modalities: frozenset[str] = frozenset(),
) -> tuple[int, int, int, int]:
    """Per-instance token rates for a fixed modality list (no concrete sample).

    Returns ``(spacetime_bandsets, space_only_bandsets, static_bandsets,
    time_only_bandsets)``. The token count for a shape ``(h_w_p, t)`` is::

        space_only_bandsets * h_w_p**2 + static_bandsets                  # fixed
        + t * (spacetime_bandsets * h_w_p**2 + time_only_bandsets)        # per-t

    Mirrors the accounting in :func:`_get_max_t_within_token_budget` but over a
    modality list, so the dataloader can invert the budget (max_t given hw_p and
    max_hw_p given t) before a sample is in hand. ``exclude_modalities`` are
    dropped from the accounting (e.g. decode-only map targets never encoded).
    """
    st_bs = so_bs = static_bs = time_bs = 0
    for attribute in modalities:
        if attribute in ("timestamps", "latlon") or attribute in exclude_modalities:
            continue
        modality_spec = Modality.get(attribute)
        num_band_sets = (
            tokenization_config.get_num_bandsets(attribute)
            if tokenization_config is not None
            else modality_spec.num_band_sets
        )
        if modality_spec.is_spacetime_varying:
            st_bs += num_band_sets
        elif modality_spec.is_space_only_varying:
            so_bs += num_band_sets
        elif modality_spec.is_time_only_varying:
            time_bs += num_band_sets
        elif modality_spec.is_static_in_space_and_time:
            static_bs += num_band_sets
    return st_bs, so_bs, static_bs, time_bs


def _get_max_t_within_token_budget(
    sample: OlmoEarthSample,
    h_w_p: int,
    max_tokens_per_instance: int,
    tokenization_config: TokenizationConfig | None = None,
    exclude_modalities: frozenset[str] = frozenset(),
) -> int:
    """Find max t possible when subsetting.

    Given a sampled h_w_p (the number of tokens along the h and w dimensions)
    return the maximum t allowed within the max_tokens budget so that the
    patchified OlmoEarthSample will have fewer than max_tokens tokens.

    ``exclude_modalities`` are not counted against the budget. This is used to
    keep decode-only map modalities (which are never encoded) from consuming the
    encoder token budget.

    This function assumes we apply (H, W, T=1 patchifying)
    """
    from math import floor

    used_tokens = 0
    time_multiply_tokens = 0
    for attribute in sample.as_dict().keys():
        if attribute in ("timestamps", "latlon"):
            continue
        if attribute in exclude_modalities:
            continue
        modality_spec = Modality.get(attribute)
        num_band_sets = (
            tokenization_config.get_num_bandsets(attribute)
            if tokenization_config is not None
            else modality_spec.num_band_sets
        )
        if modality_spec.is_spacetime_varying:
            time_multiply_tokens += (h_w_p**2) * num_band_sets
        elif modality_spec.is_space_only_varying:
            used_tokens += (h_w_p**2) * num_band_sets
        elif modality_spec.is_time_only_varying:
            time_multiply_tokens += num_band_sets
        elif modality_spec.is_static_in_space_and_time:
            used_tokens += num_band_sets
    if time_multiply_tokens == 0:
        return 1
    remaining_tokens = max_tokens_per_instance - used_tokens
    max_t_within_budget = remaining_tokens / time_multiply_tokens
    if max_t_within_budget < 1:
        raise ValueError(
            f"patch_size too small for this sample and budget, h_w_p: {h_w_p}, max_tokens: {max_tokens_per_instance}"
        )

    return min(floor(max_t_within_budget), sample.time)


def get_valid_start_ts(
    missing_timesteps: dict[str, Any], max_t: int, current_length: int
) -> list[int]:
    """Get valid starting timesteps."""
    if current_length > max_t:
        if not missing_timesteps:
            valid_start_ts = list(range(current_length - max_t + 1))
        else:
            start_ts = set()
            for modality in missing_timesteps:
                valid_timesteps = np.flatnonzero(missing_timesteps[modality])
                valid_timesteps = valid_timesteps[
                    valid_timesteps + max_t <= current_length
                ]
                start_ts.update(valid_timesteps)
            valid_start_ts = list(start_ts)
    else:
        valid_start_ts = [0]
    if len(valid_start_ts) == 0:
        logger.warning(
            f"No valid start timesteps found for {missing_timesteps} with max_t {max_t} and current_length {current_length}"
        )
        raise ValueError(
            f"No valid start timesteps found for {missing_timesteps} with max_t {max_t} and current_length {current_length}"
        )
    return sorted(valid_start_ts)


def subset_sample_default(
    sample: OlmoEarthSample,
    patch_size: int,
    max_tokens_per_instance: int | None,
    sampled_hw_p: int,
    current_length: int,
    missing_timesteps_masks: dict[str, Any] | None = None,
    tokenization_config: TokenizationConfig | None = None,
    target_t: int | None = None,
    budget_exclude_modalities: frozenset[str] = frozenset(),
) -> OlmoEarthSample:
    """Subset a OlmoEarthSample using default rectangular cropping.

    Args:
        sample: The sample to subset.
        patch_size: The patch size being applied to this sample.
        max_tokens_per_instance: The token budget when subsetting. This is used
            to determine the maximum number of timesteps possible for a given
            height and width. If None, this operation is a no-op.
        sampled_hw_p: The number of tokens in the height and width dimensions.
        current_length: The current maximum sequence length of the sample.
        missing_timesteps_masks: A dictionary of missing timesteps masks.
        tokenization_config: Optional tokenization config for custom band groupings.
        target_t: Optional requested number of timesteps. The number of timesteps
            used is ``min(target_t, budget_max_t)`` so the budget is always a hard
            cap even when the caller requests more. If None, the budget max is used
            (the historical behaviour).
        budget_exclude_modalities: Modalities not counted against the token budget
            (e.g. decode-only maps that are never encoded).

    Returns:
        A subsetted OlmoEarthSample with rectangular cropping applied.
    """
    if max_tokens_per_instance is None:
        return sample
    if missing_timesteps_masks is None:
        missing_timesteps_masks = {}

    max_t = _get_max_t_within_token_budget(
        sample,
        sampled_hw_p,
        max_tokens_per_instance,
        tokenization_config,
        exclude_modalities=budget_exclude_modalities,
    )
    if target_t is not None:
        max_t = min(max_t, target_t)
    valid_start_ts = get_valid_start_ts(missing_timesteps_masks, max_t, current_length)
    start_t = np.random.choice(valid_start_ts)
    new_data_dict: dict[str, ArrayTensor] = {}

    sampled_hw = sampled_hw_p * patch_size
    start_h = np.random.choice(sample.height - sampled_hw + 1)
    start_w = np.random.choice(sample.width - sampled_hw + 1)

    for attribute, modality in sample.as_dict().items():
        assert modality is not None
        if attribute == "timestamps":
            new_data_dict[attribute] = modality[start_t : start_t + max_t]
            continue
        if attribute == "latlon":
            new_data_dict[attribute] = modality
            continue
        modality_spec = Modality.get(attribute)
        if modality_spec.is_spacetime_varying:
            new_data_dict[attribute] = modality[
                start_h * modality_spec.image_tile_size_factor : (start_h + sampled_hw)
                * modality_spec.image_tile_size_factor,
                start_w * modality_spec.image_tile_size_factor : (start_w + sampled_hw)
                * modality_spec.image_tile_size_factor,
                start_t : start_t + max_t,
            ]
        elif modality_spec.is_space_only_varying:
            new_data_dict[attribute] = modality[
                start_h * modality_spec.image_tile_size_factor : (start_h + sampled_hw)
                * modality_spec.image_tile_size_factor,
                start_w * modality_spec.image_tile_size_factor : (start_w + sampled_hw)
                * modality_spec.image_tile_size_factor,
            ]
        elif modality_spec.is_time_only_varying:
            new_data_dict[attribute] = modality[start_t : start_t + max_t]
        elif modality_spec.is_static_in_space_and_time:
            new_data_dict[attribute] = modality

    return OlmoEarthSample(**new_data_dict)


def subset_sample_cutmix(
    sample: OlmoEarthSample,
    patch_size: int,
    max_tokens_per_instance: int | None,
    sampled_hw_p: int,
    current_length: int,
    missing_timesteps_masks: dict[str, Any] | None = None,
    tokenization_config: TokenizationConfig | None = None,
    target_t: int | None = None,
    budget_exclude_modalities: frozenset[str] = frozenset(),
) -> OlmoEarthSample:
    """Subset a OlmoEarthSample using CutMix patch sampling.

    Args:
        sample: The sample to subset.
        patch_size: The patch size being applied to this sample.
        max_tokens_per_instance: The token budget when subsetting. This is used
            to determine the maximum number of timesteps possible for a given
            height and width. If None, this operation is a no-op.
        sampled_hw_p: The number of tokens in the height and width dimensions.
        current_length: The current maximum sequence length of the sample.
        missing_timesteps_masks: A dictionary of missing timesteps masks.
        tokenization_config: Optional tokenization config for custom band groupings.
        target_t: Optional requested number of timesteps. The number used is
            ``min(target_t, budget_max_t)`` so the budget is always a hard cap. If
            None, the budget max is used (the historical behaviour).
        budget_exclude_modalities: Modalities not counted against the token budget
            (e.g. decode-only maps that are never encoded).

    Returns:
        A subsetted OlmoEarthSample with CutMix patch sampling applied.
    """
    if max_tokens_per_instance is None:
        return sample
    if missing_timesteps_masks is None:
        missing_timesteps_masks = {}

    max_t = _get_max_t_within_token_budget(
        sample,
        sampled_hw_p,
        max_tokens_per_instance,
        tokenization_config,
        exclude_modalities=budget_exclude_modalities,
    )
    if target_t is not None:
        max_t = min(max_t, target_t)
    valid_start_ts = get_valid_start_ts(missing_timesteps_masks, max_t, current_length)
    start_t = np.random.choice(valid_start_ts)
    new_data_dict: dict[str, ArrayTensor] = {}

    height_p, width_p = sample.height // patch_size, sample.width // patch_size
    h_p_indices = np.random.choice(height_p, size=sampled_hw_p, replace=False)
    w_p_indices = np.random.choice(width_p, size=sampled_hw_p, replace=False)
    h_indices = [
        i
        for h_p in h_p_indices
        for i in range(h_p * patch_size, (h_p + 1) * patch_size)
    ]
    w_indices = [
        i
        for w_p in w_p_indices
        for i in range(w_p * patch_size, (w_p + 1) * patch_size)
    ]
    hh, ww = np.meshgrid(h_indices, w_indices, indexing="ij")

    for attribute, modality in sample.as_dict().items():
        assert modality is not None
        if attribute == "timestamps":
            new_data_dict[attribute] = modality[start_t : start_t + max_t]
            continue
        if attribute == "latlon":
            new_data_dict[attribute] = modality
            continue
        modality_spec = Modality.get(attribute)
        if modality_spec.is_spacetime_varying:
            new_data_dict[attribute] = modality[
                hh * modality_spec.image_tile_size_factor,
                ww * modality_spec.image_tile_size_factor,
                start_t : start_t + max_t,
            ]
        elif modality_spec.is_space_only_varying:
            new_data_dict[attribute] = modality[
                hh * modality_spec.image_tile_size_factor,
                ww * modality_spec.image_tile_size_factor,
            ]
        elif modality_spec.is_time_only_varying:
            new_data_dict[attribute] = modality[start_t : start_t + max_t]
        elif modality_spec.is_static_in_space_and_time:
            new_data_dict[attribute] = modality

    return OlmoEarthSample(**new_data_dict)


def one_hot_worldcover(raw: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """One-hot encode raw ESA WorldCover class codes.

    The worldcover modality is stored on disk as a single band holding one class code per
    pixel. The derived worldcover_onehot modality expands that band into one channel per
    class in WORLDCOVER_CLASSES order.

    Args:
        raw: array of class codes with the band axis as the last dim of size 1,
            e.g. [H, W, 1, 1]. MISSING_VALUE marks missing pixels.
        dtype: the dtype of the output array.

    Returns:
        Array with the trailing band axis expanded to len(WORLDCOVER_CLASSES). Missing
        pixels are MISSING_VALUE across all channels (so the existing missing-value
        handling continues to work); codes not in WORLDCOVER_CLASSES map to all-zero.
    """
    # Drop the single-band axis: [H, W, 1, 1] -> [H, W, 1].
    codes = raw[..., 0]
    missing = codes == MISSING_VALUE
    onehot = np.zeros((*codes.shape, len(WORLDCOVER_CLASSES)), dtype=dtype)
    for idx, code in enumerate(WORLDCOVER_CLASSES):
        onehot[..., idx] = codes == code
    # Preserve missingness so it is recognised by the missing mask downstream.
    onehot[missing] = MISSING_VALUE
    return onehot


# Union-timeline key = ordinal day * this + same-day occurrence rank.
_MAX_CAPTURES_PER_DAY = 64


def timestamps_to_ordinals(timestamps: np.ndarray) -> np.ndarray:
    """``(T, 3)`` ``[day, month-1, year]`` timestamps -> ``(T,)`` proleptic ordinal days."""
    return np.array(
        [date(int(y), int(m) + 1, int(d)).toordinal() for d, m, y in timestamps],
        dtype=np.int64,
    )


def build_union_timeline(
    timestamps_per_modality: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Merge per-modality capture dates into one shared, chronological timeline.

    Captures of different modalities on the same calendar day share a step. A
    modality with k captures on one day occupies k consecutive steps of that day
    (its j-th same-day capture shares a step with the other modalities' j-th
    same-day captures), so every capture keeps its own step.

    Args:
        timestamps_per_modality: modality -> (T_m, 3) ``[day, month-1, year]``,
            chronological.

    Returns:
        ``(timestamps, step_index)``: the union timeline ``(T_u, 3)`` in the same
        format, and modality -> ``(T_m,)`` union step of each of its captures.
    """
    keys_per_modality: dict[str, np.ndarray] = {}
    triples: list[np.ndarray] = []
    for modality, ts in timestamps_per_modality.items():
        ordinals = timestamps_to_ordinals(ts)
        # Occurrence rank of each capture within its day (0 for the first).
        rank = np.zeros(len(ordinals), dtype=np.int64)
        for i in range(1, len(ordinals)):
            if ordinals[i] == ordinals[i - 1]:
                rank[i] = rank[i - 1] + 1
        keys_per_modality[modality] = ordinals * _MAX_CAPTURES_PER_DAY + rank
        triples.append(np.asarray(ts))
    if not keys_per_modality:
        return np.zeros((0, 3), dtype=np.int64), {}
    all_keys = np.concatenate(list(keys_per_modality.values()))
    union_keys, first = np.unique(all_keys, return_index=True)
    timestamps = np.concatenate(triples)[first]
    step_index = {
        modality: np.searchsorted(union_keys, keys)
        for modality, keys in keys_per_modality.items()
    }
    return timestamps, step_index


class GetItemArgs(NamedTuple):
    """Arguments for the __getitem__ method of the OlmoEarthDataset."""

    idx: int
    patch_size: int
    sampled_hw_p: int
    token_budget: int | None = None
    tokenization_config: TokenizationConfig | None = None
    target_t: int | None = None
    budget_exclude_modalities: frozenset[str] = frozenset()
    # Per-modality-timestamps datasets only: length of the random time range
    # (days) to take every capture from. None = the sample's whole span.
    time_range_days: float | None = None


# TODO should training modalities be str or modality_spec
class OlmoEarthDataset(Dataset):
    """OlmoEarth Pretrain dataset."""

    def __init__(
        self,
        h5py_dir: UPath,
        training_modalities: list[str],
        dtype: np.dtype,
        max_sequence_length: int = MAX_SEQUENCE_LENGTH,
        normalize: bool = True,
        cache_dir: UPath | None = None,
        samples_per_sec: float | None = None,
        dataset_percentage: float = 1.0,
        seed: int = 0,
        apply_cutmix: bool = False,
        filter_idx_file: str | None = None,
        per_modality_timestamps: bool = False,
    ):
        """Initialize the dataset.

        To use an already created h5py directory, set h5py_dir to the path of the h5py directory.
        To use a raw tile directory, set tile_path to the path of the tile directory, this will create the h5py files in a prepare step before training.
        Warning from OLMo-core:
            In distributed settings, be sure that the :data:`work_dir` is shared among all local ranks
            and :data:`fs_local_rank` is set accordingly. Once those fields are set you should then call
            :meth:`prepare()` in the main process before doing anything else.

        Args:
            h5py_dir: The path to the h5py directory containing preprocessed data.
            training_modalities: The modalities to use for training.
            dtype: The dtype of the data.
            max_sequence_length: The maximum sequence length that we pad all time dimensions to.
            normalize: If True, apply normalization to the data, if False, do not apply
                normalization.
            cache_dir: optional local directory to cache the H5 files.
            samples_per_sec: throttle to reading this many samples per second. This
                throttling only applies when reading from the h5py_dir, not the
                cache_dir (if set).
            dataset_percentage: The percentage of the dataset to use.
            seed: For selecting the dataset percentage.
            apply_cutmix: Whether or not to apply CutMix augmentation during subsetting.
            filter_idx_file: If not None, filters indices by the values in this numpy array
            per_modality_timestamps: The h5 files store every capture of each
                multitemporal modality with its own ``timestamps_<modality>`` (the
                every-capture corpus) instead of one shared ``timestamps`` grid with
                missing-timestep masks. Samples are assembled on a per-sample union
                timeline; see ``_getitem_per_modality_timestamps``.

        Returns:
            None
        """
        self.h5py_dir = h5py_dir
        if not self.h5py_dir.exists():
            raise FileNotFoundError(f"H5PY directory does not exist: {self.h5py_dir}")
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.training_modalities = training_modalities

        self.dtype = dtype
        self.normalize = normalize
        self.dataset_percentage = dataset_percentage
        self.seed = seed
        if self.normalize:
            self.normalizer_predefined = Normalizer(Strategy.PREDEFINED)
            self.normalizer_computed = Normalizer(Strategy.COMPUTED)
        self.max_sequence_length = max_sequence_length

        if samples_per_sec is None:
            self.sec_per_sample = None
        else:
            self.sec_per_sample = 1 / samples_per_sec
        self.last_read_time = time.time()

        self.sample_indices: np.ndarray | None = None
        self.latlon_distribution: np.ndarray | None = None
        self.apply_cutmix = apply_cutmix
        self.filter_idx_file = filter_idx_file
        if filter_idx_file is not None:
            self.indices_to_filter: np.ndarray | None = np.load(filter_idx_file)
            assert isinstance(self.indices_to_filter, np.ndarray), (
                f"Expected filter_idx_file to point to a np.ndarray, got {type(self.indices_to_filter)} instead."
            )
        else:
            self.indices_to_filter = None
        self.per_modality_timestamps = per_modality_timestamps

    @property
    def fingerprint_version(self) -> str:
        """The version of the fingerprint."""
        return "v0.1"

    @property
    def fingerprint(self) -> str:
        """Can be used to identify/compare a dataset."""
        if not self.is_dataset_prepared:
            raise RuntimeError("Dataset must be prepared before creating a fingerprint")
        sha256_hash = hashlib.sha256()
        # Parse from the h5py_dir
        supported_modalities_folder = self.h5py_dir.parent.name
        supported_modalities = supported_modalities_folder.split("_")
        # join back sentinel_l2a and openstreetmap_raster if applicable
        if "l2a" in supported_modalities:
            supported_modalities.remove("l2a")
            supported_modalities.remove("sentinel2")
            supported_modalities.append("sentinel2_l2a")
        if "raster" in supported_modalities:
            supported_modalities.remove("raster")
            supported_modalities.remove("openstreetmap")
            supported_modalities.append("openstreetmap_raster")

        if "naip" in supported_modalities and "10" in supported_modalities:
            supported_modalities.remove("naip")
            supported_modalities.remove("10")
            supported_modalities.append("naip_10")
        # latlons are saved with every h5py file, see
        # olmoearth_pretrain.dataset.convert_to_h5py.ConvertToH5py._create_h5_file
        supported_modalities.append("latlon")
        num_samples = int(self.h5py_dir.name)

        tile_path = self.h5py_dir.parent.parent.parent

        if self.filter_idx_file is not None:
            filter_file_string = f",filter_idx_file={self.filter_idx_file}"
        else:
            filter_file_string = ""

        sha256_hash.update(
            f"tile_path={tile_path},"
            f"supported_modalities={sorted(supported_modalities)},"
            f"sample_size={num_samples},"
            f"dtype={self.dtype}"
            f"{filter_file_string}".encode()
        )
        return sha256_hash.hexdigest()

    @property
    def sample_metadata_path(self) -> UPath:
        """Get the path to the sample metadata file."""
        return self.h5py_dir / ConvertToH5py.sample_metadata_fname

    @property
    def latlon_distribution_path(self) -> UPath:
        """Get the path to the latlon distribution file."""
        return self.h5py_dir / ConvertToH5py.latlon_distribution_fname

    @property
    def is_dataset_prepared(self) -> bool:
        """Check if the dataset is prepared."""
        return self.sample_indices is not None

    def _filter_sample_indices_for_training(self) -> None:
        """Filter the sample indices for training.

        Updates the sample indices numpy array to only include the indices we want to train on.
        """
        # Read the metadata CSV
        # TODO: Pandas can't read gcs upaths
        metadata_df = pd.read_csv(str(self.sample_metadata_path))
        logger.info(f"Metadata CSV has {len(metadata_df)} samples")
        logger.info(f"columns: {metadata_df.columns}")

        # Get the indices of samples that don't have any training modalities that are
        # spacetime varying. We want to remove these samples.
        # Skip derived modalities (ignore_when_parsing=True) since they don't have
        # columns in the metadata CSV.
        spacetime_varying_training_modalities = [
            modality
            for modality in self.training_modalities
            if Modality.get(modality).is_spacetime_varying
            and not Modality.get(modality).ignore_when_parsing
        ]
        if len(spacetime_varying_training_modalities) == 0:
            # Allowed for label-only reads (e.g. PretrainSubsetDataset reading
            # a static raster like worldcover/srtm/openstreetmap_raster). Without
            # any spacetime-varying modality there is nothing to filter against,
            # so keep all sample indices.
            logger.warning(
                "no spacetime varying modalities are specified for training; "
                "skipping spacetime-varying sample filter"
            )
            return
        no_spacetime_varying_indices = metadata_df[
            metadata_df[spacetime_varying_training_modalities].sum(axis=1) == 0
        ].index

        # Filter these indices out
        logger.info(
            f"Filtering out {len(no_spacetime_varying_indices)} samples without any training modalities"
        )
        self.sample_indices = np.setdiff1d(
            self.sample_indices, no_spacetime_varying_indices
        )
        logger.info(
            f"Filtered {len(no_spacetime_varying_indices)} samples to {self.sample_indices.shape} samples"
        )
        if self.indices_to_filter is not None:
            self.sample_indices = np.intersect1d(
                self.sample_indices, self.indices_to_filter
            )

            logger.info(
                f"Intersected {len(self.indices_to_filter)} samples to yield {self.sample_indices.shape} samples"
            )

    def _filter_sample_indices_by_dataset_percentage(self) -> None:
        """Filter the sample indices for dataset percentage."""
        assert self.sample_indices is not None, (
            "Sample indices must be set before filtering by dataset percentage"
        )
        if self.dataset_percentage < 1.0:
            rng = get_rng(self.seed)
            num_samples = len(self.sample_indices)
            self.sample_indices = rng.choice(
                self.sample_indices,
                size=int(len(self.sample_indices) * self.dataset_percentage),
                replace=False,
            )
            logger.info(
                f"Picked {len(self.sample_indices)} samples from {num_samples} samples"
            )

    def prepare(self) -> None:
        """Prepare the dataset.

        THIS SHOULD BE CALLED BY THE MAIN PROCESS ONLY and should happen
        before any other process tries to use the dataset
        """
        logger.info("Preparing dataset...")
        if self.is_dataset_prepared:
            logger.info("Dataset is already prepared")
            return

        num_samples = int(self.h5py_dir.name)
        self.latlon_distribution = self.get_geographic_distribution()
        self.sample_indices = np.arange(num_samples)
        self._filter_sample_indices_for_training()
        self._filter_sample_indices_by_dataset_percentage()
        self.latlon_distribution = self.latlon_distribution[self.sample_indices]

    def get_geographic_distribution(self) -> np.ndarray:
        """Get the geographic distribution of the dataset.

        Returns:
            numpy.ndarray: Array of shape (N, 2) containing [latitude, longitude]
            coordinates for each of the N samples in the dataset.
        """
        if self.latlon_distribution_path.exists():
            with self.latlon_distribution_path.open("rb") as f:
                return np.load(f)

    def __len__(self) -> int:
        """Get the length of the dataset."""
        if self.sample_indices is None:
            raise ValueError("Dataset is not prepared")
        return self.sample_indices.shape[0]

    def normalize_image(self, modality: ModalitySpec, image: np.ndarray) -> np.ndarray:
        """Normalize the image."""
        # Try computed strategy first, if it fails, try predefined strategy
        # TODO: we can also make modality norm strategy configurable later
        try:
            return self.normalizer_computed.normalize(modality, image)
        except Exception:
            return self.normalizer_predefined.normalize(modality, image)

    def _compute_ndvi(
        self,
        s2_data: np.ndarray,
        missing_modalities: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        """Compute NDVI from raw Sentinel-2 L2A bands.

        NDVI = (NIR - Red) / (NIR + Red) where NIR=B08 (index 3) and Red=B04 (index 2).
        If either band has MISSING_VALUE at a pixel, NDVI is set to MISSING_VALUE there.

        Args:
            s2_data: Raw (un-normalized) S2 L2A data, shape [H, W, T, C].
            missing_modalities: List of modalities that are entirely missing.

        Returns:
            Tuple of (ndvi array [H, W, T, 1], updated missing_modalities).
        """
        s2_band_order = Modality.SENTINEL2_L2A.band_order
        red = s2_data[..., s2_band_order.index("B04")]
        nir = s2_data[..., s2_band_order.index("B08")]

        missing = (red == MISSING_VALUE) | (nir == MISSING_VALUE)

        denom = nir + red
        safe_denom = np.where(np.abs(denom) < 1e-10, 1.0, denom)
        ndvi = (nir - red) / safe_denom
        ndvi = np.where(np.abs(denom) < 1e-10, 0.0, ndvi)
        ndvi = np.where(missing, MISSING_VALUE, ndvi)

        # Remove "ndvi" from missing_modalities since we computed it
        updated_missing = [m for m in missing_modalities if m != "ndvi"]
        return ndvi[..., np.newaxis].astype(self.dtype), updated_missing

    def _fill_missing_timesteps(
        self,
        modality_data: np.ndarray,
        missing_timestep_mask: np.ndarray,
    ) -> np.ndarray:
        """Fill the missing timesteps with the missing value."""
        # cast to appropriate dtype to prevent overflow from missing values
        modality_data = modality_data.astype(self.dtype)
        # Get the shape of the data to create properly sized temporal layers
        h, w, t, c = modality_data.shape

        full_timesteps_data = np.full(
            (h, w, self.max_sequence_length, c),
            MISSING_VALUE,
            dtype=self.dtype,
        )

        # Copy the existing data to the appropriate timestep positions
        present_indices = np.where(missing_timestep_mask)[0]
        num_to_copy = min(len(present_indices), t)
        if num_to_copy > 0:
            full_timesteps_data[:, :, present_indices[:num_to_copy], :] = modality_data[
                :, :, :num_to_copy, :
            ]

        return full_timesteps_data

    def _fill_missing_modality(
        self, modality: str, height: int | None, width: int | None, time: int
    ) -> np.ndarray:
        """Fill an array of shape of modality with the missing value."""
        expected_shape = OlmoEarthSample.compute_expected_shape(
            modality, height, width, time
        )
        logger.debug(f"Filling {modality} with shape {expected_shape}")
        return np.full(
            expected_shape,
            fill_value=MISSING_VALUE,
            dtype=self.dtype,
        )

    @staticmethod
    def extract_hwt_from_sample_dict(
        sample_dict: dict[str, Any],
    ) -> tuple[int, int, int]:
        """Extract h, w, t from sample_dict."""
        time = sample_dict["timestamps"].shape[0]
        for mod_name, mod_data in sample_dict.items():
            if mod_name == "timestamps":
                continue
            mod_spec = Modality.get(mod_name)
            if mod_spec.is_spatial and mod_data is not None:
                # shape is (H, W, T, C) without batch dim
                height = mod_data.shape[0] // mod_spec.image_tile_size_factor
                width = mod_data.shape[1] // mod_spec.image_tile_size_factor
                return height, width, time
        raise ValueError("Expected sample dict to have at least one spatial modality")

    def fill_sample_with_missing_values(
        self, sample_dict: dict[str, Any], missing_timesteps_masks: dict[str, Any]
    ) -> tuple[OlmoEarthSample, list[str]]:
        """Fill the sample with missing values."""
        assert sample_dict["timestamps"].shape[0] == self.max_sequence_length, (
            f"Timestamps shape {sample_dict['timestamps'].shape[0]} does not match max_sequence_length {self.max_sequence_length}"
        )
        missing_modalities = []

        height, width, time = self.extract_hwt_from_sample_dict(sample_dict)

        for modality in self.training_modalities:
            # If one modality is completely missing, we need to fill it all with missing values
            if modality not in sample_dict.keys():
                logger.debug(f"Filling {modality} with missing values")
                sample_dict[modality] = self._fill_missing_modality(
                    modality, height, width, time
                )
                missing_modalities.append(modality)
                continue

            # For multi-temporal modalities, we need to handle missing timesteps
            # The missing_timesteps_masks indicates which timesteps are present (True) or missing (False)
            if modality in missing_timesteps_masks:
                mask = missing_timesteps_masks[modality]
                modality_data = sample_dict[modality]
                # cast to appropriate dtype to prevent overflow from missing values
                modality_data = modality_data.astype(self.dtype)

                # As long as the #timesteps is less than the max_sequence_length, we will impute by missing value
                has_missing_timesteps = (
                    not np.all(mask) or len(mask) < self.max_sequence_length
                )
                if has_missing_timesteps:
                    # By default, we will fill missing timesteps with the missing value
                    modality_data = self._fill_missing_timesteps(modality_data, mask)
                # Update the sample dictionary with the potentially imputed data
                sample_dict[modality] = modality_data
        return OlmoEarthSample(**sample_dict), missing_modalities

    def _pad_timestamps(
        self, sample_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], int]:
        """Pad the timestamps to the max_sequence_length."""
        timestamps_data = sample_dict["timestamps"]
        current_length = timestamps_data.shape[0]
        if current_length < self.max_sequence_length:
            pad_width = ((0, self.max_sequence_length - current_length), (0, 0))
            # We pad at the end with copies of the last timestep
            padded_timestamps = np.pad(
                timestamps_data, pad_width=pad_width, mode="edge"
            )
            sample_dict["timestamps"] = padded_timestamps
        return sample_dict, current_length

    def _apply_throttling(self) -> None:
        """Apply read throttling.

        This function is called when reading a sample from the h5py_dir, and it applies
        the configured throttling.
        """
        if self.sec_per_sample is None:
            return
        elapsed = time.time() - self.last_read_time
        time_to_sleep = self.sec_per_sample - elapsed
        self.last_read_time = time.time()
        logger.info(f"{elapsed} elapsed since last read, sleeping for {time_to_sleep}")
        if time_to_sleep <= 0:
            return
        time.sleep(time_to_sleep)

    def _local_h5_path(self, h5_file_path: UPath) -> UPath:
        """Apply read throttling and, if a cache dir is set, return the cached copy."""
        if self.cache_dir is not None:
            cache_file_path = self.cache_dir / h5_file_path.name
            logger.debug(f"Caching H5 file {h5_file_path} to {cache_file_path}")
            if not cache_file_path.exists():
                self._apply_throttling()
                # Copy to a temp file first and then atomically rename it to avoid
                # concurrency issues.
                tmp_file_path = self.cache_dir / (h5_file_path.name + ".tmp")
                with h5_file_path.open("rb") as src, tmp_file_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                tmp_file_path.rename(cache_file_path)
            h5_file_path = cache_file_path

        else:
            self._apply_throttling()
        return h5_file_path

    def read_h5_file(
        self, h5_file_path: UPath
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Read the h5 file."""
        h5_file_path = self._local_h5_path(h5_file_path)
        sample_dict = {}
        with h5_file_path.open("rb") as f:
            with h5py.File(f, "r") as h5file:
                logger.debug(
                    f"Reading h5 file {h5_file_path} with keys {h5file.keys()}"
                )
                # timestamps should not be a floating string
                sample_dict = {
                    k: v[()]
                    for k, v in h5file.items()
                    if k in self.training_modalities
                    # TODO: Fix the floating string issue
                    or k in ["timestamps"]
                }

                # worldcover_onehot is derived from the raw worldcover band: it is not
                # stored on disk, so read the worldcover dataset and one-hot encode it.
                # If worldcover is absent the modality is treated as missing downstream.
                if (
                    Modality.WORLDCOVER_ONEHOT.name in self.training_modalities
                    and Modality.WORLDCOVER.name in h5file
                ):
                    sample_dict[Modality.WORLDCOVER_ONEHOT.name] = one_hot_worldcover(
                        h5file[Modality.WORLDCOVER.name][()], self.dtype
                    )

                if (
                    missing_mask_group_name
                    := ConvertToH5py.missing_timesteps_mask_group_name
                ) in h5file:
                    missing_timesteps_masks = {
                        k: v[()]
                        for k, v in h5file[missing_mask_group_name].items()
                        if k in self.training_modalities
                    }
                else:
                    # To preserve backwards compatibility, we set missing_timesteps_masks to an empty dict if it doesn't exist in file
                    missing_timesteps_masks = {}
        return sample_dict, missing_timesteps_masks

    def _get_h5_file_path(self, index: int) -> UPath:
        """Get the h5 file path."""
        return self.h5py_dir / ConvertToH5py.sample_file_pattern.format(index=index)

    @staticmethod
    def _crop_timestamps_and_masks(
        timestamps: np.ndarray, missing_timesteps_masks: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Crop the timestamps to the first and last valid timestep of the present modalities."""
        # Assumes that the missing timesteps masks has already been filtered for training modalities
        # get first present timestep
        if not missing_timesteps_masks:
            first_valid_timestep = 0
            last_valid_timestep = MAX_SEQUENCE_LENGTH
        else:
            # Timestep masks are the same length as the timestamps
            first_valid_timestep = MAX_SEQUENCE_LENGTH
            last_valid_timestep = 0
            for timestep_mask in missing_timesteps_masks.values():
                valid_timesteps = np.where(timestep_mask)[0]
                if len(valid_timesteps) > 0:
                    first_valid_timestep = min(first_valid_timestep, valid_timesteps[0])
                    last_valid_timestep = max(last_valid_timestep, valid_timesteps[-1])
        timestamps = timestamps[first_valid_timestep : last_valid_timestep + 1]
        for modality, timestep_mask in missing_timesteps_masks.items():
            missing_timesteps_masks[modality] = timestep_mask[
                first_valid_timestep : last_valid_timestep + 1
            ]
        return timestamps, missing_timesteps_masks

    def read_h5_file_per_modality_timestamps(
        self, h5_file_path: UPath
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Read the training modalities and each multitemporal one's own timestamps.

        Returns ``(data, timestamps)``: modality -> raw array as stored, and
        multitemporal modality -> ``(T_m, 3)`` timestamps. Modalities absent from
        the file are absent from both dicts.
        """
        h5_file_path = self._local_h5_path(h5_file_path)
        data: dict[str, np.ndarray] = {}
        timestamps: dict[str, np.ndarray] = {}
        with h5_file_path.open("rb") as f:
            with h5py.File(f, "r") as h5file:
                for modality in self.training_modalities:
                    if modality not in h5file:
                        continue
                    data[modality] = h5file[modality][()]
                    if Modality.get(modality).is_multitemporal:
                        timestamps[modality] = h5file[f"timestamps_{modality}"][()]
        return data, timestamps

    @staticmethod
    def _select_time_range(
        ordinals: dict[str, np.ndarray], time_range_days: float | None
    ) -> dict[str, np.ndarray]:
        """Pick a random range of ``time_range_days`` and keep every capture in it.

        The range length is capped at the sample's span (first to last capture
        over all modalities) and its position is uniform within the span. A
        position whose range holds no capture is redrawn (up to 8 times, then the
        whole span is used). Returns modality -> indices of the kept captures.
        """
        everything = {m: np.arange(len(o)) for m, o in ordinals.items()}
        non_empty = [o for o in ordinals.values() if len(o) > 0]
        if time_range_days is None or not non_empty:
            return everything
        first = min(int(o[0]) for o in non_empty)
        last = max(int(o[-1]) for o in non_empty)
        length = min(float(time_range_days), float(last - first))
        for _ in range(8):
            start = first + np.random.random() * (last - first - length)
            kept = {
                m: np.flatnonzero((o >= start) & (o <= start + length))
                for m, o in ordinals.items()
            }
            if any(len(k) > 0 for k in kept.values()):
                return kept
        return everything

    def _getitem_per_modality_timestamps(
        self, index: int, args: GetItemArgs
    ) -> dict[str, Any]:
        """Assemble a sample from per-modality capture timelines.

        1. Random spatial crop of ``sampled_hw_p * patch_size`` pixels.
        2. Random time range of ``args.time_range_days``; every capture in it.
        3. The kept captures are merged into a shared union timeline (see
           ``build_union_timeline``); each modality is MISSING at the steps where
           it has no capture, so the downstream shared-timestamps code (masking,
           month encodings, 3D RoPE) applies unchanged.
        4. Token budget on REAL tokens: a step costs ``hw_p**2 * bandsets`` for
           each encoded modality captured at it. The kept steps are the run that
           starts at a uniformly random step and extends forward while the running
           total fits ``args.token_budget`` (always at least one step).

        Only real captures are normalized (before they are placed on the
        timeline), so the MISSING padding costs no normalization work.
        """

        def prepare(modality: str, values: np.ndarray) -> np.ndarray:
            values = values.astype(self.dtype)
            if not self.normalize:
                return values
            return self.normalize_image(Modality.get(modality), values).astype(
                self.dtype
            )

        data, timestamps = self.read_h5_file_per_modality_timestamps(
            self._get_h5_file_path(index)
        )
        sampled_hw = args.sampled_hw_p * args.patch_size
        tile_h, tile_w = next(
            v.shape[:2] for m, v in data.items() if Modality.get(m).is_spatial
        )
        start_h = np.random.choice(tile_h - sampled_hw + 1)
        start_w = np.random.choice(tile_w - sampled_hw + 1)

        ordinals = {m: timestamps_to_ordinals(ts) for m, ts in timestamps.items()}
        kept = self._select_time_range(ordinals, args.time_range_days)
        union_ts, step_index = build_union_timeline(
            {m: timestamps[m][kept[m]] for m in timestamps}
        )

        def num_bandsets(modality: str) -> int:
            if args.tokenization_config is not None:
                return args.tokenization_config.get_num_bandsets(modality)
            return Modality.get(modality).num_band_sets

        hw_p_sq = args.sampled_hw_p**2
        step_tokens = np.zeros(len(union_ts), dtype=np.int64)
        fixed_tokens = 0
        for modality in data:
            if modality in args.budget_exclude_modalities:
                continue
            if modality in step_index:
                step_tokens[step_index[modality]] += hw_p_sq * num_bandsets(modality)
            else:
                fixed_tokens += hw_p_sq * num_bandsets(modality)
        first_step, last_step = 0, len(union_ts) - 1
        if args.token_budget is not None and len(union_ts) > 0:
            first_step = last_step = int(np.random.randint(len(union_ts)))
            total = fixed_tokens + step_tokens[first_step]
            while (
                last_step + 1 < len(union_ts)
                and total + step_tokens[last_step + 1] <= args.token_budget
            ):
                last_step += 1
                total += step_tokens[last_step]
        num_steps = last_step - first_step + 1

        sample_dict: dict[str, Any] = {
            "timestamps": union_ts[first_step : last_step + 1]
        }
        crop = np.s_[start_h : start_h + sampled_hw, start_w : start_w + sampled_hw]
        for modality in self.training_modalities:
            spec = Modality.get(modality)
            if spec.image_tile_size_factor != 1:
                raise NotImplementedError(
                    f"per_modality_timestamps does not support {modality} "
                    f"(image_tile_size_factor={spec.image_tile_size_factor})"
                )
            if spec.is_spacetime_varying:
                out = np.full(
                    (sampled_hw, sampled_hw, num_steps, spec.num_bands),
                    MISSING_VALUE,
                    dtype=self.dtype,
                )
                steps = step_index.get(modality, np.zeros(0, dtype=np.int64))
                in_run = (steps >= first_step) & (steps <= last_step)
                if in_run.any():
                    captures = kept[modality][in_run]
                    out[:, :, steps[in_run] - first_step] = prepare(
                        modality, data[modality][crop][:, :, captures]
                    )
                sample_dict[modality] = out
            elif spec.is_space_only_varying:
                if modality in data:
                    sample_dict[modality] = prepare(modality, data[modality][crop])
                else:
                    sample_dict[modality] = np.full(
                        (sampled_hw, sampled_hw, 1, spec.num_bands),
                        MISSING_VALUE,
                        dtype=self.dtype,
                    )
            else:
                raise NotImplementedError(
                    f"per_modality_timestamps only supports spatial modalities, got {modality}"
                )
        return sample_dict

    def __getitem__(self, args: GetItemArgs) -> tuple[int, OlmoEarthSample]:
        """Get the sample at the given index."""
        if hasattr(self, "sample_indices") and self.sample_indices is not None:
            index = self.sample_indices[args.idx]
        else:
            index = args.idx

        if self.per_modality_timestamps:
            sample_dict = self._getitem_per_modality_timestamps(index, args)
            return args.patch_size, OlmoEarthSample(**sample_dict)

        h5_file_path = self._get_h5_file_path(index)

        sample_dict, missing_timesteps_masks = self.read_h5_file(h5_file_path)
        timestamps, missing_timesteps_masks = self._crop_timestamps_and_masks(
            sample_dict["timestamps"], missing_timesteps_masks
        )
        sample_dict["timestamps"] = timestamps
        sample_dict, current_length = self._pad_timestamps(sample_dict)
        # fill sample currently takes like .08 seconds which may bottleneck smaller models
        sample, missing_modalities = self.fill_sample_with_missing_values(
            sample_dict, missing_timesteps_masks
        )

        if self.apply_cutmix:
            subset_sample = subset_sample_cutmix(
                sample,
                patch_size=args.patch_size,
                max_tokens_per_instance=args.token_budget,
                sampled_hw_p=args.sampled_hw_p,
                current_length=current_length,
                missing_timesteps_masks=missing_timesteps_masks,
                tokenization_config=args.tokenization_config,
                target_t=args.target_t,
                budget_exclude_modalities=args.budget_exclude_modalities,
            )
        else:
            subset_sample = subset_sample_default(
                sample,
                patch_size=args.patch_size,
                max_tokens_per_instance=args.token_budget,
                sampled_hw_p=args.sampled_hw_p,
                current_length=current_length,
                missing_timesteps_masks=missing_timesteps_masks,
                tokenization_config=args.tokenization_config,
                target_t=args.target_t,
                budget_exclude_modalities=args.budget_exclude_modalities,
            )

        sample_dict = subset_sample.as_dict()

        # Compute NDVI from raw (un-normalized) S2 L2A bands if requested
        if (
            "ndvi" in sample_dict
            and "sentinel2_l2a" in sample_dict
            and "sentinel2_l2a" not in missing_modalities
        ):
            sample_dict["ndvi"], missing_modalities = self._compute_ndvi(
                sample_dict["sentinel2_l2a"], missing_modalities
            )

        if self.normalize:
            for modality_name in sample_dict.keys():
                if modality_name == "timestamps":
                    continue
                # DO NOT NORMALIZE MISSING MODALITIES otherwise the MISSING_VALUE will be normalized
                if modality_name in missing_modalities:
                    logger.debug(
                        f"Skipping normalization for {modality_name} because it is in missing_modalities"
                    )
                    continue
                logger.debug(f"Normalizing {modality_name}")
                modality_data = sample_dict[modality_name]
                missing_mask = modality_data == MISSING_VALUE
                normalized_data = self.normalize_image(
                    Modality.get(modality_name), modality_data
                )
                # Sentinel Values must be reset after normalization so they can be recognized by missing mask
                sample_dict[modality_name] = np.where(
                    missing_mask, modality_data, normalized_data
                ).astype(self.dtype)

        return args.patch_size, OlmoEarthSample(**sample_dict)


@dataclass
class OlmoEarthDatasetConfig(Config):
    """Configuration for the OlmoEarthDataset."""

    h5py_dir: str
    training_modalities: list[str]
    dtype: str = "float32"
    normalize: bool = True
    cache_dir: str | None = None
    samples_per_sec: float | None = None
    dataset_percentage: float = 1.0
    seed: int = 0
    apply_cutmix: bool = False
    filter_idx_file: str | None = None
    per_modality_timestamps: bool = False

    def get_numpy_dtype(self) -> np.dtype:
        """Get the numpy dtype."""
        if self.dtype == "float16":
            return np.float16
        elif self.dtype == "float32":
            return np.float32
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

    def validate(self) -> None:
        """Validate the configuration and build kwargs.

        Args:
            kwargs: Dictionary of arguments to validate

        Raises:
            ValueError: If any arguments are invalid
        """
        # Validate supported_modalities
        if not isinstance(self.training_modalities, list):
            raise ValueError("training_modalities must be a list")

    @property
    def h5py_dir_upath(self) -> UPath:
        """Get the h5py directory."""
        return UPath(self.h5py_dir)

    @property
    def cache_dir_upath(self) -> UPath:
        """Get the cache directory."""
        return UPath(self.cache_dir)

    def build(self) -> OlmoEarthDataset:
        """Build the dataset."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs["h5py_dir"] = self.h5py_dir_upath
        kwargs["cache_dir"] = (
            self.cache_dir_upath if self.cache_dir is not None else None
        )
        kwargs["dtype"] = self.get_numpy_dtype()
        logger.info(f"OlmoEarthDataset kwargs: {kwargs}")
        return OlmoEarthDataset(**kwargs)


HeliosSample = _deprecated_class_alias(
    OlmoEarthSample, "helios.data.dataset.HeliosSample"
)
HeliosDataset = _deprecated_class_alias(
    OlmoEarthDataset, "helios.data.dataset.HeliosDataset"
)
HeliosDatasetConfig = _deprecated_class_alias(
    OlmoEarthDatasetConfig, "helios.data.dataset.HeliosDatasetConfig"
)
