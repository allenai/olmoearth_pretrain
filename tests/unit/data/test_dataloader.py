"""Unit tests for dataloader module."""

import functools
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from olmoearth_pretrain.data.collate import (
    collate_double_masked_batched,
    collate_single_masked_batched,
)
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataloader import (
    OlmoEarthDataLoader,
    OlmoEarthDataLoaderConfig,
    _IterableDatasetWrapper,
)
from olmoearth_pretrain.data.dataset import OlmoEarthDataset, OlmoEarthSample
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.train.masking import MaskingConfig


def test_get_batch_item_params_iterator(tmp_path: Path, setup_h5py_dir: Path) -> None:
    """Test the _get_batch_item_params_iterator function."""
    # Setup test data
    """Test the OlmoEarthDataLoader class."""
    training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.OPENSTREETMAP_RASTER.name,
    ]
    dataset = OlmoEarthDataset(
        h5py_dir=setup_h5py_dir,
        training_modalities=training_modalities,
        dtype=np.float32,
    )
    masking_strategy = MaskingConfig(strategy_config={"type": "random"}).build()
    collator = functools.partial(
        collate_single_masked_batched, transform=None, masking_strategy=masking_strategy
    )

    dataset.prepare()
    dataloader = OlmoEarthDataLoader(
        dataset=dataset,
        work_dir=tmp_path,
        global_batch_size=1,
        dp_world_size=1,
        dp_rank=0,
        fs_local_rank=0,
        seed=0,
        shuffle=True,
        num_workers=0,
        collator=collator,
        target_device_type="cpu",
        token_budget=1000000,
        min_patch_size=1,
        max_patch_size=1,
        sampled_hw_p_list=[256],
        masking_strategy=masking_strategy,
        num_masked_views=1,
    )

    dataloader.reshuffle()

    indices = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    patch_size = [1, 2, 3]
    sampled_hw_p = [4, 5, 6]
    rank_batch_size = 3

    # Set a fixed random seed for reproducibility
    dw = _IterableDatasetWrapper(dataloader)

    # Get iterator
    iterator = dw._get_batch_item_params_iterator(
        indices, patch_size, sampled_hw_p, rank_batch_size
    )

    # First batch (should all have the same patch_size and sampled_hw_p)
    first_batch = [next(iterator) for _ in range(3)]

    # Check that all items in first batch have the same patch_size and sampled_hw_p
    first_patch_size = first_batch[0][1]
    first_sampled_hw_p = first_batch[0][2]
    assert all(item[1] == first_patch_size for item in first_batch)
    assert all(item[2] == first_sampled_hw_p for item in first_batch)

    # Second batch (should have different patch_size and sampled_hw_p)
    second_batch = [next(iterator) for _ in range(3)]

    # Check that all items in second batch have the same patch_size and sampled_hw_p
    second_patch_size = second_batch[0][1]
    second_sampled_hw_p = second_batch[0][2]
    assert all(item[1] == second_patch_size for item in second_batch)
    assert all(item[2] == second_sampled_hw_p for item in second_batch)

    # Check that the patch_size or sampled_hw_p changed between batches
    assert (first_patch_size != second_patch_size) or (
        first_sampled_hw_p != second_sampled_hw_p
    )

    # Test that all indices are yielded
    remaining = list(iterator)
    assert len(remaining) == 4  # remaining 4 indices

    # Test that the indices are correct
    all_indices = [item[0] for item in first_batch + second_batch + remaining]
    assert all_indices == list(indices)

    # Test that the third batch has consistent parameters
    if len(remaining) >= 3:
        third_batch = remaining[:3]
        third_patch_size = third_batch[0][1]
        third_sampled_hw_p = third_batch[0][2]
        assert all(item[1] == third_patch_size for item in third_batch)
        assert all(item[2] == third_sampled_hw_p for item in third_batch)


def _build_shape_sampling_dataloader(
    tmp_path: Path,
    setup_h5py_dir: Path,
    *,
    token_budget: int,
    sampled_hw_p_list: list[int],
    time_priority_prob: float,
    exclude_only_decode_from_budget: bool,
    patch_size_probs: list[float] | None = None,
    temporal_bias: float = 0.0,
    min_tokens_per_instance: int = 0,
    min_patch_size: int = 1,
    max_patch_size: int = 1,
    tile_size: int = 256,
) -> OlmoEarthDataLoader:
    """Build a dataloader exercising the (patch_size, hw_p, t) shape sampler."""
    training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.OPENSTREETMAP_RASTER.name,
    ]
    dataset = OlmoEarthDataset(
        h5py_dir=setup_h5py_dir,
        training_modalities=training_modalities,
        dtype=np.float32,
    )
    masking_strategy = MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode",
            "only_decode_modalities": [
                Modality.WORLDCOVER.name,
                Modality.OPENSTREETMAP_RASTER.name,
            ],
        }
    ).build()
    collator = functools.partial(
        collate_single_masked_batched, transform=None, masking_strategy=masking_strategy
    )
    dataset.prepare()
    return OlmoEarthDataLoader(
        dataset=dataset,
        work_dir=tmp_path,
        global_batch_size=1,
        dp_world_size=1,
        dp_rank=0,
        fs_local_rank=0,
        seed=0,
        shuffle=True,
        num_workers=0,
        collator=collator,
        target_device_type="cpu",
        token_budget=token_budget,
        min_patch_size=min_patch_size,
        max_patch_size=max_patch_size,
        sampled_hw_p_list=sampled_hw_p_list,
        patch_size_probs=patch_size_probs,
        time_priority_prob=time_priority_prob,
        temporal_bias=temporal_bias,
        min_tokens_per_instance=min_tokens_per_instance,
        max_timesteps=12,
        tile_size=tile_size,
        exclude_only_decode_from_budget=exclude_only_decode_from_budget,
        masking_strategy=masking_strategy,
        num_masked_views=1,
    )


def test_shape_sampler_emits_target_t_and_respects_budget(
    tmp_path: Path, setup_h5py_dir: Path
) -> None:
    """target_t is emitted, capped by the budget, and hw_p>12 is reachable."""
    dl = _build_shape_sampling_dataloader(
        tmp_path,
        setup_h5py_dir,
        token_budget=4096,
        sampled_hw_p_list=[4, 8, 16, 24],
        time_priority_prob=0.5,
        exclude_only_decode_from_budget=True,
    )
    dl.reshuffle()
    dw = _IterableDatasetWrapper(dl)

    st, so = dl._st_bandsets, dl._so_bandsets
    static, tbs = dl._static_bandsets, dl._time_bandsets
    assert dl.token_budget is not None
    budget = cast(int, dl.token_budget)

    def budget_max_t(hw: int) -> int:
        per_t = st * hw * hw + tbs
        remaining = budget - (so * hw * hw + static)
        return min(dl.max_timesteps, remaining // per_t)

    items = list(
        dw._get_batch_item_params_iterator(
            np.arange(400), dl.patch_sizes, dl.sampled_hw_p_list, rank_batch_size=4
        )
    )

    assert all(len(it) == 4 for it in items)
    hw_seen: set[int] = set()
    t_by_hw: dict[int, set[int]] = {}
    for _idx, ps, hw, t in items:
        assert 1 <= t <= budget_max_t(hw), f"t={t} exceeds budget cap for hw={hw}"
        assert hw * ps <= dl.tile_size
        hw_seen.add(hw)
        t_by_hw.setdefault(hw, set()).add(t)
    # Large grids (>12, impossible under the old IMAGE_TILE_SIZE cap here) are reachable.
    assert max(hw_seen) >= 16
    # target_t is now an independent axis: for a grid that admits a full year it is
    # not pinned to the budget maximum, so we observe multiple distinct t values.
    assert any(len(ts) > 1 for hw, ts in t_by_hw.items() if budget_max_t(hw) > 1)
    # And a full-year sequence is sampled for at least one feasible grid.
    assert any(dl.max_timesteps in ts for ts in t_by_hw.values())


def test_min_tokens_floor_and_temporal_bias(
    tmp_path: Path, setup_h5py_dir: Path
) -> None:
    """The token floor removes tiny shapes; temporal_bias favours fuller sequences."""
    dl = _build_shape_sampling_dataloader(
        tmp_path,
        setup_h5py_dir,
        token_budget=8192,
        sampled_hw_p_list=[1, 2, 4, 8, 12],
        time_priority_prob=0.5,
        exclude_only_decode_from_budget=True,
        temporal_bias=3.0,
        min_tokens_per_instance=36,
    )
    dl.reshuffle()
    dw = _IterableDatasetWrapper(dl)

    st = dl._st_bandsets  # spacetime band-sets (maps excluded)
    items = list(
        dw._get_batch_item_params_iterator(
            np.arange(600), dl.patch_sizes, dl.sampled_hw_p_list, rank_batch_size=4
        )
    )
    tokens = [(hw, t, st * hw * hw * t) for _idx, _ps, hw, t in items]
    # Floor holds: no shape costs fewer than min_tokens, so the hw=1,t=1 corner is gone.
    assert all(tok >= 36 for _hw, _t, tok in tokens)
    assert not any(hw == 1 and t == 1 for hw, t, _tok in tokens)
    # hw=1 costs st tokens/timestep, so a 36-token floor forces t >= ceil(36/st):
    # small grids are pushed onto long sequences.
    min_t_h1 = -(-36 // st)
    assert all(t >= min_t_h1 for hw, t, _tok in tokens if hw == 1)
    # Temporal bias pushes the mean sequence length well above the uniform midpoint.
    mean_t = float(np.mean([t for _hw, t, _tok in tokens]))
    assert mean_t > 7.0


def test_exclude_only_decode_frees_budget(tmp_path: Path, setup_h5py_dir: Path) -> None:
    """Excluding decode-only maps from the budget lowers the space-only rate."""
    with_maps = _build_shape_sampling_dataloader(
        tmp_path / "a",
        setup_h5py_dir,
        token_budget=4096,
        sampled_hw_p_list=[8, 16],
        time_priority_prob=0.0,
        exclude_only_decode_from_budget=False,
    )
    without_maps = _build_shape_sampling_dataloader(
        tmp_path / "b",
        setup_h5py_dir,
        token_budget=4096,
        sampled_hw_p_list=[8, 16],
        time_priority_prob=0.0,
        exclude_only_decode_from_budget=True,
    )
    assert without_maps.budget_exclude_modalities == frozenset(
        [Modality.WORLDCOVER.name, Modality.OPENSTREETMAP_RASTER.name]
    )
    # Space-only band-set rate drops once the maps stop counting against budget.
    assert without_maps._so_bandsets < with_maps._so_bandsets


def _create_test_dataloader(
    tmp_path: Path,
    seed: int = 42,
    shuffle: bool = True,
    num_dataset_repeats_per_epoch: int = 1,
    global_batch_size: int = 2,
) -> OlmoEarthDataLoader:
    """Helper function to create a test dataloader with common parameters."""

    class MockDataset(OlmoEarthDataset):
        def __init__(self, length: int) -> None:
            self.length = length

        def __len__(self) -> int:
            return self.length

    dataset = MockDataset(length=20)
    masking_strategy = MaskingConfig(strategy_config={"type": "random"}).build()
    collator = functools.partial(
        collate_single_masked_batched, transform=None, masking_strategy=masking_strategy
    )
    return OlmoEarthDataLoader(
        dataset=dataset,
        work_dir=tmp_path,
        global_batch_size=global_batch_size,
        dp_world_size=1,
        dp_rank=0,
        fs_local_rank=0,
        seed=seed,
        shuffle=shuffle,
        num_workers=0,
        collator=collator,
        target_device_type="cpu",
        token_budget=1000000,
        min_patch_size=1,
        max_patch_size=1,
        sampled_hw_p_list=[256],
        num_dataset_repeats_per_epoch=num_dataset_repeats_per_epoch,
        masking_strategy=masking_strategy,
        num_masked_views=1,
    )


def test_build_global_indices_same_seed_epoch_deterministic(tmp_path: Path) -> None:
    """Test that same seed and epoch produce identical indices."""
    dataloader1 = _create_test_dataloader(tmp_path, seed=42)
    dataloader1._epoch = 1
    indices1 = dataloader1._build_global_indices()

    dataloader2 = _create_test_dataloader(tmp_path, seed=42)
    dataloader2._epoch = 1
    indices2 = dataloader2._build_global_indices()

    np.testing.assert_array_equal(
        indices1, indices2, "Same seed and epoch should produce identical indices"
    )


def test_build_global_indices_different_epochs_different_shuffle(
    tmp_path: Path,
) -> None:
    """Test that different epochs produce different shuffled orders."""
    dataloader1 = _create_test_dataloader(tmp_path, seed=42)
    dataloader1._epoch = 1
    indices1 = dataloader1._build_global_indices()

    dataloader2 = _create_test_dataloader(tmp_path, seed=42)
    dataloader2._epoch = 2
    indices2 = dataloader2._build_global_indices()

    # Should have same length but different order (with high probability)
    assert len(indices1) == len(indices2), (
        "Different epochs should produce same length indices"
    )

    assert not np.array_equal(indices1, indices2), (
        "Different epochs should produce different shuffled order"
    )


def test_build_global_indices_no_shuffle_consistent(tmp_path: Path) -> None:
    """Test that no shuffle produces identical results regardless of epoch."""
    dataloader1 = _create_test_dataloader(tmp_path, seed=42, shuffle=False)
    dataloader1._epoch = 1
    indices1 = dataloader1._build_global_indices()

    dataloader2 = _create_test_dataloader(tmp_path, seed=42, shuffle=False)
    dataloader2._epoch = 5
    indices2 = dataloader2._build_global_indices()

    np.testing.assert_array_equal(
        indices1,
        indices2,
        "No shuffle should produce identical results regardless of epoch",
    )


def test_build_global_indices_different_seeds_different_results(tmp_path: Path) -> None:
    """Test that different seeds produce different results."""
    dataloader1 = _create_test_dataloader(tmp_path, seed=42)
    dataloader1._epoch = 1
    indices1 = dataloader1._build_global_indices()

    dataloader2 = _create_test_dataloader(tmp_path, seed=999)
    dataloader2._epoch = 1
    indices2 = dataloader2._build_global_indices()

    # Different seeds should produce different results (with high probability)
    assert not np.array_equal(indices1, indices2), (
        "Different seeds should produce different results"
    )


def test_build_global_indices_multiple_repeats_deterministic(tmp_path: Path) -> None:
    """Test that multiple dataset repeats are deterministic."""
    dataloader1 = _create_test_dataloader(
        tmp_path, seed=42, num_dataset_repeats_per_epoch=3
    )
    dataloader1._epoch = 1
    indices1 = dataloader1._build_global_indices()

    dataloader2 = _create_test_dataloader(
        tmp_path, seed=42, num_dataset_repeats_per_epoch=3
    )
    dataloader2._epoch = 1
    indices2 = dataloader2._build_global_indices()

    np.testing.assert_array_equal(
        indices1,
        indices2,
        "Same parameters should produce identical results with multiple repeats",
    )


def test_build_global_indices_properties_validation(tmp_path: Path) -> None:
    """Test that built indices have expected properties and constraints."""
    dataloader = _create_test_dataloader(tmp_path, seed=42)
    dataloader._epoch = 1
    indices = dataloader._build_global_indices()

    # Basic properties
    assert len(indices) > 0, "Indices should not be empty"
    assert isinstance(indices, np.ndarray), "Indices should be numpy array"
    assert indices.dtype == np.uint32, "Indices should be uint32 type"

    # All indices should be valid dataset indices
    assert np.all(indices >= 0), "All indices should be non-negative"
    assert np.all(indices < len(dataloader.dataset)), (
        "All indices should be within dataset bounds"
    )

    # Length should be divisible by global batch size (due to cropping)
    assert len(indices) % dataloader.global_batch_size == 0, (
        "Indices length should be divisible by global batch size"
    )


class TestDataLoaderConfigValidation:
    """Tests for OlmoEarthDataLoaderConfig validation."""

    def test_config_validation_requires_masking_config(self) -> None:
        """Test that masking_config is always required."""
        config = OlmoEarthDataLoaderConfig(
            work_dir="/tmp/test",
            global_batch_size=2,
            min_patch_size=1,
            max_patch_size=8,
            sampled_hw_p_list=[4],
            seed=42,
            num_masked_views=1,
            masking_config=None,  # Not provided
        )

        with pytest.raises(ValueError, match="masking_config must be provided"):
            config.validate()

    def test_config_validation_invalid_num_masked_views(self) -> None:
        """Test that invalid num_masked_views raises ValueError."""
        config = OlmoEarthDataLoaderConfig(
            work_dir="/tmp/test",
            global_batch_size=2,
            min_patch_size=1,
            max_patch_size=8,
            sampled_hw_p_list=[4],
            seed=42,
            num_masked_views=3,  # Invalid
            masking_config=MaskingConfig(strategy_config={"type": "random"}),
        )

        with pytest.raises(ValueError, match="num_masked_views must be 1 or 2"):
            config.validate()

    def test_config_validation_invalid_num_masked_views_zero(self) -> None:
        """Test that num_masked_views=0 raises ValueError."""
        config = OlmoEarthDataLoaderConfig(
            work_dir="/tmp/test",
            global_batch_size=2,
            min_patch_size=1,
            max_patch_size=8,
            sampled_hw_p_list=[4],
            seed=42,
            num_masked_views=0,  # Invalid (legacy mode removed)
            masking_config=MaskingConfig(strategy_config={"type": "random"}),
        )

        with pytest.raises(ValueError, match="num_masked_views must be 1 or 2"):
            config.validate()

    def test_config_validation_valid_single_masked(self) -> None:
        """Test that valid single masked config passes validation."""
        config = OlmoEarthDataLoaderConfig(
            work_dir="/tmp/test",
            global_batch_size=2,
            min_patch_size=1,
            max_patch_size=8,
            sampled_hw_p_list=[4],
            seed=42,
            num_masked_views=1,
            masking_config=MaskingConfig(strategy_config={"type": "random"}),
        )

        # Should not raise
        config.validate()

    def test_config_validation_valid_double_masked(self) -> None:
        """Test that valid double masked config passes validation."""
        config = OlmoEarthDataLoaderConfig(
            work_dir="/tmp/test",
            global_batch_size=2,
            min_patch_size=1,
            max_patch_size=8,
            sampled_hw_p_list=[4],
            seed=42,
            num_masked_views=2,
            masking_config=MaskingConfig(strategy_config={"type": "random"}),
            masking_config_b=MaskingConfig(strategy_config={"type": "time"}),
        )

        # Should not raise
        config.validate()


class TestGetMockBatch:
    """Tests for OlmoEarthDataLoader.get_mock_batch method."""

    def test_get_mock_batch_single_masked(
        self, tmp_path: Path, setup_h5py_dir: Path
    ) -> None:
        """Test get_mock_batch returns correct format for single masked mode."""
        import functools

        training_modalities = [
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.WORLDCOVER.name,
        ]
        dataset = OlmoEarthDataset(
            h5py_dir=setup_h5py_dir,
            training_modalities=training_modalities,
            dtype=np.float32,
        )
        dataset.prepare()

        masking_strategy = MaskingConfig(strategy_config={"type": "random"}).build()

        # Use batched collator (as the config.build() method does)
        collator = functools.partial(
            collate_single_masked_batched,
            transform=None,
            masking_strategy=masking_strategy,
        )

        dataloader = OlmoEarthDataLoader(
            dataset=dataset,
            work_dir=tmp_path,
            global_batch_size=2,
            min_patch_size=1,
            max_patch_size=1,
            sampled_hw_p_list=[8],
            token_budget=1000000,
            seed=42,
            collator=collator,
            num_masked_views=1,
            masking_strategy=masking_strategy,
        )

        mock_batch = dataloader.get_mock_batch()

        # Should return (patch_size, MaskedOlmoEarthSample)
        assert len(mock_batch) == 2
        patch_size, sample = mock_batch
        assert patch_size == 1
        assert isinstance(sample, MaskedOlmoEarthSample)

    def test_get_mock_batch_double_masked(
        self, tmp_path: Path, setup_h5py_dir: Path
    ) -> None:
        """Test get_mock_batch returns correct format for double masked mode."""
        import functools

        training_modalities = [
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.WORLDCOVER.name,
        ]
        dataset = OlmoEarthDataset(
            h5py_dir=setup_h5py_dir,
            training_modalities=training_modalities,
            dtype=np.float32,
        )
        dataset.prepare()

        masking_strategy = MaskingConfig(strategy_config={"type": "random"}).build()

        # Use batched collator (as the config.build() method does)
        collator = functools.partial(
            collate_double_masked_batched,
            transform=None,
            masking_strategy=masking_strategy,
            masking_strategy_b=None,
        )

        dataloader = OlmoEarthDataLoader(
            dataset=dataset,
            work_dir=tmp_path,
            global_batch_size=2,
            min_patch_size=1,
            max_patch_size=1,
            sampled_hw_p_list=[8],
            token_budget=1000000,
            seed=42,
            collator=collator,
            num_masked_views=2,
            masking_strategy=masking_strategy,
        )

        mock_batch = dataloader.get_mock_batch()

        # Should return (patch_size, MaskedOlmoEarthSample, MaskedOlmoEarthSample)
        assert len(mock_batch) == 3
        patch_size, sample_a, sample_b = mock_batch
        assert patch_size == 1
        assert isinstance(sample_a, MaskedOlmoEarthSample)
        assert isinstance(sample_b, MaskedOlmoEarthSample)


class TestCollateSingleMaskedBatched:
    """Tests for collate_single_masked_batched function."""

    def test_collate_single_masked_batched_applies_masking(self) -> None:
        """Test that collate_single_masked_batched applies masking to the batch."""
        # Create raw OlmoEarthSamples (numpy arrays, as they come from the dataset)
        sample1 = OlmoEarthSample(
            sentinel2_l2a=np.ones((8, 8, 12, 13), dtype=np.float32),
            latlon=np.array([0.5, 0.5], dtype=np.float32),
            timestamps=np.array([[1, 1, 2020] for _ in range(12)], dtype=np.int32),
        )
        sample2 = OlmoEarthSample(
            sentinel2_l2a=np.ones((8, 8, 12, 13), dtype=np.float32) * 2,
            latlon=np.array([0.6, 0.6], dtype=np.float32),
            timestamps=np.array([[1, 1, 2020] for _ in range(12)], dtype=np.int32),
        )

        masking_strategy = MaskingConfig(strategy_config={"type": "random"}).build()

        batch = [(1, sample1), (1, sample2)]
        patch_size, collated = collate_single_masked_batched(
            batch, transform=None, masking_strategy=masking_strategy
        )

        assert patch_size == 1
        assert isinstance(collated, MaskedOlmoEarthSample)
        assert collated.sentinel2_l2a is not None
        assert collated.sentinel2_l2a_mask is not None
        # Batch dimension should be present
        assert collated.sentinel2_l2a.shape[0] == 2
        assert collated.sentinel2_l2a_mask.shape[0] == 2
        assert collated.latlon is not None
        assert collated.latlon.shape == (2, 2)

    def test_collate_single_masked_batched_with_transform(self) -> None:
        """Test that collate_single_masked_batched applies transform before masking."""
        from olmoearth_pretrain.data.transform import TransformConfig

        sample1 = OlmoEarthSample(
            sentinel2_l2a=np.ones((8, 8, 12, 13), dtype=np.float32),
            latlon=np.array([0.5, 0.5], dtype=np.float32),
            timestamps=np.array([[1, 1, 2020] for _ in range(12)], dtype=np.int32),
        )
        sample2 = OlmoEarthSample(
            sentinel2_l2a=np.ones((8, 8, 12, 13), dtype=np.float32) * 2,
            latlon=np.array([0.6, 0.6], dtype=np.float32),
            timestamps=np.array([[1, 1, 2020] for _ in range(12)], dtype=np.int32),
        )

        transform = TransformConfig(transform_type="no_transform").build()
        masking_strategy = MaskingConfig(strategy_config={"type": "random"}).build()

        batch = [(1, sample1), (1, sample2)]
        patch_size, collated = collate_single_masked_batched(
            batch, transform=transform, masking_strategy=masking_strategy
        )

        assert patch_size == 1
        assert isinstance(collated, MaskedOlmoEarthSample)
        assert collated.sentinel2_l2a is not None
        assert collated.sentinel2_l2a_mask is not None


class TestCollateDoubleMaskedBatched:
    """Tests for collate_double_masked_batched function."""

    def test_collate_double_masked_batched_applies_two_masks(self) -> None:
        """Test that collate_double_masked_batched applies two independent masks."""
        sample1 = OlmoEarthSample(
            sentinel2_l2a=np.ones((8, 8, 12, 13), dtype=np.float32),
            latlon=np.array([0.5, 0.5], dtype=np.float32),
            timestamps=np.array([[1, 1, 2020] for _ in range(12)], dtype=np.int32),
        )
        sample2 = OlmoEarthSample(
            sentinel2_l2a=np.ones((8, 8, 12, 13), dtype=np.float32) * 2,
            latlon=np.array([0.6, 0.6], dtype=np.float32),
            timestamps=np.array([[1, 1, 2020] for _ in range(12)], dtype=np.int32),
        )

        masking_strategy = MaskingConfig(strategy_config={"type": "random"}).build()

        batch = [(1, sample1), (1, sample2)]
        patch_size, collated_a, collated_b = collate_double_masked_batched(
            batch,
            transform=None,
            masking_strategy=masking_strategy,
            masking_strategy_b=None,  # Uses same strategy
        )

        assert patch_size == 1
        assert isinstance(collated_a, MaskedOlmoEarthSample)
        assert isinstance(collated_b, MaskedOlmoEarthSample)
        assert collated_a.sentinel2_l2a is not None
        assert collated_b.sentinel2_l2a is not None
        assert collated_a.sentinel2_l2a_mask is not None
        assert collated_b.sentinel2_l2a_mask is not None
        # Batch dimension should be present
        assert collated_a.sentinel2_l2a.shape[0] == 2
        assert collated_b.sentinel2_l2a.shape[0] == 2
        # The two masks should be different (independent random sampling)
        assert not torch.equal(
            collated_a.sentinel2_l2a_mask, collated_b.sentinel2_l2a_mask
        )

    def test_collate_double_masked_batched_different_strategies(self) -> None:
        """Test that collate_double_masked_batched can use different strategies."""
        sample1 = OlmoEarthSample(
            sentinel2_l2a=np.ones((8, 8, 12, 13), dtype=np.float32),
            latlon=np.array([0.5, 0.5], dtype=np.float32),
            timestamps=np.array([[1, 1, 2020] for _ in range(12)], dtype=np.int32),
        )
        sample2 = OlmoEarthSample(
            sentinel2_l2a=np.ones((8, 8, 12, 13), dtype=np.float32) * 2,
            latlon=np.array([0.6, 0.6], dtype=np.float32),
            timestamps=np.array([[1, 1, 2020] for _ in range(12)], dtype=np.int32),
        )

        masking_strategy_a = MaskingConfig(
            strategy_config={"type": "random", "encode_ratio": 0.3, "decode_ratio": 0.7}
        ).build()
        masking_strategy_b = MaskingConfig(
            strategy_config={"type": "random", "encode_ratio": 0.7, "decode_ratio": 0.3}
        ).build()

        batch = [(1, sample1), (1, sample2)]
        patch_size, collated_a, collated_b = collate_double_masked_batched(
            batch,
            transform=None,
            masking_strategy=masking_strategy_a,
            masking_strategy_b=masking_strategy_b,
        )

        assert patch_size == 1
        assert isinstance(collated_a, MaskedOlmoEarthSample)
        assert isinstance(collated_b, MaskedOlmoEarthSample)
        assert collated_a.sentinel2_l2a_mask is not None
        assert collated_b.sentinel2_l2a_mask is not None
