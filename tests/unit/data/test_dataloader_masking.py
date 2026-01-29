"""Unit tests for dataloader-side masking functionality."""

from pathlib import Path

import numpy as np
import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataloader import (
    OlmoEarthDataLoader,
    OlmoEarthDataLoaderConfig,
)
from olmoearth_pretrain.data.dataset import (
    OlmoEarthDataset,
    OlmoEarthSample,
    collate_double_masked_batched,
    collate_olmoearth_pretrain,
    collate_single_masked_batched,
)
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.train.masking import MaskingConfig


class TestDataLoaderConfigValidation:
    """Tests for OlmoEarthDataLoaderConfig validation."""

    def test_config_validation_requires_masking_config(self) -> None:
        """Test that masking_config is required when num_masked_views > 0."""
        config = OlmoEarthDataLoaderConfig(
            work_dir="/tmp/test",
            global_batch_size=2,
            min_patch_size=1,
            max_patch_size=8,
            sampled_hw_p_list=[4],
            seed=42,
            num_masked_views=1,  # Requires masking_config
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

        with pytest.raises(ValueError, match="num_masked_views must be 0, 1, or 2"):
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

    def test_get_mock_batch_legacy_mode(
        self, tmp_path: Path, setup_h5py_dir: Path
    ) -> None:
        """Test get_mock_batch returns correct format in legacy mode."""
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

        dataloader = OlmoEarthDataLoader(
            dataset=dataset,
            work_dir=tmp_path,
            global_batch_size=2,
            min_patch_size=1,
            max_patch_size=1,
            sampled_hw_p_list=[8],
            token_budget=1000000,
            seed=42,
            collator=collate_olmoearth_pretrain,
            num_masked_views=0,  # Legacy mode
        )

        mock_batch = dataloader.get_mock_batch()

        # Should return (patch_size, OlmoEarthSample)
        assert len(mock_batch) == 2
        patch_size, sample = mock_batch
        assert patch_size == 1
        assert isinstance(sample, OlmoEarthSample)

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
