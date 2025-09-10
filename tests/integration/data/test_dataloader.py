"""Test the HeliosDataloader class."""

from pathlib import Path

import numpy as np
import pytest

from helios.data.concat import HeliosConcatDatasetConfig
from helios.data.constants import Modality
from helios.data.dataloader import HeliosDataLoader, HeliosDataLoaderConfig
from helios.data.dataset import HeliosDataset, HeliosDatasetConfig, collate_helios


def test_helios_dataloader(tmp_path: Path, setup_h5py_dir: Path) -> None:
    """Test the HeliosDataloader class."""
    training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.OPENSTREETMAP_RASTER.name,
    ]
    dataset_config = HeliosDatasetConfig(
        h5py_dir=str(setup_h5py_dir),
        training_modalities=training_modalities,
    )
    dataset = dataset_config.build()
    dataset.prepare()
    assert isinstance(dataset, HeliosDataset)
    dataloader_config = HeliosDataLoaderConfig(
        work_dir=str(tmp_path),
        global_batch_size=1,
        seed=0,
        shuffle=True,
        num_workers=0,
        target_device_type="cpu",
        token_budget=1000000,
        min_patch_size=1,
        max_patch_size=1,
        sampled_hw_p_list=[6],
    )
    dataloader = dataloader_config.build(dataset, collate_helios)
    dataloader.reshuffle()
    batches_processed = 0
    for batch in dataloader:
        batches_processed += 1

    state_dict = dataloader.state_dict()
    dataloader.reset()
    dataloader.load_state_dict(state_dict)
    assert dataloader.batches_processed == batches_processed

    assert batches_processed == 1


def test_helios_dataloader_dataset_percentage(
    tmp_path: Path, setup_h5py_dir_20_samples: Path
) -> None:
    """Test the HeliosDataloader class."""
    training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.OPENSTREETMAP_RASTER.name,
    ]
    dataset_config = HeliosDatasetConfig(
        h5py_dir=str(setup_h5py_dir_20_samples),
        training_modalities=training_modalities,
        dataset_percentage=0.5,
        seed=42,
    )
    dataset = dataset_config.build()
    dataset.prepare()
    len_dataset = len(dataset)
    assert len_dataset == 10
    assert isinstance(dataset, HeliosDataset)
    dataloader_config = HeliosDataLoaderConfig(
        work_dir=str(tmp_path),
        global_batch_size=1,
        seed=0,
        shuffle=True,
        num_workers=0,
        target_device_type="cpu",
        token_budget=1000000,
        min_patch_size=1,
        max_patch_size=1,
        sampled_hw_p_list=[6],
    )
    dataloader = dataloader_config.build(dataset, collate_helios)
    len_dataloader = len(dataloader)
    assert len_dataloader == 10

    dataloader.reshuffle()
    batches_processed = 0
    for batch in dataloader:
        batches_processed += 1

    assert dataloader.batches_processed == batches_processed


@pytest.mark.parametrize("dp_world_size", [2, 8])
def test_helios_dataloader_dataset_percentage_bigger_world_size(
    tmp_path: Path, setup_h5py_dir_100_samples: Path, dp_world_size: int
) -> None:
    """Test the HeliosDataloader class with different world sizes."""
    training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.OPENSTREETMAP_RASTER.name,
    ]
    dataset_config = HeliosDatasetConfig(
        h5py_dir=str(setup_h5py_dir_100_samples),
        training_modalities=training_modalities,
        dataset_percentage=0.5,
        seed=42,
    )
    dataset = dataset_config.build()
    dataset.prepare()
    len_dataset = len(dataset)
    assert len_dataset == 50
    assert isinstance(dataset, HeliosDataset)
    dataloader = HeliosDataLoader(
        dataset=dataset,
        work_dir=str(tmp_path),
        global_batch_size=16,
        dp_world_size=dp_world_size,
        dp_rank=0,
        collator=collate_helios,
        seed=0,
        shuffle=True,
        num_workers=0,
        target_device_type="cpu",
        token_budget=1000000,
        min_patch_size=1,
        max_patch_size=1,
        sampled_hw_p_list=[6],
        fs_local_rank=0,
    )
    len_dataloader = len(dataloader)
    assert len_dataloader == 3

    dataloader.reshuffle()
    batches_processed = 0
    for batch in dataloader:
        batches_processed += 1
    assert dataloader.batches_processed == batches_processed


def test_dataset_percentage_consistent_across_epochs(
    tmp_path: Path, setup_h5py_dir_100_samples: Path
) -> None:
    """Test that different epochs with same dataset percentage yield same unique indices."""
    training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.OPENSTREETMAP_RASTER.name,
    ]

    # Helper to create dataset and dataloader with shared config
    def make_dataset_and_dataloader(
        work_dir: Path,
    ) -> tuple[HeliosDataset, HeliosDataLoader]:
        dataset_config = HeliosDatasetConfig(
            h5py_dir=str(setup_h5py_dir_100_samples),
            training_modalities=training_modalities,
            dataset_percentage=0.5,
            seed=42,
        )
        dataset = dataset_config.build()
        dataset.prepare()
        dataloader_config = HeliosDataLoaderConfig(
            work_dir=str(work_dir),
            global_batch_size=4,
            seed=42,
            shuffle=True,
            num_workers=0,
            target_device_type="cpu",
            token_budget=1000000,
            min_patch_size=1,
            max_patch_size=1,
            sampled_hw_p_list=[6],
        )
        dataloader = dataloader_config.build(dataset, collate_helios)
        return dataset, dataloader

    dataset1, dataloader1 = make_dataset_and_dataloader(tmp_path / "epoch1")
    dataset2, dataloader2 = make_dataset_and_dataloader(tmp_path / "epoch2")

    # Reshuffle for different epochs
    dataloader1.reshuffle(epoch=1)
    dataloader2.reshuffle(epoch=2)

    # The underlying dataset should have the same sample_indices since same seed was used for filtering but not neccesarily same order
    # so check unique-ness
    assert (
        np.unique(dataset1.sample_indices).shape[0]
        == np.unique(dataset2.sample_indices).shape[0]
    )


def test_concat_dataset_percentage_filtering(
    tmp_path: Path, setup_h5py_dir_20_samples: Path, setup_h5py_dir_100_samples: Path
) -> None:
    """Test that dataset percentage filtering works with HeliosConcatDataset."""
    training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.OPENSTREETMAP_RASTER.name,
    ]

    # Create dataset configs
    dataset_configs = [
        HeliosDatasetConfig(
            h5py_dir=str(setup_h5py_dir_20_samples),
            training_modalities=training_modalities,
        ),
        HeliosDatasetConfig(
            h5py_dir=str(setup_h5py_dir_100_samples),
            training_modalities=training_modalities,
        ),
    ]

    # Build concat dataset
    concat_config = HeliosConcatDatasetConfig(
        dataset_configs=dataset_configs, dataset_percentage=0.5, seed=42
    )
    concat_dataset = concat_config.build()
    concat_dataset.prepare()
    dataloader_config = HeliosDataLoaderConfig(
        work_dir=str(tmp_path),
        global_batch_size=8,
        seed=42,
        shuffle=True,
        num_workers=0,
        target_device_type="cpu",
        token_budget=1000000,
        min_patch_size=1,
        max_patch_size=1,
        sampled_hw_p_list=[6],
    )
    dataloader = dataloader_config.build(concat_dataset, collate_helios)

    dataloader.reshuffle(epoch=1)

    # Total concat dataset should also be filtered
    assert len(concat_dataset) == 60  # 10 + 50

    # Test that we can iterate through the dataloader
    batches_processed = 0
    for batch in dataloader:
        batches_processed += 1
        assert batch is not None

    assert batches_processed > 0
