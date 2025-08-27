"""Test the HeliosDataloader class."""

from pathlib import Path

import numpy as np
import pytest

from helios.data.constants import Modality
from helios.data.dataloader import HeliosDataLoader
from helios.data.dataset import HeliosDataset, collate_helios


def test_helios_dataloader(tmp_path: Path, setup_h5py_dir: Path) -> None:
    """Test the HeliosDataloader class."""
    training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.OPENSTREETMAP_RASTER.name,
    ]
    dataset = HeliosDataset(
        h5py_dir=setup_h5py_dir,
        training_modalities=training_modalities,
        dtype=np.float32,
    )

    dataset.prepare()
    assert isinstance(dataset, HeliosDataset)
    dataloader = HeliosDataLoader(
        dataset=dataset,
        work_dir=tmp_path,
        global_batch_size=1,
        dp_world_size=1,
        dp_rank=0,
        fs_local_rank=0,
        seed=0,
        shuffle=True,
        num_workers=0,
        collator=collate_helios,
        target_device_type="cpu",
        token_budget=1000000,
        min_patch_size=1,
        max_patch_size=1,
        sampled_hw_p_list=[6],
    )

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
    dataset = HeliosDataset(
        h5py_dir=setup_h5py_dir_20_samples,
        training_modalities=training_modalities,
        dtype=np.float32,
    )

    dataset.prepare()
    len_dataset = len(dataset)
    assert len_dataset == 20
    assert isinstance(dataset, HeliosDataset)
    dataloader = HeliosDataLoader(
        dataset=dataset,
        work_dir=tmp_path,
        global_batch_size=1,
        dp_world_size=1,
        dp_rank=0,
        fs_local_rank=0,
        seed=0,
        shuffle=True,
        num_workers=0,
        collator=collate_helios,
        target_device_type="cpu",
        token_budget=1000000,
        min_patch_size=1,
        max_patch_size=1,
        sampled_hw_p_list=[6],
        dataset_percentage=0.5,
    )
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
    dataset = HeliosDataset(
        h5py_dir=setup_h5py_dir_100_samples,
        training_modalities=training_modalities,
        dtype=np.float32,
    )

    dataset.prepare()
    len_dataset = len(dataset)
    assert len_dataset == 100
    assert isinstance(dataset, HeliosDataset)
    dataloader = HeliosDataLoader(
        dataset=dataset,
        work_dir=tmp_path,
        global_batch_size=16,
        dp_world_size=dp_world_size,
        dp_rank=0,
        fs_local_rank=0,
        seed=0,
        shuffle=True,
        num_workers=0,
        collator=collate_helios,
        target_device_type="cpu",
        token_budget=1000000,
        min_patch_size=1,
        max_patch_size=1,
        sampled_hw_p_list=[6],
        dataset_percentage=0.5,
    )
    len_dataloader = len(dataloader)
    assert len_dataloader == 3

    dataloader.reshuffle()
    batches_processed = 0
    for batch in dataloader:
        batches_processed += 1
    assert dataloader.batches_processed == batches_processed
