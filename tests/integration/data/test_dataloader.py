"""Test the HeliosDataloader class."""

from pathlib import Path

import pytest

from helios.data.constants import Modality
from helios.data.dataloader import HeliosDataLoader
from helios.data.dataset import HeliosDataset, collate_helios
from helios.dataset.convert_to_h5py import ConvertToH5py


@pytest.fixture
def setup_h5py_dir(
    tmp_path: Path, prepare_samples_and_supported_modalities: tuple
) -> None:
    """Setup the h5py directory."""
    prepare_samples, supported_modalities = prepare_samples_and_supported_modalities
    prepared_samples = prepare_samples(tmp_path)
    convert_to_h5py = ConvertToH5py(
        tile_path=tmp_path,
        supported_modalities=[m for m in supported_modalities if m != Modality.LATLON],
        multiprocessed_h5_creation=False,
    )
    convert_to_h5py.prepare_h5_dataset(prepared_samples)


def test_helios_dataloader(tmp_path: Path, setup_h5py_dir: None) -> None:
    """Test the HeliosDataloader class."""
    training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.OPENSTREETMAP_RASTER.name,
    ]
    print(f"Training modalities: {training_modalities}")
    h5py_dir = tmp_path / "h5py_data" / "_".join(sorted(training_modalities)) / "1"
    dataset = HeliosDataset(
        h5py_dir=h5py_dir,
        training_modalities=training_modalities,
        dtype="float32",
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
        sampled_hw_p_list=[256],
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
