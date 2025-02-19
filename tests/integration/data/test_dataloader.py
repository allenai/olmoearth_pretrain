"""Test the HeliosDataloader class."""

from pathlib import Path

from torch.utils.data import default_collate

from helios.data.dataloader import HeliosDataLoader
from helios.data.dataset import HeliosDataset


def test_helios_dataloader(
    tmp_path: Path, prepare_samples_and_supported_modalities: tuple
) -> None:
    """Test the HeliosDataloader class."""
    prepare_samples, supported_modalities = prepare_samples_and_supported_modalities
    samples = prepare_samples(tmp_path)
    dataset = HeliosDataset(
        samples=samples,
        tile_path=tmp_path,
        supported_modalities=supported_modalities,
    )
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
        num_threads=1,
        num_workers=0,
        collator=default_collate,
        target_device_type="cpu",
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


# TODO: Add test for global indices
# TODO: Add test for multi-threading
