"""Unit tests for GeoAwareDataLoader."""

from pathlib import Path

import numpy as np

from helios.data.constants import Modality
from helios.data.dataset import HeliosDataset
from helios.data.geo_aware_data_loader import GeoAwareDataLoader, GeoAwareDataLoaderConfig
from helios.data.dataset import collate_helios

def test_geo_aware_data_loader(tmp_path: Path, setup_h5py_dir: Path) -> None:
    """Test the GeoAwareDataLoader class."""
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

    data_loader_config = GeoAwareDataLoaderConfig(work_dir=tmp_path,
        global_batch_size=1,
        min_patch_size=1,
        max_patch_size=8,
        sampled_hw_p_list=[4, 5, 6],
        seed=0,
        num_neighbors=10000,
        min_neighbor_radius=1000.0,
        max_neighbor_radius=100_000.0,
    )
    data_loader = data_loader_config.build(dataset=dataset, collator=collate_helios)
    assert isinstance(data_loader, GeoAwareDataLoader)
