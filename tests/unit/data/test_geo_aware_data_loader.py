"""Unit tests for GeoAwareDataLoader."""

from pathlib import Path

import numpy as np

from helios.data.constants import Modality
from helios.data.dataset import HeliosDatasetConfig
from helios.data.geo_aware_data_loader import GeoAwareDataLoader, GeoAwareDataLoaderConfig
from helios.data.dataset import collate_helios

def test_geo_aware_data_loader(tmp_path: Path) -> None:
    """Test the GeoAwareDataLoader class."""
    training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.OPENSTREETMAP_RASTER.name,
    ]
    # what If I use a real dir for now
    dataset = HeliosDatasetConfig(
        h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/era5_10_landsat_naip_10_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/1138828",
        training_modalities=training_modalities,
    ).build()

    data_loader_config = GeoAwareDataLoaderConfig(work_dir=tmp_path,
        global_batch_size=8,
        min_patch_size=1,
        max_patch_size=8,
        sampled_hw_p_list=[4, 5, 6],
        seed=0,
        num_neighbors=10000,
        token_budget=15000,
        min_neighbor_radius=1000.0,
        max_neighbor_radius=100_000.0,
    )
    data_loader = data_loader_config.build(dataset=dataset, collator=collate_helios)
    assert isinstance(data_loader, GeoAwareDataLoader)
    data_loader.reshuffle()

    # now get the iterator
    iterator = iter(data_loader)
    print(f"Iterator type: {type(iterator)}")
    sample = next(iterator)
    assert isinstance(sample, HeliosSample)
