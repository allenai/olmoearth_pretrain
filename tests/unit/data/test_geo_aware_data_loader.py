"""Unit tests for GeoAwareDataLoader."""

from pathlib import Path

import numpy as np

from helios.data.constants import Modality
from helios.data.dataset import HeliosDatasetConfig
from helios.data.geo_aware_data_loader import GeoAwareDataLoader, GeoAwareDataLoaderConfig
from helios.data.dataset import collate_helios, HeliosSample
import logging
from helios.data.utils import plot_latlon_distribution
from helios.data.concat import HeliosConcatDatasetConfig

logger = logging.getLogger(__name__)

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
        global_batch_size=64,
        min_patch_size=1,
        max_patch_size=8,
        sampled_hw_p_list=[4, 5, 6],
        seed=0,
        token_budget=15000,
        min_neighbor_radius=100.0,
        max_neighbor_radius=10000.0,
        neighbor_percentage=0.5,
    )
    data_loader = data_loader_config.build(dataset=dataset, collator=collate_helios)
    assert isinstance(data_loader, GeoAwareDataLoader)
    data_loader.reshuffle()

    # now get the iterator
    iterator = iter(data_loader)
    patch_size, sample = next(iterator)
    assert isinstance(sample, HeliosSample)
    assert isinstance(patch_size, int)
    # latlons = sample.latlon
    # logger.info(f"latlons: {latlons}")
    # # plot the first 16 points first
    # import uuid
    # uuid = str(uuid.uuid4())[:8]
    # fig = plot_latlon_distribution(latlons[32:], f"latlon distribution_neighbors_{uuid}", s=1.0)
    # fig.savefig(f"./latlon_distribution_neighbors_{uuid}.png")

    # fig = plot_latlon_distribution(latlons, f"latlon distribution_all_{uuid}", s=1.0)
    # fig.savefig(f"./latlon_distribution_all_{uuid}.png")


# test with a concat dataset
def test_geo_aware_data_loader_concat(tmp_path: Path) -> None:
    """Test the GeoAwareDataLoader class."""
    training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.OPENSTREETMAP_RASTER.name,
    ]
    # what If I use a real dir for now
    dataset_configs = [
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/era5_10_landsat_naip_10_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/1138828",
            training_modalities=training_modalities,
        ),
    ]
    dataset = HeliosConcatDatasetConfig(dataset_configs=dataset_configs).build()

    data_loader_config = GeoAwareDataLoaderConfig(work_dir=tmp_path,
        global_batch_size=64,
        min_patch_size=1,
        max_patch_size=8,
        sampled_hw_p_list=[4, 5, 6],
        seed=0,
        token_budget=15000,
        min_neighbor_radius=100.0,
        max_neighbor_radius=10000.0,
        neighbor_percentage=0.5,
    )
    data_loader = data_loader_config.build(dataset=dataset, collator=collate_helios)
    assert isinstance(data_loader, GeoAwareDataLoader)
    data_loader.reshuffle()

    # now get the iterator
    iterator = iter(data_loader)
    patch_size, sample = next(iterator)
    assert isinstance(sample, HeliosSample)
    assert isinstance(patch_size, int)

def test_concat_dataset_latlon_pickle(tmp_path: Path) -> None:
    training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.OPENSTREETMAP_RASTER.name,
    ]
    # what If I use a real dir for now
    dataset_configs = [
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/era5_10_landsat_naip_10_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/1138828",
            training_modalities=training_modalities,
        ),
    ]
    dataset = HeliosConcatDatasetConfig(dataset_configs=dataset_configs).build()

    data_loader_config = GeoAwareDataLoaderConfig(work_dir=tmp_path,
        global_batch_size=64,
        min_patch_size=1,
        max_patch_size=8,
        sampled_hw_p_list=[4, 5, 6],
        seed=0,
        token_budget=15000,
        min_neighbor_radius=100.0,
        max_neighbor_radius=10000.0,
        neighbor_percentage=0.5,
    )
    data_loader = data_loader_config.build(dataset=dataset, collator=collate_helios)

    data_loader.reshuffle()
    data_loader_torch = data_loader._iter_batches()
    import pickle
    state = pickle.loads(pickle.dumps(data_loader))  # simulate spawn pickling
    print("has latlon_distribution?", hasattr(state.dataset, "latlon_distribution"),
        state.dataset.latlon_distribution is not None)
    print("dict keys:", state.__dict__.keys())

def test_local_donut_indices(tmp_path: Path) -> None:
    """Test the local donut indices."""
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
        global_batch_size=64,
        min_patch_size=1,
        max_patch_size=8,
        sampled_hw_p_list=[4, 5, 6],
        seed=0,
        neighbor_percentage=0.5,
        token_budget=15000,
        min_neighbor_radius=1000.0,
        max_neighbor_radius=100_000.0,
    )
    data_loader = data_loader_config.build(dataset=dataset, collator=collate_helios)
    data_loader.reshuffle()
    global_indices = data_loader.get_global_indices()
    indices = data_loader._get_local_instance_indices(global_indices)
    anchor_index = indices[0]
    # import uuid
    # uuid = str(uuid.uuid4())[:8]
    ring_neighbors = data_loader.get_per_instance_donut_indices(anchor_index, indices)
    logger.info(f"Ring neighbors: {ring_neighbors}")
    # get the latlons of each of them
    latlons = data_loader.get_latlons(ring_neighbors)
    # assert that the latlons are all within 1 degree of the anchor index
    anchor_latlon = data_loader.get_latlons(anchor_index)
    logger.info(f"Anchor latlon: {anchor_latlon}")
    logger.info(f"Latlons: {latlons}")
    diffs = np.abs(latlons - anchor_latlon).sum(axis=1)
    logger.info(f"Differences: {diffs[:10]} shape: {diffs.shape}")
    sum_difs = np.sum(diffs > 2.0)
    logger.info(f"num diffs > 2.0: {sum_difs}")
    assert sum_difs == 0
