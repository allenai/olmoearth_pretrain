"""Unit tests for the Helios Dataset Visualization."""

import os
from pathlib import Path

from helios.data.constants import Modality
from helios.data.dataset import HeliosDataset
from helios.data.visualize import visualize_sample


def test_visualize_sample(
    setup_h5py_dir: Path,
) -> None:
    """Test the visualize_sample function."""
    tmp_path = Path("./test_vis")
    os.makedirs(tmp_path, exist_ok=True)
    training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.OPENSTREETMAP_RASTER.name,
    ]
    dataset = HeliosDataset(
        h5py_dir=setup_h5py_dir,
        dtype="float32",
        training_modalities=training_modalities,
    )
    dataset.prepare()
    for i in range(len(dataset)):
        visualize_sample(
            dataset,
            i,
            tmp_path / "visualizations_predefined",
        )
        assert (tmp_path / "visualizations_predefined" / f"sample_{i}.png").exists()
