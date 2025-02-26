"""Unit tests for the Helios Dataset Visualization."""

import os
from pathlib import Path

from helios.data.dataset import HeliosDataset
from helios.data.normalize import Normalizer, Strategy
from helios.data.visualize import visualize_sample


def test_visualize_sample(
    prepare_samples_and_supported_modalities: tuple, tmp_path: Path
) -> None:
    """Test the visualize_sample function."""
    tmp_path = Path("./test_vis")
    os.makedirs(tmp_path, exist_ok=True)
    prepare_samples, supported_modalities = prepare_samples_and_supported_modalities
    samples = prepare_samples(tmp_path)
    dataset = HeliosDataset(
        samples=samples,
        supported_modalities=supported_modalities,
        tile_path=tmp_path,
        dtype="float32",
    )
    for i in range(len(samples)):
        visualize_sample(
            dataset,
            i,
            Normalizer(Strategy.PREDEFINED),
            tmp_path / "visualizations_predefined",
        )
        assert (tmp_path / "visualizations_predefined" / f"sample_{i}.png").exists()

    for i in range(len(samples)):
        visualize_sample(
            dataset,
            i,
            Normalizer(Strategy.COMPUTED),
            tmp_path / "visualizations_computed",
        )
        assert (tmp_path / "visualizations_computed" / f"sample_{i}.png").exists()
