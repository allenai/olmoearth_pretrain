"""Test GeoBench dataset."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from olmoearth_pretrain.evals.datasets import GeobenchDataset
from olmoearth_pretrain.evals.datasets.configs import dataset_to_config
from olmoearth_pretrain.evals.datasets.utils import eval_collate_fn


@pytest.fixture
def geobench_dir() -> Path:
    """Fixture providing path to test dataset index."""
    return Path("tests/fixtures/sample_geobench")


def test_geobench_dataset(geobench_dir: Path) -> None:
    """Test the dataset works."""
    d = GeobenchDataset(
        dataset="m-eurosat",
        geobench_dir=geobench_dir,
        split="train",
        label_fraction=0.01,
    )
    sample, _ = d[0]
    assert isinstance(sample.sentinel2_l2a, torch.Tensor)
    assert sample.sentinel2_l2a.shape == (64, 64, 1, 12)


def test_geobench_dataset_and_dataloader(geobench_dir: Path) -> None:
    """Test the dataloader (and specifically the collate fn) works."""
    d = DataLoader(
        GeobenchDataset(
            dataset="m-eurosat",
            geobench_dir=geobench_dir,
            split="train",
            label_fraction=0.01,
        ),
        collate_fn=eval_collate_fn,
        batch_size=1,
        shuffle=False,
    )
    sample, _ = next(iter(d))
    assert isinstance(sample.sentinel2_l2a, torch.Tensor)
    assert sample.sentinel2_l2a.shape == (1, 64, 64, 1, 12)


def test_eurosat_band_names_are_corrected(geobench_dir: Path) -> None:
    """GeoBench mislabels EuroSAT's bands; we must undo it, not inherit it.

    GeoBench names EuroSAT's 13 tif channels from a list with B8A ninth, but the
    tifs put B8A last, so its last five names are rotated by one. Bands 1-8 are
    untouched and the last five are pulled from the channel that actually holds
    them -- in particular the band we emit as B12 must come from GeoBench's
    channel 11, not channel 12 (which is really B8A).
    """
    d = GeobenchDataset(
        dataset="m-eurosat",
        geobench_dir=geobench_dir,
        split="train",
        label_fraction=0.01,
    )
    assert d.band_indices == [0, 1, 2, 3, 4, 5, 6, 7, 12, 8, 9, 10, 11]


def test_band_name_corrections_reject_a_collision() -> None:
    """Two channels claiming the same band is what a config typo looks like."""
    from olmoearth_pretrain.evals.datasets.geobench_dataset import (
        _apply_band_name_corrections,
    )

    names = ["a", "b", "c"]
    # Applied simultaneously, so a swap is a swap and does not cascade.
    assert _apply_band_name_corrections(names, {"a": "b", "b": "a"}) == ["b", "a", "c"]
    # Renaming a channel out of the band set is legitimate: it may not be one.
    assert _apply_band_name_corrections(names, {"a": "not a band"}) == [
        "not a band",
        "b",
        "c",
    ]
    with pytest.raises(ValueError, match="more than one channel"):
        _apply_band_name_corrections(names, {"a": "c"})
    with pytest.raises(ValueError, match="does not have"):
        _apply_band_name_corrections(names, {"z": "a"})


def test_brick_kiln_corrections_and_imputes_cover_every_band() -> None:
    """m-brick-kiln's config must reconstruct all 13 S2 slots without data.

    GeoBench named m-brick-kiln's 13 channels as the MSI band set, but the
    source exported B1-B5, B7, B8A, B8, B11, B12 and three 8-bit true-colour
    renders. So the corrections drop three channels rather than permuting them,
    and B06/B09/B10 have to come from imputes. This checks the two halves of the
    config agree, which is the part a future edit is most likely to break.
    """
    from olmoearth_pretrain.evals.datasets.constants import EVAL_S2_BAND_NAMES
    from olmoearth_pretrain.evals.datasets.geobench_dataset import (
        _apply_band_name_corrections,
        _validate_band_coverage,
    )

    config = dataset_to_config("m-brick-kiln")
    # GeoBench labels the channels with its canonical 13-band list, in order.
    corrected = _apply_band_name_corrections(
        list(EVAL_S2_BAND_NAMES), config.band_name_corrections
    )
    present = [n for n in EVAL_S2_BAND_NAMES if n in corrected]
    assert present == [
        "01 - Coastal aerosol",
        "02 - Blue",
        "03 - Green",
        "04 - Red",
        "05 - Vegetation Red Edge",
        "07 - Vegetation Red Edge",
        "08 - NIR",
        "08A - Vegetation Red Edge",
        "11 - SWIR",
        "12 - SWIR",
    ]
    # Raises if any of the 13 slots is neither present nor imputed.
    _validate_band_coverage(
        present, config.imputes, list(EVAL_S2_BAND_NAMES), "m-brick-kiln"
    )


def test_band_coverage_rejects_an_uncovered_band() -> None:
    """A band that is neither present nor imputed must fail loudly."""
    from olmoearth_pretrain.evals.datasets.geobench_dataset import (
        _validate_band_coverage,
    )

    with pytest.raises(ValueError, match="neither"):
        _validate_band_coverage(["a", "b"], [], ["a", "b", "c"], "fake")
    with pytest.raises(ValueError, match="does not have"):
        _validate_band_coverage(["a", "b"], [("z", "c")], ["a", "b", "c"], "fake")
