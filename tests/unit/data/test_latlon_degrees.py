"""latlon must reach the model in raw DEGREES through the real dataset path.

The geographic encodings (unit-sphere xyz, multifreq sphere expansions)
assume degrees; min-max normalizing latlon to [0, 1] silently degenerates
them (x ~ 1, y/z ~ 0). This test goes through OlmoEarthDataset.__getitem__
with normalize=True — not hand-built tensors — so the exemption is exercised
where it lives.
"""

from typing import Any

import numpy as np
import pytest
from upath import UPath

from olmoearth_pretrain.data.dataset import GetItemArgs, OlmoEarthDataset

LAT, LON = 47.6, -122.3


@pytest.fixture
def dataset_with_fake_reader(tmp_path: UPath) -> OlmoEarthDataset:
    """A normalize=True dataset whose H5 reader is replaced with fixed data."""
    dataset = OlmoEarthDataset(
        h5py_dir=UPath(tmp_path),
        training_modalities=["sentinel2_l2a", "latlon"],
        dtype=np.float32,
        normalize=True,
    )

    h, w, t = 8, 8, 4
    sample_dict = {
        "sentinel2_l2a": np.random.uniform(0, 4000, (h, w, t, 12)).astype(np.float32),
        "latlon": np.array([LAT, LON], dtype=np.float32),
        "timestamps": np.array([[15, m, 2023] for m in range(t)], dtype=np.int32),
    }

    def fake_read_h5_file(h5_file_path: UPath) -> tuple[dict[str, Any], dict[str, Any]]:
        return {k: v.copy() for k, v in sample_dict.items()}, {}

    dataset.read_h5_file = fake_read_h5_file  # type: ignore[method-assign]
    dataset._get_h5_file_path = lambda index: UPath("/nonexistent.h5")  # type: ignore[method-assign]
    return dataset


def test_latlon_stays_in_degrees(dataset_with_fake_reader: OlmoEarthDataset) -> None:
    """__getitem__ with normalize=True must NOT min-max normalize latlon."""
    args = GetItemArgs(idx=0, patch_size=4, sampled_hw_p=2, token_budget=None)
    _, sample = dataset_with_fake_reader[args]
    assert sample.latlon is not None
    np.testing.assert_allclose(np.asarray(sample.latlon), [LAT, LON], rtol=0, atol=1e-5)


def test_other_modalities_still_normalized(
    dataset_with_fake_reader: OlmoEarthDataset,
) -> None:
    """The exemption is latlon-specific: S2 data is still normalized."""
    args = GetItemArgs(idx=0, patch_size=4, sampled_hw_p=2, token_budget=None)
    _, sample = dataset_with_fake_reader[args]
    s2 = np.asarray(sample.sentinel2_l2a)
    # Raw S2 was uniform in [0, 4000]; normalized output must be small.
    assert np.abs(s2).max() < 100.0
