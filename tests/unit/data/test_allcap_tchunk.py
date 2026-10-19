"""The per-capture (``_tchunk``) allcap layout gives bit-identical samples to the original."""

from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import pytest
from upath import UPath

from olmoearth_pretrain.data.allcap import (
    ALLCAP_SAMPLE_DIR_ENV,
    ALLCAP_SELECTIONS,
    SlotSelection,
    _read_series,
    _TimeSubset,
    read_allcap_sample,
    read_allcap_sample_two_draws,
)
from olmoearth_pretrain.data.dataset import GetItemArgs, OlmoEarthDataset

from .test_allcap import MODS, _write_allcap_h5

TS = ("sentinel2_l2a", "sentinel2_scl", "sentinel1", "landsat_l2")


def _rewrite(src: Path, dst: Path, per_capture: bool) -> None:
    """Copy ``src`` like production: zstd-3, time series in one chunk (original) or one per capture."""
    z = hdf5plugin.Zstd(clevel=3)
    with h5py.File(src, "r") as f, h5py.File(dst, "w") as g:
        for k in f:
            a = f[k][()]
            if k in TS:
                t = 1 if per_capture else a.shape[2]
                g.create_dataset(k, data=a, chunks=(a.shape[0], a.shape[1], t, a.shape[3]), **z)
            else:
                g.create_dataset(k, data=a)


@pytest.fixture
def two_layouts(tmp_path: Path) -> tuple[Path, Path]:
    """The same sample in the original layout (h5py dir) and in a per-capture copy."""
    raw = tmp_path / "raw.h5"
    _write_allcap_h5(raw, np.random.default_rng(3))
    orig, tchunk = tmp_path / "orig" / "1", tmp_path / "tchunk" / "1"
    orig.mkdir(parents=True)
    tchunk.mkdir(parents=True)
    _rewrite(raw, orig / "sample_0.h5", per_capture=False)
    _rewrite(raw, tchunk / "sample_0.h5", per_capture=True)
    import pandas as pd

    pd.DataFrame(
        [{"sample_index": 0, "sentinel2_l2a": 1, "sentinel1": 1, "landsat_l2": 1, "worldcover": 1, "srtm": 1}]
    ).to_csv(orig / "sample_metadata.csv", index=False)
    np.save(orig / "latlon_distribution.npy", np.array([[10.0, 20.0]]))
    return orig, tchunk


def _assert_same(a: tuple[dict, dict], b: tuple[dict, dict]) -> None:
    for x, y in zip(a, b):
        assert set(x) == set(y)
        for k in x:
            xa, ya = np.asarray(x[k]), np.asarray(y[k])
            assert xa.dtype == ya.dtype and xa.shape == ya.shape, k
            np.testing.assert_array_equal(xa, ya, err_msg=k)


@pytest.mark.parametrize("selection", ALLCAP_SELECTIONS)
@pytest.mark.parametrize("mods", [MODS, ["sentinel2_l2a", "worldcover"]])
def test_single_draw_identical(two_layouts: tuple[Path, Path], selection: str, mods: list[str]) -> None:
    """Same RNG state, same selection: identical sample and masks from both layouts."""
    orig, tchunk = two_layouts
    for seed in range(5):
        out = []
        for d in (orig, tchunk):
            with h5py.File(d / "sample_0.h5", "r") as f:
                out.append(read_allcap_sample(f, mods, selection, 0.52, np.random.default_rng(seed)))
        _assert_same(*out)


def test_two_draws_identical(two_layouts: tuple[Path, Path]) -> None:
    """Arm C/C0's two views, read once, are identical across layouts."""
    orig, tchunk = two_layouts
    for seed in range(5):
        out = []
        for d in (orig, tchunk):
            rngs = tuple(np.random.default_rng(seed).spawn(2))
            with h5py.File(d / "sample_0.h5", "r") as f:
                out.append(read_allcap_sample_two_draws(f, MODS, "random_clear", 0.52, rngs))
        for va, vb in zip(*out):
            _assert_same(va, vb)


def test_per_capture_reads_only_used_captures(two_layouts: tuple[Path, Path]) -> None:
    """On the per-capture layout only the selected captures are read; the original reads all."""
    orig, tchunk = two_layouts
    sel = SlotSelection(
        days=np.arange(12) * 30, s2_idx=np.array([3, -1, 7] + [-1] * 9), s1_idx=np.full(12, -1)
    )
    with h5py.File(tchunk / "sample_0.h5", "r") as f:
        s = _read_series(f, MODS, [sel])
    assert isinstance(s["sentinel2_l2a"], _TimeSubset)
    np.testing.assert_array_equal(s["sentinel2_l2a"].kept, [3, 7])
    assert s["sentinel2_l2a"].data.shape[2] == 2
    assert "sentinel1" not in s
    with h5py.File(orig / "sample_0.h5", "r") as f:
        full = _read_series(f, MODS, [sel])["sentinel2_l2a"]
    assert isinstance(full, np.ndarray) and full.shape[2] == 40
    idx = np.array([3, 7])
    np.testing.assert_array_equal(s["sentinel2_l2a"][:, :, idx, :], full[:, :, idx, :])


def test_dataset_reads_sample_dir_with_fallback(
    two_layouts: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """OE_ALLCAP_SAMPLE_DIR: read the copy when the sample is there, else the h5py dir."""
    orig, tchunk = two_layouts
    monkeypatch.setenv(ALLCAP_SAMPLE_DIR_ENV, str(tchunk))
    ds = OlmoEarthDataset(h5py_dir=UPath(orig), training_modalities=MODS, dtype=np.float32,
                          normalize=False, allcap_selection="random_clear", allcap_cloud_tau=0.52)
    ds.prepare()
    assert ds._get_h5_file_path(0) == UPath(tchunk) / "sample_0.h5"
    (tchunk / "sample_0.h5").rename(tchunk / "moved.h5")
    assert ds._get_h5_file_path(0) == UPath(orig) / "sample_0.h5"
    (tchunk / "moved.h5").rename(tchunk / "sample_0.h5")
    # end to end: same getitem output from both layouts under the same global seed
    outs = []
    for env in (str(tchunk), None):
        if env:
            monkeypatch.setenv(ALLCAP_SAMPLE_DIR_ENV, env)
        else:
            monkeypatch.delenv(ALLCAP_SAMPLE_DIR_ENV)
        d = OlmoEarthDataset(h5py_dir=UPath(orig), training_modalities=MODS, dtype=np.float32,
                             normalize=False, allcap_selection="random_clear", allcap_cloud_tau=0.52)
        d.prepare()
        np.random.seed(0)
        outs.append(d[GetItemArgs(idx=0, patch_size=1, sampled_hw_p=4, token_budget=10000)][1])
    for k in ("timestamps", "sentinel2_l2a", "sentinel1", "worldcover", "srtm"):
        np.testing.assert_array_equal(np.asarray(getattr(outs[0], k)), np.asarray(getattr(outs[1], k)), err_msg=k)
