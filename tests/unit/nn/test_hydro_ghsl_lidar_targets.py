"""Tests for the hand / hydro / ghsl / lidar_height target heads (card t-v3sdsf)."""

import h5py
import hdf5plugin
import numpy as np
import torch
from upath import UPath

from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality
from olmoearth_pretrain.data.dataset import OlmoEarthDataset
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.supervision_head import (
    SupervisionHead,
    SupervisionModalityConfig,
    SupervisionTaskType,
    compute_supervision_loss,
)

B, P, D, MPS = 2, 4, 8, 8
H = P * MPS  # 32 px at 10 m


def _configs() -> dict[str, SupervisionModalityConfig]:
    cfgs = {
        name: SupervisionModalityConfig(
            task_type=SupervisionTaskType.REGRESSION,
            num_output_channels=Modality.get(name).num_bands,
            regression_loss_type="l1",
        )
        for name in ["hand", "hydro", "ghsl"]
    }
    cfgs["lidar_height"] = SupervisionModalityConfig(
        task_type=SupervisionTaskType.REGRESSION,
        num_output_channels=3,
        regression_loss_type="l1",
        laplacian_loss=True,
    )
    return cfgs


def _batch(lidar: bool, lidar_missing_frac: float = 0.5) -> MaskedOlmoEarthSample:
    kw = {"timestamps": torch.tensor([[1, 1, 2023]]).expand(B, -1, -1)}
    for name in ["hand", "hydro", "ghsl"]:
        c = Modality.get(name).num_bands
        kw[name] = torch.rand(B, H, H, 1, c)
        kw[name + "_mask"] = torch.full((B, H, H, 1, 1), MaskValue.DECODER.value)
    if lidar:
        x = torch.rand(B, 4 * H, 4 * H, 1, 3)
        # sample 0 has lidar on its left part only; sample 1 has none
        cut = int(4 * H * (1 - lidar_missing_frac))
        x[0, :, cut:] = MISSING_VALUE
        x[1] = MISSING_VALUE
        kw["lidar_height"] = x
        kw["lidar_height_mask"] = torch.full(
            (B, 4 * H, 4 * H, 1, 1), MaskValue.DECODER.value
        )
    return MaskedOlmoEarthSample(**kw)


def test_heads_shapes_and_finite_loss() -> None:
    """All four heads predict at their target grid and give finite, nonzero losses."""
    head = SupervisionHead(_configs(), embedding_dim=D, max_patch_size=MPS)
    grid = torch.randn(B, P, P, D, requires_grad=True)
    batch = _batch(lidar=True)
    preds = head(grid, batch)
    assert preds["hand"].shape == (B, H, H, 1, 1)
    assert preds["hydro"].shape == (B, H, H, 1, 2)
    assert preds["ghsl"].shape == (B, H, H, 1, 2)
    assert preds["lidar_height"].shape == (B, 4 * H, 4 * H, 1, 3)
    total, per = compute_supervision_loss(preds, batch, head)
    assert torch.isfinite(total)
    for name in ["hand", "hydro", "ghsl", "lidar_height"]:
        assert per[name] > 0, name
    total.backward()
    assert grid.grad is not None and torch.isfinite(grid.grad).all()


def test_lidar_missing_pixels_do_not_contribute() -> None:
    """Changing predictions only where lidar is missing leaves the lidar loss unchanged."""
    head = SupervisionHead(_configs(), embedding_dim=D, max_patch_size=MPS)
    batch = _batch(lidar=True)
    preds = head(torch.randn(B, P, P, D), batch)
    _, per = compute_supervision_loss(preds, batch, head)
    missing = (batch.lidar_height == MISSING_VALUE).any(-1, keepdim=True)
    preds2 = dict(preds)
    preds2["lidar_height"] = torch.where(
        missing, preds["lidar_height"] + 100.0, preds["lidar_height"]
    )
    _, per2 = compute_supervision_loss(preds2, batch, head)
    assert torch.allclose(per["lidar_height"], per2["lidar_height"])


def test_absent_lidar_gives_zero_loss() -> None:
    """A batch without lidar (most batches' samples) contributes zero lidar loss."""
    head = SupervisionHead(_configs(), embedding_dim=D, max_patch_size=MPS)
    batch = _batch(lidar=False)
    preds = head(torch.randn(B, P, P, D, requires_grad=True), batch)
    total, per = compute_supervision_loss(preds, batch, head)
    assert per["lidar_height"] == 0
    total.backward()


def test_dataset_reads_aux_h5(tmp_path) -> None:
    """read_h5_file adds training modalities from aux_h5_dir/sample_{i}.h5."""
    main_dir = tmp_path / "main" / "1"
    aux_dir = tmp_path / "aux"
    main_dir.mkdir(parents=True)
    aux_dir.mkdir()
    with h5py.File(main_dir / "sample_0.h5", "w") as f:
        f.create_dataset("srtm", data=np.ones((8, 8, 1, 1), np.int32))
        f.create_dataset("timestamps", data=np.zeros((12, 3), np.int32))
    hand = np.full((8, 8, 1, 1), 0.5, np.float32)
    lidar = np.zeros((32, 32, 1, 3), np.float32)
    with h5py.File(aux_dir / "sample_0.h5", "w") as f:
        f.create_dataset("hand", data=hand, **hdf5plugin.Zstd(clevel=3))
        f.create_dataset("lidar_height", data=lidar, **hdf5plugin.Zstd(clevel=3))
        f.create_dataset("ghsl", data=np.zeros((8, 8, 1, 2), np.float32))
    ds = OlmoEarthDataset(
        h5py_dir=UPath(main_dir),
        training_modalities=["srtm", "hand", "lidar_height"],
        dtype=np.float32,
        aux_h5_dir=str(aux_dir),
    )
    sample, _ = ds.read_h5_file(UPath(main_dir / "sample_0.h5"))
    assert set(sample) == {"srtm", "hand", "lidar_height", "timestamps"}
    np.testing.assert_array_equal(sample["hand"], hand)
    # No side file -> modality simply absent.
    with h5py.File(main_dir / "sample_1.h5", "w") as f:
        f.create_dataset("srtm", data=np.ones((8, 8, 1, 1), np.int32))
        f.create_dataset("timestamps", data=np.zeros((12, 3), np.int32))
    sample, _ = ds.read_h5_file(UPath(main_dir / "sample_1.h5"))
    assert "hand" not in sample
