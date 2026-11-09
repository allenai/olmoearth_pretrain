"""Tests for per-pixel text labels."""

from pathlib import Path

import h5py
import numpy as np
from upath import UPath

from olmoearth_pretrain.data.constants import MISSING_VALUE
from olmoearth_pretrain.data.dataset import GetItemArgs, OlmoEarthDataset
from olmoearth_pretrain.data.text_labels import (
    CEREAL,
    CROP,
    LAND_COVER,
    OSM_CHANNEL,
    OSM_NAMES,
    LabelVocabulary,
    layer_label_names,
)
from olmoearth_pretrain.dataset.convert_to_h5py import ConvertToH5py


def test_vocabulary_drops_generic_and_absence_names() -> None:
    """Generic, absence and repeated names get no id; underscores read as spaces."""
    vocab = LabelVocabulary(
        ["Background", "non-water", "not forest", "Port_areas", "port areas", "water"]
    )
    assert vocab.names == ["Port areas", "water"]
    assert vocab.id("port-areas") == 0
    assert vocab.id("other") is None


def test_layer_raster_picks_one_label_per_channel() -> None:
    """Each channel gets its layer's label; specific crops and rare OSM objects win."""
    vocab = LabelVocabulary(layer_label_names())
    worldcover = np.full((2, 2, 1, 1), 10, dtype=np.uint8)
    worldcover[1, 1] = 0
    cdl = np.full((2, 2, 1, 1), 24, dtype=np.uint8)
    cereal = np.zeros((2, 2, 1, 8), dtype=np.float32)
    cereal[0, 0, 0, [0, 2]] = 90  # temporary crops and maize
    osm = np.zeros((2, 2, 1, 30), dtype=np.uint8)
    osm[0, 0, 0, [OSM_NAMES.index("road"), OSM_NAMES.index("wind turbine")]] = 1
    raster = vocab.layer_raster(
        {
            "worldcover": worldcover,
            "cdl": cdl,
            "worldcereal": cereal,
            "openstreetmap_raster": osm,
        }
    )
    assert raster is not None
    assert raster.shape == (2, 2, 1, 4)
    assert raster[0, 0, 0, LAND_COVER] == vocab.id("tree cover")
    assert raster[1, 1, 0, LAND_COVER] == MISSING_VALUE
    assert raster[0, 0, 0, CROP] == vocab.id("winter wheat")
    assert raster[0, 0, 0, CEREAL] == vocab.id("maize")
    assert raster[0, 1, 0, CEREAL] == MISSING_VALUE
    assert raster[0, 0, 0, OSM_CHANNEL] == vocab.id("wind turbine")
    assert vocab.layer_raster({"cdl": np.zeros((2, 2, 1, 1), np.uint8)}) is None


def _write(path: Path, label: np.ndarray | None) -> None:
    with h5py.File(path, "w") as f:
        f["sentinel2_l2a"] = np.ones((8, 8, 2, 12), dtype=np.uint16)
        f["worldcover"] = np.full((8, 8, 1, 1), 80, dtype=np.uint8)
        f["timestamps"] = np.array([[1, 0, 2020], [1, 1, 2020]])
        if label is not None:
            f["text_label"] = label


def test_dataset_derives_or_reads_text_label(tmp_path: Path) -> None:
    """Pretraining windows get labels from their layers; stored labels are kept."""
    vocab_dir = tmp_path / "vocab"
    vocab = LabelVocabulary([*layer_label_names(), "solar farm"])
    vocab.save(str(vocab_dir))
    h5py_dir = tmp_path / "h5"
    h5py_dir.mkdir()
    stored = np.full((8, 8, 1, 4), MISSING_VALUE, dtype=np.float32)
    stored[..., 0, LAND_COVER] = vocab.id("solar farm")
    _write(h5py_dir / ConvertToH5py.sample_file_pattern.format(index=0), None)
    _write(h5py_dir / ConvertToH5py.sample_file_pattern.format(index=1), stored)
    dataset = OlmoEarthDataset(
        h5py_dir=UPath(h5py_dir),
        training_modalities=["sentinel2_l2a", "worldcover", "text_label"],
        dtype=np.float32,
        normalize=False,
        label_vocab_dir=str(vocab_dir),
    )
    args = GetItemArgs(idx=0, patch_size=1, sampled_hw_p=8, token_budget=10_000)
    _, derived = dataset[args]
    _, read = dataset[args._replace(idx=1)]
    assert derived.text_label is not None and read.text_label is not None
    assert (derived.text_label[..., LAND_COVER] == vocab.id("open water")).all()
    assert (derived.text_label[..., CROP] == MISSING_VALUE).all()
    assert (read.text_label[..., LAND_COVER] == vocab.id("solar farm")).all()
