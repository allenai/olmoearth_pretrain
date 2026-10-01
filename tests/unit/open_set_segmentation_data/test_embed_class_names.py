"""Tests for the class-name text embedding script (stub encoder, no model download)."""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import torch

from olmoearth_pretrain.open_set_segmentation_data.embed_class_names import (
    OUTPUT_STEM,
    class_texts,
    embed_class_names,
)
from olmoearth_pretrain.train.open_set_probe import load_text_embeddings


def _mapping() -> dict:
    return {
        "open_set": {
            "num_classes": 3,
            "classes": [
                {"global_id": 1, "slug": "crops", "local_id": 1, "name": "Maize"},
                {"global_id": 0, "slug": "crops", "local_id": 0, "name": "Wheat"},
                {
                    "global_id": 2,
                    "slug": None,
                    "local_id": None,
                    "name": "forest_disturbance",
                    "concept": "forest_disturbance",
                    "members": [
                        {"slug": "wind", "local_id": 0, "name": "wind damage"},
                        {"slug": "beetle", "local_id": 0, "name": "bark beetle"},
                    ],
                },
            ],
            "training_datasets": [],
        },
        "open_set_regression": {"datasets": []},
    }


def _registry() -> dict:
    return {
        "datasets": [
            {"slug": "crops", "name": "Crop Types (Kenya)"},
            {"slug": "wind", "name": "Wind"},
            {"slug": "beetle", "name": "Beetle"},
        ]
    }


def test_class_texts_follow_global_id_order_and_template() -> None:
    """Texts are ordered by global id; merged classes list their member names."""
    texts = class_texts(_mapping(), _registry())
    assert texts == [
        "Wheat; Crop Types (Kenya)",
        "Maize; Crop Types (Kenya)",
        "forest disturbance; bark beetle, wind damage",
    ]


def test_class_texts_reject_non_contiguous_ids() -> None:
    """Rows must line up with global ids 0..num_classes-1."""
    mapping = _mapping()
    mapping["open_set"]["classes"][0]["global_id"] = 5
    with pytest.raises(ValueError, match="global ids"):
        class_texts(mapping, _registry())


def test_embed_writes_normalized_npy_and_sidecar(tmp_path: Path) -> None:
    """The .npy is float16 + unit-norm and the sidecar pins the mapping hash."""
    mapping_path = tmp_path / "class_mapping.json"
    mapping_path.write_text(json.dumps(_mapping()))
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(_registry()))

    seen: list[str] = []

    def stub_encoder(texts: Sequence[str]) -> np.ndarray:
        seen.extend(texts)
        rng = np.random.default_rng(0)
        return rng.normal(size=(len(texts), 6)) * 3.0

    npy_path, sidecar_path = embed_class_names(
        mapping_path, registry_path, tmp_path / "out", stub_encoder, "stub-model"
    )
    assert len(seen) == 3
    assert npy_path.name == f"{OUTPUT_STEM}.npy"
    array = np.load(npy_path)
    assert array.dtype == np.float16 and array.shape == (3, 6)
    np.testing.assert_allclose(
        np.linalg.norm(array.astype(np.float32), axis=1), 1.0, atol=1e-2
    )
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["model_name"] == "stub-model"
    assert (
        sidecar["class_mapping_sha256"]
        == hashlib.sha256(mapping_path.read_bytes()).hexdigest()
    )
    assert sidecar["num_classes"] == 3 and sidecar["dim"] == 6

    # Round trip through the probe-side loader.
    loaded = load_text_embeddings(npy_path, sidecar["class_mapping_sha256"], 3)
    assert loaded.shape == (3, 6) and loaded.dtype == torch.float32
    with pytest.raises(ValueError, match="expected \\(4, dim\\)"):
        load_text_embeddings(npy_path, sidecar["class_mapping_sha256"], 4)
