"""OSM regions as a clipearth.train source."""

import json
from pathlib import Path

import numpy as np

from olmoearth_pretrain.clipearth.train import load_osm_regions


def test_load_osm_regions_joins_texts_and_scales_by_norm(tmp_path: Path) -> None:
    """Regions without a text line are dropped; embeddings are unit vector x length."""
    emb = np.eye(3, 128, dtype=np.float32)
    np.savez(
        tmp_path / "r.npz",
        id=np.array([5, 7, 9]),
        embedding=emb,
        embedding_norm=np.array([0.5, 0.9, 1.0], np.float32),
        split=np.array(["train", "osm_val", "train"]),
    )
    with open(tmp_path / "t.jsonl", "w") as f:
        for i, cat in ((5, "landfill"), (9, "cemetery")):
            levels = {"name": [cat], "short": [], "detailed": [f"A {cat}."]}
            f.write(json.dumps({"id": i, "category": cat, "levels": levels}) + "\n")
    out = load_osm_regions(str(tmp_path / "r.npz"), str(tmp_path / "t.jsonl"))
    assert out["category"].tolist() == ["landfill", "cemetery"]
    assert out["split"].tolist() == ["train", "train"]
    np.testing.assert_allclose(out["embedding"][0], emb[0] * 0.5)
    assert out["levels"][1]["detailed"] == ["A cemetery."]
