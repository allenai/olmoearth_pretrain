"""Pixel samples and the context head of clipearth.train."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from olmoearth_pretrain.clipearth.train import Aligner, load_pixel_samples


def test_load_pixel_samples_joins_texts_and_scales_by_norm(tmp_path: Path) -> None:
    """Joined on text_id, generic labels dropped; pixel and context are unit x length."""
    pixel = np.eye(3, 128, dtype=np.float32)
    context = np.eye(3, 128, k=5, dtype=np.float32)
    np.savez(
        tmp_path / "p.npz",
        id=np.array([1, 2, 3]),
        pixel=pixel,
        pixel_norm=np.array([0.9, 0.8, 0.7], np.float32),
        context=context,
        context_norm=np.array([0.5, 0.4, 0.3], np.float32),
        text_id=np.array([10, 11, 12]),
        split=np.array([1, 0, 0], np.int32),
        split_names=np.array(["test", "train"]),
    )
    with open(tmp_path / "p.jsonl", "w") as f:
        for i, name in ((10, "shrimp ponds"), (11, "background"), (12, "cemetery")):
            levels = {"name": [name], "short": [], "detailed": []}
            record = {"text_id": i, "category": name, "levels": levels}
            f.write(json.dumps(record) + "\n")
    out = load_pixel_samples(str(tmp_path / "p.npz"), str(tmp_path / "p.jsonl"))
    assert out["id"].tolist() == [1, 3]
    assert out["split"].tolist() == ["train", "test"]
    np.testing.assert_allclose(out["embedding"][1], pixel[2] * 0.7)
    np.testing.assert_allclose(out["context"][0], context[0] * 0.5)
    assert out["text_id"].tolist() == [10, 12]
    assert out["texts"][12]["name"] == ["cemetery"]


def test_context_head_shapes_and_zero_start() -> None:
    """One head per width; at initialization the context changes nothing."""
    model = Aligner(128, 32, 512, 64, native=True, context_widths=[64, 128])
    assert sorted(model.context) == ["128", "64"]
    x = F.normalize(torch.randn(5, 128), dim=-1)
    c = F.normalize(torch.randn(5, 128), dim=-1)
    for dims, width in ((None, 128), (64, 64)):
        out = model.embed_image(x[:, :width], dims, c[:, :width])
        assert out.shape == (5, width)
        torch.testing.assert_close(out, model.embed_image(x[:, :width], dims))
    torch.nn.init.normal_(model.context["64"].weight)
    moved = model.embed_image(x[:, :64], 64, c[:, :64])
    assert not torch.allclose(moved, model.embed_image(x[:, :64], 64))
    torch.testing.assert_close(moved.norm(dim=-1), torch.ones(5))
