"""Distractors (extra negatives) and synonyms (extra positives) of class names."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from olmoearth_pretrain.clipearth.geo_train import CellIndex, Samples, build_batch
from olmoearth_pretrain.clipearth.train import (
    add_synonyms,
    contrastive_loss,
    load_distractors,
)


def test_distractors_penalize_only_the_samples_they_belong_to() -> None:
    """A look-alike distractor raises the loss of its own sample, not of the others."""
    image = F.normalize(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), dim=-1)
    text = F.normalize(torch.tensor([[1.0, 0.1, 0.0], [0.0, 1.0, 0.1]]), dim=-1)
    labels = torch.tensor([0, 1])
    scale = torch.tensor(10.0)
    look_alike = F.normalize(torch.tensor([[1.0, 0.0, 0.1]]), dim=-1)

    base = contrastive_loss(image, text, labels, scale)
    own = contrastive_loss(
        image, text, labels, scale, look_alike, torch.tensor([[True], [False]])
    )
    other = contrastive_loss(
        image, text, labels, scale, look_alike, torch.tensor([[False], [True]])
    )

    assert own > base + 0.1
    assert abs(other - base) < 0.01


def test_geo_batches_add_a_class_distractors_to_its_own_members(tmp_path: Path) -> None:
    """Only members of a class with distractors get them, as extra candidate slots."""
    path = tmp_path / "d.jsonl"
    path.write_text(
        json.dumps({"name": "glacier", "easy": ["town"], "hard": ["snow cover"]}) + "\n"
    )
    samples = Samples()
    groups = [samples.group({"name": [g]}) for g in ["glacier", "glacier", "lake"]]
    samples.distractors = load_distractors(
        str(path), 0.0, samples.group_levels, samples.text_index
    )
    samples.add(
        "p",
        np.eye(3, 8, dtype=np.float32) + 0.1,
        np.array([0.01, 0.02, 0.03]),
        np.zeros(3),
        np.full(3, 2020),
        "pixel",
        np.array(groups),
        1.0,
        True,
    )
    samples.finish()
    extra = {samples.text_index["town"], samples.text_index["snow cover"]}

    batch = build_batch(
        samples,
        CellIndex(samples, 0.05),
        {"cell": np.arange(3)},
        {"cell": np.cumsum(np.ones(3))},
        2,
        3,
        0,
        2,
        np.random.default_rng(0),
    )

    for row, member in enumerate(batch["member"]):
        if member < 0:
            continue
        slots = set(batch["slot_text"][row].tolist())
        assert (extra <= slots) == (samples.group[member] == groups[0])


def test_synonyms_join_their_class_name_level_except_the_held_out_last(
    tmp_path: Path,
) -> None:
    """A class's groups gain its synonyms; the last one is held out; others untouched."""
    path = tmp_path / "s.jsonl"
    path.write_text(
        json.dumps({"name": "Maize", "synonyms": ["corn", "MAIZE", "Zea mays"]}) + "\n"
    )
    text_index = {"Maize": 0, "a satellite image of Maize": 1, "lake": 2}
    group_levels = [{"name": [0, 1], "short": [2]}, {"name": [2]}]

    changed = add_synonyms(str(path), group_levels, text_index, hold_last=True)

    assert changed == 1
    assert [t for t, i in text_index.items() if i in group_levels[0]["name"]] == [
        "Maize",
        "a satellite image of Maize",
        "corn",
    ]
    assert group_levels[1] == {"name": [2]}
    assert "Zea mays" not in text_index


def test_related_classes_are_not_pushed_apart() -> None:
    """Excluding a same-meaning class's text lowers the loss and keeps it finite."""
    image = F.normalize(torch.tensor([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]]), dim=-1)
    text = F.normalize(torch.tensor([[1.0, 0.0, 0.0], [0.95, 0.05, 0.0]]), dim=-1)
    labels = torch.tensor([0, 1])  # "Maize" and "corn" from two datasets
    scale = torch.tensor(20.0)
    related = torch.tensor([[False, True], [True, False]])

    pushed = contrastive_loss(image, text, labels, scale)
    masked = contrastive_loss(image, text, labels, scale, exclude=related)

    assert torch.isfinite(masked)
    assert masked < pushed - 0.1
