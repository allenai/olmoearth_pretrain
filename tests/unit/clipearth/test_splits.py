"""Held-out classes and held-out locations from a split spec."""

import json
from pathlib import Path

import numpy as np

from olmoearth_pretrain.clipearth.splits import (
    BUFFER,
    HELD_CLASS,
    HELD_LOCATION,
    SplitSpec,
)
from olmoearth_pretrain.clipearth.train import add_synonyms, load_distractors

REFERENCE = Path(
    "/weka/dfive-default/agent/piperbots/2026-11-16-piperw-clean-splits/split_test_points.json"
)


def spec() -> SplitSpec:
    """A small spec with the clean split's location rule."""
    return SplitSpec(
        ["fish pond", "pond", "Airport"],
        region_deg=2.0,
        buffer_deg=0.5,
        held_fraction=0.1,
    )


def test_location_matches_the_reference_points() -> None:
    """Points checked against the split script the spec was made with."""
    got = spec().location(
        np.array([63.222, 25.9898, -23.5788]), np.array([44.8725, 171.8212, -158.3851])
    )
    assert got.tolist() == [HELD_LOCATION, BUFFER, "train"]
    if REFERENCE.exists():
        ref = json.loads(REFERENCE.read_text())
        expected = [{"held": HELD_LOCATION}.get(s, s) for s in ref["split"]]
        assert (
            spec().location(np.array(ref["lat"]), np.array(ref["lon"])).tolist()
            == expected
        )


def test_a_held_class_is_held_wherever_it_is() -> None:
    """Held classes override the location split."""
    s = spec()
    held = s.held_mask(np.array(["Fish Ponds", "fish pond", "lake", "airport"]))
    assert held.tolist() == [False, True, False, True]
    out = s.assign(
        held,
        np.full(4, -23.5788),
        np.full(4, -158.3851),
        np.array(["train", "train", "val", "train"]),
    )
    assert out.tolist() == ["train", HELD_CLASS, "val", HELD_CLASS]


def test_free_text_mentions_match_whole_words_and_plurals() -> None:
    """Captions match held names as whole words, plurals included."""
    s = spec()
    assert s.mentions_held("Two fish ponds by the airports.")
    assert not s.mentions_held("A pondering airportless town")


def test_held_names_are_never_distractors_or_synonyms(tmp_path: Path) -> None:
    """Held names never enter training as distractors or synonyms."""
    text_index = {"lake": 0}
    group_levels = [{"name": [0]}]
    d = tmp_path / "d.jsonl"
    d.write_text(
        json.dumps({"name": "lake", "easy": ["pond", "river"], "hard": []}) + "\n"
    )
    out = load_distractors(str(d), 0.0, group_levels, text_index, spec().held_names)
    assert [t for t, i in text_index.items() if i in out[0]] == ["river"]
    syn = tmp_path / "s.jsonl"
    syn.write_text(json.dumps({"name": "lake", "synonyms": ["pond", "tarn"]}) + "\n")
    add_synonyms(str(syn), group_levels, text_index, False, spec().held_names)
    assert "pond" not in text_index and "tarn" in text_index
