"""The location-conditioned text head of clipearth.geo_train."""

import numpy as np
import torch
import torch.nn.functional as F

from olmoearth_pretrain.clipearth.geo_train import (
    SCALES,
    YEARS,
    CellIndex,
    GeoTextHead,
    Samples,
    batch_loss,
    build_batch,
    condition,
    condition_dim,
    make_frequencies,
)
from olmoearth_pretrain.clipearth.train import Aligner


def test_condition_layout_unknowns_and_smoothness() -> None:
    """Unknown location and year have their own flags; features are smooth everywhere."""
    freqs = make_frequencies(per_scale=4)
    lat = np.array([10.0, np.nan, 89.999, 0.0, 0.0])
    lon = np.array([20.0, 20.0, 0.0, 179.999, -179.999])
    c = condition(
        lat,
        lon,
        np.array([2017, 2025, 0, 2030, 2020]),
        np.array([0, 1, 0, 0, 1]),
        freqs,
    )
    assert c.shape == (5, condition_dim(freqs))
    n = 2 * len(freqs)
    assert not c[1, :n].any() and c[1, n] == 1 and c[0, n] == 0
    years = c[:, n + 1 : n + 2 + len(YEARS)]
    assert years.argmax(1).tolist() == [0, 8, 9, 9, 3]
    np.testing.assert_array_equal(c[:, -len(SCALES) :].argmax(1), [0, 1, 0, 0, 1])
    # Across the antimeridian 0.002 degrees (~220 m) apart: within a
    # 12 km wavelength, a phase change of ~0.1.
    assert np.abs(c[3, :n] - c[4, :n]).max() < 0.15
    pole = condition(
        np.array([89.999]), np.array([120.0]), np.zeros(1), np.zeros(1), freqs
    )
    assert np.abs(pole[0, :n] - c[2, :n]).max() < 0.15


def test_head_starts_as_the_native_aligner_text_head() -> None:
    """With the FiLM output at zero, the head is the initializing aligner's text head."""
    aligner = Aligner(128, 32, 512, 64, native=True)
    model = GeoTextHead(32, 64, 128, make_frequencies(per_scale=2))
    model.load_state_dict(
        {k: v for k, v in aligner.state_dict().items() if k.startswith("text.")},
        strict=False,
    )
    t = F.normalize(torch.randn(6, 32), dim=-1)
    freqs = model.frequencies.numpy()
    cond = torch.from_numpy(
        condition(
            np.full(6, 45.0),
            np.linspace(-170, 170, 6),
            np.full(6, 2020),
            np.zeros(6),
            freqs,
        )
    )
    torch.testing.assert_close(model(t, cond), aligner.embed_text(t))
    torch.testing.assert_close(model(t, cond, 64), aligner.embed_text(t, 64))
    torch.nn.init.normal_(model.film[2].weight)
    moved = model(t, cond)
    assert not torch.allclose(moved, aligner.embed_text(t), atol=1e-3)


def _samples(lat: list[float], groups: list[str]) -> Samples:
    samples = Samples()
    ids = np.array([samples.group({"name": [g]}) for g in groups])
    n = len(lat)
    samples.add(
        "p",
        np.eye(n, 8, dtype=np.float32) + 0.1,
        np.array(lat),
        np.zeros(n),
        np.full(n, 2020),
        "pixel",
        ids,
        1.0,
        True,
    )
    samples.finish()
    return samples


def test_cells_and_batch_slots() -> None:
    """Cell-mates come from other groups in the cell; slots list the cell's texts first."""
    #            cell 0 (lat 0.01-0.03)           cell 2 (lat ~0.11)
    samples = _samples(
        [0.01, 0.02, 0.02, 0.03, 0.11, 0.12], ["a", "a", "b", "c", "d", "a"]
    )
    cells = CellIndex(samples, 0.05)
    rng = np.random.default_rng(0)
    for _ in range(10):
        mates = cells.mates(0, 4, rng)
        assert mates[0] == 0 and sorted(samples.group[mates[1:]]) == [1, 2]
    pools = {"cell": np.arange(6)}
    cdfs = {"cell": np.cumsum(np.ones(6))}
    batch = build_batch(samples, cells, pools, cdfs, 3, 4, 0, 5, rng)
    k = 4
    slot, member, target = batch["slot_text"], batch["member"], batch["target"]
    for row in range(len(member)):
        if member[row] < 0:
            assert target[row] == -1 and (slot[row] < 0).all()
            continue
        own = samples.candidates[samples.group[member[row]]]["name"][0]
        assert slot[row, target[row]] == own
        usable = slot[row][slot[row] >= 0]
        assert len(usable) == len(set(usable))
        cell = member[row - row % k : row - row % k + k]
        cell_texts = [
            samples.candidates[samples.group[m]]["name"][0] for m in cell if m >= 0
        ]
        assert set(slot[row, :k][slot[row, :k] >= 0]) == set(cell_texts)
    np.testing.assert_array_equal(
        batch["unique"][batch["text_pos"]][slot >= 0], slot[slot >= 0]
    )


def test_batch_loss_by_hand() -> None:
    """One cell of three members (texts 7, 8, 7) and a padding row, plus random slots."""
    slot_text = torch.tensor(
        [[7, 8, -1, -1, 9], [7, 8, -1, -1, 5], [7, 8, -1, -1, -1], [-1] * 5]
    )
    target = torch.tensor([0, 1, 0, -1])
    logits = torch.randn(4, 5)
    i2t, t2i = batch_loss(logits, slot_text, target, 4, 1)
    m = logits.masked_fill(slot_text < 0, float("-inf"))
    expected_i2t = F.cross_entropy(m[:3], target[:3])
    torch.testing.assert_close(i2t, expected_i2t)
    # Text 7 (column 0): members 0 and 2 positive among 0-2; text 8: member 1.
    col0 = m[:3, 0].log_softmax(0)
    col1 = m[:3, 1].log_softmax(0)
    expected_t2i = (-(col0[0] + col0[2]) / 2 - col1[1]) / 2
    torch.testing.assert_close(t2i, expected_t2i)
    single = torch.tensor([[7, -1, -1, -1, 9], [-1] * 5, [-1] * 5, [-1] * 5])
    assert batch_loss(logits, single, torch.tensor([0, -1, -1, -1]), 4, 1)[1] is None
