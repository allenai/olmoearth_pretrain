"""Every-capture eval inputs (``evals/datasets/every_capture.py``), pure parts."""

from datetime import datetime

import numpy as np

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.datasets.every_capture import (
    S2_ABSENT_BANDS,
    _merge_same_day,
    every_capture_source,
    union_timeline_sample,
)


def _valid(a: np.ndarray) -> np.ndarray:
    return (a > 0).any(axis=-1)


def test_same_day_scenes_are_mosaicked_and_empty_days_dropped() -> None:
    """Two tiles on one day fill each other's gaps; a mostly-empty day is dropped."""
    a = np.zeros((2, 2, 1), np.float32)
    a[0] = 1.0  # top row
    b = np.zeros((2, 2, 1), np.float32)
    b[:, :] = 2.0  # full, but later tile
    sparse = np.zeros((2, 2, 1), np.float32)
    sparse[0, 0] = 3.0  # 25% valid
    t0, t1 = datetime(2020, 5, 1, 10), datetime(2020, 5, 1, 11)
    out, valid, days = _merge_same_day(
        [b, a, sparse], [t1, t0, datetime(2020, 6, 1)], _valid, 0.5
    )
    assert days == [datetime(2020, 5, 1)]
    # The earlier scene (a) wins where valid; b fills the bottom row.
    np.testing.assert_array_equal(out[0][..., 0], [[1.0, 1.0], [2.0, 2.0]])
    assert valid[0].all()


def test_union_timeline_places_each_modality_on_its_own_days() -> None:
    """S2 and S1 share a step on the same day; elsewhere the other is invalid."""
    s2 = np.ones((1, 1, 2, len(Modality.SENTINEL2_L2A.band_order)), np.float32)
    s1 = np.full((1, 1, 2, 2), 5.0, np.float32)
    d = [datetime(2020, 1, 1), datetime(2020, 1, 5), datetime(2020, 1, 9)]
    series = {
        Modality.SENTINEL2_L2A.name: (s2, np.ones((1, 1, 2), bool), [d[0], d[1]]),
        Modality.SENTINEL1.name: (s1, np.ones((1, 1, 2), bool), [d[1], d[2]]),
    }
    values, valid, days = union_timeline_sample(
        series, [Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name], 1, 1
    )
    assert days == d
    assert valid[Modality.SENTINEL2_L2A.name][0, 0].tolist() == [True, True, False]
    assert valid[Modality.SENTINEL1.name][0, 0].tolist() == [False, True, True]
    assert values[Modality.SENTINEL1.name][0, 0, 2, 0] == 5.0


def test_absent_modality_and_mapping() -> None:
    """A modality with no captures is all-invalid; B01/B09 are the absent S2 bands."""
    values, valid, days = union_timeline_sample({}, [Modality.SENTINEL1.name], 2, 2)
    assert len(days) == 1 and not valid[Modality.SENTINEL1.name].any()
    order = Modality.SENTINEL2_L2A.band_order
    assert sorted(order[i] for i in S2_ABSENT_BANDS) == ["B01", "B09"]
    src = every_capture_source("africa_crop_mask_year_aligned", "/stage/africa")
    assert (src.root, src.group) == ("/stage/africa", "africa_crop_mask_tessera_v2")
    assert (
        every_capture_source("pastis_year_aligned", None).group == "pastis_tessera_v2"
    )
