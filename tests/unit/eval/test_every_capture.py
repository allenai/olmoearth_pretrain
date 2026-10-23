"""Every-capture eval inputs (``evals/datasets/every_capture.py``), pure parts."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import (
    MaskedOlmoEarthSample,
    MaskValue,
    OlmoEarthSample,
)
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


def test_every_capture_sample_puts_landsat_l2_captures_on_the_day_timeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """every_capture + twin_landsat_l2: each L2 capture gets its own day step."""
    from olmoearth_pretrain.data.normalize import Normalizer, Strategy
    from olmoearth_pretrain.evals.datasets import rslearn_dataset

    s2_days = [datetime(2019, 1, 5), datetime(2019, 3, 1)]
    l2_days = [datetime(2019, 1, 5), datetime(2019, 2, 10)]
    n_s2 = len(Modality.SENTINEL2_L2A.band_order)
    n_l2 = len(Modality.LANDSAT_L2.band_order)
    s2 = np.full((2, 2, 2, n_s2), 1000.0, np.float32)
    l2 = np.full((2, 2, 2, n_l2), 9000.0, np.float32)
    l2_valid = np.ones((2, 2, 2), dtype=bool)
    l2_valid[0, 0, 1] = False  # one missing pixel on 2019-02-10
    monkeypatch.setattr(
        rslearn_dataset,
        "read_every_capture",
        lambda *a: {
            Modality.SENTINEL2_L2A.name: (s2, np.ones((2, 2, 2), bool), s2_days)
        },
    )
    monkeypatch.setattr(
        rslearn_dataset, "read_landsat_l2", lambda *a: (l2, l2_valid, l2_days)
    )
    ds = rslearn_dataset.RslearnToOlmoEarthDataset.__new__(
        rslearn_dataset.RslearnToOlmoEarthDataset
    )
    ds.every_capture_reader = object()  # type: ignore[assignment]
    ds.twin_landsat_l2_reader = object()  # type: ignore[assignment]
    ds._tiles_per_side = 1
    ds._every_capture_cache = None
    ds.input_modalities = [Modality.SENTINEL2_L2A.name, Modality.LANDSAT_L2.name]
    ds.norm_stats_from_pretrained = True
    ds.normalizer_computed = Normalizer(Strategy.COMPUTED)

    sample, _ = ds._every_capture_sample("w", None, None, torch.zeros(2, 2))

    # Union of days: 01-05 (both), 02-10 (L2 only), 03-01 (S2 only).
    assert sample.timestamps.tolist() == [[5, 0, 2019], [10, 1, 2019], [1, 2, 2019]]
    missing = MaskValue.MISSING.value
    assert sample.landsat_l2_mask is not None
    assert sample.sentinel2_l2a_mask is not None
    l2_mask = sample.landsat_l2_mask[..., 0]
    assert (l2_mask[:, :, 0] != missing).all()
    assert l2_mask[0, 0, 1] == missing and (l2_mask[1, 1, 1] != missing)
    assert (l2_mask[:, :, 2] == missing).all()
    s2_mask = sample.sentinel2_l2a_mask[..., 0]
    assert (s2_mask[:, :, 1] == missing).all()


def test_real_capture_dates_move_monthly_steps_to_their_scenes(tmp_path: Path) -> None:
    """Each modality's monthly timestep lands on its own scene's acquisition day."""
    from olmoearth_pretrain.evals.datasets import rslearn_dataset

    window = tmp_path / "windows" / "g" / "w"
    window.mkdir(parents=True)

    def layer(name: str, times: list[str]) -> dict:
        groups = [[{"geometry": {"time_range": [t, t]}}] for t in times]
        return {"layer_name": name, "serialized_item_groups": groups}

    (window / "items.json").write_text(
        json.dumps(
            [
                layer("sentinel2_l2a_mo01", ["2019-01-18T10:44:31+00:00"]),
                layer("sentinel2_l2a_mo02", ["2019-02-20T10:44:31+00:00"]),
                layer("sentinel1_mo01", ["2019-01-28T05:00:00+00:00"]),
                layer("sentinel1_mo02", []),  # no S1 scene in period 2
            ]
        )
    )
    ds = rslearn_dataset.RslearnToOlmoEarthDataset.__new__(
        rslearn_dataset.RslearnToOlmoEarthDataset
    )
    ds.input_modalities = [Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name]
    ds.real_capture_dates = rslearn_dataset.CaptureDateSource(
        root=str(tmp_path),
        layers={
            Modality.SENTINEL2_L2A.name: ["sentinel2_l2a_mo01", "sentinel2_l2a_mo02"],
            Modality.SENTINEL1.name: ["sentinel1_mo01", "sentinel1_mo02"],
        },
    )
    ds._capture_dates_cache = None
    n_s2, n_s1 = (len(Modality.get(m).band_order) for m in ds.input_modalities)
    s2 = torch.arange(2, dtype=torch.float32).view(1, 1, 2, 1).expand(2, 2, 2, n_s2)
    s1 = torch.full((2, 2, 2, n_s1), 7.0)
    sample = MaskedOlmoEarthSample.from_olmoearthsample(
        OlmoEarthSample(
            sentinel2_l2a=s2.clone(),
            sentinel1=s1,
            timestamps=torch.tensor([[1, 0, 2019], [31, 0, 2019]]),
        )
    )
    out = ds._retime_to_capture_dates(sample, "g/w", [])

    assert out.timestamps is not None
    # Days: S2 01-18, S1 01-28, S2 02-20 (S1 period 2 keeps its start, 01-31).
    assert out.timestamps.tolist() == [
        [18, 0, 2019],
        [28, 0, 2019],
        [31, 0, 2019],
        [20, 1, 2019],
    ]
    missing = MaskValue.MISSING.value
    assert out.sentinel2_l2a_mask is not None and out.sentinel1_mask is not None
    s2_present = (out.sentinel2_l2a_mask[0, 0, :, 0] != missing).tolist()
    assert s2_present == [True, False, False, True]
    assert out.sentinel2_l2a is not None
    assert out.sentinel2_l2a[0, 0, 3, 0] == 1.0  # period 2's values moved with it
    s1_present = (out.sentinel1_mask[0, 0, :, 0] != missing).tolist()
    assert s1_present == [False, True, True, False]
