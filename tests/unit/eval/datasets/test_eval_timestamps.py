"""Eval datasets with fabricated dates must carry the year=0 "year unknown" sentinel.

get_simple_temporal_encoding treats year == 0 as the in-band "year unknown"
sentinel (year_valid channel = 0). Tasks with fabricated dates should present
that trained state instead of a confident wrong year, while keeping their
month/day for the annual-phase channels.
"""

import torch
from einops import repeat

from olmoearth_pretrain.evals.datasets.floods_dataset import Sen1Floods11Dataset
from olmoearth_pretrain.evals.datasets.geobench_dataset import GeobenchDataset
from olmoearth_pretrain.evals.datasets.mados_dataset import MADOSDataset
from olmoearth_pretrain.evals.datasets.rslearn_dataset import get_timestamps
from olmoearth_pretrain.nn.encodings import get_simple_temporal_encoding


def _assert_fabricated_default(day_month_year: list[int]) -> None:
    """Fabricated (day, month0, year) defaults keep day/month but zero the year."""
    assert len(day_month_year) == 3
    day, month, year = day_month_year
    assert day == 1  # day-of-month untouched
    assert month == 6  # month stays 0-indexed (6 == July) and untouched
    assert year == 0  # the "year unknown" sentinel


def test_geobench_default_timestamp_year_unknown() -> None:
    """GeoBench fabricates dates; its default year must be the unknown sentinel."""
    _assert_fabricated_default(GeobenchDataset.default_day_month_year)


def test_mados_default_timestamp_year_unknown() -> None:
    """MADOS fabricates dates; its default year must be the unknown sentinel."""
    _assert_fabricated_default(MADOSDataset.default_day_month_year)


def test_floods_default_timestamp_year_unknown() -> None:
    """Sen1Floods11 fabricates dates; its default year must be the unknown sentinel."""
    _assert_fabricated_default(Sen1Floods11Dataset.default_day_month_year)


def test_fabricated_timestamp_hits_year_unknown_encoding() -> None:
    """The fabricated triple encodes to year_valid=0 with a live annual phase."""
    # Mirror how the datasets build their timestamp tensor in __getitem__.
    timestamp = repeat(
        torch.tensor(GeobenchDataset.default_day_month_year), "d -> t d", t=1
    ).long()
    enc = get_simple_temporal_encoding(timestamp)
    assert enc.shape == (1, 4)
    assert enc[0, 0] == 0.0  # scaled-year channel zeroed
    assert enc[0, 3] == 0.0  # year_valid indicator says "unknown"
    # The annual-phase channels still reflect the (real-ish) month/day.
    assert enc[0, 1] != 0.0 or enc[0, 2] != 0.0


def test_rslearn_synthetic_timestamps_year_unknown() -> None:
    """The default monthly ramp is synthetic: year column must be 0."""
    timestamps = torch.stack(
        get_timestamps("2022-09-01", "2023-09-01", num_timesteps=12)
    )
    assert timestamps.shape == (12, 3)
    assert (timestamps[:, 0] == 1).all()  # ramp starts each month on day 1
    # Months stay 0-indexed: Sep (8) ... Dec (11), then Jan (0) ... Aug (7).
    expected_months = torch.tensor([8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7])
    assert (timestamps[:, 1] == expected_months).all()
    assert (timestamps[:, 2] == 0).all()  # synthetic dates -> year unknown


def test_rslearn_real_dates_keep_year() -> None:
    """dates_are_real=True preserves the absolute years of the window."""
    timestamps = torch.stack(
        get_timestamps(
            "2022-09-01", "2023-09-01", num_timesteps=12, dates_are_real=True
        )
    )
    expected_years = torch.tensor([2022] * 4 + [2023] * 8)
    assert (timestamps[:, 2] == expected_years).all()
    # day/month identical to the synthetic variant.
    synthetic = torch.stack(get_timestamps("2022-09-01", "2023-09-01", 12))
    assert (timestamps[:, :2] == synthetic[:, :2]).all()
