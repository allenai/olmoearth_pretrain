"""Unit tests for LCC change-prediction point sampling."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
import rasterio.transform
from upath import UPath

from olmoearth_pretrain.dataset_creation.create_windows.from_open_set import (
    iter_sparse_samples,
)
from olmoearth_pretrain.open_set_segmentation_data.io import write_points_table
from olmoearth_pretrain.open_set_segmentation_data.sample_lcc_change import (
    NUM_SUMMARY_BANDS,
    POST_CLASS_BAND,
    SLUG,
    TS_POST_MONTH_BAND,
    TS_PRE_MONTH_BAND,
    count_change_pixels_per_cell,
    month_value_to_date,
    process_tile,
)

CELL = 32


def test_month_value_to_date() -> None:
    """Month values decode to the first day of 1 + months since Jan 2015."""
    assert month_value_to_date(1) == datetime(2015, 1, 1, tzinfo=UTC)
    assert month_value_to_date(13) == datetime(2016, 1, 1, tzinfo=UTC)
    assert month_value_to_date(50) == datetime(2019, 2, 1, tzinfo=UTC)


def test_count_change_pixels_per_cell() -> None:
    """Per-cell counts only include real-category pixels within full cells."""
    post_class = np.ones((70, 70), dtype=np.uint8)  # all "none"
    post_class[0:10, 0:10] = 3  # 100 change pixels in cell (0, 0)
    post_class[40:45, 40:45] = 2  # 25 change pixels in cell (1, 1)
    post_class[65:70, :] = 4  # beyond the last full 32px cell row: ignored
    counts = count_change_pixels_per_cell(post_class, CELL)
    assert counts.shape == (2, 2)
    assert counts[0, 0] == 100 and counts[1, 1] == 25
    assert counts[0, 1] == 0 and counts[1, 0] == 0


def _write_summary_tif(path: Path) -> None:
    """Write a synthetic 64x64 summary raster with 2x2 32px cells.

    Cell (0,0): 150 change pixels with months set -> accepted.
    Cell (0,1): 20 change pixels -> rejected (below threshold).
    Cell (1,0): no prediction (all 0) -> rejected.
    Cell (1,1): 150 change pixels but month bands 0 -> accepted then skipped.
    """
    arr = np.zeros((NUM_SUMMARY_BANDS, 64, 64), dtype=np.uint8)
    post_class = arr[POST_CLASS_BAND - 1]
    pre_months = arr[TS_PRE_MONTH_BAND - 1]
    post_months = arr[TS_POST_MONTH_BAND - 1]

    post_class[0:32, 0:32] = 1  # "none"
    post_class[0:15, 0:10] = 3  # 150 pixels of new_building
    pre_months[0:15, 0:10] = 50  # change starts 2019-02
    post_months[0:15, 0:10] = 55  # change ends 2019-07

    post_class[0:32, 32:64] = 1
    post_class[0:4, 32:37] = 3  # only 20 change pixels

    post_class[32:64, 32:64] = 5  # change but no usable month predictions

    transform = rasterio.transform.from_origin(500000, 200000, 10, 10)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=64,
        height=64,
        count=NUM_SUMMARY_BANDS,
        dtype="uint8",
        crs="EPSG:32636",
        transform=transform,
    ) as dst:
        dst.write(arr)


def test_process_tile(tmp_path: Path) -> None:
    """Accepted cells become points with month-derived pre/post time ranges."""
    tif_path = tmp_path / "EPSG:32636_500000_-200000_summary.tif"
    _write_summary_tif(tif_path)

    points, num_accepted = process_tile(
        str(tif_path),
        min_change_pixels=100,
        max_per_tile=10,
        seed=0,
        cell_size=CELL,
    )

    # Cells (0,0) and (1,1) pass the threshold; (1,1) is then skipped because
    # it has no usable month predictions.
    assert num_accepted == 2
    assert len(points) == 1
    pt = points[0]

    # Absolute pixel coords of cell (0,0): origin (500000m, 200000m) at 10m.
    assert pt["id"] == "32636_50000_-20000"
    assert pt["label"] is None and pt["time_range"] is None
    assert pt["num_change_pixels"] == 150
    assert pt["post_category"] == "new_building"
    # UTM 36N around 1.8N latitude.
    assert 30 < pt["lon"] < 36 and 0 < pt["lat"] < 4

    # Pre range ends when the change starts; post range starts the month after
    # the change completes; both span 180 days.
    pre_end = datetime(2019, 2, 1, tzinfo=UTC)
    post_start = datetime(2019, 8, 1, tzinfo=UTC)
    assert pt["pre_time_range"] == (pre_end - timedelta(days=180), pre_end)
    assert pt["post_time_range"] == (post_start, post_start + timedelta(days=180))


def test_max_per_tile_applies_per_category(tmp_path: Path) -> None:
    """The per-tile cap limits each dominant category independently."""
    arr = np.zeros((NUM_SUMMARY_BANDS, 64, 64), dtype=np.uint8)
    post_class = arr[POST_CLASS_BAND - 1]
    pre_months = arr[TS_PRE_MONTH_BAND - 1]
    post_months = arr[TS_POST_MONTH_BAND - 1]

    post_class[:, :] = 1  # "none" everywhere
    # Two cells dominated by new_building (3) and one by new_infrastructure (5).
    post_class[0:15, 0:10] = 3  # cell (0, 0)
    post_class[0:15, 32:42] = 3  # cell (0, 1)
    post_class[32:47, 0:10] = 5  # cell (1, 0)
    change = post_class > 1
    pre_months[change] = 50
    post_months[change] = 55

    tif_path = tmp_path / "EPSG:32636_500000_-200000_summary.tif"
    transform = rasterio.transform.from_origin(500000, 200000, 10, 10)
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        width=64,
        height=64,
        count=NUM_SUMMARY_BANDS,
        dtype="uint8",
        crs="EPSG:32636",
        transform=transform,
    ) as dst:
        dst.write(arr)

    points, num_accepted = process_tile(
        str(tif_path), min_change_pixels=100, max_per_tile=1, seed=0, cell_size=CELL
    )
    assert num_accepted == 3
    # One of the two new_building cells is dropped by the cap, but the
    # new_infrastructure cell is kept because the cap is per category.
    categories = sorted(pt["post_category"] for pt in points)
    assert categories == ["new_building", "new_infrastructure"]


def test_points_round_trip_into_from_open_set(tmp_path: Path) -> None:
    """Written points are readable by from_open_set's sparse sample iterator."""
    tif_path = tmp_path / "EPSG:32636_500000_-200000_summary.tif"
    _write_summary_tif(tif_path)
    points, _ = process_tile(
        str(tif_path), min_change_pixels=100, max_per_tile=10, seed=0, cell_size=CELL
    )

    datasets_root = UPath(tmp_path / "datasets")
    write_points_table(SLUG, "change", points, root=datasets_root)

    samples = list(iter_sparse_samples(datasets_root, SLUG))
    assert len(samples) == 1
    sample = samples[0]
    assert sample["kind"] == "sparse"
    assert sample["slug"] == SLUG
    assert sample["sample_id"] == "32636_50000_-20000"
    assert sample["time_range"] is None
    assert sample["pre_time_range"] == [
        (datetime(2019, 2, 1, tzinfo=UTC) - timedelta(days=180)).isoformat(),
        datetime(2019, 2, 1, tzinfo=UTC).isoformat(),
    ]
    assert sample["post_time_range"] == [
        datetime(2019, 8, 1, tzinfo=UTC).isoformat(),
        (datetime(2019, 8, 1, tzinfo=UTC) + timedelta(days=180)).isoformat(),
    ]
