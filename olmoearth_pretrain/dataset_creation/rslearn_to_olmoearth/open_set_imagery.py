"""Convert open-set period-mosaic imagery layers to the OlmoEarth Pretrain format.

The open-set dataset materializes each multitemporal modality (Sentinel-2 L2A,
Sentinel-1, Landsat) as a SINGLE ``MOSAIC`` layer with ``period_duration=30d`` and
``include_partial_periods`` (see ``config_open_set.json``), so the number of timesteps
follows the label's own time range: one mosaic per ~30-day period, one mosaic for a
sub-30-day range, etc. This reads that layer's period groups and writes them as the
modality's multitemporal series, keyed by the window's ``example_id``.

Paired pre/post change samples consist of TWO windows sharing one ``example_id`` (with
a ``paired_part`` option of "pre" or "post"; see ``create_windows.from_open_set``).
Their period mosaics are merged in chronological order into a single multitemporal
series for that example_id.

``--modality open_set_change_boundary`` writes, for each paired example only, the
static ``open_set_change_boundary`` modality: the pre/post boundary date (the windows'
shared ``time`` option) as a ``[day, month, year]`` vector in the ``timestamps``
convention. Training uses it to tell "before" timesteps from "after" ones (see
``data.dataset.subset_sample_default`` and ``train.open_set_probe``).

The static modalities (worldcover, srtm, cdl, worldcereal, wri_canopy_height_map,
openstreetmap) reuse their existing conversion scripts unchanged (they skip the
secondary window of each pair; see ``cli.filter_paired_secondary_windows``).
"""

import argparse
import csv
import multiprocessing

import numpy as np
import tqdm
from rslearn.dataset import Dataset, Window
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_array import RasterArray
from upath import UPath

from olmoearth_pretrain.data.constants import Modality, TimeSpan
from olmoearth_pretrain.dataset.utils import get_modality_fname
from olmoearth_pretrain.open_set_segmentation_data.pretrain_constants import (
    OPEN_SET_WINDOW_SIZE,
)

from ..constants import GEOTIFF_RASTER_FORMAT, METADATA_COLUMNS
from ..util import get_modality_temp_meta_fname, get_window_metadata
from .cli import add_common_arguments
from .multitemporal_raster import (
    convert_paired_period_mosaic,
    convert_period_mosaic,
    get_adjusted_projection_and_bounds,
)

# CLI modality choice -> (rslearn layer name, ModalitySpec name). The layer name matches
# the modality name in config_open_set.json.
MODALITIES = {
    "sentinel2_l2a": Modality.SENTINEL2_L2A,
    "sentinel1": Modality.SENTINEL1,
    "landsat": Modality.LANDSAT,
}

# Derived (not materialized) modality: the pre/post boundary of paired change samples.
CHANGE_BOUNDARY_MODALITY_NAME = Modality.OPEN_SET_CHANGE_BOUNDARY.name


def is_paired_group(windows: list[Window]) -> bool:
    """Whether a per-example window group is a paired pre/post change sample."""
    return any(bool(w.options.get("paired_part")) for w in windows)


def convert_change_boundary(windows: list[Window], olmoearth_path: UPath) -> None:
    """Write the ``open_set_change_boundary`` modality for one paired example.

    Both windows of a pair carry the pre/post boundary as their ``time`` option (see
    ``create_windows.from_open_set._create_paired_windows``), which is what
    ``get_window_metadata(...).time`` returns. It is written as a 1x1 pixel, 3-band
    raster ``[day, month - 1, year]`` -- the same convention as the ``timestamps``
    written to the H5s -- so a timestep is post-change iff ``timestamp >= boundary``.
    Non-paired examples are skipped (the modality is missing-filled for them).

    Args:
        windows: the rslearn window(s) making up one example.
        olmoearth_path: OlmoEarth Pretrain dataset path to write to.
    """
    if not is_paired_group(windows):
        return
    modality = Modality.OPEN_SET_CHANGE_BOUNDARY
    assert len(modality.band_sets) == 1
    band_set = modality.band_sets[0]

    windows = sorted(windows, key=lambda w: w.time_range[0])
    window = windows[0]
    window_metadata = get_window_metadata(window)
    if window_metadata.example_id is None:
        raise ValueError(f"paired window {window.name} is missing an example_id option")
    boundary = window_metadata.time
    for other in windows[1:]:
        if get_window_metadata(other).time != boundary:
            raise ValueError(
                f"paired windows {window.name} and {other.name} disagree on the "
                "pre/post boundary time"
            )

    adjusted_projection, adjusted_bounds = get_adjusted_projection_and_bounds(
        modality, band_set, window.projection, window.bounds
    )
    if (
        adjusted_bounds[2] - adjusted_bounds[0] != 1
        or adjusted_bounds[3] - adjusted_bounds[1] != 1
    ):
        raise ValueError(
            f"expected a 1x1 pixel change-boundary raster, got bounds {adjusted_bounds}"
        )
    array = np.array(
        [boundary.day, boundary.month - 1, boundary.year], dtype=np.int32
    ).reshape(3, 1, 1)

    dst_fname = get_modality_fname(
        olmoearth_path,
        modality,
        TimeSpan.STATIC,
        window_metadata,
        band_set.get_resolution(),
        "tif",
    )
    GEOTIFF_RASTER_FORMAT.encode_raster(
        path=dst_fname.parent,
        projection=adjusted_projection,
        bounds=adjusted_bounds,
        raster=RasterArray(chw_array=array),
        fname=dst_fname.name,
    )

    start_time = min(w.time_range[0] for w in windows)
    end_time = max(w.time_range[1] for w in windows)
    metadata_fname = get_modality_temp_meta_fname(
        olmoearth_path, modality, TimeSpan.STATIC, window_metadata.example_id
    )
    metadata_fname.parent.mkdir(parents=True, exist_ok=True)
    with metadata_fname.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerow(
            dict(
                example_id=window_metadata.example_id,
                crs=window_metadata.crs,
                col=window_metadata.col,
                row=window_metadata.row,
                tile_time=window_metadata.time.isoformat(),
                image_idx="0",
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
            )
        )


def convert_open_set_imagery(
    windows: list[Window], olmoearth_path: UPath, modality_name: str
) -> None:
    """Convert one example's period-mosaic layer for the given modality.

    Args:
        windows: the rslearn window(s) making up one example: a single window for
            regular samples, or the pre/post window pair of a change sample (merged
            into one multitemporal series).
        olmoearth_path: OlmoEarth Pretrain dataset path to write to.
        modality_name: one of ``sentinel2_l2a``, ``sentinel1``, ``landsat``, or
            ``open_set_change_boundary``.
    """
    if modality_name == CHANGE_BOUNDARY_MODALITY_NAME:
        convert_change_boundary(windows, olmoearth_path)
        return
    modality = MODALITIES[modality_name]
    if len(windows) == 1:
        convert_period_mosaic(
            windows[0],
            olmoearth_path,
            layer_name=modality_name,
            modality=modality,
            image_tile_size=OPEN_SET_WINDOW_SIZE,
        )
    else:
        convert_paired_period_mosaic(
            windows,
            olmoearth_path,
            layer_name=modality_name,
            modality=modality,
            image_tile_size=OPEN_SET_WINDOW_SIZE,
        )


def group_windows_by_example(windows: list[Window]) -> list[list[Window]]:
    """Group windows into per-example lists.

    Paired pre/post windows (those with a ``paired_part`` option) are grouped by their
    shared ``example_id``; all other windows form singleton groups.
    """
    groups: dict[str, list[Window]] = {}
    singles: list[list[Window]] = []
    for window in windows:
        if window.options.get("paired_part"):
            groups.setdefault(window.options["example_id"], []).append(window)
        else:
            singles.append([window])
    return singles + list(groups.values())


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser(
        description="Convert open-set period-mosaic imagery"
    )
    add_common_arguments(parser, default_groups=["open_set"])
    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        choices=sorted(MODALITIES.keys()) + [CHANGE_BOUNDARY_MODALITY_NAME],
        help=(
            "Which multitemporal modality layer to convert, or "
            f"{CHANGE_BOUNDARY_MODALITY_NAME} to write the paired samples' pre/post "
            "boundary"
        ),
    )
    args = parser.parse_args()

    dataset = Dataset(UPath(args.ds_path))
    olmoearth_path = UPath(args.olmoearth_path)

    windows = dataset.load_windows(
        workers=args.workers, show_progress=True, groups=args.groups
    )
    jobs = []
    for window_group in group_windows_by_example(windows):
        if args.modality == CHANGE_BOUNDARY_MODALITY_NAME and not is_paired_group(
            window_group
        ):
            continue
        jobs.append(
            dict(
                windows=window_group,
                olmoearth_path=olmoearth_path,
                modality_name=args.modality,
            )
        )

    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, convert_open_set_imagery, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
    p.join()
