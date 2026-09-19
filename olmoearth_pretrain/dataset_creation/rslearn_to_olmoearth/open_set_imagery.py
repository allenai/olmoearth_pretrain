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

The static modalities (worldcover, srtm, cdl, worldcereal, wri_canopy_height_map,
openstreetmap) reuse their existing conversion scripts unchanged (they skip the
secondary window of each pair; see ``cli.filter_paired_secondary_windows``).
"""

import argparse
import multiprocessing

import tqdm
from rslearn.dataset import Dataset, Window
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.open_set_segmentation_data.pretrain_constants import (
    OPEN_SET_WINDOW_SIZE,
)

from .cli import add_common_arguments
from .multitemporal_raster import convert_paired_period_mosaic, convert_period_mosaic

# CLI modality choice -> (rslearn layer name, ModalitySpec name). The layer name matches
# the modality name in config_open_set.json.
MODALITIES = {
    "sentinel2_l2a": Modality.SENTINEL2_L2A,
    "sentinel1": Modality.SENTINEL1,
    "landsat": Modality.LANDSAT,
}


def convert_open_set_imagery(
    windows: list[Window], olmoearth_path: UPath, modality_name: str
) -> None:
    """Convert one example's period-mosaic layer for the given modality.

    Args:
        windows: the rslearn window(s) making up one example: a single window for
            regular samples, or the pre/post window pair of a change sample (merged
            into one multitemporal series).
        olmoearth_path: OlmoEarth Pretrain dataset path to write to.
        modality_name: one of ``sentinel2_l2a``, ``sentinel1``, ``landsat``.
    """
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
        choices=sorted(MODALITIES.keys()),
        help="Which multitemporal modality layer to convert",
    )
    args = parser.parse_args()

    dataset = Dataset(UPath(args.ds_path))
    olmoearth_path = UPath(args.olmoearth_path)

    windows = dataset.load_windows(
        workers=args.workers, show_progress=True, groups=args.groups
    )
    jobs = []
    for window_group in group_windows_by_example(windows):
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
