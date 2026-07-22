"""Convert open-set period-mosaic imagery layers to the OlmoEarth Pretrain format.

The open-set dataset materializes each multitemporal modality (Sentinel-2 L2A,
Sentinel-1, Landsat) as a SINGLE ``MOSAIC`` layer with ``period_duration=30d`` and
``include_partial_periods`` (see ``config_open_set.json``), so the number of timesteps
follows the label's own time range: one mosaic per ~30-day period, one mosaic for a
sub-30-day range, etc. This reads that layer's period groups and writes them as the
modality's multitemporal series, keyed by the window's ``example_id``.

The static modalities (worldcover, srtm, cdl, worldcereal, wri_canopy_height_map,
openstreetmap) reuse their existing conversion scripts unchanged.
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
from .multitemporal_raster import convert_period_mosaic

# CLI modality choice -> (rslearn layer name, ModalitySpec name). The layer name matches
# the modality name in config_open_set.json.
MODALITIES = {
    "sentinel2_l2a": Modality.SENTINEL2_L2A,
    "sentinel1": Modality.SENTINEL1,
    "landsat": Modality.LANDSAT,
}


def convert_open_set_imagery(
    window: Window, olmoearth_path: UPath, modality_name: str
) -> None:
    """Convert one window's period-mosaic layer for the given modality.

    Args:
        window: the rslearn window to read data from.
        olmoearth_path: OlmoEarth Pretrain dataset path to write to.
        modality_name: one of ``sentinel2_l2a``, ``sentinel1``, ``landsat``.
    """
    modality = MODALITIES[modality_name]
    convert_period_mosaic(
        window,
        olmoearth_path,
        layer_name=modality_name,
        modality=modality,
        image_tile_size=OPEN_SET_WINDOW_SIZE,
    )


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

    jobs = []
    for window in dataset.load_windows(
        workers=args.workers, show_progress=True, groups=args.groups
    ):
        jobs.append(
            dict(
                window=window,
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
