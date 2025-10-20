"""Post-process ingested Sentinel-2 L2A data into the OlmoEarth Pretrain dataset."""

import argparse
import multiprocessing

import tqdm
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from olmoearth_pretrain.data.constants import Modality

from .multitemporal_raster import convert_freq, convert_monthly

# rslearn layer for frequent data.
LAYER_FREQ = "sentinel2_l2a_freq"

# rslearn layer prefix for monthly data.
LAYER_MONTHLY = "sentinel2_l2a"


def convert_sentinel2_l2a(window_path: UPath, helios_path: UPath) -> None:
    """Add Sentinel-2 data for this window to the OlmoEarth Pretrain dataset.

    Args:
        window_path: the rslearn window directory to read data from.
        helios_path: OlmoEarth Pretrain dataset path to write to.
    """
    convert_freq(
        window_path,
        helios_path,
        LAYER_FREQ,
        Modality.SENTINEL2_L2A,
        missing_okay=True,
        unprepared_okay=True,
    )
    convert_monthly(window_path, helios_path, LAYER_MONTHLY, Modality.SENTINEL2_L2A)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser(
        description="Post-process OlmoEarth Pretrain data",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        help="Source rslearn dataset path",
        required=True,
    )
    parser.add_argument(
        "--helios_path",
        type=str,
        help="Destination OlmoEarth Pretrain dataset path",
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers to use",
        default=32,
    )
    args = parser.parse_args()

    ds_path = UPath(args.ds_path)
    helios_path = UPath(args.helios_path)

    metadata_fnames = ds_path.glob("windows/res_10/*/metadata.json")
    jobs = []
    for metadata_fname in metadata_fnames:
        jobs.append(
            dict(
                window_path=metadata_fname.parent,
                helios_path=helios_path,
            )
        )

    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, convert_sentinel2_l2a, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
