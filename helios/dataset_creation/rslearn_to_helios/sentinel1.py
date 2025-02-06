"""Post-process ingested Landsat data into the Helios dataset."""

import argparse
import multiprocessing

import tqdm
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from .multitemporal_raster import BandSet, convert_freq, convert_monthly

BAND_SETS = [
    BandSet(["vv", "vh"], 10),
]

# rslearn layer for frequent data.
LAYER_FREQ = "sentinel1_freq"

# rslearn layer prefix for monthly data.
LAYER_MONTHLY = "sentinel1"

# Modality for frequent data in the output Helios dataset.
MODALITY_FREQ = "10_sentinel1_freq"

# Modality for monthly data in the output Helios dataset.
MODALITY_MONTHLY = "10_sentinel1_monthly"


def convert_sentinel1(window_path: UPath, helios_path: UPath) -> None:
    """Add Landsat data for this window to the Helios dataset.

    Args:
        window_path: the rslearn window directory to read data from.
        helios_path: Helios dataset path to write to.
    """
    convert_freq(window_path, helios_path, LAYER_FREQ, MODALITY_FREQ, BAND_SETS)
    convert_monthly(
        window_path, helios_path, LAYER_MONTHLY, MODALITY_MONTHLY, BAND_SETS
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser(
        description="Post-process Helios data",
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
        help="Destination Helios dataset path",
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
    outputs = star_imap_unordered(p, convert_sentinel1, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
