"""Post-process ingested Landsat data into the OlmoEarth Pretrain dataset."""

import argparse
import multiprocessing

import tqdm
from rslearn.dataset import Dataset, Window
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from olmoearth_pretrain.data.constants import Modality

from .multitemporal_raster import convert_freq, convert_monthly

# rslearn layer for frequent data.
LAYER_FREQ = "landsat_freq"

# rslearn layer prefix for monthly data.
LAYER_MONTHLY = "landsat"


def convert_landsat(
    window: Window,
    olmoearth_path: UPath,
    convert_frequent: bool = True,
    convert_monthly_data: bool = True,
) -> None:
    """Add Landsat data for this window to the OlmoEarth Pretrain dataset.

    Args:
        window: the rslearn window to read data from.
        olmoearth_path: OlmoEarth Pretrain dataset path to write to.
        convert_frequent: whether to convert the two-week frequent layer.
        convert_monthly_data: whether to convert the one-year monthly layers.
    """
    if convert_frequent:
        convert_freq(
            window,
            olmoearth_path,
            LAYER_FREQ,
            Modality.LANDSAT,
            missing_okay=True,
        )
    if convert_monthly_data:
        convert_monthly(window, olmoearth_path, LAYER_MONTHLY, Modality.LANDSAT)


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
        "--olmoearth_path",
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
    parser.add_argument(
        "--group",
        type=str,
        nargs="+",
        help="rslearn window group(s) to convert",
        default=["res_10"],
    )
    parser.add_argument(
        "--skip-freq",
        action="store_true",
        help="Skip conversion of the landsat_freq two-week layer",
    )
    parser.add_argument(
        "--skip-monthly",
        action="store_true",
        help="Skip conversion of the landsat_mo* monthly layers",
    )
    args = parser.parse_args()

    if args.skip_freq and args.skip_monthly:
        raise ValueError(
            "at least one of frequent or monthly Landsat conversion is required"
        )

    dataset = Dataset(UPath(args.ds_path))
    olmoearth_path = UPath(args.olmoearth_path)

    jobs = []
    for window in dataset.load_windows(
        workers=args.workers, show_progress=True, groups=args.group
    ):
        jobs.append(
            dict(
                window=window,
                olmoearth_path=olmoearth_path,
                convert_frequent=not args.skip_freq,
                convert_monthly_data=not args.skip_monthly,
            )
        )

    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, convert_landsat, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
