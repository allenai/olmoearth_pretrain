"""Post-process ingested Sentinel-2 L2A data into the OlmoEarth Pretrain dataset."""

import argparse
import multiprocessing

import tqdm
from rslearn.dataset import Dataset, Window
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from olmoearth_pretrain.data.constants import Modality, TimeSpan

from .multitemporal_raster import convert_freq, convert_monthly

# rslearn layer for frequent data.
LAYER_FREQ = "sentinel2_l2a_freq"

# rslearn layer prefix for monthly data.
LAYER_MONTHLY = "sentinel2_l2a"


def convert_sentinel2_l2a(window: Window, olmoearth_path: UPath) -> None:
    """Add Sentinel-2 data for this window to the OlmoEarth Pretrain dataset.

    Args:
        window: the rslearn window to read data from.
        olmoearth_path: OlmoEarth Pretrain dataset path to write to.
    """
    try:
        convert_freq(
            window,
            olmoearth_path,
            LAYER_FREQ,
            Modality.SENTINEL2_L2A,
            missing_okay=True,
            unprepared_okay=True,
        )
        convert_monthly(window, olmoearth_path, LAYER_MONTHLY, Modality.SENTINEL2_L2A)
    except Exception as e:
        print(f"warning: error handling window {window.name}: {e}")


def convert_sentinel2_l2a_highfreq(
    window: Window,
    olmoearth_path: UPath,
    layer_name: str = LAYER_MONTHLY,
    time_span: TimeSpan = TimeSpan.HIGH_FREQ,
) -> None:
    """Add high-frequency Sentinel-2 data for this window."""
    try:
        convert_freq(
            window,
            olmoearth_path,
            layer_name,
            Modality.SENTINEL2_L2A,
            missing_okay=True,
            unprepared_okay=True,
            time_span=time_span,
            use_group_time_ranges=True,
        )
    except Exception as e:
        print(f"warning: error handling window {window.name}: {e}")


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
        "--mode",
        type=str,
        choices=["legacy", "highfreq"],
        help="Use legacy two-week/monthly layers or one high-frequency yearly layer",
        default="legacy",
    )
    parser.add_argument(
        "--highfreq_layer",
        type=str,
        help="rslearn layer name to use with --mode highfreq",
        default=LAYER_MONTHLY,
    )
    parser.add_argument(
        "--highfreq_time_span",
        type=str,
        help="OlmoEarth time span to write with --mode highfreq",
        default=TimeSpan.HIGH_FREQ.value,
    )
    args = parser.parse_args()

    dataset = Dataset(UPath(args.ds_path))
    olmoearth_path = UPath(args.olmoearth_path)
    convert_fn = (
        convert_sentinel2_l2a
        if args.mode == "legacy"
        else convert_sentinel2_l2a_highfreq
    )

    jobs = []
    for window in dataset.load_windows(
        workers=args.workers, show_progress=True, groups=args.group
    ):
        job = dict(window=window, olmoearth_path=olmoearth_path)
        if args.mode == "highfreq":
            job["layer_name"] = args.highfreq_layer
            job["time_span"] = TimeSpan(args.highfreq_time_span)
        jobs.append(job)

    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, convert_fn, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
