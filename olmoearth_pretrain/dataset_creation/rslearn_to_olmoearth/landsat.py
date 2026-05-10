"""Post-process ingested Landsat data into the OlmoEarth Pretrain dataset."""

import argparse
import hashlib
import logging
import multiprocessing

import tqdm
from rslearn.dataset import Dataset, Window
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from olmoearth_pretrain.data.constants import Modality, TimeSpan

from ..util import get_modality_temp_meta_fname
from .multitemporal_raster import convert_freq, convert_monthly

logger = logging.getLogger(__name__)

# rslearn layer for frequent data.
LAYER_FREQ = "landsat_freq"

# rslearn layer prefix for monthly data.
LAYER_MONTHLY = "landsat"


def _is_window_completed(
    window: Window,
    olmoearth_path: UPath,
    convert_frequent: bool,
    convert_monthly_data: bool,
) -> bool:
    """Check whether the output files for this window already exist.

    Uses the per-window metadata CSV as a proxy: it is written last, so its
    presence implies the data files have been written too.
    """
    if convert_frequent:
        meta = get_modality_temp_meta_fname(
            olmoearth_path, Modality.LANDSAT, TimeSpan.TWO_WEEK, window.name
        )
        if not meta.exists():
            return False
    if convert_monthly_data:
        meta = get_modality_temp_meta_fname(
            olmoearth_path, Modality.LANDSAT, TimeSpan.YEAR, window.name
        )
        if not meta.exists():
            return False
    return True


def convert_landsat(
    window: Window,
    olmoearth_path: UPath,
    convert_frequent: bool = True,
    convert_monthly_data: bool = True,
    skip_completed: bool = False,
) -> None:
    """Add Landsat data for this window to the OlmoEarth Pretrain dataset.

    Args:
        window: the rslearn window to read data from.
        olmoearth_path: OlmoEarth Pretrain dataset path to write to.
        convert_frequent: whether to convert the two-week frequent layer.
        convert_monthly_data: whether to convert the one-year monthly layers.
        skip_completed: skip this window if its output files already exist.
    """
    if skip_completed and _is_window_completed(
        window, olmoearth_path, convert_frequent, convert_monthly_data
    ):
        logger.info("skipping already-completed window %s", window.name)
        return

    if convert_frequent:
        convert_freq(
            window,
            olmoearth_path,
            LAYER_FREQ,
            Modality.LANDSAT,
            missing_okay=True,
            unprepared_okay=True,
        )
    if convert_monthly_data:
        convert_monthly(window, olmoearth_path, LAYER_MONTHLY, Modality.LANDSAT)


def _window_shard(window_name: str, shard_cnt: int) -> int:
    """Deterministically assign a window to a shard index."""
    digest = hashlib.sha256(window_name.encode()).digest()
    return int.from_bytes(digest[:8], "big") % shard_cnt


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
    parser.add_argument(
        "--shard_ix",
        type=int,
        default=None,
        help="Index of the shard this worker should process (0-based)",
    )
    parser.add_argument(
        "--shard_cnt",
        type=int,
        default=None,
        help="Total number of shards to split windows across",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip windows whose output files already exist",
    )
    args = parser.parse_args()

    if args.skip_freq and args.skip_monthly:
        raise ValueError(
            "at least one of frequent or monthly Landsat conversion is required"
        )

    if (args.shard_ix is None) != (args.shard_cnt is None):
        raise ValueError("--shard_ix and --shard_cnt must be provided together")
    if args.shard_cnt is not None and not (0 <= args.shard_ix < args.shard_cnt):
        raise ValueError(
            f"--shard_ix must be in [0, {args.shard_cnt}), got {args.shard_ix}"
        )

    dataset = Dataset(UPath(args.ds_path))
    olmoearth_path = UPath(args.olmoearth_path)

    jobs = []
    for window in dataset.load_windows(
        workers=args.workers, show_progress=True, groups=args.group
    ):
        if args.shard_cnt is not None:
            if _window_shard(window.name, args.shard_cnt) != args.shard_ix:
                continue

        jobs.append(
            dict(
                window=window,
                olmoearth_path=olmoearth_path,
                convert_frequent=not args.skip_freq,
                convert_monthly_data=not args.skip_monthly,
                skip_completed=args.skip_completed,
            )
        )

    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, convert_landsat, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
