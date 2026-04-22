"""Post-process ingested Sentinel-2 L2A data into the OlmoEarth Pretrain dataset."""

import argparse
import multiprocessing

import tqdm
from rslearn.dataset import Dataset, Window
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from olmoearth_pretrain.data.constants import Modality

from .multitemporal_raster import convert_freq, convert_monthly, convert_temporal_stack

# rslearn layer for frequent data.
LAYER_FREQ = "sentinel2_l2a_freq"

# rslearn layer name (new single-layer layout) / prefix (legacy _moNN layout).
LAYER_NAME = "sentinel2_l2a"


def convert_sentinel2_l2a(
    window: Window,
    olmoearth_path: UPath,
    use_temporal_stack: bool = True,
) -> None:
    """Add Sentinel-2 data for this window to the OlmoEarth Pretrain dataset.

    Args:
        window: the rslearn window to read data from.
        olmoearth_path: OlmoEarth Pretrain dataset path to write to.
        use_temporal_stack: if True, use the new single-layer temporal stack layout.
            If False, fall back to the legacy 12 separate ``_moNN`` layers.
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
    except Exception as e:
        print(
            f"warning: got error {e} while converting frequent data for window {window.name}"
        )

    try:
        if use_temporal_stack:
            convert_temporal_stack(
                window, olmoearth_path, LAYER_NAME, Modality.SENTINEL2_L2A,
                missing_okay=True,
            )
        else:
            convert_monthly(window, olmoearth_path, LAYER_NAME, Modality.SENTINEL2_L2A)
    except Exception as e:
        print(
            f"warning: got error {e} while converting monthly data for window {window.name}"
        )


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
        "--legacy-monthly",
        action="store_true",
        help="Use legacy 12 separate _moNN layers instead of temporal stack",
    )
    args = parser.parse_args()

    dataset = Dataset(UPath(args.ds_path))
    olmoearth_path = UPath(args.olmoearth_path)
    use_temporal_stack = not args.legacy_monthly

    jobs = []
    for window in dataset.load_windows(
        workers=args.workers, show_progress=True, groups=["res_10"]
    ):
        jobs.append(
            dict(
                window=window,
                olmoearth_path=olmoearth_path,
                use_temporal_stack=use_temporal_stack,
            )
        )

    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, convert_sentinel2_l2a, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
