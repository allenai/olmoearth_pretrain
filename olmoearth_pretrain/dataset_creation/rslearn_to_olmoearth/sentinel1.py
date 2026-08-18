"""Post-process ingested Sentinel-1 data into the OlmoEarth Pretrain dataset."""

import argparse
import multiprocessing

import tqdm
from rslearn.dataset import Dataset, Window
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from olmoearth_pretrain.data.constants import SENTINEL1_NODATA, Modality

from .multitemporal_raster import (
    convert_allcap,
    convert_freq,
    convert_monthly,
    convert_temporal_stack,
)

# rslearn layer for frequent data.
LAYER_FREQ = "sentinel1_freq"

# rslearn layer name (new single-layer layout) / prefix (legacy _moNN layout).
LAYER_NAME = "sentinel1"


def convert_sentinel1(
    window: Window,
    olmoearth_path: UPath,
    use_temporal_stack: bool = True,
    use_allcap: bool = False,
) -> None:
    """Add Sentinel-1 data for this window to the OlmoEarth Pretrain dataset.

    Args:
        window: the rslearn window to read data from.
        olmoearth_path: OlmoEarth Pretrain dataset path to write to.
        use_temporal_stack: if True, use the new single-layer temporal stack layout.
            If False, fall back to the legacy 12 separate ``_moNN`` layers.
        use_allcap: if True, the layer stores every individual capture (one item per
            group). Captures containing nodata pixels are skipped so the resulting
            stack stays valid under the dB conversion at h5 time.
    """
    if use_allcap:
        try:
            convert_allcap(
                window,
                olmoearth_path,
                LAYER_NAME,
                Modality.SENTINEL1,
                missing_okay=True,
                unprepared_okay=True,
                skip_nodata_value=SENTINEL1_NODATA,
            )
        except Exception as e:
            print(
                f"warning: got error {e} while converting allcap sentinel1 "
                f"data for window {window.name}"
            )
        return

    try:
        convert_freq(
            window,
            olmoearth_path,
            LAYER_FREQ,
            Modality.SENTINEL1,
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
                window,
                olmoearth_path,
                LAYER_NAME,
                Modality.SENTINEL1,
                missing_okay=True,
            )
        else:
            convert_monthly(window, olmoearth_path, LAYER_NAME, Modality.SENTINEL1)
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
    outputs = star_imap_unordered(p, convert_sentinel1, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
