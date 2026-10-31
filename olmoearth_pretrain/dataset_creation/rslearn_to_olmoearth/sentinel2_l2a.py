"""Post-process ingested Sentinel-2 L2A data into the OlmoEarth Pretrain dataset."""

import argparse
import multiprocessing

import tqdm
from rslearn.dataset import Dataset, Window
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from olmoearth_pretrain.data.constants import Modality

from .cli import add_common_arguments, add_layout_argument
from .multitemporal_raster import convert_multitemporal

# rslearn layer prefix for monthly data (also the layer name for period_mosaic).
LAYER_MONTHLY = "sentinel2_l2a"


def convert_sentinel2_l2a(
    window: Window, olmoearth_path: UPath, layout: str = "monthly"
) -> None:
    """Add Sentinel-2 L2A data for this window to the OlmoEarth Pretrain dataset.

    Args:
        window: the rslearn window to read data from.
        olmoearth_path: OlmoEarth Pretrain dataset path to write to.
        layout: "monthly" (layers <prefix>_mo01..mo12) or "period_mosaic" (one
            MOSAIC layer <prefix> with one item group per period).
    """
    try:
        convert_multitemporal(
            window, olmoearth_path, LAYER_MONTHLY, Modality.SENTINEL2_L2A, layout
        )
    except Exception as e:
        print(f"warning: error handling window {window.name}: {e}")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser(
        description="Post-process OlmoEarth Pretrain data",
    )
    add_common_arguments(parser)
    add_layout_argument(parser)
    args = parser.parse_args()

    dataset = Dataset(UPath(args.ds_path))
    olmoearth_path = UPath(args.olmoearth_path)

    jobs = []
    for window in dataset.load_windows(
        workers=args.workers, show_progress=True, groups=["res_10"]
    ):
        jobs.append(
            dict(
                window=window,
                olmoearth_path=olmoearth_path,
                layout=args.layout,
            )
        )

    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, convert_sentinel2_l2a, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
