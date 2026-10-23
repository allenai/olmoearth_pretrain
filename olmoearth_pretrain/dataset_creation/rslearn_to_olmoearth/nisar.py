"""Post-process NISAR L2 GCOV data into the OlmoEarth Pretrain dataset."""

import argparse
import multiprocessing

import tqdm
from rslearn.dataset import Dataset, Window
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from olmoearth_pretrain.data.constants import Modality

from .cli import add_common_arguments
from .multitemporal_raster import convert_monthly

# Layer prefix (layers are named <prefix>_mo01 .. <prefix>_mo12) for each modality.
LAYER_PREFIXES = {
    Modality.NISAR_ASC: "nisar_asc",
    Modality.NISAR_DESC: "nisar_desc",
}


def convert_nisar(window: Window, olmoearth_path: UPath) -> None:
    """Add NISAR ascending and descending data for this window to the OlmoEarth Pretrain dataset.

    Args:
        window: the rslearn window to read data from.
        olmoearth_path: OlmoEarth Pretrain dataset path to write to.
    """
    for modality, layer_prefix in LAYER_PREFIXES.items():
        try:
            convert_monthly(window, olmoearth_path, layer_prefix, modality)
        except Exception as e:
            print(
                f"warning: got error {e} while converting {modality.name} data for window {window.name}"
            )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser(
        description="Post-process OlmoEarth Pretrain data",
    )
    add_common_arguments(parser)
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
            )
        )

    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, convert_nisar, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
