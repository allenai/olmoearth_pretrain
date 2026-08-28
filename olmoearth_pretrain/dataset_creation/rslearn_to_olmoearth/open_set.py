"""Convert the open-set label layers into the OlmoEarth Pretrain dataset format.

Reads the ``open_set`` (classification) or ``open_set_regression`` label layer that was
written into each window by ``create_windows.from_open_set`` (via ``window.data``), and
writes it out in the OlmoEarth Pretrain format, keyed by the window's ``example_id``
(``slug_sampleid``).

Each window carries exactly one of the two layers, so this script skips windows that do
not have the requested layer. Run it once per layer (``--layer open_set`` and
``--layer open_set_regression``).
"""

import argparse
import csv
import multiprocessing

import tqdm
from rslearn.dataset import Dataset, Window
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from olmoearth_pretrain.data.constants import Modality, ModalitySpec, TimeSpan
from olmoearth_pretrain.dataset.utils import get_modality_fname

from ..constants import GEOTIFF_RASTER_FORMAT, METADATA_COLUMNS
from ..util import get_modality_temp_meta_fname, get_window_metadata
from .cli import add_common_arguments

# Maps the CLI --layer choice to (rslearn layer name, ModalitySpec).
LAYER_TO_MODALITY: dict[str, ModalitySpec] = {
    "open_set": Modality.OPEN_SET,
    "open_set_regression": Modality.OPEN_SET_REGRESSION,
}


def convert_open_set(window: Window, olmoearth_path: UPath, layer_name: str) -> None:
    """Convert one window's open-set label layer to the OlmoEarth Pretrain format.

    Args:
        window: the rslearn window to read the label layer from.
        olmoearth_path: OlmoEarth Pretrain dataset path to write to.
        layer_name: the label layer to convert ("open_set" or "open_set_regression").
    """
    if not window.is_layer_completed(layer_name):
        return

    modality = LAYER_TO_MODALITY[layer_name]
    window_metadata = get_window_metadata(window)
    assert len(modality.band_sets) == 1
    band_set = modality.band_sets[0]

    image = window.data.read_raster(layer_name, band_set.bands, GEOTIFF_RASTER_FORMAT)

    dst_fname = get_modality_fname(
        olmoearth_path,
        modality,
        TimeSpan.STATIC,
        window_metadata,
        band_set.get_resolution(),
        "tif",
    )
    GEOTIFF_RASTER_FORMAT.encode_raster(
        path=dst_fname.parent,
        projection=window.projection,
        bounds=window.bounds,
        raster=image,
        fname=dst_fname.name,
    )

    assert window.time_range is not None
    start_time, end_time = window.time_range
    metadata_fname = get_modality_temp_meta_fname(
        olmoearth_path, modality, TimeSpan.STATIC, window.name
    )
    metadata_fname.parent.mkdir(parents=True, exist_ok=True)
    with metadata_fname.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerow(
            dict(
                example_id=window_metadata.example_id or "",
                crs=window_metadata.crs,
                col=window_metadata.col,
                row=window_metadata.row,
                tile_time=window_metadata.time.isoformat(),
                image_idx="0",
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
            )
        )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser(description="Convert open-set label layers")
    add_common_arguments(parser, default_groups=["open_set"])
    parser.add_argument(
        "--layer",
        type=str,
        default="open_set",
        choices=sorted(LAYER_TO_MODALITY.keys()),
        help="Which label layer to convert",
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
                layer_name=args.layer,
            )
        )

    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, convert_open_set, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
    p.join()
