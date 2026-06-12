"""Post-process ingested Sentinel-2 data into the OlmoEarth Pretrain dataset."""

from rslearn.dataset import Window
from upath import UPath

from olmoearth_pretrain.modalities import Modality

from .multitemporal_raster import convert_freq, convert_monthly
from .runner import run_window_converter

# rslearn layer for frequent data.
LAYER_FREQ = "sentinel2_freq"

# rslearn layer prefix for monthly data.
LAYER_MONTHLY = "sentinel2"


def convert_sentinel2(window: Window, olmoearth_path: UPath) -> None:
    """Add Sentinel-2 data for this window to the OlmoEarth Pretrain dataset.

    Args:
        window: the rslearn window to read data from.
        olmoearth_path: OlmoEarth Pretrain dataset path to write to.
    """
    convert_freq(
        window,
        olmoearth_path,
        LAYER_FREQ,
        Modality.SENTINEL2,
        missing_okay=True,
        unprepared_okay=True,
    )
    convert_monthly(window, olmoearth_path, LAYER_MONTHLY, Modality.SENTINEL2)


if __name__ == "__main__":
    run_window_converter(convert_sentinel2, groups=["res_10"])
