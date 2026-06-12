"""Post-process ingested Sentinel-2 L2A data into the OlmoEarth Pretrain dataset."""

from rslearn.dataset import Window
from upath import UPath

from olmoearth_pretrain.modalities import Modality

from .multitemporal_raster import convert_freq, convert_monthly
from .runner import run_window_converter

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


if __name__ == "__main__":
    run_window_converter(convert_sentinel2_l2a, groups=["res_10"])
