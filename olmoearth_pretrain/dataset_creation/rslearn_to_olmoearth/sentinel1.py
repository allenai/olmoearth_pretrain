"""Post-process ingested Landsat data into the OlmoEarth Pretrain dataset."""

from rslearn.dataset import Window
from upath import UPath

from olmoearth_pretrain.modalities import Modality

from .multitemporal_raster import convert_freq, convert_monthly
from .runner import run_window_converter

# rslearn layer for frequent data.
LAYER_FREQ = "sentinel1_freq"

# rslearn layer prefix for monthly data.
LAYER_MONTHLY = "sentinel1"


def convert_sentinel1(window: Window, olmoearth_path: UPath) -> None:
    """Add Landsat data for this window to the OlmoEarth Pretrain dataset.

    Args:
        window: the rslearn window to read data from.
        olmoearth_path: OlmoEarth Pretrain dataset path to write to.
    """
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
        convert_monthly(window, olmoearth_path, LAYER_MONTHLY, Modality.SENTINEL1)
    except Exception as e:
        print(
            f"warning: got error {e} while converting monthly data for window {window.name}"
        )


if __name__ == "__main__":
    run_window_converter(convert_sentinel1, groups=["res_10"])
