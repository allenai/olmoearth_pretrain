r"""Run the conversion of a dataset to h5py files.

This script is used to convert a dataset to h5py files.

The modalities support can be changed in the script or by overriding the supported_modality_names argument with an escaped list

Usage:
    python run_h5_conversion.py --tile-path=TILE_PATH --supported-modality-names="\[sentinel2_l2a,sentinel1,worldcover\]"
"""

import logging
import sys
from collections.abc import Callable
from typing import Any

from olmo_core.utils import prepare_cli_environment

from helios.data.constants import Modality
from helios.dataset.convert_to_h5py import ConvertToH5pyConfig

logger = logging.getLogger(__name__)


def build_default_config() -> ConvertToH5pyConfig:
    """Build the default configuration for H5 conversion."""
    return ConvertToH5pyConfig(
        tile_path="",
        supported_modality_names=[
            # Modality.NAIP_10.name,
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.WORLDCOVER.name,
            Modality.SRTM.name,
            Modality.LANDSAT.name,
            Modality.OPENSTREETMAP_RASTER.name,
        ],
        multiprocessed_h5_creation=True,
    )


def main(config_builder: Callable = build_default_config, *args: Any) -> None:
    """Parse arguments, build config, and run the H5 conversion."""
    prepare_cli_environment()

    script, *overrides = sys.argv

    # Create the configuration object from arguments
    default_config = config_builder()
    config = default_config.merge(overrides)
    logger.info(f"Configuration overrides: {overrides}")
    logger.info(f"Configuration loaded: {config}")

    # Build and run the converter
    converter = config.build()
    logger.info("Starting H5 conversion process...")
    converter.run()
    logger.info("H5 conversion process finished.")


if __name__ == "__main__":
    main()
