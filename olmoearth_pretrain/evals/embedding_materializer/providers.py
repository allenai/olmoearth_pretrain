"""Window iteration and raster-layer writing for rslearn eval datasets."""

import logging

import numpy as np
from rslearn.dataset import Dataset, Window
from rslearn.dataset.storage.file import FileWindowStorage
from rslearn.utils.raster_array import RasterArray, RasterMetadata
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

from olmoearth_pretrain.data.constants import ModalitySpec

logger = logging.getLogger(__name__)


def get_target_year(window: Window, year_override: int | None = None) -> int | None:
    """Compute the target product year for a window.

    Args:
        window: the rslearn window.
        year_override: if set, this year is used for every window.

    Returns:
        the explicit override year if given; otherwise the year of the
        midpoint of the window's time range, or None if the window has no
        time range.
    """
    if year_override is not None:
        return year_override
    if window.time_range is None:
        return None
    start, end = window.time_range
    midpoint = start + (end - start) / 2
    return midpoint.year


class RslearnWindowProvider:
    """Iterates windows of an rslearn dataset and writes embedding layers.

    Embedding rasters are written the same way rslearn materializes layers:
    windows/<group>/<window>/layers/<layer>/<bands>/geotiff.tif, with the
    layer marked completed afterwards so rslearn treats it as materialized.
    """

    def __init__(self, ds_path: UPath | str, groups: list[str] | None = None) -> None:
        """Initialize an RslearnWindowProvider.

        Args:
            ds_path: path to the rslearn dataset root. If config.json exists
                there, the dataset's configured window storage is used;
                otherwise the default file-based window storage is assumed.
            groups: optional list of window groups to restrict iteration to.
        """
        self.ds_path = UPath(ds_path)
        self.groups = groups
        if (self.ds_path / "config.json").exists():
            self._storage = Dataset(self.ds_path).storage
        else:
            logger.warning(
                f"No config.json at {self.ds_path}; using file-based window storage."
            )
            self._storage = FileWindowStorage(self.ds_path)
        self._raster_format = GeotiffRasterFormat()

    def load_windows(self) -> list[Window]:
        """Load the dataset's windows (optionally restricted to groups).

        Returns:
            list of rslearn Windows.
        """
        return self._storage.get_windows(groups=self.groups)

    def is_layer_written(self, window: Window, layer_name: str) -> bool:
        """Check whether the embedding layer is already materialized.

        Args:
            window: the rslearn window.
            layer_name: the layer name (usually the modality name).

        Returns:
            whether the layer was previously written and marked completed.
        """
        return window.is_layer_completed(layer_name)

    def write_embedding(
        self,
        window: Window,
        modality: ModalitySpec,
        array: np.ndarray,
        nodata_value: float,
        layer_name: str | None = None,
    ) -> UPath:
        """Write an embedding array as a raster layer of the window.

        The array is written as a GeoTIFF with the modality's band names,
        mirroring rslearn's materialized-layer storage layout, and the layer
        is marked completed.

        Args:
            window: the rslearn window to write to.
            modality: the modality spec; its band order names the bands.
            array: float32 (C, H, W) array aligned to the window's
                bounds/projection, with C == len(modality.band_order).
            nodata_value: nodata value recorded in the GeoTIFF.
            layer_name: layer name; defaults to the modality name.

        Returns:
            the directory the raster was written to.

        Raises:
            ValueError: if the array shape does not match the modality band
                count or the window bounds.
        """
        if layer_name is None:
            layer_name = modality.name
        bands = modality.band_order
        expected_shape = (
            len(bands),
            window.bounds[3] - window.bounds[1],
            window.bounds[2] - window.bounds[0],
        )
        if array.shape != expected_shape:
            raise ValueError(
                f"window {window.group}/{window.name}: expected array of shape "
                f"{expected_shape}, got {array.shape}"
            )

        raster_dir = window.get_raster_dir(layer_name, bands)
        self._raster_format.encode_raster(
            raster_dir,
            window.projection,
            window.bounds,
            RasterArray(
                chw_array=array.astype(np.float32),
                time_range=window.time_range,
                metadata=RasterMetadata(nodata_value=nodata_value),
            ),
        )
        window.mark_layer_completed(layer_name)
        logger.debug(
            f"Wrote {layer_name} layer for window {window.group}/{window.name} "
            f"to {raster_dir}"
        )
        return raster_dir
