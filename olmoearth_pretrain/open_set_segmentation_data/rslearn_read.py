"""Helpers to read local rslearn datasets (window metadata + label layers)."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from rslearn.utils.geometry import Projection
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath


def iter_windows_metadata(ds_path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield each window's metadata.json dict, augmented with _group/_name/_path.

    Reads files directly (fast) rather than instantiating a Dataset. Works for the
    standard on-disk layout windows/<group>/<name>/metadata.json.
    """
    root = Path(ds_path) / "windows"
    for group in sorted(root.iterdir()):
        if not group.is_dir():
            continue
        for w in sorted(group.iterdir()):
            mp = w / "metadata.json"
            if mp.exists():
                with open(mp) as f:
                    md = json.load(f)
                md["_group"] = group.name
                md["_name"] = w.name
                md["_path"] = str(w)
                yield md


def read_label_vector(window_path: str | Path, layer: str = "label") -> list:
    """Read a vector label layer's features (as rslearn Feature objects)."""
    path = UPath(window_path) / "layers" / layer / "data.geojson"
    return GeojsonVectorFormat().decode_from_file(path)


def read_label_raster(
    window_path: str | Path,
    projection: Projection,
    bounds: tuple[int, int, int, int],
    layer: str = "label_raster",
    band: str = "label",
) -> RasterArray:
    """Read a raster label layer into a RasterArray aligned to projection/bounds."""
    raster_dir = UPath(window_path) / "layers" / layer / band
    return GeotiffRasterFormat().decode_raster(raster_dir, projection, bounds)
