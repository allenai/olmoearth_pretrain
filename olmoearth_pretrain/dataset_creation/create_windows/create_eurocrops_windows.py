"""Create EuroCrops windows.

We use the lower-level create_window function instead of
create_windows_with_highres_time since we need to set the timestamp to match the time
of the EuroCrops data.

The EuroCrops data must first be ingested into a tile store. We did this using the
osm_sampling dataset.
"""

import multiprocessing
import random
import sys
from datetime import UTC, datetime

import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

from olmoearth_pretrain.dataset.parse import WindowMetadata
from olmoearth_pretrain.dataset_creation.create_windows.util import create_window

RESOLUTION = 10
TILE_SIZE = 256


def option_to_tile(geojson_fname: UPath) -> tuple[str, int, int, datetime]:
    """Convert each EuroCrops example to a pre-training dataset 256x256 tile."""
    features = GeojsonVectorFormat().decode_from_file(geojson_fname)
    feature = random.choice(features)
    wgs84_geom = STGeometry(
        feature.geometry.projection,
        feature.geometry.shp.centroid,
        None,
    ).to_projection(WGS84_PROJECTION)
    dst_proj = get_utm_ups_projection(
        wgs84_geom.shp.x, wgs84_geom.shp.y, RESOLUTION, -RESOLUTION
    )
    dst_geom = wgs84_geom.to_projection(dst_proj)
    col = int(dst_geom.shp.x) // TILE_SIZE
    row = int(dst_geom.shp.y) // TILE_SIZE
    # The parent folder name is the year for this data.
    # So put timestamp as the middle of that year.
    ts = datetime(int(geojson_fname.parent.name), 7, 1, tzinfo=UTC)

    return (str(dst_proj.crs), col, row, ts)


if __name__ == "__main__":
    # This should be path like /weka/dfive-default/helios/dataset_creation/presto/tiles/eurocrops/
    eurocrops_path = UPath(sys.argv[1])
    ds_path = UPath(sys.argv[2])
    num_windows = int(sys.argv[3])

    # Get all of the GeoJSON file options.
    options = list(eurocrops_path.glob("*/*.geojson"))

    # Pick num_windows options.
    # We may end up with fewer windows but it is approximate.
    options = random.sample(options, num_windows)

    # Now for each option we need to convert it to a WindowMetadata.
    # First we get (CRS string, col, row, time) from each option and make sure they are
    # unique.
    p = multiprocessing.Pool(64)
    outputs = p.imap_unordered(option_to_tile, options)
    unique_tiles = set(tqdm.tqdm(outputs, total=len(options), desc="Getting tiles"))

    # Finally we can create these windows.
    jobs = [
        dict(ds_path=ds_path, metadata=WindowMetadata(crs, RESOLUTION, col, row, ts))
        for crs, col, row, ts in unique_tiles
    ]
    outputs = star_imap_unordered(p, create_window, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs), desc="Creating windows"):
        pass
    p.close()
