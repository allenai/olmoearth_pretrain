"""Create res_10 (+res_160) rslearn windows for MP-16 pretraining points.

Reuses henry's create_window (new storage= API) but bypasses the NAIP-based
time selection: assigns each point a random RECENT center time (shared per
coarse 160 m tile) so Sentinel-2/-1/Landsat all exist. No high-res/NAIP.
"""

import argparse
import json
import random
from datetime import UTC, datetime, timedelta

import shapely
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from upath import UPath

from olmoearth_pretrain.dataset.utils import WindowMetadata
from olmoearth_pretrain.dataset_creation.constants import WINDOW_SIZE
from olmoearth_pretrain.dataset_creation.create_windows.util import create_window

RES = 10
COARSE_FACTOR = 16  # 160m / 10m
START_TIME = datetime(2021, 1, 1, tzinfo=UTC)
END_TIME = datetime(2024, 1, 1, tzinfo=UTC)


def tile_for(lon: float, lat: float) -> tuple[str, int, int]:
    proj = get_utm_ups_projection(lon, lat, RES, -RES)
    g = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), None).to_projection(proj)
    return str(proj.crs), int(g.shp.x) // WINDOW_SIZE, int(g.shp.y) // WINDOW_SIZE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_path", required=True)
    ap.add_argument("--fname", required=True, help="JSON list of [lon, lat]")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ds_path = UPath(args.ds_path)
    lonlats = json.load(open(args.fname))

    tiles = sorted({tile_for(lon, lat) for lon, lat in lonlats})
    coarse_time: dict[tuple[str, int, int], datetime] = {}
    secs = int((END_TIME - START_TIME).total_seconds())

    n = 0
    for crs, col, row in tiles:
        ckey = (crs, col // COARSE_FACTOR, row // COARSE_FACTOR)
        if ckey not in coarse_time:
            t = START_TIME + timedelta(seconds=rng.randint(0, secs))
            coarse_time[ckey] = datetime(t.year, t.month, t.day, tzinfo=UTC)
        meta = WindowMetadata(
            crs=crs, resolution=RES, col=col, row=row, time=coarse_time[ckey]
        )
        create_window(ds_path, meta)  # creates res_10 (+ res_160)
        n += 1
    print(
        f"created windows for {n} distinct 10m tiles "
        f"from {len(lonlats)} points ({len(coarse_time)} coarse tiles)"
    )


if __name__ == "__main__":
    main()
