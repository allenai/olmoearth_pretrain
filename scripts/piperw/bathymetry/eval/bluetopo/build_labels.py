"""BlueTopo elevation on the 10 m pixel grid of shallow US pretraining samples.

Writes labels.npz: per sample, BlueTopo elevation and uncertainty (m, NaN = no data) on the
sample's 128x128 10 m grid, plus the finest contributing tile resolution.

Sample grid: latlon_distribution.npy holds the centre of each sample's parent 256x256 UTM
tile (10 m); sample n is quadrant n % 4 (as in build_gebco_patches.py). The parent tile is
placed in the UTM zone of its centre.

  python build_labels.py [max_samples | all]

"all" drops the GEBCO shallow-water filter (every sample centred in a BlueTopo tile) and
writes labels_all.npz. Links whose tile version was replaced on S3 are re-resolved by listing
the tile's prefix.
"""

import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyogrio
import rasterio
import shapely
from pyproj import CRS, Transformer
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds

HERE = Path(
    "/weka/dfive-default/agent/piperbots/2026-10-29-piperw-bathymetry-direct-supervision/bluetopo_eval"
)
NOAA = HERE.parent / "noaa"
H5 = Path(
    "/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828"
)
GEBCO = "/weka/dfive-default/agent/piperbots/2026-10-26-piperw-bathymetry-target/gebco_patches.npy"
PIX, SUB = 10.0, 128
GDAL_ENV = dict(
    AWS_NO_SIGN_REQUEST="YES",
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_HTTP_MAX_RETRY="5",
    GDAL_HTTP_RETRY_DELAY="2",
)


def utm_crs(lat: float, lon: float) -> CRS:
    """UTM zone CRS containing (lat, lon)."""
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def sample_grid(idx: int, lat: float, lon: float) -> tuple[CRS, rasterio.Affine]:
    """CRS and 10 m affine transform of sample idx's 128x128 quadrant."""
    crs = utm_crs(lat, lon)
    e, n = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    q = idx % 4
    west = e - SUB * PIX + (q % 2) * SUB * PIX
    north = n + SUB * PIX - (q // 2) * SUB * PIX
    return crs, from_origin(west, north, PIX, PIX)


BUCKET = "https://noaa-ocs-nationalbathymetry-pds.s3.amazonaws.com"
_resolved: dict[str, str] = {}


def current_link(link: str) -> str:
    """The newest GeoTIFF under the link's tile prefix (the tile scheme can be stale)."""
    if link not in _resolved:
        prefix = "/".join(link.split("/")[-3:-1]) + "/"
        with urllib.request.urlopen(  # nosec
            f"{BUCKET}/?list-type=2&prefix={prefix}", timeout=60
        ) as f:
            keys = re.findall(r"<Key>([^<]+\.tiff?)</Key>", f.read().decode())
        _resolved[link] = f"{BUCKET}/{sorted(keys)[-1]}" if keys else link
    return _resolved[link]


def open_tile(link: str):
    """Open a remote BlueTopo COG."""
    try:
        return rasterio.open("/vsicurl/" + link)
    except rasterio.errors.RasterioIOError:
        return rasterio.open("/vsicurl/" + current_link(link))


def read_sample(idx, lat, lon, links, resolutions):
    """BlueTopo elevation and uncertainty resampled onto one sample's grid."""
    crs, transform = sample_grid(idx, lat, lon)
    bounds = (
        transform.c,
        transform.f - SUB * PIX,
        transform.c + SUB * PIX,
        transform.f,
    )
    elev = np.full((SUB, SUB), np.nan, np.float32)
    unc = np.full((SUB, SUB), np.nan, np.float32)
    best_res = np.inf
    # finest tiles last so they overwrite coarser ones
    for link, res in sorted(zip(links, resolutions), key=lambda x: -x[1]):
        with rasterio.Env(**GDAL_ENV), open_tile(link) as src:
            left, b, r, t = transform_bounds(crs, src.crs, *bounds)
            win = (
                src.window(left - 50, b - 50, r + 50, t + 50)
                .round_offsets()
                .round_lengths()
            )
            data = (
                src.read([1, 2], window=win, boundless=True, masked=True)
                .astype(np.float32)
                .filled(np.nan)
            )
            src_transform = src.window_transform(win)
            for band, out in ((0, elev), (1, unc)):
                dst = np.full((SUB, SUB), np.nan, np.float32)
                reproject(
                    data[band],
                    dst,
                    src_transform=src_transform,
                    src_crs=src.crs,
                    src_nodata=np.nan,
                    dst_transform=transform,
                    dst_crs=crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.average,
                )
                fill = np.isfinite(dst)
                out[fill] = dst[fill]
            if np.isfinite(data[0]).any():
                best_res = min(best_res, res)
    return elev, unc, best_res, crs.to_epsg(), np.array(transform)[:6]


def main() -> None:
    """Build BlueTopo label patches for the pretraining samples inside BlueTopo tiles."""
    latlon = np.load(H5 / "latlon_distribution.npy")
    geb = np.load(GEBCO, mmap_mode="r")
    idx = np.load(NOAA / "bluetopo_sample_idx.npy")
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg != "all":
        idx = idx[((geb[idx] < 0) & (geb[idx] >= -30)).any(axis=(1, 2))]
    if arg not in (None, "all"):
        idx = np.random.default_rng(0).choice(idx, int(sys.argv[1]), replace=False)
    gdf = pyogrio.read_dataframe(next(NOAA.glob("BlueTopo_Tile_Scheme_*.gpkg")))
    gdf = gdf[gdf["GeoTIFF_Link"].notna()].reset_index(drop=True)
    tree = shapely.STRtree(gdf.geometry.values)
    links = gdf["GeoTIFF_Link"].values
    res_m = gdf["Resolution"].str.rstrip("m").astype(float).values

    def job(i):
        lat, lon = map(float, latlon[i])
        crs, transform = sample_grid(i, lat, lon)
        box = shapely.box(
            *transform_bounds(
                crs,
                "EPSG:4326",
                transform.c,
                transform.f - SUB * PIX,
                transform.c + SUB * PIX,
                transform.f,
            )
        )
        hits = tree.query(box, predicate="intersects")
        try:
            return i, read_sample(i, lat, lon, links[hits], res_m[hits])
        except Exception as e:  # noqa: BLE001
            print(f"sample {i}: {e!r}", flush=True)
            return i, None

    out = {k: [] for k in ["idx", "elev", "unc", "res", "epsg", "transform"]}
    with ThreadPoolExecutor(32) as pool:
        for k, (i, r) in enumerate(pool.map(job, idx)):
            if (k + 1) % 200 == 0:
                print(f"{k + 1}/{len(idx)}", flush=True)
            if r is None:
                continue
            elev, unc, res, epsg, transform = r
            for key, v in zip(
                out,
                (
                    i,
                    elev.astype(np.float16),
                    unc.astype(np.float16),
                    res,
                    epsg,
                    transform,
                ),
            ):
                out[key].append(v)
    name = "labels.npz" if arg is None else f"labels_{arg}.npz"
    np.savez(HERE / name, **{k: np.stack(v) for k, v in out.items()})
    valid = np.isfinite(np.stack(out["elev"]).astype(np.float32))
    print(
        f"wrote {name}: {len(out['idx'])} samples, mean valid pixel fraction {valid.mean():.3f}"
    )


if __name__ == "__main__":
    main()
