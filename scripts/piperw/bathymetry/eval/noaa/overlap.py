"""Pretraining samples inside BlueTopo tiles, by tile resolution, and how many are shallow per GEBCO."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyogrio
import shapely

H5 = Path(
    "/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828"
)
GEBCO = "/weka/dfive-default/agent/piperbots/2026-10-26-piperw-bathymetry-target/gebco_patches.npy"

gdf = pyogrio.read_dataframe(next(Path(".").glob("BlueTopo_Tile_Scheme_*.gpkg")))
gdf = gdf[gdf["GeoTIFF_Link"].notna()]
print("delivered tiles:", len(gdf), dict(Counter(gdf["Resolution"])))
latlon = np.load(H5 / "latlon_distribution.npy")
pts = shapely.points(latlon[:, 1], latlon[:, 0])
tree = shapely.STRtree(gdf.geometry.values)
pi, ti = tree.query(pts, predicate="within")
first = {}
for p, t in zip(pi, ti):
    first.setdefault(p, t)
idx = np.array(sorted(first))
res = np.array(gdf["Resolution"].values)[[first[i] for i in idx]]
geb = np.load(GEBCO, mmap_mode="r")
shallow = ((geb[idx] < 0) & (geb[idx] >= -30)).any(axis=(1, 2))
out = {
    "n_samples": int(len(latlon)),
    "in_bluetopo": int(len(idx)),
    "in_bluetopo_with_gebco_0_30m_block": int(shallow.sum()),
    "by_resolution": {r: int((res == r).sum()) for r in sorted(set(res))},
    "shallow_by_resolution": {
        r: int(((res == r) & shallow).sum()) for r in sorted(set(res))
    },
}
all_shallow = ((geb < 0) & (geb >= -30)).any(axis=(1, 2))
out["all_samples_with_gebco_0_30m_block"] = int(all_shallow.sum())
print(json.dumps(out, indent=1))
json.dump(out, open("overlap.json", "w"), indent=1)
np.save("bluetopo_sample_idx.npy", idx)
