"""Model-free per-pixel input for the labelled samples: the raw S2 median.

Writes inputs.npz (same sample order as labels.npz): s2_median [N, 128, 128, 12] float16
(raw reflectance, median over valid timesteps).

  python inputs.py [labels file]
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from upath import UPath

from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality
from olmoearth_pretrain.data.dataset import GetItemArgs, OlmoEarthDataset

HERE = Path(
    "/weka/dfive-default/agent/piperbots/2026-10-29-piperw-bathymetry-direct-supervision/bluetopo_eval"
)
H5 = (
    "/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828"
)


def main() -> None:
    """Save the S2 inputs for the labelled samples."""
    labels = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "labels.npz"
    idx = np.load(labels)["idx"]
    ds = OlmoEarthDataset(
        h5py_dir=UPath(H5),
        training_modalities=[Modality.SENTINEL2_L2A.name],
        dtype=np.float32,
        normalize=False,
    )

    def job(i):
        _, s = ds[GetItemArgs(idx=int(i), patch_size=4, sampled_hw_p=32)]
        s2 = s.sentinel2_l2a.astype(np.float32)  # H W T C
        s2[(s2 == MISSING_VALUE) | (s2 <= 0)] = np.nan
        return np.nanmedian(s2, axis=2).astype(np.float16)

    with ThreadPoolExecutor(16) as pool:
        res = list(pool.map(job, idx))
    out = labels.with_name(labels.name.replace("labels", "inputs"))
    np.savez(out, idx=idx, s2_median=np.stack(res))
    print(f"wrote {out.name}: {len(idx)} samples")


if __name__ == "__main__":
    main()
