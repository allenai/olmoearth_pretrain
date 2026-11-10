"""OlmoEarth inputs for the TerraMesh-Masks eval tiles, from TerraMesh's own S2L2A / S1RTC.

  python -m olmoearth_pretrain.evals.terramesh.prep --masks <dir> --raw <dir> --out <dir>

``--masks`` holds the eval's mask GeoTIFFs (<file_id>.tif). ``--raw`` holds the TerraMesh
val shards (HF ibm-esa-geospatial/TerraMesh, CC-BY-SA) as val/{S2L2A,S1RTC}/<tar> and an
eval_shards.csv mapping file_id -> tar. Each mask covers 256 x 256 px of a 264 x 264
TerraMesh sample; the crop offset comes from the zarr's pixel-centre coordinates against
the mask's transform. Writes <out>/<file_id>.npz with

  sentinel2_l2a  (256, 256, 1, 12) uint16, OlmoEarth band order, raw L2A counts
  sentinel1      (256, 256, 1, 2) float32 dB (vv, vh), MajorTOM tiles only
  timestamp      [day, month 0-11, year] of the S2 acquisition
  no_data        (256, 256) bool, from the S2 cloud mask's no_data class
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

SIZE = 256
# Modality.SENTINEL2_L2A band order.
S2_ORDER = [
    "B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09",
]  # fmt: skip
NO_DATA_CLASS = 6


def open_member(tar: tarfile.TarFile, name: str) -> Any:
    """Open a zipped zarr inside a shard."""
    import zarr

    member = tar.extractfile(name)
    if member is None:
        raise KeyError(name)
    store = zarr.storage.ZipStore(io.BytesIO(member.read()), mode="r")
    return zarr.open_group(store, mode="r")


def offsets(group: Any, mask_path: str) -> tuple[int, int]:
    """Row and column of the mask's top-left pixel in the sample."""
    import rasterio

    with rasterio.open(mask_path) as src:
        left, top = src.transform.c, src.transform.f
    x, y = np.asarray(group["x"]), np.asarray(group["y"])
    col = (left + 5 - x[0]) / 10
    row = (y[0] - (top - 5)) / 10
    if abs(col - round(col)) > 0.05 or abs(row - round(row)) > 0.05:
        raise ValueError(f"mask not on the sample's pixel grid ({row}, {col})")
    return int(round(row)), int(round(col))


def main() -> None:
    """Write one input file per eval tile and a summary next to them."""
    p = argparse.ArgumentParser()
    p.add_argument("--masks", required=True)
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    shards = pd.read_csv(os.path.join(a.raw, "eval_shards.csv"))
    by_tar = defaultdict(list)
    for file_id, tar in zip(shards.file_id, shards.tar):
        by_tar[tar].append(file_id)
    report: dict = {"written": 0, "with_s1": 0, "problems": []}
    for tar_name, file_ids in sorted(by_tar.items()):
        s2_tar = tarfile.open(os.path.join(a.raw, "val", "S2L2A", tar_name))
        s1_path = os.path.join(a.raw, "val", "S1RTC", tar_name)
        s1_tar = tarfile.open(s1_path) if os.path.exists(s1_path) else None
        for file_id in file_ids:
            mask_path = os.path.join(a.masks, f"{file_id}.tif")
            try:
                g = open_member(s2_tar, f"{file_id}.zarr.zip")
                r, c = offsets(g, mask_path)
                names = [str(b) for b in g["band"][:]]
                bands = np.asarray(g["bands"][0])[:, r : r + SIZE, c : c + SIZE]
                s2 = np.stack([bands[names.index(b)] for b in S2_ORDER], axis=-1)
                cloud = np.asarray(g["cloud_mask"][0])[r : r + SIZE, c : c + SIZE]
                t = pd.Timestamp(int(np.asarray(g["time"])[0]))
                out = {
                    "sentinel2_l2a": np.clip(s2, 0, None).astype(np.uint16)[
                        :, :, None, :
                    ],
                    "timestamp": np.array([t.day, t.month - 1, t.year], dtype=np.int32),
                    "no_data": cloud == NO_DATA_CLASS,
                }
                if s1_tar is not None:
                    s1g = open_member(s1_tar, f"{file_id}.zarr.zip")
                    r1, c1 = offsets(s1g, mask_path)
                    s1 = np.asarray(s1g["bands"][0], dtype=np.float32)[
                        :, r1 : r1 + SIZE, c1 : c1 + SIZE
                    ]
                    if np.isfinite(s1).mean() > 0.99:
                        out["sentinel1"] = np.nan_to_num(np.moveaxis(s1, 0, -1))[
                            :, :, None, :
                        ]
                        report["with_s1"] += 1
                np.savez(os.path.join(a.out, f"{file_id}.npz"), **out)
                report["written"] += 1
            except (KeyError, ValueError) as e:
                report["problems"].append(
                    f"{file_id}: {type(e).__name__}: {str(e)[:100]}"
                )
    json.dump(report, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(
        report["written"], "written,", report["with_s1"], "with S1,",
        len(report["problems"]), "problems",
    )  # fmt: skip


if __name__ == "__main__":
    main()
