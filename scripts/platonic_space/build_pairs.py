"""Sample paired pixel triples (AEF, OlmoEarth v1.3, TESSERA v2; 2024 products) for vec2vec.

Source (read-only): piperw's 2024 product reads,
  /weka/dfive-default/piperw/shared_embedding_space/pilot2024/out/<i>.npz
  aef (64,128,128) int8 north-up, oe (128,128,128) int8 (-128 = nodata for both),
  tess (128,128,128) int8 + tess_scale (128,128) float32 (non-finite scale = nodata).
Dequantised as piperw's student/pack.py does: AEF and OE with the power scheme
sign(f) * f^2, f = x / 127.5; TESSERA as int8 * scale. The split is piperw's (GEA-17's
0.5-degree test blocks, plus a salted 1/8 of the other blocks for val), so our test tiles
are the student's.

Output <out_dir>/{train,val,test}.npz:
  aef [N,64] / oe [N,128] / tess [N,128] float16 (tess NaN where TESSERA has no data),
  tile [N] int32 (footprint id i), lon / lat [N] float32 (tile centre).
Only pixels where AEF and OE are both valid are sampled, up to --px per tile.

Usage: python build_pairs.py <out_dir> [--px 1024] [--workers 0] [--clean_ids ids.npy]
"""

import argparse
import glob
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, "/weka/dfive-default/agent/piperbots/2026-10-26-piperw-rs-imagenet")
from gea_grid import block_of, gea_test  # noqa: E402

SRC = "/weka/dfive-default/piperw/shared_embedding_space/pilot2024/out"


def split_of(lon: float, lat: float) -> str:
    """Piperw's student/pack.py split_of, verbatim."""
    b = int(block_of(lon, lat))
    if gea_test(np.array([b]))[0]:
        return "test"
    h = int.from_bytes(
        hashlib.sha256(f"emb2024-val:{b}".encode()).digest()[:8], "little"
    )
    return "val" if h % 1000 < 125 else "train"


def power_dequant(x: np.ndarray) -> np.ndarray:
    """AEF / OE int8 -> float, AEF's power scheme."""
    f = x.astype(np.float32) / 127.5
    return np.sign(f) * f * f


def one(args: tuple[str, int, int]) -> dict | None:
    """Sample one tile's valid pixels; None when it has none."""
    path, px, seed = args
    i = int(os.path.basename(path).removesuffix(".npz"))
    z = np.load(path)
    aef, oe = z["aef"], z["oe"]
    ok = (aef[0] != -128) & (oe[0] != -128)
    flat = np.flatnonzero(ok.ravel())
    if not len(flat):
        return None
    rng = np.random.default_rng(seed + i)
    pick = np.sort(rng.choice(flat, size=min(px, len(flat)), replace=False))
    sc = z["tess_scale"].ravel()[pick]
    tess = z["tess"].reshape(128, -1)[:, pick].astype(np.float32) * sc[None]
    tess[:, ~np.isfinite(sc)] = np.nan
    lon, lat = float(z["lon"]), float(z["lat"])
    return {
        "aef": power_dequant(aef.reshape(64, -1)[:, pick]).T.astype(np.float16),
        "oe": power_dequant(oe.reshape(128, -1)[:, pick]).T.astype(np.float16),
        "tess": tess.T.astype(np.float16),
        "tile": np.full(len(pick), i, np.int32),
        "lon": np.full(len(pick), lon, np.float32),
        "lat": np.full(len(pick), lat, np.float32),
        "split": split_of(lon, lat),
    }


def main() -> None:
    """Build the train / val / test pair files."""
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    p.add_argument("--px", type=int, default=1024, help="pixels sampled per tile")
    p.add_argument(
        "--workers", type=int, default=0, help="0 = every CPU this process sees"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--clean_ids", default=None, help="npy of allowed tile ids (others dropped)"
    )
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    paths = sorted(glob.glob(f"{SRC}/*.npz"))
    if a.clean_ids:
        keep = set(np.load(a.clean_ids).tolist())
        n0 = len(paths)
        paths = [q for q in paths if int(os.path.basename(q)[:-4]) in keep]
        print(f"clean_ids: kept {len(paths)} of {n0} tiles", flush=True)
    if a.limit:
        paths = paths[: a.limit]
    workers = a.workers or min(64, len(os.sched_getaffinity(0)))
    parts: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    with ProcessPoolExecutor(workers) as ex:
        for k, r in enumerate(
            ex.map(one, [(q, a.px, a.seed) for q in paths], chunksize=8)
        ):
            if r is not None:
                parts[r.pop("split")].append(r)
            if k % 1000 == 0:
                print(f"{k}/{len(paths)}", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    for split, rs in parts.items():
        arrs = {key: np.concatenate([r[key] for r in rs]) for key in rs[0]}
        tess_ok = np.isfinite(arrs["tess"][:, 0]).mean()
        tmp = f"{a.out_dir}/{split}.tmp.npz"
        np.savez(tmp, **arrs)
        os.replace(tmp, f"{a.out_dir}/{split}.npz")
        print(
            f"{split}: {len(rs)} tiles, {len(arrs['aef'])} px, tess valid {tess_ok:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
