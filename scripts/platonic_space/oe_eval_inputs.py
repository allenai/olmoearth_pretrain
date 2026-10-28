"""Read the public OlmoEarth v1.3 product at every eval window (the translator's OE input basis).

The translator trains on the v1.3 *product* (int8 annual zarr), so the eval windows need the
same product, not a forward pass (Ilse's harness exports are in a different basis: cosine .02).

- AEF year-aligned tasks: the window's labelled pixel (emb_row, emb_col) in the window's year,
  read from source.coop `olmoearth-v1-3-embeddings/embeddings.zarr`. The eval windows sit on
  the zarr's 10 m UTM grid; row / col / zone per window were precomputed in
  investigation/windows/<task>.parquet (see investigation/FINDINGS.md, B).
- PASTIS: the already-materialized `pastis_year_aligned_oe13cdnzarr` layer (float32,
  dequantised, nodata -1.0; bit-identical to the zarr), whole 128x128 windows.

Writes <out_dir>/<task>_year_aligned.npz: ids (window_id "<group>/<name>"), emb float32
(power-dequantised; NaN where the product has no data), has_obs bool. AEF-task rows are
written in parts of --part rows first, so a restarted job skips the finished parts.

Usage: python oe_eval_inputs.py <task> [--threads 48]
"""

import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

SESSION = "/weka/dfive-default/agent/hadriens/2026-10-28-hadriens-platonic-space"
WINDOWS = f"{SESSION}/investigation/windows"
OUT = "/weka/dfive-default/hadriens/platonic_space/eval_inputs/oe13_product"
PASTIS_LAYER = "/weka/dfive-default/olmoearth/eval_datasets/pastis_year_aligned_oe13cdnzarr/windows"
YEAR0 = 2017


def power_dequant(q: np.ndarray) -> np.ndarray:
    """int8 product -> float (AEF's power scheme, the zarr's `signed_power`)."""
    f = q.astype(np.float32) / 127.5
    return np.sign(f) * f * f


def read_points(
    task: str, threads: int, part: int, out_dir: str, limit: int = 0
) -> None:
    """Labelled pixel of every window of an AEF task, in parts, then one npz."""
    import zarr
    from obstore.store import S3Store

    store = S3Store(
        "olmoearth",
        endpoint="https://data.source.coop",
        skip_signature=True,
        region="us-west-2",
    )
    oe = zarr.open_group(
        zarr.storage.ObjectStore(store, read_only=True),
        path="olmoearth-v1-3-embeddings/embeddings.zarr",
        mode="r",
    )
    arrs = {z: oe[f"utm{z:02d}"]["embeddings"] for z in range(1, 61)}
    t = pd.read_parquet(f"{WINDOWS}/{task}.parquet")
    if limit:
        t = t.iloc[:limit]
    jobs = list(zip(t.zone_crs, t.year, t.oe_row_crs, t.oe_col_crs))

    def one(job: tuple[int, int, int, int]) -> np.ndarray:
        z, y, r, c = job
        for attempt in range(5):
            try:
                return np.asarray(arrs[int(z)][int(y) - YEAR0, :, int(r), int(c)])
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)

    parts_dir = f"{out_dir}/parts/{task}"
    os.makedirs(parts_dir, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(threads) as ex:
        for p0 in range(0, len(jobs), part):
            f = f"{parts_dir}/{p0:07d}.npy"
            if os.path.exists(f):
                continue
            q = np.stack(list(ex.map(one, jobs[p0 : p0 + part])))
            np.save(f + ".tmp.npy", q)
            os.replace(f + ".tmp.npy", f)
            print(
                f"{task} {p0 + len(q)}/{len(jobs)} {time.time() - t0:.0f}s valid {(q != -128).all(1).mean():.3f}",
                flush=True,
            )
    q = np.concatenate(
        [np.load(f"{parts_dir}/{p0:07d}.npy") for p0 in range(0, len(jobs), part)]
    )
    ok = (q != -128).all(1)
    emb = power_dequant(q)
    emb[~ok] = np.nan
    write(out_dir, task, t.window_id.to_numpy(), emb, ok)


def read_pastis(out_dir: str, threads: int, limit: int = 0) -> None:
    """Whole 128x128 windows from the materialized oe13cdnzarr layer."""
    import rasterio

    t = pd.read_parquet(f"{WINDOWS}/pastis.parquet")
    if limit:
        t = t.iloc[:limit]

    def one(w: str) -> np.ndarray:
        (tif,) = glob.glob(f"{PASTIS_LAYER}/{w}/layers/olmoearth_emb/*/geotiff.tif")
        with rasterio.open(tif) as ds:
            return ds.read().astype(np.float32).transpose(1, 2, 0)

    with ThreadPoolExecutor(threads) as ex:
        emb = np.stack(list(ex.map(one, t.window_id)))
    ok = ~(emb == -1.0).all(-1)
    emb[~ok] = np.nan
    write(out_dir, "pastis", t.window_id.to_numpy(), emb.astype(np.float16), ok)


def write(
    out_dir: str, task: str, ids: np.ndarray, emb: np.ndarray, ok: np.ndarray
) -> None:
    """Scorer-layout npz, atomically."""
    out = f"{out_dir}/{task}_year_aligned.npz"
    np.savez(out + ".tmp.npz", ids=ids.astype(str), emb=emb, has_obs=ok)
    os.replace(out + ".tmp.npz", out)
    print(f"wrote {out}: {len(ids)} windows, has_obs {ok.mean():.4f}", flush=True)


def main() -> None:
    """Read one task's OE inputs."""
    p = argparse.ArgumentParser()
    p.add_argument("task")
    p.add_argument("--threads", type=int, default=48)
    p.add_argument("--part", type=int, default=4000)
    p.add_argument("--out_dir", default=OUT)
    p.add_argument("--limit", type=int, default=0, help="first N windows (smoke tests)")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.task == "pastis":
        read_pastis(a.out_dir, a.threads, a.limit)
    else:
        read_points(a.task, a.threads, a.part, a.out_dir, a.limit)


if __name__ == "__main__":
    main()
