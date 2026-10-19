"""Rewrite AllCaptures h5 files with small chunks (same arrays, different block layout).

The AllCaptures corpus stores every array as ONE zstd chunk (the whole 128 x 128 x
all-captures tile), so a sample that keeps a small crop or a few captures still reads
and decompresses the whole array. This rewrites each ``[H, W, T, C]`` array with chunks
of ``SPATIAL x SPATIAL x TIME x C`` (zstd level 3, as the source); every other dataset,
group and attribute is copied unchanged. Each written file is checked array-by-array
against its source.

``subset`` mode builds a benchmark pair from ``--n`` random samples: ``<out>/orig``
(symlinks to the source files) and ``<out>/rechunk`` (rewritten copies), both laid out
as a dataset dir (``.../<modalities>/<n>/sample_{i}.h5`` + metadata + latlons).

Usage: ``python rechunk_allcap_h5.py subset --src <h5py_dir> --out <dir> --n 12000``.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import pandas as pd

SPATIAL = 32
TIME = 16
ZSTD_LEVEL = 3


def _chunks(shape: tuple[int, ...]) -> tuple[int, ...] | None:
    """Chunk shape for a ``[H, W, T, C]`` array, or None to copy as stored."""
    if len(shape) != 4:
        return None
    h, w, t, c = shape
    return (min(SPATIAL, h), min(SPATIAL, w), max(1, min(TIME, t)), c)


def rechunk_file(src: Path, dst: Path) -> dict:
    """Write ``dst`` = ``src`` with small chunks; verify every dataset matches."""
    tmp = dst.with_suffix(".h5.tmp")
    with h5py.File(src, "r") as fin, h5py.File(tmp, "w") as fout:
        fout.attrs.update(fin.attrs)

        def copy(name: str, obj: h5py.HLObject) -> None:
            if isinstance(obj, h5py.Group):
                fout.require_group(name).attrs.update(obj.attrs)
                return
            chunks = _chunks(obj.shape)
            if chunks is None:
                fin.copy(obj, fout, name=name)
                return
            out = fout.create_dataset(
                name,
                data=obj[()],
                chunks=chunks,
                **hdf5plugin.Zstd(clevel=ZSTD_LEVEL),
            )
            out.attrs.update(obj.attrs)

        fin.visititems(copy)
    # Verify before the rename, so a dst file is always a checked copy.
    with h5py.File(src, "r") as a, h5py.File(tmp, "r") as b:
        names: list[str] = []
        a.visititems(
            lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None
        )
        for name in names:
            x, y = a[name], b[name]
            if x.shape != y.shape or x.dtype != y.dtype:
                raise ValueError(f"{src}:{name} shape/dtype differ")
            if not np.array_equal(x[()], y[()], equal_nan=x.dtype.kind == "f"):
                raise ValueError(f"{src}:{name} values differ")
    tmp.rename(dst)
    return {"src_mb": src.stat().st_size / 1e6, "dst_mb": dst.stat().st_size / 1e6}


def _subset_one(job: tuple[int, int, str, str, str]) -> dict:
    i, source_index, src_dir, orig_dir, rechunk_dir = job
    src = Path(src_dir) / f"sample_{source_index}.h5"
    link = Path(orig_dir) / f"sample_{i}.h5"
    if not link.exists():
        os.symlink(src, link)
    dst = Path(rechunk_dir) / f"sample_{i}.h5"
    if dst.exists():
        return {"src_mb": src.stat().st_size / 1e6, "dst_mb": dst.stat().st_size / 1e6}
    return rechunk_file(src, dst)


def subset(src_dir: Path, out: Path, n: int, workers: int, seed: int) -> None:
    """Build ``orig`` (symlinks) and ``rechunk`` dataset dirs of ``n`` random samples."""
    metadata = pd.read_csv(src_dir / "sample_metadata.csv")
    latlons = np.load(src_dir / "latlon_distribution.npy")
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(len(metadata), n, replace=False))
    tail = Path(src_dir.parent.parent.name) / src_dir.parent.name / str(n)
    dirs = {}
    for variant in ("orig", "rechunk"):
        d = out / variant / tail
        d.mkdir(parents=True, exist_ok=True)
        sub = metadata.iloc[picked].reset_index(drop=True)
        if "sample_index" in sub.columns:
            sub["sample_index"] = np.arange(n)
        sub.to_csv(d / "sample_metadata.csv", index=False)
        np.save(d / "latlon_distribution.npy", latlons[picked])
        dirs[variant] = d
    jobs = [
        (i, int(s), str(src_dir), str(dirs["orig"]), str(dirs["rechunk"]))
        for i, s in enumerate(picked)
    ]
    t0 = time.perf_counter()
    sizes = []
    with Pool(workers) as pool:
        for k, size in enumerate(pool.imap_unordered(_subset_one, jobs, chunksize=4)):
            sizes.append(size)
            if (k + 1) % 500 == 0:
                print(
                    json.dumps(
                        {"done": k + 1, "s": round(time.perf_counter() - t0, 1)}
                    ),
                    flush=True,
                )
    src_mb = float(np.sum([s["src_mb"] for s in sizes]))
    dst_mb = float(np.sum([s["dst_mb"] for s in sizes]))
    print(
        json.dumps(
            {
                "n": n,
                "orig": str(dirs["orig"]),
                "rechunk": str(dirs["rechunk"]),
                "src_mb_per_file": round(src_mb / n, 1),
                "dst_mb_per_file": round(dst_mb / n, 1),
                "size_ratio": round(dst_mb / src_mb, 3),
                "s": round(time.perf_counter() - t0, 1),
            }
        ),
        flush=True,
    )


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["subset"])
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=12000)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    subset(args.src, args.out, args.n, args.workers, args.seed)


if __name__ == "__main__":
    main()
