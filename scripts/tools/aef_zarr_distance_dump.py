r"""AEF eval tasks from the public v1.3 geozarr, with unlabeled pools at increasing distance.

For each AEF supplemental task (point labels: one labelled centre pixel per
window), reads the OlmoEarth v1.3 embedding at every sampled window centre for
the window's label year straight from the geozarr store, so the labelled points
and the unlabeled pools come from the same source. Writes, per task:

- ``<out>/olmoearth/<task>.npz``: ``train_*`` / ``val_*`` embeddings and labels
  (the format open_set_eval_aef.py reads); windows with no embedding for their
  year are skipped and counted.
- ``<out>/pools/<task>_d<D>.npz``: ``box_olmoearth`` unlabeled pixels drawn
  uniformly within D km (Chebyshev) of a random train window, in that window's
  label year (D = 0: inside the window footprint). Read as random
  ``--block``-pixel blocks around the drawn point.

Window listing is parallel (``--list_workers``) and each split is capped at
``--max_per_split`` random windows, since the large tasks hold tens of
thousands.

    uv run --with 'zarr>=3.1' --with google-crc32c --with aiohttp \
        python scripts/tools/aef_zarr_distance_dump.py --zarr_url ... --out /weka/.../aef_zarr
"""

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rslearn.dataset import Dataset, Window
from upath import UPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bake_olmoearth_zarr_embeddings import (  # noqa: E402
    RESOLUTION,
    ZARR_NODATA,
    ZarrReader,
    dequantize,
)

from olmoearth_pretrain.evals.embedding_materializer.providers import (  # noqa: E402
    get_target_year,
)

logger = logging.getLogger(__name__)

ROOT = "/weka/dfive-default/olmoearth/eval_datasets"
TASKS = [
    "africa_crop_mask",
    "canada_crops_coarse",
    "canada_crops_fine",
    "descals",
    "ethiopia_crops",
    "glance",
    "lcmap_lu",
    "us_trees",
]


def centre_label(window: Window) -> int | None:
    """The label at the window centre from its label_raster layer."""
    tifs = sorted(Path(str(window.get_layer_dir("label_raster"))).glob("*/*.tif"))
    if not tifs:
        return None
    with rasterio.open(tifs[0]) as src:
        a = src.read(1)
    return int(a[a.shape[0] // 2, a.shape[1] // 2])


def main() -> None:
    """Dump labelled points and distance pools for every task."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--zarr_url", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--max_per_split", type=int, default=5000)
    parser.add_argument("--buffers_km", type=float, nargs="+", default=[0, 1, 5, 20])
    parser.add_argument("--pool_pixels", type=int, default=10000)
    parser.add_argument("--block", type=int, default=32)
    parser.add_argument("--per_block", type=int, default=8)
    parser.add_argument("--list_workers", type=int, default=32)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--nodata_label", type=int, default=255)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    (out / "olmoearth").mkdir(parents=True, exist_ok=True)
    (out / "pools").mkdir(parents=True, exist_ok=True)
    reader = ZarrReader(args.zarr_url)
    rng = np.random.default_rng(args.seed)

    def centre_embedding(w: Window, year: int) -> np.ndarray | None:
        """Dequantized (128,) embedding at the window centre, or None if nodata."""
        emb, yrs, x0, y0 = reader._zone(w.projection.crs.to_epsg())
        if year not in yrs:
            return None
        left, top, right, bottom = w.bounds
        c = (left + right) // 2 - round(x0 / RESOLUTION)
        r = (top + bottom) // 2 + round(y0 / RESOLUTION)
        v = np.asarray(emb[yrs.index(year), :, r, c])
        return None if (v == ZARR_NODATA).all() else dequantize(v)

    for task in args.tasks:
        ds = f"{ROOT}/{task}_year_aligned"
        windows = Dataset(UPath(ds)).storage.get_windows(workers=args.list_workers)
        by_split: dict[str, list[Window]] = {}
        for w in windows:
            by_split.setdefault(
                str(w.options.get("eval_split", w.options.get("split"))), []
            ).append(w)
        logger.info(
            f"{task}: {len(windows)} windows, splits "
            f"{ {k: len(v) for k, v in by_split.items()} }"
        )
        payload: dict[str, np.ndarray] = {"dataset": np.array(f"{task}_year_aligned")}
        train_windows: list[tuple[Window, int]] = []
        for split in ("train", "val"):
            ws = by_split.get(split, [])
            if len(ws) > args.max_per_split:
                ws = [
                    ws[i]
                    for i in rng.choice(len(ws), args.max_per_split, replace=False)
                ]

            def point(w: Window) -> tuple | None:
                try:
                    year = get_target_year(w)
                    lab = centre_label(w)
                    if lab is None or lab == args.nodata_label or lab < 0:
                        return None
                    e = centre_embedding(w, year)
                    return None if e is None else (e, lab, w, year)
                except Exception as exc:  # report, keep going
                    logger.warning(f"{task}/{w.name}: {type(exc).__name__}: {exc}")
                    return None

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                pts = [p for p in ex.map(point, ws) if p is not None]
            payload[f"{split}_embeddings"] = (
                np.stack([p[0] for p in pts]).astype(np.float32)
                if pts
                else np.zeros((0, 128))
            )
            payload[f"{split}_labels"] = np.array([p[1] for p in pts], dtype=np.int64)
            if split == "train":
                train_windows = [(p[2], p[3]) for p in pts]
            logger.info(
                f"{task} {split}: {len(pts)}/{len(ws)} windows with an embedding"
            )
        np.savez_compressed(out / "olmoearth" / f"{task}.npz", **payload)
        if not train_windows:
            continue

        for d in args.buffers_km:
            buf = int(round(d * 1000 / RESOLUTION))
            n_blocks = int(np.ceil(args.pool_pixels / args.per_block * 1.4))
            picks = rng.integers(0, len(train_windows), n_blocks)

            def block(i: int, seed: int) -> np.ndarray | None:
                w, year = train_windows[i]
                left, top, right, bottom = w.bounds
                brng = np.random.default_rng(seed)
                c = int(brng.integers(left - buf, right + buf))
                r = int(brng.integers(top - buf, bottom + buf))
                emb, yrs, x0, y0 = reader._zone(w.projection.crs.to_epsg())
                if year not in yrs:
                    return None
                half = args.block // 2
                cols = np.arange(c - half, c + half)
                rows = np.arange(r - half, r + half)
                r0 = rows[0] + round(y0 / RESOLUTION)
                c0 = cols[0] - round(x0 / RESOLUTION)
                try:
                    slab = np.asarray(
                        emb[
                            yrs.index(year),
                            :,
                            r0 : r0 + args.block,
                            c0 : c0 + args.block,
                        ]
                    )
                except Exception:
                    return None
                ok = ~(slab == ZARR_NODATA).all(axis=0)
                # within the buffer of this window (Chebyshev, pixels)
                rr, cc = np.meshgrid(
                    rows[: slab.shape[1]], cols[: slab.shape[2]], indexing="ij"
                )
                dx = np.maximum(np.maximum(left - cc, cc - (right - 1)), 0)
                dy = np.maximum(np.maximum(top - rr, rr - (bottom - 1)), 0)
                ok &= np.maximum(dx, dy) <= buf
                yy, xx = np.nonzero(ok)
                if len(yy) == 0:
                    return None
                take = brng.choice(len(yy), min(args.per_block, len(yy)), replace=False)
                return dequantize(slab[:, yy[take], xx[take]]).T.astype(np.float16)

            seeds = rng.integers(1 << 31, size=n_blocks)
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                parts = [p for p in ex.map(block, picks, seeds) if p is not None]
            pool = np.concatenate(parts) if parts else np.zeros((0, 128), np.float16)
            pool = pool[: args.pool_pixels]
            np.savez_compressed(
                out / "pools" / f"{task}_d{d:g}.npz", box_olmoearth=pool
            )
            logger.info(f"{task} pool d{d:g} km: {len(pool)} pixels")


if __name__ == "__main__":
    main()
