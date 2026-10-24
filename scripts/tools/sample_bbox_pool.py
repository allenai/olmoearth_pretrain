r"""Sample an unlabeled pool from anywhere in the PASTIS region, not just its tiles.

The open-set dump's uniform sample only covers PASTIS tiles, which sit on
declared farmland. This draws unlabeled OlmoEarth v1.3 embeddings from the
public geozarr store anywhere inside the bounding box of the PASTIS tiles in
each UTM zone (towns, forest, water included), for the year the tiles use.

To keep reads cheap it reads random ``--block`` x ``--block`` blocks and keeps
``--per_block`` random valid pixels from each. Pixels inside a val or test
tile are dropped so the evaluation stays clean; pixels inside train tiles are
kept and flagged. Zones get blocks in proportion to their bounding-box area.

Writes an npz with ``box_olmoearth`` (float16, dequantized like the bake
script), ``box_epsg``, ``box_col``, ``box_row`` (pixel coordinates in the
zone's 10 m grid) and ``box_in_train_tile``.

    uv run --with 'zarr>=3.1' --with google-crc32c --with aiohttp \
        python scripts/tools/sample_bbox_pool.py --zarr_url ... --out bbox_pool.npz
"""

import argparse
import logging
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from rslearn.dataset import Dataset
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

OE_DS = "/weka/dfive-default/olmoearth/eval_datasets/pastis_year_aligned_oe13cdnzarr"


def main() -> None:
    """Sample blocks inside each zone's PASTIS bounding box and write the pool."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--zarr_url", required=True)
    parser.add_argument("--ds_path", default=OE_DS)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_pixels", type=int, default=100000)
    parser.add_argument("--block", type=int, default=64)
    parser.add_argument("--per_block", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    windows = Dataset(UPath(args.ds_path)).storage.get_windows()
    by_zone: dict[int, list] = defaultdict(list)
    years = set()
    for w in windows:
        by_zone[w.projection.crs.to_epsg()].append(w)
        years.add(get_target_year(w))
    if len(years) != 1:
        raise ValueError(f"expected one target year, got {years}")
    year = years.pop()
    zones = {}
    for epsg, ws in by_zone.items():
        b = np.array([w.bounds for w in ws])
        split = [w.options.get("eval_split", w.options.get("split")) for w in ws]
        zones[epsg] = {
            "bbox": (b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()),
            "held_out": b[[s in ("val", "test") for s in split]],
            "train": b[[s == "train" for s in split]],
        }
        x0, y0, x1, y1 = zones[epsg]["bbox"]
        logger.info(
            f"EPSG:{epsg}: {len(ws)} tiles, bbox {(x1 - x0) * RESOLUTION / 1000:.0f} x "
            f"{(y1 - y0) * RESOLUTION / 1000:.0f} km"
        )
    areas = {
        e: (z["bbox"][2] - z["bbox"][0]) * (z["bbox"][3] - z["bbox"][1])
        for e, z in zones.items()
    }
    total = sum(areas.values())
    n_blocks = int(
        np.ceil(args.n_pixels / args.per_block * 1.3)
    )  # headroom for nodata / held-out
    plan = []
    for epsg, z in zones.items():
        x0, y0, x1, y1 = z["bbox"]
        n = int(round(n_blocks * areas[epsg] / total))
        cols = rng.integers(x0, x1 - args.block, n)
        rows = rng.integers(y0, y1 - args.block, n)
        plan += [
            (epsg, int(c), int(r), int(rng.integers(1 << 31)))
            for c, r in zip(cols, rows)
        ]
    logger.info(f"year {year}: reading {len(plan)} blocks of {args.block} px")

    reader = ZarrReader(args.zarr_url)

    def inside(boxes: np.ndarray, col: np.ndarray, row: np.ndarray) -> np.ndarray:
        """Whether each (col, row) pixel falls in any of the [x0, y0, x1, y1) boxes."""
        if len(boxes) == 0:
            return np.zeros(len(col), dtype=bool)
        c, r = col[:, None], row[:, None]
        return (
            (c >= boxes[:, 0])
            & (c < boxes[:, 2])
            & (r >= boxes[:, 1])
            & (r < boxes[:, 3])
        ).any(axis=1)

    def read_block(item: tuple) -> tuple | None:
        """Up to per_block valid, non-held-out pixels from one block."""
        epsg, col0, row0, seed = item
        emb, yrs, x0, y0 = reader._zone(epsg)
        r0 = row0 + round(y0 / RESOLUTION)
        c0 = col0 - round(x0 / RESOLUTION)
        slab = np.asarray(
            emb[yrs.index(year), :, r0 : r0 + args.block, c0 : c0 + args.block]
        )
        valid = ~(slab == ZARR_NODATA).all(axis=0)
        rr, cc = np.nonzero(valid)
        if len(rr) == 0:
            return None
        col, row = cc + col0, rr + row0
        keep = ~inside(zones[epsg]["held_out"], col, row)
        rr, cc, col, row = rr[keep], cc[keep], col[keep], row[keep]
        if len(rr) == 0:
            return None
        take = np.random.default_rng(seed).choice(
            len(rr), min(args.per_block, len(rr)), replace=False
        )
        x = dequantize(slab[:, rr[take], cc[take]]).T
        return (
            x.astype(np.float16),
            np.full(len(take), epsg),
            col[take],
            row[take],
            inside(zones[epsg]["train"], col[take], row[take]),
        )

    parts = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(read_block, plan), start=1):
            if res is not None:
                parts.append(res)
            if i % 500 == 0:
                logger.info(
                    f"{i}/{len(plan)} blocks, {sum(len(p[0]) for p in parts)} pixels"
                )
    cat = [np.concatenate([p[k] for p in parts]) for k in range(5)]
    n = min(args.n_pixels, len(cat[0]))
    sel = rng.choice(len(cat[0]), n, replace=False)
    np.savez_compressed(
        args.out,
        box_olmoearth=cat[0][sel],
        box_epsg=cat[1][sel],
        box_col=cat[2][sel],
        box_row=cat[3][sel],
        box_in_train_tile=cat[4][sel],
        year=np.array(year),
    )
    logger.info(
        f"wrote {args.out}: {n} pixels, {cat[4][sel].mean():.1%} inside train tiles, "
        f"zones {sorted(set(cat[1][sel].tolist()))}"
    )


if __name__ == "__main__":
    main()
