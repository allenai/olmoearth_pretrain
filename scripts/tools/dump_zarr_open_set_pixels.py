r"""Dump open-set pixel samples for a PASTIS-format dataset straight from the v1.3 geozarr.

Same output format as dump_pastis_open_set_pixels.py (per-class ``pix_*`` and
uniform ``uni_*`` tables, ``window_split``), but the OlmoEarth v1.3 embeddings
are read from the public geozarr store for each window's target year instead
of a baked layer, and the label raster is remapped with ``--class_id_mapping``
(source id -> benchmark id; unmapped ids become ``--void``). Built for
PLANTEUR (pastis2_drom_2019: 26-class source taxonomy, 8 benchmark classes,
255 void), where no baked embedding layer exists.

Only the ``olmoearth`` product is written (``pix_olmoearth``, ``uni_olmoearth``).

    uv run --with 'zarr>=3.1' --with google-crc32c --with aiohttp \
        python scripts/tools/dump_zarr_open_set_pixels.py --zarr_url ... \
        --ds_path /weka/.../pastis2_drom_2019 --label_band class_id \
        --class_id_mapping '{"0":0,"1":1,"12":2,"16":3,"19":4,"20":5,"21":6,"23":7}' \
        --out planteur_open_set.npz
"""

import argparse
import json
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
    ZARR_NODATA,
    ZarrReader,
    dequantize,
)

from olmoearth_pretrain.evals.embedding_materializer.providers import (  # noqa: E402
    get_target_year,
)

logger = logging.getLogger(__name__)


def read_label(window: Window, layer: str, band: str) -> np.ndarray:
    """The (H, W) label raster of a window."""
    path = Path(str(window.get_layer_dir(layer))) / band / "geotiff.tif"
    with rasterio.open(path) as src:
        return src.read(1)


def main() -> None:
    """Sample per-class and uniform pixels from every window and write the npz."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--zarr_url", required=True)
    parser.add_argument("--ds_path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label_layer", default="label")
    parser.add_argument("--label_band", default="label")
    parser.add_argument("--class_id_mapping", default=None, help="JSON {source id: id}")
    parser.add_argument("--void", type=int, default=255)
    parser.add_argument("--pixels_per_class", type=int, default=32)
    parser.add_argument("--uniform_per_window", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    lut = None
    if args.class_id_mapping:
        mapping = {int(k): int(v) for k, v in json.loads(args.class_id_mapping).items()}
        lut = np.full(256, args.void, dtype=np.int32)
        for k, v in mapping.items():
            lut[k] = v
    windows = Dataset(UPath(args.ds_path)).storage.get_windows()
    windows.sort(key=lambda w: (w.group, w.name))
    logger.info(f"{len(windows)} windows")
    reader = ZarrReader(args.zarr_url)

    def handle(item: tuple[int, Window]) -> dict | None:
        idx, w = item
        rng = np.random.default_rng(args.seed + idx)
        try:
            lab = read_label(w, args.label_layer, args.label_band).astype(np.int64)
            if lut is not None:
                lab = lut[np.clip(lab, 0, 255)]
            slab = np.asarray(reader.read(w, get_target_year(w)))
        except Exception as e:
            logger.warning(f"{w.group}/{w.name}: {type(e).__name__}: {e}")
            return None
        valid = ~(slab == ZARR_NODATA).all(axis=0)
        out: dict = {"pix": [], "uni": []}
        base = valid & (lab != args.void)
        for cls in np.unique(lab[base]):
            rr, cc = np.nonzero(base & (lab == cls))
            take = rng.choice(
                len(rr), min(args.pixels_per_class, len(rr)), replace=False
            )
            out["pix"] += [
                (idx, int(cls), int(r), int(c)) for r, c in zip(rr[take], cc[take])
            ]
        rr, cc = np.nonzero(valid)
        take = rng.choice(len(rr), min(args.uniform_per_window, len(rr)), replace=False)
        out["uni"] = [
            (idx, int(lab[r, c]), int(r), int(c)) for r, c in zip(rr[take], cc[take])
        ]
        for t in ("pix", "uni"):
            pts = out[t]
            out[f"{t}_emb"] = (
                dequantize(slab[:, [p[2] for p in pts], [p[3] for p in pts]]).T.astype(
                    np.float16
                )
                if pts
                else np.zeros((0, slab.shape[0]), np.float16)
            )
        out["valid_frac"] = float(valid.mean())
        return out

    rows = {"pix": [], "uni": []}
    embs = {"pix": [], "uni": []}
    valid_fracs, skipped = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for done, out in enumerate(ex.map(handle, enumerate(windows)), start=1):
            if out is None:
                skipped += 1
            else:
                valid_fracs.append(out["valid_frac"])
                for t in rows:
                    rows[t] += out[t]
                    embs[t].append(out[f"{t}_emb"])
            if done % 200 == 0 or done == len(windows):
                logger.info(
                    f"{done}/{len(windows)} windows, {len(rows['pix'])} pixels, "
                    f"{len(rows['uni'])} uniform, {skipped} skipped"
                )
    meta = [
        {
            "group": w.group,
            "name": w.name,
            "split": w.options.get("eval_split", w.options.get("split")),
            "options": {k: str(v) for k, v in w.options.items()},
        }
        for w in windows
    ]
    payload = {}
    for t in rows:
        a = np.array(rows[t], dtype=np.int64).reshape(-1, 4)
        payload.update(
            {
                f"{t}_window": a[:, 0],
                f"{t}_label": a[:, 1],
                f"{t}_row": a[:, 2],
                f"{t}_col": a[:, 3],
                f"{t}_olmoearth": np.concatenate(embs[t])
                if embs[t]
                else np.zeros((0, 128), np.float16),
            }
        )
    np.savez_compressed(
        args.out,
        **payload,
        window_split=np.array([str(m["split"]) for m in meta]),
        windows_json=np.array(json.dumps(meta)),
        args_json=np.array(json.dumps(vars(args))),
    )
    counts = np.bincount(payload["pix_label"], minlength=1)
    uni = np.bincount(payload["uni_label"].clip(0, 255), minlength=256)
    logger.info(
        f"wrote {args.out}: windows with embeddings {len(valid_fracs)}/{len(windows)} "
        f"(mean valid fraction {np.mean(valid_fracs) if valid_fracs else 0:.2f}), "
        f"pixels per class {counts.tolist()}, uniform per class "
        f"{ {int(i): int(n) for i, n in enumerate(uni) if n} }"
    )


if __name__ == "__main__":
    main()
