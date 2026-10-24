r"""Dump sampled PASTIS pixel embeddings (OlmoEarth v1.3 / AEF / Tessera v2) for open-set experiments.

Reads the pastis_year_aligned_oe13cdnzarr windows (OlmoEarth-v1_3-Base from the
public-CDN geozarr store, see bake_olmoearth_zarr_embeddings.py) and, for the
same windows, the ``gse`` (AEF) and ``tessera_v2`` layers of the source
pastis_year_aligned dataset. Every product is on the window's 10 m grid, so a
pixel index means the same location in all three.

Two tables are written to one .npz:

- pixels: per window, up to ``--pixels_per_class`` random pixels of every class
  present (labels 0..18; 19 is void), restricted to pixels where OlmoEarth and
  AEF are both valid. Tessera v2 is optional in the source dataset, so it has
  its own validity mask.
- segments: the mean embedding of every connected component of one class
  (4-connectivity, at least ``--min_segment_px`` valid pixels). PASTIS parcels
  are not in the dataset, so these stand in for parcels; adjacent parcels of
  the same crop merge into one segment.

Embeddings are stored as float16. Run on a Weka-mounted machine::

    python scripts/tools/dump_pastis_open_set_pixels.py --out /weka/.../pastis_open_set.npz
"""

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rslearn.dataset import Dataset, Window
from scipy import ndimage
from upath import UPath

logger = logging.getLogger(__name__)

OE_DS = "/weka/dfive-default/olmoearth/eval_datasets/pastis_year_aligned_oe13cdnzarr"
SRC_DS = "/weka/dfive-default/olmoearth/eval_datasets/pastis_year_aligned"
PRODUCTS = {"olmoearth": "olmoearth_emb", "aef": "gse", "tessera_v2": "tessera_v2"}
LABEL_LAYER = "label"
VOID = 19
NUM_CLASSES = 19


def read_layer(window: Window, layer: str) -> np.ndarray | None:
    """The (C, H, W) raster of a completed layer, or None if it is missing."""
    if not window.is_layer_completed(layer):
        return None
    tifs = sorted(Path(str(window.get_layer_dir(layer))).glob("*/geotiff.tif"))
    if len(tifs) != 1:
        raise ValueError(f"{window.name}/{layer}: expected one geotiff, got {tifs}")
    with rasterio.open(tifs[0]) as src:
        return src.read()


def valid_mask(array: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    """Pixels that are finite and not the all-band -1.0 / 0.0 nodata fill."""
    if array is None:
        return np.zeros(shape, dtype=bool)
    finite = np.isfinite(array).all(axis=0)
    fill = (array == -1.0).all(axis=0) | (array == 0.0).all(axis=0)
    return finite & ~fill


def main() -> None:
    """Sample pixels and segment means from every window and write the .npz."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--oe_ds_path", default=OE_DS)
    parser.add_argument("--src_ds_path", default=SRC_DS)
    parser.add_argument("--pixels_per_class", type=int, default=32)
    parser.add_argument("--min_segment_px", type=int, default=10)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    oe_windows = Dataset(UPath(args.oe_ds_path)).storage.get_windows()
    src_windows = {
        (w.group, w.name): w
        for w in Dataset(UPath(args.src_ds_path)).storage.get_windows()
    }
    oe_windows.sort(key=lambda w: (w.group, w.name))
    logger.info(f"{len(oe_windows)} windows")

    def handle(item: tuple[int, Window]) -> dict[str, list]:
        idx, window = item
        rng = np.random.default_rng(args.seed + idx)
        src = src_windows[(window.group, window.name)]
        label = read_layer(window, LABEL_LAYER)[0]
        arrays = {
            "olmoearth": read_layer(window, PRODUCTS["olmoearth"]),
            "aef": read_layer(src, PRODUCTS["aef"]),
            "tessera_v2": read_layer(src, PRODUCTS["tessera_v2"]),
        }
        valid = {k: valid_mask(v, label.shape) for k, v in arrays.items()}
        base = (label != VOID) & valid["olmoearth"] & valid["aef"]

        out: dict[str, list] = {"pix": [], "seg": []}
        for cls in np.unique(label[base]):
            rows, cols = np.nonzero(base & (label == cls))
            take = rng.choice(
                len(rows), min(args.pixels_per_class, len(rows)), replace=False
            )
            for r, c in zip(rows[take], cols[take]):
                out["pix"].append(
                    (idx, int(cls), int(r), int(c), bool(valid["tessera_v2"][r, c]))
                )
            components, n = ndimage.label(base & (label == cls))
            for comp in range(1, n + 1):
                m = components == comp
                size = int(m.sum())
                if size < args.min_segment_px:
                    continue
                tess_m = m & valid["tessera_v2"]
                out["seg"].append(
                    {
                        "window": idx,
                        "label": int(cls),
                        "size": size,
                        "olmoearth": arrays["olmoearth"][:, m].mean(axis=1),
                        "aef": arrays["aef"][:, m].mean(axis=1),
                        "tessera_v2": arrays["tessera_v2"][:, tess_m].mean(axis=1)
                        if tess_m.sum() >= args.min_segment_px
                        else None,
                    }
                )
        out["pix_emb"] = {
            k: np.stack([arrays[k][:, r, c] for (_, _, r, c, _) in out["pix"]])
            if out["pix"] and arrays[k] is not None
            else None
            for k in PRODUCTS
        }
        return out

    windows_meta = [
        {
            "group": w.group,
            "name": w.name,
            "split": w.options.get("eval_split", w.options.get("split")),
            "options": {k: str(v) for k, v in w.options.items()},
        }
        for w in oe_windows
    ]
    dims = {"olmoearth": 128, "aef": 64, "tessera_v2": 128}
    pix_rows: list[tuple] = []
    pix_emb: dict[str, list[np.ndarray]] = {k: [] for k in PRODUCTS}
    seg_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for done, out in enumerate(
            executor.map(handle, enumerate(oe_windows)), start=1
        ):
            n = len(out["pix"])
            pix_rows.extend(out["pix"])
            for k in PRODUCTS:
                e = out["pix_emb"][k]
                pix_emb[k].append(
                    e if e is not None else np.full((n, dims[k]), np.nan, np.float32)
                )
            seg_rows.extend(out["seg"])
            if done % 200 == 0 or done == len(oe_windows):
                logger.info(
                    f"{done}/{len(oe_windows)} windows, "
                    f"{len(pix_rows)} pixels, {len(seg_rows)} segments"
                )

    pix = np.array([r[:4] for r in pix_rows], dtype=np.int32)
    tess_valid = np.array([r[4] for r in pix_rows], dtype=bool)
    seg_emb = {
        k: np.stack(
            [
                s[k] if s[k] is not None else np.full(dims[k], np.nan, np.float32)
                for s in seg_rows
            ]
        ).astype(np.float16)
        for k in PRODUCTS
    }
    np.savez_compressed(
        args.out,
        pix_window=pix[:, 0],
        pix_label=pix[:, 1],
        pix_row=pix[:, 2],
        pix_col=pix[:, 3],
        pix_tessera_valid=tess_valid,
        **{f"pix_{k}": np.concatenate(pix_emb[k]).astype(np.float16) for k in PRODUCTS},
        seg_window=np.array([s["window"] for s in seg_rows], dtype=np.int32),
        seg_label=np.array([s["label"] for s in seg_rows], dtype=np.int32),
        seg_size=np.array([s["size"] for s in seg_rows], dtype=np.int32),
        **{f"seg_{k}": v for k, v in seg_emb.items()},
        window_split=np.array([str(m["split"]) for m in windows_meta]),
        windows_json=np.array(json.dumps(windows_meta)),
        args_json=np.array(json.dumps(vars(args))),
    )
    counts = np.bincount(pix[:, 1], minlength=NUM_CLASSES)
    splits = {
        s: int((np.array([m["split"] for m in windows_meta]) == s).sum())
        for s in {str(m["split"]) for m in windows_meta}
    }
    logger.info(
        f"wrote {args.out}: pixels per class {counts.tolist()}, "
        f"window splits {splits}, tessera-valid {tess_valid.mean():.3f}"
    )


if __name__ == "__main__":
    main()
