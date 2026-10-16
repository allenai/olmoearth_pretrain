"""Export a text-search index of OlmoEarth embeddings for the search page.

Items are the four 64 px (640 m) quadrants of every embedded window: the mean of the
quadrant's 16 tile embeddings, centered and projected by a trained aligner into the
shared text-image space, reduced by PCA and stored as int8. Classes are the aligner's
classes with their projected text embeddings in the same reduced space, so a page can
score ``query = sum(weight * class)`` against every item with one dot product.

Written to ``--out_dir``:
    items_<k>.bin int8 ``[N_windows * 4, dim]`` split into files of at most
                  ``--max_file_mb``; item ``4 * w + q`` is quadrant ``q`` (row-major) of
                  window ``w``
    latlon.bin    float32 ``[N_windows * 4, 2]`` quadrant centres
    thumbs.jpg    16 px RGB thumbnail of every window, ``--sheet_cols`` per row
    meta.json     windows (name, dataset, split, labels), classes (name, short
                  description, vector), dims and the int8 scale

Run:
  python -m olmoearth_pretrain.clipearth.export_search
      --emb_dir <dir> --aligner <run>/aligner.pt --out_dir <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from olmoearth_pretrain.clipearth.open_set import MOLMO_SET_DIR
from olmoearth_pretrain.clipearth.train import (
    CATEGORIES,
    VISDESC_KEY,
    Aligner,
    class_text_matrix,
    encode_texts,
    text_views,
)

logger = logging.getLogger(__name__)

THUMB = 16


def load_tiles(emb_dir: str) -> dict[str, np.ndarray]:
    """Quadrant embeddings and centres for every window in the embedding parts."""
    windows, splits, slugs, quads, latlon = [], [], [], [], []
    region_window, region_class = [], []
    for path in sorted(glob.glob(os.path.join(emb_dir, "*.npz"))):
        if path.endswith(".tmp.npz"):
            continue
        part = np.load(path)
        n = len(part["window"])
        if not n:
            continue
        loc = part["tile_loc"].astype(np.float32).reshape(n, 2, 4, 2, 4, -1)
        quads.append(loc.mean((2, 4)).reshape(n * 4, -1))
        ll = part["tile_lat_lon"].reshape(n, 2, 4, 2, 4, 2)
        latlon.append(ll.mean((2, 4)).reshape(n * 4, 2))
        windows += part["window"].tolist()
        splits += part["window_split"].tolist()
        slugs += part["window_slug"].tolist()
        region_window += part["region_window"].tolist()
        region_class += part["region_class"].tolist()
    return {
        "window": np.array(windows),
        "split": np.array(splits),
        "slug": np.array(slugs),
        "quad": np.concatenate(quads),
        "latlon": np.concatenate(latlon).astype(np.float32),
        "region_window": np.array(region_window),
        "region_class": np.array(region_class),
    }


def thumbnail(name: str) -> np.ndarray:
    """16 px RGB from Molmo's rendered image of the window (grey if missing)."""
    path = os.path.join(MOLMO_SET_DIR, "images", f"{name}.jpg")
    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB").resize((THUMB, THUMB), Image.BOX))
    except OSError:
        return np.full((THUMB, THUMB, 3), 128, np.uint8)


def main() -> None:
    """Build the index."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--aligner", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--sheet_cols", type=int, default=256)
    parser.add_argument("--max_file_mb", type=float, default=14)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.aligner, map_location="cpu", weights_only=False)
    train_args = ckpt["args"]
    class_names = ckpt["class_names"]
    visdesc = json.load(open(CATEGORIES)).get(VISDESC_KEY, {})
    texts: list[str] = []
    class_views: list[list[int]] = []
    for name in class_names:
        views = text_views(name, visdesc)
        class_views.append(list(range(len(texts), len(texts) + len(views))))
        texts += views
    view_emb = torch.from_numpy(encode_texts(texts, train_args["text_model"], "cpu"))

    data = load_tiles(args.emb_dir)
    logger.info(f"{len(data['window'])} windows, {len(data['quad'])} quadrants")
    model = Aligner(
        data["quad"].shape[1],
        view_emb.shape[1],
        train_args["dim"],
        train_args["hidden"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        class_emb = class_text_matrix(model, view_emb, class_views).numpy()
        quad = torch.from_numpy(data["quad"] - ckpt["image_mean"])
        items = torch.cat([model.embed_image(b) for b in quad.split(65536)]).numpy()

    # PCA on the items; classes go through the same basis.
    mean = items.mean(0)
    _, _, vt = np.linalg.svd(
        items[:: max(1, len(items) // 200000)] - mean, full_matrices=False
    )
    basis = vt[: args.dim].T
    items_r = F.normalize(torch.from_numpy((items - mean) @ basis), dim=-1).numpy()
    classes_r = F.normalize(
        torch.from_numpy((class_emb - mean) @ basis), dim=-1
    ).numpy()

    # How much the reduction changes each class's top results.
    overlaps = []
    for c in range(0, len(class_names), max(1, len(class_names) // 200)):
        full = np.argsort(-(items @ class_emb[c]))[:100]
        reduced = np.argsort(-(items_r @ classes_r[c]))[:100]
        overlaps.append(len(np.intersect1d(full, reduced)) / 100)
    logger.info(f"top-100 overlap after PCA to {args.dim}: {np.mean(overlaps):.3f}")

    scale = 127 / np.abs(items_r).max()
    packed = (items_r * scale).round().astype(np.int8)
    rows_per_file = int(args.max_file_mb * 1e6) // args.dim // 4 * 4
    item_files = []
    for k, start in enumerate(range(0, len(packed), rows_per_file)):
        item_files.append(f"items_{k}.bin")
        packed[start : start + rows_per_file].tofile(
            os.path.join(args.out_dir, item_files[-1])
        )
    data["latlon"].tofile(os.path.join(args.out_dir, "latlon.bin"))

    with ThreadPoolExecutor(64) as pool:
        thumbs = list(pool.map(thumbnail, data["window"]))
    rows = -(-len(thumbs) // args.sheet_cols)
    sheet = np.zeros((rows * THUMB, args.sheet_cols * THUMB, 3), np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, args.sheet_cols)
        sheet[r * THUMB : (r + 1) * THUMB, c * THUMB : (c + 1) * THUMB] = t
    Image.fromarray(sheet).save(os.path.join(args.out_dir, "thumbs.jpg"), quality=88)

    index = {name: i for i, name in enumerate(class_names)}
    labels: dict[str, set[int]] = {}
    for w, c in zip(data["region_window"], data["region_class"]):
        if c in index:
            labels.setdefault(str(w), set()).add(index[c])
    datasets = sorted(set(data["slug"].tolist()))
    dataset_index = {d: i for i, d in enumerate(datasets)}
    meta = {
        "dim": args.dim,
        "item_files": item_files,
        "scale": float(scale),
        "thumb": THUMB,
        "sheet_cols": args.sheet_cols,
        "pca_top100_overlap": float(np.mean(overlaps)),
        "datasets": datasets,
        "windows": [
            [str(w), dataset_index[str(s)], str(sp)[0], sorted(labels.get(str(w), ()))]
            for w, s, sp in zip(data["window"], data["slug"], data["split"])
        ],
        "classes": [
            {
                "name": name,
                "desc": (visdesc.get(name) or {}).get("short", ""),
                "vec": [round(float(v), 4) for v in classes_r[i]],
            }
            for i, name in enumerate(class_names)
        ],
    }
    json.dump(
        meta, open(os.path.join(args.out_dir, "meta.json"), "w"), separators=(",", ":")
    )
    for f in sorted(os.listdir(args.out_dir)):
        logger.info(
            f"{f}: {os.path.getsize(os.path.join(args.out_dir, f)) / 1e6:.1f} MB"
        )


if __name__ == "__main__":
    main()
