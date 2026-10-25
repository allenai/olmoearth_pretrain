"""Optional dump of the embeddings a probe sees, for offline analyses.

Enabled by ``OE_DUMP_EMBEDDINGS_DIR`` (see evaluator_callback). Classification
tasks are saved whole: ``<split>_embeddings`` / ``<split>_labels``. Dense
(segmentation) tasks would be far too large, so they are subsampled per tile in
the format of scripts/tools/dump_pastis_open_set_pixels.py:

- ``pix_*``: up to ``OE_DUMP_PER_CLASS`` pixels of every labelled class present
  in each tile;
- ``uni_*``: ``OE_DUMP_UNIFORM`` uniformly drawn pixels per tile, any label;
- ``<table>_window``: a window id (tile index // ``OE_DUMP_TILES_PER_WINDOW``,
  so the tiles of one stored window share an id), and ``window_split``.

The product key is ``olmoearth`` (``pix_olmoearth``, ``uni_olmoearth``), so the
open-set scripts read it like the PASTIS dump.
"""

import logging
import os

import numpy as np
import torch
from upath import UPath

logger = logging.getLogger(__name__)


def _as_np(t: torch.Tensor | None) -> np.ndarray:
    """CPU numpy copy; empty when the split was not computed."""
    return np.zeros(0) if t is None else t.detach().float().cpu().numpy()


def _dense_tables(
    splits: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]],
    num_classes: int,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Per-class and uniform pixel samples from (N, H, W, D) embeddings."""
    per_class = int(os.environ.get("OE_DUMP_PER_CLASS", "4"))
    uniform = int(os.environ.get("OE_DUMP_UNIFORM", "4"))
    tiles_per_window = int(os.environ.get("OE_DUMP_TILES_PER_WINDOW", "1"))
    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {"pix": [], "uni": []}
    embs: dict[str, list] = {"pix": [], "uni": []}
    window_split: list[str] = []
    window_base = 0
    for split, (emb, lab) in splits.items():
        if emb is None or lab is None:
            continue
        e, y = _as_np(emb), lab.detach().cpu().numpy()
        if e.ndim != 4:
            raise ValueError(f"expected (N, H, W, D) embeddings, got {e.shape}")
        if y.shape[1:3] != e.shape[1:3]:
            # labels at a finer grid than the embeddings: take the pixel centres
            sh, sw = y.shape[1] // e.shape[1], y.shape[2] // e.shape[2]
            y = y[:, sh // 2 :: sh, sw // 2 :: sw]
        n_windows = int(np.ceil(len(e) / tiles_per_window))
        window_split += [split] * n_windows
        for i in range(len(e)):
            wid = window_base + i // tiles_per_window
            yi = y[i]
            known = (yi >= 0) & (yi < num_classes)
            for c in np.unique(yi[known]):
                rr, cc = np.nonzero(yi == c)
                take = rng.choice(len(rr), min(per_class, len(rr)), replace=False)
                rows["pix"] += [(wid, int(c))] * len(take)
                embs["pix"].append(e[i, rr[take], cc[take]])
            flat = rng.choice(yi.size, min(uniform, yi.size), replace=False)
            rr, cc = np.unravel_index(flat, yi.shape)
            rows["uni"] += [(wid, int(yi[r, c_])) for r, c_ in zip(rr, cc)]
            embs["uni"].append(e[i, rr, cc])
        window_base += n_windows
    out = {"window_split": np.array(window_split)}
    for t in ("pix", "uni"):
        a = np.array(rows[t], dtype=np.int64).reshape(-1, 2)
        out[f"{t}_window"] = a[:, 0]
        out[f"{t}_label"] = a[:, 1]
        out[f"{t}_olmoearth"] = np.concatenate(embs[t]).astype(np.float16)
    return out


def dump_probe_embeddings(
    dump_dir: str,
    dataset: str,
    evaluation_name: str,
    splits: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]],
    num_classes: int,
) -> None:
    """Write the probe's embeddings for one task to ``<dump_dir>/<name>.npz``."""
    path = UPath(dump_dir) / f"{evaluation_name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    first = next(e for e, _ in splits.values() if e is not None)
    if first.ndim == 4:
        payload = _dense_tables(splits, num_classes)
    else:
        payload = {
            f"{split}_{kind}": _as_np(t)
            for split, (emb, lab) in splits.items()
            for kind, t in (("embeddings", emb), ("labels", lab))
        }
    with path.open("wb") as f:
        np.savez(
            f,
            dataset=np.array(dataset),
            evaluation_name=np.array(evaluation_name),
            **payload,
        )
    logger.info(f"Dumped {evaluation_name} embeddings to {path}")
