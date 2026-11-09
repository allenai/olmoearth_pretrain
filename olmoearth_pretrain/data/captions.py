"""Grid captions for pretraining windows, and the caption_cell raster that carries them.

Each window has one caption for the whole window and one per cell of a 4 x 4 grid (row 0
north, column 0 west); an empty caption has nothing to say and is masked. The captions sit
outside the H5 files in a CaptionStore, and the dataset adds a caption_cell raster instead:
per pixel, the window id and the index of the cell the pixel falls in. Cropping, CutMix and
flips then move caption positions along with the imagery, and patch_caption_keys reads the
caption each token should match.

Cell captions don't mention their position, so they survive flips. Window captions use
compass directions and describe the whole window, so they only fit an unflipped full window.

Build a store from the caption JSONL files ({"id", "window", "cells"} per line):

    python -m olmoearth_pretrain.data.captions <captions_dir> <out_dir> --h5py_dir <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from olmoearth_pretrain.data.constants import MISSING_VALUE

GRID = 4
# The window id travels as float32, which holds integers exactly below 2**24.
MAX_WINDOWS = 2**24


class CaptionStore:
    """Captions by window id, memory-mapped so dataloader workers share them."""

    def __init__(self, path: str):
        """Open the store written by build_caption_store."""
        self.path = path
        self.meta = json.load(open(os.path.join(path, "meta.json")))
        self._text: np.memmap | None = None
        self._bounds: np.ndarray | None = None

    def _open(self) -> tuple[np.memmap, np.ndarray]:
        # Opened on first use so each worker maps the files itself.
        if self._text is None or self._bounds is None:
            self._text = np.memmap(os.path.join(self.path, "text.bin"), mode="r")
            self._bounds = np.load(os.path.join(self.path, "bounds.npy"), mmap_mode="r")
        return self._text, self._bounds

    def __getstate__(self) -> dict:
        """Pickle without the memory maps."""
        return {**self.__dict__, "_text": None, "_bounds": None}

    def __len__(self) -> int:
        """The number of window ids the store covers, captioned or not."""
        return self.meta["num_windows"]

    def get(self, window_id: int) -> tuple[str, list[str]]:
        """The window caption and the cell captions, row-major; "" means masked."""
        text, bounds = self._open()
        b = bounds[window_id]
        parts = [bytes(text[b[k] : b[k + 1]]).decode() for k in range(GRID * GRID + 1)]
        return parts[0], parts[1:]

    def cells_present(self, window_id: int) -> np.ndarray:
        """Which cells have a caption, row-major, without decoding any text."""
        _, bounds = self._open()
        b = bounds[window_id]
        return b[2:] > b[1:-1]


def build_caption_store(
    captions_dir: str, out_dir: str, h5py_dir: str, num_windows: int
) -> None:
    """Write text.bin (UTF-8 captions back to back) and bounds.npy ([num_windows, 18])."""
    if num_windows > MAX_WINDOWS:
        raise ValueError(f"{num_windows} windows don't fit a float32 window id")
    os.makedirs(out_dir, exist_ok=True)
    rows: dict[int, list[bytes]] = {}
    for path in sorted(glob.glob(os.path.join(captions_dir, "*.jsonl"))):
        for line in open(path):
            r = json.loads(line)
            if len(r["cells"]) != GRID * GRID:
                raise ValueError(f"window {r['id']} has {len(r['cells'])} cells")
            rows[r["id"]] = [s.encode() for s in [r["window"], *r["cells"]]]
    if rows and max(rows) >= num_windows:
        raise ValueError(f"window id {max(rows)} is beyond {num_windows} windows")

    sizes = np.zeros((num_windows, GRID * GRID + 1), dtype=np.int64)
    for i, parts in rows.items():
        sizes[i] = [len(p) for p in parts]
    bounds = np.zeros((num_windows, GRID * GRID + 2), dtype=np.int64)
    ends = np.cumsum(sizes.ravel()).reshape(sizes.shape)
    bounds[:, 1:] = ends
    bounds[1:, 0] = ends[:-1, -1]

    with open(os.path.join(out_dir, "text.bin.tmp"), "wb") as f:
        for i in range(num_windows):
            for p in rows.get(i, ()):
                f.write(p)
    np.save(os.path.join(out_dir, "bounds.npy"), bounds)
    os.replace(os.path.join(out_dir, "text.bin.tmp"), os.path.join(out_dir, "text.bin"))
    meta = {"h5py_dir": h5py_dir, "num_windows": num_windows, "captioned": len(rows)}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)


def caption_cell_raster(
    window_id: int, present: np.ndarray, height: int, width: int
) -> np.ndarray:
    """[H, W, 1, 2] float32: window id and cell index per pixel, MISSING_VALUE where masked."""
    rows = np.arange(height) * GRID // height
    cols = np.arange(width) * GRID // width
    cell = rows[:, None] * GRID + cols[None, :]
    out = np.empty((height, width, 1, 2), dtype=np.float32)
    out[..., 0, 0] = window_id
    out[..., 0, 1] = cell
    out[~present[cell]] = MISSING_VALUE
    return out


def patch_caption_keys(caption_cell: torch.Tensor, patch_size: int) -> torch.Tensor:
    """[B, H, W, 1, 2] -> [B, H/p, W/p, 2] (window id, cell) at each patch's centre pixel.

    A patch can straddle two cells when the crop isn't aligned to the grid; its centre
    decides. Rows with MISSING_VALUE have no caption to match.
    """
    c = patch_size // 2
    return caption_cell[:, c::patch_size, c::patch_size, 0, :]


def main() -> None:
    """Build a CaptionStore from the command line."""
    p = argparse.ArgumentParser()
    p.add_argument("captions_dir")
    p.add_argument("out_dir")
    p.add_argument("--h5py_dir", required=True)
    p.add_argument("--num_windows", type=int)
    a = p.parse_args()
    num_windows = a.num_windows or len(
        glob.glob(os.path.join(a.h5py_dir, "sample_*.h5"))
    )
    build_caption_store(a.captions_dir, a.out_dir, a.h5py_dir, num_windows)


if __name__ == "__main__":
    main()
