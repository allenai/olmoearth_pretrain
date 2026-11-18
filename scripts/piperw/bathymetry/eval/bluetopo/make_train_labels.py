"""Split BlueTopo samples by 1-degree cell into training labels and a held-out eval set.

30% of the cells containing a BlueTopo sample (seed 0) are held out: they never get a
training label, and probe.py --holdout scores only on them. The rest become the dataset's
bluetopo_dir: idx.npy (sorted sample indices) and depth_dm.npy (int16 [M, 128, 128], depth in
decimetres, -1 = no label). A pixel is labelled if BlueTopo has it below 0 m and at most
MAX_DEPTH deep; samples whose finest tile is coarser than 8 m are dropped.

  python make_train_labels.py [labels file] [out dir]
"""

import json
import sys
from pathlib import Path

import numpy as np
from probe import H5, MAX_TILE_RES

HERE = Path(
    "/weka/dfive-default/agent/piperbots/2026-10-29-piperw-bathymetry-direct-supervision/bluetopo_eval"
)
HOLDOUT_FRACTION = 0.3
MAX_DEPTH = 40.0


def cells_of(idx: np.ndarray) -> np.ndarray:
    """1-degree cell id of each sample."""
    latlon = np.load(H5 / "latlon_distribution.npy")[idx]
    return np.floor(latlon[:, 0]).astype(int) * 1000 + np.floor(latlon[:, 1]).astype(
        int
    )


def holdout_cells() -> set[int]:
    """Fixed by the full BlueTopo sample list, so every caller gets the same split."""
    path = HERE / "holdout_cells.json"
    if path.exists():
        return set(json.loads(path.read_text()))
    all_idx = np.load(HERE.parent / "noaa" / "bluetopo_sample_idx.npy")
    cells = np.unique(cells_of(all_idx))
    rng = np.random.default_rng(0)
    held = sorted(
        int(c)
        for c in rng.choice(cells, round(HOLDOUT_FRACTION * len(cells)), replace=False)
    )
    path.write_text(json.dumps(held))
    return set(held)


def main() -> None:
    """Write BlueTopo training labels, excluding held-out cells."""
    labels = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "labels_all.npz"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "bluetopo_train"
    held = holdout_cells()
    lab = np.load(labels)
    idx = lab["idx"]
    train = ~np.isin(cells_of(idx), list(held)) & (lab["res"] <= MAX_TILE_RES)
    elev = lab["elev"][train].astype(np.float32)
    depth = -elev
    valid = np.isfinite(depth) & (depth > 0) & (depth <= MAX_DEPTH)
    depth_dm = np.where(valid, np.round(depth * 10), -1).astype(np.int16)
    order = np.argsort(idx[train])
    out.mkdir(exist_ok=True)
    np.save(out / "idx.npy", idx[train][order].astype(np.int64))
    np.save(out / "depth_dm.npy", depth_dm[order])
    stats = {
        "labels": str(labels),
        "holdout_cells": len(held),
        "n_samples_in": int(len(idx)),
        "n_train_samples": int(train.sum()),
        "labelled_pixel_fraction": float(valid.mean()),
        "train_samples_with_1pct_labelled": int((valid.mean((1, 2)) >= 0.01).sum()),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
