"""Is BlueTopo aligned with the S2 pixels?

Agreement of S2 NDWI water with BlueTopo elev < 0 under integer pixel shifts of the label grid; the peak should be at (0, 0).

  python check_alignment.py [suffix, e.g. _40]
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(
    "/weka/dfive-default/agent/piperbots/2026-10-29-piperw-bathymetry-direct-supervision/bluetopo_eval"
)
S = 4

suffix = sys.argv[1] if len(sys.argv) > 1 else ""
lab = np.load(HERE / f"labels{suffix}.npz")
inp = np.load(HERE / f"inputs{suffix}.npz")
assert (lab["idx"] == inp["idx"]).all()
elev = lab["elev"].astype(np.float32)
s2 = inp["s2_median"].astype(np.float32)
g, n = s2[..., 1], s2[..., 3]
water = (g - n) / (g + n) > 0

scores = np.zeros((2 * S + 1, 2 * S + 1))
for dy in range(-S, S + 1):
    for dx in range(-S, S + 1):
        e = np.roll(elev, (dy, dx), axis=(1, 2))[:, S:-S, S:-S]
        w = water[:, S:-S, S:-S]
        ok = np.isfinite(e) & np.isfinite(g[:, S:-S, S:-S])
        bw = e < 0
        # balanced accuracy, per sample, over samples with both classes present
        accs = []
        for k in range(len(e)):
            m = ok[k]
            if bw[k][m].all() or not bw[k][m].any() or m.sum() < 500:
                continue
            accs.append(0.5 * (w[k][m & bw[k]].mean() + (~w[k])[m & ~bw[k]].mean()))
        scores[dy + S, dx + S] = np.mean(accs) if accs else np.nan
print(f"samples with land and water in BlueTopo: {len(accs)}")
print("rows: label shifted down by dy (-S..S); cols: right by dx")
print(np.round(scores, 3))
dy, dx = np.unravel_index(np.nanargmax(scores), scores.shape)
print(f"best shift dy={dy - S}, dx={dx - S}")
