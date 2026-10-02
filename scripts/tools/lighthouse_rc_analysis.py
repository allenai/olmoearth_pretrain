"""Halo convergence and chunk seams of RC Lighthouse runs (float16 ``.npy`` outputs).

For every window and configuration written by ``lighthouse_aoi_inference.py
--save_npy``:

* agreement: per-pixel cosine against a reference configuration (by default the
  Lighthouse run with the largest halo) and against the tiled run;
* chunk seams: mean ``|E[x+1] - E[x]|`` across the chunk-core boundaries over the
  same statistic at all other positions with the same phase mod 4 (~1 = seamless);
* tile seams: the same at the tiled merge seams (``lighthouse_aoi_analysis``'s
  ``tile12``), so tiled and Lighthouse runs are scored on one statistic.

Writes ``rc_analysis.json`` in ``--out_dir`` and prints a table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lighthouse_aoi_analysis import diff_profile, tile_seam_positions  # noqa: E402
from lighthouse_aoi_inference import parse_config  # noqa: E402


def seam_ratio(prof: np.ndarray, seams: np.ndarray) -> float:
    """Mean diff at ``seams`` over non-seam diffs at the same phase mod 4."""
    if not seams.any():
        return float("nan")
    idx = np.arange(prof.size)
    phase = np.nonzero(seams)[0] % 4
    same = np.isin(idx % 4, phase) & ~seams
    return float(prof[seams].mean() / prof[same].mean())


def chunk_seam_positions(n: int, core: int) -> np.ndarray:
    """Adjacent differences that straddle a chunk-core boundary."""
    mask = np.zeros(n - 1, bool)
    mask[np.arange(core, n, core) - 1] = True
    return mask


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel cosine of two (H, W, C) rasters."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return num / np.maximum(den, 1e-8)


def main() -> None:
    """Score every configuration of every window."""
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--timings", default="timings.json")
    p.add_argument("--reference", default=None, help="config to compare against")
    a = p.parse_args()
    out = Path(a.out_dir)
    timings = json.loads((out / a.timings).read_text())
    result: dict = {}
    for name, rec in timings["windows"].items():
        cfgs = list(rec["configs"])
        embs = {
            c: np.load(out / c / f"{name}.f16.npy")
            for c in cfgs
            if (out / c / f"{name}.f16.npy").exists()
        }
        lh = [c for c in embs if parse_config(c)[0] == "lh"]
        ref = a.reference or max(
            lh, key=lambda c: rec["configs"][c].get("halo_px", 0), default=None
        )
        tiled = next((c for c in embs if parse_config(c)[0] == "tiled"), None)
        rows = {}
        for c, e in embs.items():
            info = rec["configs"][c]
            row = {
                "s_per_km2": info["s_per_km2"],
                "overhead_x": info["overhead_x"],
                "halo_px": info.get("halo_px"),
                "core_px": info.get("core_px"),
                "peak_gib": info.get("peak_gib_window"),
            }
            for axis, tag in ((1, "x"), (0, "y")):
                prof = diff_profile(e.astype(np.float32), axis)
                n = prof.size + 1
                row[f"tile_seam_{tag}"] = seam_ratio(prof, tile_seam_positions(n))
                if info.get("core_px"):
                    row[f"chunk_seam_{tag}"] = seam_ratio(
                        prof, chunk_seam_positions(n, info["core_px"])
                    )
            for other, key in ((ref, "vs_ref"), (tiled, "vs_tiled")):
                if other is not None and other != c:
                    cos = cosine(e, embs[other])
                    row[f"{key}_cos_mean"] = float(cos.mean())
                    row[f"{key}_cos_p01"] = float(np.percentile(cos, 1))
            rows[c] = row
        result[name] = {"reference": ref, "tiled": tiled, "configs": rows}
        print(f"== {name} (reference {ref})")
        keys = [
            "s_per_km2",
            "overhead_x",
            "tile_seam_x",
            "chunk_seam_x",
            "vs_ref_cos_mean",
            "vs_ref_cos_p01",
            "vs_tiled_cos_mean",
        ]
        print("config".ljust(22) + "".join(k[:14].rjust(15) for k in keys))
        for c, row in rows.items():
            vals = "".join(
                (f"{row[k]:15.4f}" if isinstance(row.get(k), float) else " " * 15)
                for k in keys
            )
            print(c.ljust(22) + vals)
    (out / "rc_analysis.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
