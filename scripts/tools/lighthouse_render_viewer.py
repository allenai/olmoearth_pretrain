"""Render full-resolution PCA images + a manifest for the tiling-comparison viewer.

For every window where all requested rasters exist, writes
``<out>/<window>/<panel>.jpg`` at native resolution (one pixel per 10 m embedding)
and merges a ``<out>/manifest.json`` entry with sizes, per-panel seam metrics,
timings and run-to-run agreement.

Panels: ``s2`` (twelve-month median true colour), one per ``run:config`` (e.g.
``fastpass:tiled_ps1``), and ``rc`` (the RC's existing ``output_mlpgram1`` layer in
the AOI dataset, ws16 / overlap 4 / ps1). All embedding panels of a window share ONE
colour frame: a PCA basis and 2/98 stretch fitted on a pooled sample of this
model's panels, so a colour difference between panels is an embedding difference.
The RC is a different model with its own space, so it is drawn in its own frame.

A run's panel is skipped (and marked identical) when its int8 raster equals the
first run's for the same config, so identical rasters are stored once.

JPEG quality 92 with no chroma subsampling: subsampling would blur colour edges at
2 px, which is the scale of the seams being inspected.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from lighthouse_aoi_analysis import dequantize, s2_true_colour, seam_stats  # noqa: E402


def _pca_frame(samples: list[np.ndarray], rng: np.random.Generator) -> dict:
    x = np.concatenate(samples)
    mu = x.mean(0)
    sub = x[rng.choice(len(x), min(len(x), 120000), replace=False)] - mu
    _, svals, vt = np.linalg.svd(sub, full_matrices=False)
    basis = vt[:3].T
    proj = (x - mu) @ basis
    return {
        "mu": mu,
        "basis": basis,
        "lo": np.percentile(proj, 2, axis=0),
        "hi": np.percentile(proj, 98, axis=0),
        "explained_top3": float((svals[:3] ** 2).sum() / (svals**2).sum()),
    }


def _render(e: np.ndarray, frame: dict) -> np.ndarray:
    h, w, c = e.shape
    proj = (e.reshape(-1, c) - frame["mu"]) @ frame["basis"]
    rgb = (proj - frame["lo"]) / (frame["hi"] - frame["lo"])
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8).reshape(h, w, 3)


def _save(rgb: np.ndarray, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, quality=92, subsampling=0, optimize=True)
    return path.stat().st_size


def _read(path: str) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        q = src.read()
    return q, dequantize(q)


def main() -> None:
    """Render every complete window."""
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True, help="name=out_dir")
    p.add_argument(
        "--configs", nargs="+", default=["tiled_ps1", "tiled_ps2", "tiled_ps4"]
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--rc_layer", default="output_mlpgram1")
    p.add_argument("--out", required=True)
    p.add_argument("--windows", nargs="*", default=None)
    a = p.parse_args()
    runs = dict(r.split("=", 1) for r in a.run)
    run_names = list(runs)
    out = Path(a.out)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.setdefault("windows", {})
    timings = {}
    for name, d in runs.items():
        t = sorted(Path(d).glob("timings*.json"))
        timings[name] = json.loads(t[0].read_text()) if t else {"windows": {}}
    first = runs[run_names[0]]
    names = a.windows or sorted(
        f.stem for f in (Path(first) / a.configs[0]).glob("*.tif")
    )
    rng = np.random.default_rng(0)
    for w in names:
        paths = {(r, c): f"{runs[r]}/{c}/{w}.tif" for r in run_names for c in a.configs}
        if not all(Path(pth).exists() for pth in paths.values()):
            print("skip (incomplete)", w, flush=True)
            continue
        if (
            w in manifest["windows"]
            and manifest["windows"][w].get("complete_runs") == run_names
        ):
            print("skip (rendered)", w, flush=True)
            continue
        q, e = {}, {}
        for key, pth in paths.items():
            q[key], e[key] = _read(pth)
        h, wd, _ = e[(run_names[0], a.configs[0])].shape
        idx = rng.choice(h * wd, size=min(25000, h * wd), replace=False)
        frame = _pca_frame(
            [e[(run_names[0], c)].reshape(-1, 128)[idx] for c in a.configs], rng
        )
        panels = []
        rgb = s2_true_colour(a.dataset, w)
        panels.append(
            {
                "id": "s2",
                "label": "Sentinel-2 median",
                "file": f"{w}/s2.jpg",
                "bytes": _save(rgb, out / w / "s2.jpg"),
            }
        )
        for r in run_names:
            for c in a.configs:
                pid = f"{r}:{c}"
                entry = {
                    "id": pid,
                    "run": r,
                    "config": c,
                    "stats": seam_stats(e[(r, c)]),
                }
                base = (run_names[0], c)
                if r != run_names[0]:
                    same = bool(np.array_equal(q[(r, c)], q[base]))
                    e1, e2 = e[(r, c)], e[base]
                    cos = (e1 * e2).sum(-1) / (
                        np.linalg.norm(e1, axis=-1) * np.linalg.norm(e2, axis=-1) + 1e-8
                    )
                    entry["vs_first_run"] = {
                        "identical_int8": same,
                        "cos_mean": float(cos.mean()),
                        "cos_p01": float(np.percentile(cos, 1)),
                        "frac_px_changed": float(
                            (np.abs(q[(r, c)].astype(int) - q[base]).max(0) > 0).mean()
                        ),
                    }
                    if same:
                        entry["file"] = f"{w}/{run_names[0]}__{c}.jpg"
                        panels.append(entry)
                        continue
                fn = f"{w}/{r}__{c}.jpg"
                entry["file"] = fn
                entry["bytes"] = _save(_render(e[(r, c)], frame), out / fn)
                t = timings[r]["windows"].get(w, {}).get("configs", {}).get(c)
                if t:
                    entry["timing"] = {
                        k: t[k]
                        for k in ("forward_s", "s_per_km2", "overhead_x")
                        if k in t
                    }
                panels.append(entry)
        rc = glob.glob(
            f"{a.dataset}/windows/predict/{w}/layers/{a.rc_layer}/*/geotiff.tif"
        )
        if rc:
            _, erc = _read(rc[0])
            frc = _pca_frame([erc.reshape(-1, erc.shape[-1])[idx]], rng)
            panels.append(
                {
                    "id": "rc",
                    "label": "RC (mlpgram1) tiled ps1",
                    "file": f"{w}/rc.jpg",
                    "own_frame": True,
                    "stats": seam_stats(erc),
                    "bytes": _save(_render(erc, frc), out / w / "rc.jpg"),
                }
            )
        manifest["windows"][w] = {
            "H": h,
            "W": wd,
            "panels": panels,
            "complete_runs": run_names,
            "missing_frac": {
                r: timings[r]["windows"].get(w, {}).get("missing_frac")
                for r in run_names
            },
            "explained_top3": frame["explained_top3"],
            "gpu": timings[run_names[0]].get("gpu"),
        }
        manifest_path.write_text(json.dumps(manifest, indent=1))
        print(
            "rendered",
            w,
            sum(pp.get("bytes", 0) for pp in panels) / 1e6,
            "MB",
            flush=True,
        )


if __name__ == "__main__":
    main()
