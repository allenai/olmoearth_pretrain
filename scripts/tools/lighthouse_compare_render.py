"""Render a model-comparison viewer: one panel per run per AOI window, own colour frames.

Each run is ``label=out_dir:config`` -- the rasters ``<out_dir>/<config>/<window>.tif``
and the timings file in ``<out_dir>`` written by ``lighthouse_aoi_inference.py``.
Different runs are different models, so each panel gets its OWN PCA frame (basis and
2/98 stretch fitted on that raster): colours are comparable within a panel, not
across panels. Writes ``<out>/<window>/{s2,<label>}.jpg`` at native resolution and
``<out>/manifest.json`` with sizes, seam metrics and per-window timings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lighthouse_aoi_analysis import s2_true_colour, seam_stats  # noqa: E402
from lighthouse_render_viewer import _pca_frame, _read, _render, _save  # noqa: E402


def main() -> None:
    """Render every window that has a raster for every run."""
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True, help="label=out_dir:config")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    runs = {}
    for spec in a.run:
        label, rest = spec.split("=", 1)
        out_dir, config = rest.rsplit(":", 1)
        timings = sorted(Path(out_dir).glob("timings*.json"))
        rec = json.loads(timings[0].read_text()) if timings else {"windows": {}}
        runs[label] = (Path(out_dir), config, rec)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    first = next(iter(runs.values()))
    names = sorted(f.stem for f in (first[0] / first[1]).glob("*.tif"))
    manifest: dict = {"runs": {}, "windows": {}}
    for label, (_, config, rec) in runs.items():
        manifest["runs"][label] = {
            "config": config,
            "gpu": rec.get("gpu"),
            "torch": rec.get("torch"),
            "checkpoint": rec.get("checkpoint"),
        }
    rng = np.random.default_rng(0)
    for w in names:
        paths = {lab: d / cfg / f"{w}.tif" for lab, (d, cfg, _) in runs.items()}
        if not all(pth.exists() for pth in paths.values()):
            print("skip (incomplete)", w, flush=True)
            continue
        panels = [
            {
                "id": "s2",
                "label": "Sentinel-2 median",
                "file": f"{w}/s2.jpg",
                "bytes": _save(s2_true_colour(a.dataset, w), out / w / "s2.jpg"),
            }
        ]
        h = wd = None
        for label, pth in paths.items():
            _, e = _read(str(pth))
            h, wd = e.shape[:2]
            idx = rng.choice(h * wd, size=min(25000, h * wd), replace=False)
            frame = _pca_frame([e.reshape(-1, e.shape[-1])[idx]], rng)
            _, config, rec = runs[label]
            t = rec["windows"].get(w, {}).get("configs", {}).get(config, {})
            panels.append(
                {
                    "id": label,
                    "run": label,
                    "config": config,
                    "file": f"{w}/{label}.jpg",
                    "stats": seam_stats(e),
                    "timing": {
                        k: t[k]
                        for k in (
                            "forward_s",
                            "s_per_km2",
                            "overhead_x",
                            "halo_px",
                            "core_px",
                        )
                        if k in t
                    },
                    "explained_top3": frame["explained_top3"],
                    "bytes": _save(_render(e, frame), out / w / f"{label}.jpg"),
                }
            )
        missing = next(
            (
                r["windows"][w].get("missing_frac")
                for _, _, r in runs.values()
                if w in r["windows"]
            ),
            None,
        )
        manifest["windows"][w] = {
            "H": h,
            "W": wd,
            "panels": panels,
            "missing_frac": missing,
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print(
            "rendered",
            w,
            sum(pp.get("bytes", 0) for pp in panels) / 1e6,
            "MB",
            flush=True,
        )


if __name__ == "__main__":
    main()
