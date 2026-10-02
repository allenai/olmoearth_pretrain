"""Render RC Lighthouse runs into the AOI tiling viewer (one shared RC colour frame).

For each window with rasters in ``--run_dir`` (written by
``lighthouse_aoi_inference.py``: ``tiled_ps1`` and ``lh_ps1_*`` configs of a v1.3 RC
checkpoint), writes ``<out>/<window>/rc13__<config>.jpg`` at native resolution and
appends one panel per config to that window's entry in ``<out>/manifest.json``
(``run: "rc13"``). All panels share ONE PCA frame fitted on a pooled sample of every
config (plus the AOI dataset's production RC layer, rendered as ``rc13:production``),
so a colour difference between them is an embedding difference -- unlike the
existing ``rc`` panel, which has a frame of its own. Seam metrics come from
``lighthouse_aoi_analysis.seam_stats``; chunk-seam ratios, agreement and timings
from ``rc_analysis.json`` / the timings file.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lighthouse_aoi_analysis import seam_stats  # noqa: E402
from lighthouse_render_viewer import _pca_frame, _read, _render, _save  # noqa: E402


def main() -> None:
    """Render every window of the run."""
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--timings", default="timings.json")
    p.add_argument("--dataset", required=True)
    p.add_argument("--rc_layer", default="output_mlpgram1")
    p.add_argument("--out", required=True)
    p.add_argument("--run_name", default="rc13")
    a = p.parse_args()
    run_dir, out = Path(a.run_dir), Path(a.out)
    timings = json.loads((run_dir / a.timings).read_text())
    analysis_path = run_dir / "rc_analysis.json"
    analysis = json.loads(analysis_path.read_text()) if analysis_path.exists() else {}
    manifest_path = out / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"windows": {}}
    )
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for w, rec in timings["windows"].items():
        cfgs = [c for c in rec["configs"] if (run_dir / c / f"{w}.tif").exists()]
        e = {c: _read(str(run_dir / c / f"{w}.tif"))[1] for c in cfgs}
        prod = glob.glob(
            f"{a.dataset}/windows/predict/{w}/layers/{a.rc_layer}/*/geotiff.tif"
        )
        if prod:
            e["production"] = _read(prod[0])[1]
        h, wd, _ = next(iter(e.values())).shape
        idx = rng.choice(h * wd, size=min(25000, h * wd), replace=False)
        frame = _pca_frame([x.reshape(-1, x.shape[-1])[idx] for x in e.values()], rng)
        entry = manifest["windows"].setdefault(w, {"H": h, "W": wd, "panels": []})
        entry["panels"] = [pp for pp in entry["panels"] if pp.get("run") != a.run_name]
        extra = analysis.get(w, {}).get("configs", {})
        for c, x in e.items():
            fn = f"{w}/{a.run_name}__{c}.jpg"
            panel = {
                "id": f"{a.run_name}:{c}",
                "run": a.run_name,
                "config": c,
                "stats": seam_stats(x),
                "file": fn,
                "bytes": _save(_render(x, frame), out / fn),
            }
            t = rec["configs"].get(c)
            if t:
                panel["timing"] = {
                    k: t[k]
                    for k in (
                        "forward_s",
                        "s_per_km2",
                        "overhead_x",
                        "halo_px",
                        "core_px",
                    )
                    if k in t
                }
            if c in extra:
                panel["rc_analysis"] = extra[c]
            entry["panels"].append(panel)
        entry[f"{a.run_name}_explained_top3"] = frame["explained_top3"]
        entry[f"{a.run_name}_gpu"] = timings.get("gpu")
        print("rendered", w, sorted(e), flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
