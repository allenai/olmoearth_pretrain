"""Direct 2 px token-lattice ratio for every run raster, comparable with the seam ratio.

For an embedding raster E (H, W, C), d(i) = mean |E[..., i+1] - E[..., i]| along an
axis. At token patch size 2, the boundaries between 2 x 2 px token cells are the odd
differences (between pixels 2k+1 and 2k+2). The ratio is

    mean(d at odd i) / mean(d at even i)

over positions that are not tiling merge seams (excluded so the tiled v1.3 run's 12 px
seams cannot leak in). It has the same form as the tile-seam ratio (boundaries over
comparable non-boundaries), so the two are directly comparable; 1.00 = no lattice.

Each run is ``label=out_dir:config`` or ``label=layer:<name>`` (as in
``lighthouse_compare_render.py``). Writes ``--out`` JSON: {window: {label: {x, y}}}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lighthouse_aoi_analysis import diff_profile, tile_seam_positions  # noqa: E402
from lighthouse_render_viewer import _read  # noqa: E402


def lattice2(e: np.ndarray) -> dict[str, float]:
    """Odd-over-even adjacent-difference ratio per axis, tile seams excluded."""
    out = {}
    for axis, tag in ((1, "x"), (0, "y")):
        prof = diff_profile(e, axis)
        keep = ~tile_seam_positions(prof.size + 1)
        idx = np.arange(prof.size)
        odd = prof[(idx % 2 == 1) & keep].mean()
        even = prof[(idx % 2 == 0) & keep].mean()
        out[tag] = float(odd / even)
    return out


def main() -> None:
    """Compute the ratio for every window of every run."""
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    runs: dict[str, tuple[str, str]] = {}
    for spec in a.run:
        label, rest = spec.split("=", 1)
        if rest.startswith("layer:"):
            runs[label] = ("layer", rest[len("layer:") :])
        else:
            out_dir, config = rest.rsplit(":", 1)
            runs[label] = (out_dir, config)
    first = next(v for v in runs.values() if v[0] != "layer")
    names = sorted(f.stem for f in (Path(first[0]) / first[1]).glob("*.tif"))
    result: dict[str, dict[str, dict[str, float]]] = {}
    for w in names:
        result[w] = {}
        for label, (src, cfg) in runs.items():
            if src == "layer":
                hits = sorted(
                    (Path(a.dataset) / "windows" / "predict" / w / "layers" / cfg).glob(
                        "*/geotiff.tif"
                    )
                )
                path = hits[0] if hits else None
            else:
                path = Path(src) / cfg / f"{w}.tif"
            if path is None or not path.exists():
                continue
            result[w][label] = lattice2(_read(str(path))[1])
        print(
            w,
            {k: (round(v["x"], 3), round(v["y"], 3)) for k, v in result[w].items()},
            flush=True,
        )
    Path(a.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
