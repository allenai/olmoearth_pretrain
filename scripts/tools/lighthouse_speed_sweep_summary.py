"""Summarize a speed sweep: one row per run label, plus output cosines between runs.

Each run writes into ``<out_dir>/<label>/`` (``--out_dir`` of the driver) with
``--timings_name timings_<label>.json``. For pairs given as ``--compare a:b``
(``<label>/<config>/<window>`` on each side) it also reports the per-pixel cosine of
the saved ``.f16.npy`` embeddings. Writes ``summary.json``.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    """Print and save the sweep table."""
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--compare", nargs="*", default=[])
    a = p.parse_args()
    out = Path(a.out_dir)
    rows = []
    for f in sorted(out.glob("*/timings_*.json")):
        label = f.stem[len("timings_") :]
        rec = json.loads(f.read_text())
        for window, w in rec["windows"].items():
            for cfg, c in w["configs"].items():
                rows.append(
                    {
                        "label": label,
                        "window": window,
                        "config": cfg,
                        "s_per_km2": c["s_per_km2"],
                        "forward_s": c["forward_s"],
                        "torch": rec.get("torch"),
                        "gpu": rec.get("gpu"),
                    }
                )
    for r in rows:
        print(
            f"{r['window']:>14} {r['label']:<28} {r['config']:<34} "
            f"{r['s_per_km2']:.4f} s/km2  ({r['torch']}, {r['gpu']})",
            flush=True,
        )
    cosines = []
    for pair in a.compare:
        left, right = pair.split(":")
        try:
            x = np.load(out / f"{left}.f16.npy")
            y = np.load(out / f"{right}.f16.npy")
        except FileNotFoundError as e:
            print(f"compare {pair}: missing {e.filename}", flush=True)
            continue
        x, y = x.astype(np.float32), y.astype(np.float32)
        cos = (x * y).sum(-1) / np.maximum(
            np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1), 1e-8
        )
        row = {
            "pair": pair,
            "cos_mean": float(cos.mean()),
            "cos_p01": float(np.percentile(cos, 1)),
        }
        cosines.append(row)
        print(
            f"compare {pair}: cos mean {row['cos_mean']:.5f} p01 {row['cos_p01']:.4f}",
            flush=True,
        )
    (out / "summary.json").write_text(
        json.dumps({"rows": rows, "cosines": cosines}, indent=1)
    )


if __name__ == "__main__":
    main()
