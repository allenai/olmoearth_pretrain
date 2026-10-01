"""Print a timing table from ``lighthouse_aoi_inference.py``'s timings.json.

Per configuration: total model-forward seconds, seconds per km^2 of output, the
processed/output pixel ratio (tiled overlap or Lighthouse halo), seconds per
processed km^2 (the cost with that overhead divided out), and peak GPU memory.
"""

import json
import sys
from collections import defaultdict


def main() -> None:
    """Summarize one timings.json (path as the only argument)."""
    rec = json.load(open(sys.argv[1]))
    print(f"gpu={rec['gpu']} torch={rec['torch']} tiled={rec.get('tiled_attention')}")
    tot: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for name, w in rec["windows"].items():
        km2 = w["H"] * w["W"] / 1e4
        for cfg, i in w["configs"].items():
            t = tot[cfg]
            t["s"] += i["forward_s"]
            t["km2"] += km2
            t["proc_km2"] += i["processed_px"] / 1e4
            t["peak"] = max(t["peak"], i.get("peak_gib_window", 0.0))
            print(
                f"  {name:28s} {cfg:10s} {i['forward_s']:9.2f} s "
                f"{i['s_per_km2']:8.3f} s/km2 {i['overhead_x']:5.2f}x"
            )
    print(
        f"{'config':10s} {'total s':>9s} {'s/km2':>8s} {'overhead':>8s} {'s/proc km2':>10s} {'peak GiB':>8s}"
    )
    for cfg, t in tot.items():
        print(
            f"{cfg:10s} {t['s']:9.1f} {t['s'] / t['km2']:8.3f} "
            f"{t['proc_km2'] / t['km2']:7.2f}x {t['s'] / t['proc_km2']:10.3f} {t['peak']:8.1f}"
        )


if __name__ == "__main__":
    main()
