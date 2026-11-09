"""Every model vs each base (v1.2 base, v1.3 base = v13_sdbprobe), per task, from existing results -> vs_base.json."""

import json

import numpy as np
from summary import HERE, VESSEL_NAME, bootstrap_ci, seagrass_rows

MODELS = list(VESSEL_NAME)
BASES = ["v12_base", "v13_sdbprobe"]


def bluetopo(bt: dict, m: str, base: str) -> dict:
    """Held-out depth R² difference of model m minus base."""
    p = bt["paired"].get(f"{m} - {base} @200000")
    if p:
        return {"d": p["r2_diff"], "ci": p["r2_diff_ci95"]}
    p = bt["paired"][f"{base} - {m} @200000"]
    return {"d": -p["r2_diff"], "ci": [-p["r2_diff_ci95"][1], -p["r2_diff_ci95"][0]]}


def main() -> None:
    """Write vs_base.json comparing every model with each base."""
    bt = json.loads(
        (HERE.parent / "bluetopo_eval" / "results_holdout.json").read_text()
    )
    vessel = json.loads((HERE / "vessel_seeds" / "results.json").read_text())
    sg = seagrass_rows()
    out = {}
    for base in BASES:
        out[base] = {}
        for m in MODELS:
            if m == base:
                continue
            e = {"bluetopo": bluetopo(bt, m, base)}
            for key, metric in (("vessel_length", "r2"), ("vessel_type", "acc")):
                x, y = (
                    np.array(vessel[VESSEL_NAME[base]][metric]),
                    np.array(vessel[VESSEL_NAME[m]][metric]),
                )
                d = y.mean() - x.mean()
                h = 1.96 * np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
                e[key] = {"d": float(d), "ci": [float(d - h), float(d + h)]}
            for method in ("linprobe", "finetune"):
                d = np.array(
                    [
                        sg[(m, f, mm, s)] - v
                        for (arm, f, mm, s), v in sg.items()
                        if arm == base and mm == method and (m, f, mm, s) in sg
                    ]
                )
                e[f"seagrass_{method}"] = {
                    "d": float(d.mean()),
                    "ci": bootstrap_ci(d),
                    "n": len(d),
                }
            out[base][m] = e
    (HERE / "vs_base.json").write_text(json.dumps(out, indent=1))
    for base, rows in out.items():
        print(f"\n== vs {base}")
        for m, e in rows.items():
            print(
                f"{m:13s}",
                "  ".join(
                    f"{k}: {v['d']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]{'*' if v['ci'][0] > 0 or v['ci'][1] < 0 else ''}"
                    for k, v in e.items()
                ),
            )


if __name__ == "__main__":
    main()
